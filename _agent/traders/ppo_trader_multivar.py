
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils
from _utils.drl_utils import robust_argmax
from _utils.drl_utils import *
import asyncio
from matplotlib import pyplot as plt
import sqlalchemy
from sqlalchemy import MetaData, Column
import dataset
import ast

from itertools import product
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras as k
import tensorflow_probability as tfp
tf.get_logger().setLevel(3)

async def scale_from_dist_to_action(dist_action, a_min, a_max):
    scaled_action = a_min + (dist_action * (a_max - a_min))
    return scaled_action

async def scale_from_action_to_dist(scaled_action, a_min, a_max):

    dist_action = (scaled_action - a_min) / (a_max - a_min)
    return dist_action


async def pretend_greedy_policy(next_load, next_generation, distribution, actions):
    # ToDo: add capabilities and defaults for other action dimensions here!!
    battery_target = -(next_load - next_generation)
    target_action = {}
    target_action['storage'] = battery_target

    # rescale to [-1...1] and then
    dist_action = await scale_from_action_to_dist(battery_target, a_min=actions["storage"]['min'],
                                                  a_max=actions["storage"]['max'])
    dist_action = [dist_action]

    log_prob = distribution.log_prob(dist_action)
    log_prob = await smart_squeeze(log_prob, remaining_dims=1)

    # entropy = -log_prob[0]
    entropy = distribution.entropy()
    entropy = tf.reduce_mean(entropy)
    return target_action, dist_action, log_prob, entropy

class Trader:
    """This trader uses the proximal policy optimization algorithm (PPO) as proposed in https://arxiv.org/abs/1707.06347.
    Any liberties and further modifications to the algorithm will be attempted to be documented here
    Impelmentation is inspired by the torch implementation from cleanRL and by
    https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/tf2/agent.py
    """
    def __init__(self, bid_price, ask_price, **kwargs):
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.study_name = kwargs['study_name']
        self.status = {
            'weights_loading': False
        }

        # Initialize metrics tracking
        self.track_metrics = kwargs['track_metrics']
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        # Generate actions
        #I think we could stay with quantized actions, however I'd like to start testing on the non-quantized version ASAP so we do non quantized
        self.actions = {}
        if 'P_max' in kwargs:
            p_max = kwargs['P_max']
        else:
            p_max = 17
        for action in kwargs['actions']:
            if action == 'price':
                self.actions['price'] = {'min': ask_price, 'max': bid_price}
            if action == 'quantity':
                self.actions['quantity'] = {'min': -p_max, 'max': p_max}
            if action == 'storage':
                self.actions['storage'] = {'min': -p_max, 'max': p_max}

        # initialize all the counters we need
        self.train_step = 0
        self.total_step = 0
        self.gen = 0

        #prepare TB functionality, to open TB use the terminal command: tensorboard --logdir <dir_path>
        cwd = os.getcwd()
        experiment_path = os.path.join(cwd, self.study_name)
        trader_path = os.path.join(experiment_path, self.__participant['id'])
        self.summary_writer = tf.summary.create_file_writer(trader_path)

        # Initialize learning parameters
        self.learning = kwargs['learning']
        reward_function = kwargs['reward_function']
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        # Hyperparameters
        self.batch_size = kwargs['batch_size'] #bigger is smoother, but might require a bigger replay buffer
        self.policy_clip = kwargs['policy_clip']
        self.kl_stop = kwargs['kl_stop'] #according to baselines tends to be bewteen 0.01 and 0.05
        self.entropy_reg = kwargs['entropy_reg']
        self.gamma = kwargs['gamma']
        self.gae_lambda = kwargs['gae_lambda']

        self.max_train_steps = kwargs['max_train_steps']
        self.replay_buffer_length = kwargs['experience_replay_buffer_length']
        self.g_grad_norm = kwargs['g_grad_norm']

        #ToDo: change this to config arguments once we got this going
        self.teacher = kwargs["teacher"]
        self.teacher_sampling_rate = kwargs["teacher_sampling_rate"]
        self.observations = kwargs['observations']

        self.burn_in = kwargs['burn_in'] if 'burn_in' in kwargs else 0
        self.trajectory_length = kwargs['trajectory_length'] if 'trajectory_length' in kwargs else 1

        self.experience_replay_buffer = PPO_ExperienceReplay(max_length=self.replay_buffer_length,
                                                            action_types=self.actions,
                                                             multivariate=True,
                                                             trajectory_length=self.burn_in+self.trajectory_length)

        self.share_actor_critic = kwargs['shared_actor_critic']
        self.use_early_stop_actor = kwargs['use_early_stop_actor']

        self.aux_losses = []
        if 'aux_losses' in kwargs:
            self.aux_losses =kwargs['aux_losses']
            self.aux_targets_buffer = {}
            self.aux_losses_weights = kwargs['aux_losses_weights']

        if not self.share_actor_critic:
            self.actor_type = kwargs['actor_type']


            self.critic_patience = kwargs['critic_patience']
            self.use_early_stop_critic = kwargs['use_early_stop_critic']
            self.critic_type = kwargs['critic_type']

            actor_dict, critic_dict = build_actor_critic_models(num_inputs=len(kwargs['observations']),
                                                                hidden_actor=kwargs['actor_hidden'],
                                                                actor_type=self.actor_type,
                                                                hidden_critic=kwargs['actor_hidden'],
                                                                critic_type=self.critic_type,
                                                                num_actions=len(self.actions),
                                                                aux_losses=self.aux_losses,
                                                                share_params=False)


            self.ppo_actor = actor_dict['model']
            self.ppo_actor_dist = actor_dict['distribution']
            if self.actor_type == 'GRU':
                self.actor_states_dummy = actor_dict['initial_states_dummy']
            self.ppo_actor.compile(optimizer=k.optimizers.Adam(learning_rate= kwargs['alpha_actor'], ), )

            self.ppo_critic = critic_dict['model']
            if self.critic_type == 'GRU':
                self.critic_states_dummy = critic_dict['initial_states_dummy']
            self.ppo_critic.compile(optimizer=k.optimizers.Adam(learning_rate=kwargs['alpha_critic'], ), )

        else:
            self.actor_critic_type = kwargs['actor_critic_type']
            actor_critic_dict = build_actor_critic_models(num_inputs=len(kwargs['observations']),
                                                            hidden_actor_critic=kwargs['hidden_actor_critic'],
                                                            actor_critic_type=self.actor_critic_type,
                                                            num_actions=len(self.actions),
                                                            aux_losses=self.aux_losses,
                                                            share_params=True)

            self.ppo_actor_critic = actor_critic_dict['model']
            self.ppo_actor_dist = actor_critic_dict['distribution']
            if self.actor_critic_type == 'GRU':
                self.actor_critic_states_dummy = actor_critic_dict['initial_states_dummy']
            self.ppo_actor_critic.compile(optimizer=k.optimizers.Adam(learning_rate=kwargs['alpha_actor_critic'], ), )


        # Buffers we need for logging stuff before putting into the PPo Memory
        self.actions_buffer = {}
        self.pi_buffer = {}
        self.rewards_buffer = {}
        # self.pi_history = {}
        self.log_prob_buffer = {}
        self.value_buffer = {}
        self.observations_buffer = {}
        if self.share_actor_critic:
            if self.actor_critic_type == 'GRU':
                self.actor_critic_input_states_buffer = {}
        else:
            if self.actor_type == 'GRU':
                self.actor_input_states_buffer = {}
            if self.critic_type == 'GRU':
                self.critic_input_states_buffer = {}

        #logs we need for plotting
        self.rewards_history = []
        self.value_history = []
        self.observations_history = []
        self.net_load_history = []
        self.action_correction_distance = []

        self.actions_history = {}
        self.corrected_actions_history = {}
        self.pdf_history = {}
        for action in self.actions:
            self.actions_history[action] = []
            self.corrected_actions_history[action] = []
            # self.pdf_history[action] = {}
            # for param in ['loc', 'scale']:
            #     self.pdf_history[action][param] = []

    def __init_metrics(self):
        import sqlalchemy
        '''
        Initializes metrics to record into database
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('rewards', sqlalchemy.Float)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)

    def anneal(self, parameter:str, adjustment, mode:str='multiply', limit=None):
        if not hasattr(self, parameter):
            return False

        if mode not in ('subtract', 'multiply', 'set'):
            return False

        param_value = getattr(self, parameter)
        if mode == 'subtract':
            param_value = max(0, param_value - adjustment)

        elif mode == 'multiply':
            param_value *= adjustment

        elif mode == 'set':
            param_value = adjustment

        if limit is not None:
            param_value = max(param_value, limit)

        setattr(self, parameter, param_value)

    async def post_process_obs(self):
        timing = self.__participant['timing']
        current_round = timing['current_round']
        next_settle = timing['next_settle']
        round_duration = timing['duration']
        last_settle = timing['last_settle']
        last_round = timing['last_round']

        # adjusted_timing = timing.copy()
        # adjusted_timing.pop('timezone')
        # adjusted_timing.pop('duration')
        # sorted_timing = sorted(adjusted_timing.items(), key=lambda x: x[1])
        latest_processed_timestamp = current_round[1]
        reward = await self._rewards.calculate()
        if reward is None:
            await self.metrics.track('rewards', reward)
            return
        else:

            # align reward with action timing
            # in the current market setup the reward is for actions taken 3 steps ago
            # if self._rewards.type == 'net_profit':
            #ToDo: are we sure we're not moving this one step too far? we moved it one round up
            #ToDo:we'rescheduling forlast settle, we needto move the rewards
            reward_time_offset = current_round[1] - next_settle[1] - round_duration
            reward_timestamp = current_round[1] + 0
            self.rewards_buffer[reward_timestamp] = reward
            await self.metrics.track('rewards', reward)
            self.rewards_history.append(reward)
            latest_processed_timestamp = min(latest_processed_timestamp, reward_timestamp)

        # lets make sure the actions buffered are actually what we did ...
        actions_reference_round = current_round
        if actions_reference_round[1] in self.actions_buffer:
            # - recalculate the actually taken actions from the previous round
            # - rescale them to the equivalent distribution value
            # - recalculate logprob for the actually occured action
            # for: Battery, quantity, price, etc.

            storage_schedule = self.__participant['storage']
            storage_schedule = await storage_schedule['check_schedule'](actions_reference_round)
            actual_battery_action = storage_schedule[actions_reference_round]['energy_scheduled']
            actual_battery_action_dist_equiv = await scale_from_action_to_dist(actual_battery_action,
                                                                               a_min=self.actions['storage']['min'],
                                                                               a_max=self.actions['storage']['max'])
            storage_index = list(self.actions.keys()).index('storage') #This can create issues down the line if an individual action ahs several dims!
            pi_battery_action = self.actions_buffer[actions_reference_round[1]][storage_index]

            action_offset = np.abs(actual_battery_action_dist_equiv-pi_battery_action)
            data_for_tb = [{'name': 'action_correction_distance',
                            'data': action_offset,
                            'type': 'scalar',
                            'step': self.total_step}]
            tb_plotter(data_for_tb, self.summary_writer)

            if actual_battery_action_dist_equiv != pi_battery_action: #if true, we'll need to recalculate the
                self.actions_buffer[actions_reference_round[1]][storage_index] = actual_battery_action_dist_equiv
                corrected_actions = self.actions_buffer[actions_reference_round[1]]

                pi_t = self.pi_buffer[actions_reference_round[1]]
                pi_t = tf.expand_dims(pi_t, axis=[0])
                pi_t = tf.expand_dims(pi_t, axis=[0])

                dist = build_multivar(pi_t, self.ppo_actor_dist, self.actions)

                new_log_probs = dist.log_prob(corrected_actions)
                new_log_probs = await smart_squeeze(new_log_probs, remaining_dims=1)
                #old_log_probs = self.log_prob_buffer[last_round[1]]
                self.log_prob_buffer[actions_reference_round[1]] = new_log_probs

            # do any observation and rewards post_processing
            latest_processed_timestamp = min(actions_reference_round[1], latest_processed_timestamp)

        if len(self.aux_losses) > 0:
            if last_round[1] not in self.observations_buffer:
                return
            else:
                self.aux_targets_buffer[last_round[1]] = {}
                if 'generation' in self.aux_losses:
                    solar_idx = self.obs_order.index('generation')
                    current_solar = self.observations_buffer[current_round[1]][solar_idx]
                    self.aux_targets_buffer[last_round[1]]['generation'] = current_solar

                if 'load' in self.aux_losses:
                    load_idx = self.obs_order.index('load')
                    current_load = self.observations_buffer[current_round[1]][load_idx]
                    self.aux_targets_buffer[last_round[1]]['load'] = current_load

                latest_processed_timestamp = min(last_round[1], latest_processed_timestamp)


        if latest_processed_timestamp in self.actions_buffer:
            #log actions for later histogram plot

            for action in self.actions:
                action_index = list(self.actions.keys()).index(action)
                corrected_actions = self.actions_buffer[latest_processed_timestamp][action_index]
                scaled_action = await scale_from_action_to_dist(corrected_actions,
                                    a_min=self.actions['storage']['min'],
                                    a_max=self.actions['storage']['max'])
                self.corrected_actions_history[action].append(scaled_action)





        return latest_processed_timestamp

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        # print(self.total_step)
        if not self.learning:
            return

        #post_process_obs
        latest_updated_timestamp = await self.post_process_obs()

        if latest_updated_timestamp in self.observations_buffer and latest_updated_timestamp in self.actions_buffer and latest_updated_timestamp in self.rewards_buffer:  # we found matching ones, buffer and pop
            memory_kwargs = {}
            memory_kwargs['observations'] = self.observations_buffer[latest_updated_timestamp]
            memory_kwargs['actions_taken'] = self.actions_buffer[latest_updated_timestamp]
            memory_kwargs['log_probs'] = self.log_prob_buffer[latest_updated_timestamp]
            memory_kwargs['values'] = self.value_buffer[latest_updated_timestamp]
            memory_kwargs['rewards'] = self.rewards_buffer[latest_updated_timestamp]
            memory_kwargs['episode'] = self.gen

            if self.share_actor_critic:
                memory_kwargs['actor_critic_states'] = self.actor_critic_input_states_buffer[
                                                     latest_updated_timestamp] if self.actor_critic_type == 'GRU' else None
            else:
                memory_kwargs['critic_states'] = self.critic_input_states_buffer[
                                    latest_updated_timestamp] if self.critic_type == 'GRU' else None
                memory_kwargs['actor_states'] = self.actor_input_states_buffer[
                                   latest_updated_timestamp] if self.actor_type == 'GRU' else None

            if len(self.aux_losses) > 0:
                for aux_loss in self.aux_losses:
                    memory_kwargs[aux_loss] = self.aux_targets_buffer[latest_updated_timestamp][aux_loss]

            #ToDo: add the aux-loss variables to the memory_kwargs

            self.experience_replay_buffer.add_entry(**memory_kwargs)

            self.rewards_buffer.pop(latest_updated_timestamp)
            self.actions_buffer.pop(latest_updated_timestamp) #ToDo: check if we can pop into the above function, would look nicer
            self.log_prob_buffer.pop(latest_updated_timestamp)
            self.value_buffer.pop(latest_updated_timestamp)
            self.observations_buffer.pop(latest_updated_timestamp)
            if self.share_actor_critic:
                if self.actor_critic_type == 'GRU':
                    self.actor_critic_input_states_buffer.pop(latest_updated_timestamp)
            else:
                if self.actor_type == 'GRU':
                    self.actor_input_states_buffer.pop(latest_updated_timestamp)
                if self.critic_type == 'GRU':
                    self.critic_input_states_buffer.pop(latest_updated_timestamp)

            if self.experience_replay_buffer.should_we_learn():
                advantage_calulated = await self.experience_replay_buffer.calculate_advantage(gamma=self.gamma,
                                                                                              gae_lambda=self.gae_lambda,
                                                                                              )
                explained_variance_critic = await self.experience_replay_buffer.calculate_explained_variance()
                data_for_tb = [{'name': 'explained_variance',
                                'data': explained_variance_critic,
                                'type': 'scalar',
                                'step': self.total_step}]
                tb_plotter(data_for_tb, self.summary_writer)

                buffer_indexed = await self.experience_replay_buffer.generate_availale_indices()  # so we can caluclate the batches faster
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, func=self.train_RL_agent)

    def train_RL_agent(self):

        stop_critic_training = False
        stop_actor_training = False
        sgd_steps = self.max_train_steps
        max_train_steps = self.train_step + sgd_steps
        if not self.share_actor_critic:
            if self.use_early_stop_critic:
                critic_stopper = EarlyStopper(patience=self.critic_patience)

        while self.train_step <= max_train_steps and not (stop_actor_training and stop_critic_training):
            keys_to_fetch = ['returns', 'observations', 'advantages', 'actions_taken', 'log_probs', 'v_target']
            if self.share_actor_critic:
                keys_to_fetch.append('actor_critic_states')
            else:
                keys_to_fetch.append('critic_states')
                keys_to_fetch.append('actor_states')
            #ToDo: add the losses fetching for them aux losses
            if len(self.aux_losses) > 0:
                for aux_loss in self.aux_losses:
                    keys_to_fetch.append(aux_loss)

            batch = self.experience_replay_buffer.fetch_batch(batchsize=self.batch_size, keys=keys_to_fetch) #to see the batch data structure check this method
            observations = tf.convert_to_tensor(batch['observations'], dtype=tf.float32)
            # returns = tf.convert_to_tensor(batch['returns'], dtype=tf.float32)
            V_target = tf.convert_to_tensor(batch['v_target'])
            advantages = tf.convert_to_tensor(batch['advantages'], dtype=tf.float32)
            log_probs_old = tf.convert_to_tensor(batch['log_probs'], dtype=tf.float32)
            a_taken = tf.convert_to_tensor(batch['actions_taken'], dtype=tf.float32)

            # normalize batch advantages, https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/, detail7
            avd_batch_mean = tf.math.reduce_mean(advantages)
            adv_batch_std = tf.math.reduce_std(advantages)
            advantages = (advantages - avd_batch_mean) / tf.math.maximum(adv_batch_std, 1e-10)

            shared_inputs = {'observations': observations}

            if len(self.aux_losses) > 0:
                aux_targets = {}
                for aux_loss in self.aux_losses:
                    aux_targets[aux_loss] = tf.expand_dims(tf.convert_to_tensor(batch[aux_loss], dtype=tf.float32), axis=-1)


            if self.share_actor_critic:
                # shared_inputs
                if self.actor_critic_type == 'GRU':  # this still seems somewhat unclean
                    actor_critic_states = assemble_subdict_batch(batch['actor_critic_states'])
                    for key in self.actor_critic_states_dummy:
                        states = tf.convert_to_tensor(actor_critic_states[key])
                        states = tf.squeeze(states, axis=1)  # ToDo: this should not be necessary really ....
                        shared_inputs[key] = states

                with tf.GradientTape() as tape_AC:
                    theta_out = self.ppo_actor_critic(shared_inputs)
                    loss_critic = calculate_critic_loss(theta_critic_out=theta_out,
                                                        V_target=V_target,
                                                          burn_in=self.burn_in if self.burn_in is not None else None,
                                                          )

                    loss_actor, approx_kl, entropy, clip_frac = calculate_ppo_loss(theta_actor_out=theta_out,
                                                                                   a_taken=a_taken,
                                                                                   log_probs_old=log_probs_old,
                                                                                   advantages=advantages,
                                                                                   actor_distribution=self.ppo_actor_dist,
                                                                                   actionspace=self.actions,
                                                                                   policy_clip_ratio=self.policy_clip,
                                                                                   burn_in=self.burn_in if self.burn_in is not None else None)
                    if len(self.aux_losses) > 0:
                        aux_losses = calculate_aux_losses(theta_out=theta_out, aux_loss_targets=aux_targets, burn_in=self.burn_in)


                        data_for_tb = [] #add to the logger
                        for loss in self.aux_losses:
                            idx = self.aux_losses.index(loss)
                            data_for_tb.append({'name': loss+'_loss', 'data': aux_losses[idx], 'type': 'scalar', 'step': self.train_step})
                        tb_plotter(data_for_tb, self.summary_writer)

                        weighted_aux_losses = self.aux_losses_weights * tf.reduce_sum(aux_losses)
                    else:
                        weighted_aux_losses = 0

                    total_loss = loss_actor + 0.5*loss_critic - self.entropy_reg*entropy + weighted_aux_losses
                    data_for_tb = [{'name': 'total_loss', 'data': total_loss, 'type': 'scalar', 'step': self.train_step}]
                    tb_plotter(data_for_tb, self.summary_writer)
                self.ppo_actor_critic = apply_gradients_to_model(model=self.ppo_actor_critic,
                                                          gratient_tape=tape_AC,
                                                          loss=total_loss,
                                                          g_grad_norm=self.g_grad_norm)

            else:
                if not stop_critic_training:
                    # manage the inputs
                    critic_inputs = shared_inputs
                    if self.critic_type == 'GRU':                                                       # this still seems somewhat unclean
                        critic_states = assemble_subdict_batch(batch['critic_states'])
                        for key in self.critic_states_dummy:
                            states = tf.convert_to_tensor(critic_states[key])
                            states = tf.squeeze(states, axis=1) #ToDo: this should not be necessary really ....
                            critic_inputs[key] = states

                    with tf.GradientTape() as tape_critic:
                        # calculate critic loss and backpropagate
                        theta_critic_out = self.ppo_critic(critic_inputs)
                        loss_critic = calculate_critic_loss(theta_critic_out=theta_critic_out,
                                                            V_target=V_target,
                                                              burn_in=self.burn_in if self.burn_in is not None else None,
                                                              )

                    self.ppo_critic = apply_gradients_to_model(model=self.ppo_critic,
                                                               gratient_tape=tape_critic,
                                                               loss=loss_critic,
                                                               g_grad_norm=self.g_grad_norm)

                    # calculate the stopping crtierions
                    if self.use_early_stop_critic:
                        stop_critic_training, self.ppo_critic = critic_stopper.check_iteration(loss_critic.numpy(), self.ppo_critic)

                if not stop_actor_training:

                    actor_inputs = shared_inputs
                    if self.actor_type == 'GRU':
                        actor_states = assemble_subdict_batch(batch['actor_states'])
                        for key in self.actor_states_dummy:
                            states = tf.convert_to_tensor(actor_states[key])
                            states = tf.squeeze(states, axis=1)
                            actor_inputs[key] = states

                    with tf.GradientTape() as tape_actor:
                        theta_actor_out = self.ppo_actor(actor_inputs)
                        loss_actor, approx_kl, entropy, clip_frac = calculate_ppo_loss(theta_actor_out=theta_actor_out,
                                                                            a_taken=a_taken,
                                                                            log_probs_old=log_probs_old,
                                                                            advantages=advantages,
                                                                            actor_distribution=self.ppo_actor_dist,
                                                                            actionspace=self.actions,
                                                                            policy_clip_ratio=self.policy_clip,
                                                                            burn_in=self.burn_in if self.burn_in is not None else None)
                        loss_actor = loss_actor - self.entropy_reg * entropy


                    self.ppo_actor = apply_gradients_to_model(model=self.ppo_actor,
                                                              gratient_tape=tape_actor,
                                                              loss=loss_actor,
                                                              g_grad_norm=self.g_grad_norm)

                    # early stopping condition or keep training, consider having this a running avg of 5 or sth?

            if self.use_early_stop_actor and not stop_actor_training:
                if approx_kl.numpy() > 1.5 * self.kl_stop:
                    stop_actor_training = True
                else:
                    stop_actor_training = False

            # log
            data_for_tb = [{'name': 'critic_loss', 'data': loss_critic, 'type': 'scalar', 'step': self.train_step},
                           {'name': 'actor_loss', 'data': loss_actor, 'type': 'scalar', 'step': self.train_step}, # Main loss, if too spiky we want to see where it comes from
                           {'name': 'approx_KLD', 'data': approx_kl, 'type': 'scalar', 'step': self.train_step}, # Distance pseudometric between old and new policy, we want this to decrease over training as this would indicate convergence
                           {'name': 'policy_clip_frac', 'data': clip_frac, 'type': 'scalar', 'step': self.train_step}, #clip fraction of the policy loss
                           {'name': 'minibatch_entropy', 'data': entropy, 'type': 'scalar', 'step': self.train_step}, # Randomness of policy, we want the differential entropy to keep dropping slowly over the course of training

                           {'name': 'early_stop_actor', 'data': stop_actor_training, 'type': 'scalar', 'step': self.train_step},
                           ]
            tb_plotter(data_for_tb, self.summary_writer)

            self.train_step = self.train_step + 1


        #clear the buffer after we learned it
        self.experience_replay_buffer.clear_buffer()
        self.train_step = max_train_steps #to make sure we log all the algorithms that might be running in parallel at the same scales

    async def __sample_pi(self, pi_dict):
        dist = build_multivar(pi_dict, self.ppo_actor_dist, self.actions)

        a_dist = dist.sample(1)
        a_dist = tf.clip_by_value(a_dist, clip_value_min=1e-8, clip_value_max=0.999999)
        a_dist = await smart_squeeze(a_dist, remaining_dims=1)

        log_prob = dist.log_prob(a_dist)
        log_prob = await smart_squeeze(log_prob, remaining_dims=1)

        # entropy_proxy = -log_prob[0]
        entropy_proxy = dist.entropy()
        entropy_proxy = tf.reduce_mean(entropy_proxy)

        a_scaled = {}
        keys = list(self.actions.keys())
        for action_index in range(len(keys)):
            a = await scale_from_dist_to_action(a_dist[action_index],
                                                       a_min=self.actions[keys[action_index]]['min'],
                                                       a_max=self.actions[keys[action_index]]['max'])

            a_scaled[keys[action_index]] = a

        return a_scaled, log_prob, a_dist, entropy_proxy

    async def _query_actor(self, shared_inputs, current_round, last_settle):
        # actor stuff
        # assemble inputs
        actor_inputs = shared_inputs.copy()
        if self.actor_type == 'GRU':
            if current_round[1] not in self.actor_input_states_buffer: #we assume that we have just started or reset
                self.actor_input_states_buffer[current_round[1]] = self.actor_states_dummy

            actor_current_states = self.actor_input_states_buffer[current_round[1]]
            for key in actor_current_states:
                actor_inputs[key] = actor_current_states[key]

        actor_outputs = self.ppo_actor(actor_inputs)
        #post process outputs
        pi_dict = actor_outputs.pop('pi')
        if self.actor_type == 'GRU':
            states_actor_t = actor_outputs
            self.actor_input_states_buffer[last_settle[1]] = states_actor_t

        return pi_dict

    async def _query_critic(self, shared_inputs, current_round, last_settle):
        # assemble inputs
        critic_inputs = shared_inputs.copy()
        if self.critic_type == 'GRU':
            if current_round[1] not in self.critic_input_states_buffer: #we assume that we have just started or reset
                self.critic_input_states_buffer[current_round[1]] = self.critic_states_dummy

            critic_current_states = self.critic_input_states_buffer[current_round[1]]
            for key in critic_current_states:
                critic_inputs[key] = critic_current_states[key]
        #call critic
        critic_outputs = self.ppo_critic(critic_inputs)
        #post process critic outputs
        V_t = critic_outputs.pop('value')
        if self.critic_type == 'GRU':
            states_critic_t = critic_outputs
            self.critic_input_states_buffer[last_settle[1]] = states_critic_t
        # log
        V_t = tf.squeeze(V_t).numpy().tolist()

        return V_t

    async def _query_actor_critic(self, shared_inputs, current_round, last_settle):
        if self.actor_critic_type == 'GRU':
            if current_round[1] not in self.actor_critic_input_states_buffer:  # we assume that we have just started or reset
                self.actor_critic_input_states_buffer[current_round[1]] = self.actor_critic_states_dummy

            actor_critic_current_states = self.actor_critic_input_states_buffer[current_round[1]]
            for key in actor_critic_current_states:
                shared_inputs[key] = actor_critic_current_states[key]

        actor_critic_outputs = self.ppo_actor_critic(shared_inputs)
        V_t = actor_critic_outputs.pop('value')
        V_t = tf.squeeze(V_t).numpy().tolist()
        pi_dict = actor_critic_outputs.pop('pi')

        if self.actor_critic_type == 'GRU':
            states_AC_t = actor_critic_outputs
            self.actor_critic_input_states_buffer[last_settle[1]] = states_AC_t

        return pi_dict, V_t


    async def pre_process_obs(self, ts_obs):
        self.obs_order = []
        # current_round = self.__participant['timing']['current_round']
        # previous_round = self.__participant['timing']['last_round']
        # next_settle = self.__participant['timing']['next_settle']

        next_generation, next_load = await self.__participant['read_profile'](ts_obs)

        observations_t = []
        if not hasattr(self, 'profile_stats'):
            self.profile_stats = await self.__participant['get_profile_stats']()

        if 'generation' in self.observations:
            self.obs_order.append('generation')
            if self.profile_stats:
                avg_generation = self.profile_stats['avg_generation']
                stddev_generation = self.profile_stats['stddev_generation']
                z_next_generation = (next_generation - avg_generation) / stddev_generation
                observations_t.append(z_next_generation)
            else:
                observations_t.append(next_generation)

        if 'load' in self.observations:
            self.obs_order.append('load')
            if self.profile_stats:
                avg_load = self.profile_stats['avg_consumption']
                stddev_load = self.profile_stats['stddev_consumption']
                z_next_load = (next_load - avg_load) / stddev_load
                observations_t.append(z_next_load)
            else:
                observations_t.append(next_load)

        minutes = int(ts_obs[0] / 60)  # ToDo: there should be an inbuilt conversion for these formats
        hour = int(minutes / 60)
        day = int(hour / 24)

        if 'time_sin_hour' in self.observations:
            self.obs_order.append('time_sin_hour')
            observations_t.append(np.sin(2 * np.pi * hour / 24))
        if 'time_cos_hour' in self.observations:
            self.obs_order.append('time_cos_hour')
            observations_t.append(np.cos(2 * np.pi * hour / 24))
        if 'time_sin_day' in self.observations:
            self.obs_order.append('time_sin_day')
            observations_t.append(np.sin(2 * np.pi * day / 7))
        if 'time_cos_day' in self.observations:
            self.obs_order.append('time_cos_day')
            observations_t.append(np.cos(2 * np.pi * day / 7))

        if 'soc' in self.observations:
            self.obs_order.append('soc')
            storage_schedule = await self.__participant['storage']['check_schedule'](ts_obs)
            soc = storage_schedule[ts_obs]['projected_soc_end']
            observations_t.append(soc)

        observations_t_numpy = np.array(observations_t)
        #ToDo: move the observations extension to the batching process?
        obs_t_tensor = tf.expand_dims(observations_t, axis=0)
        obs_t_tensor = tf.expand_dims(obs_t_tensor, axis=0)


        #ToDo: add calculation for explained variance once Lab is back online, see https://github.com/ray-project/ray/blob/7f03368fc0f56fee478e9ac15576b626fb1103a9/rllib/utils/tf_utils.py
        data_for_tb = [{'name': 'obs_mean', 'data': np.mean(observations_t_numpy), 'type': 'scalar', 'step': self.total_step}, #These should be consistent-ish wrt to each other and not super spiky (think orders of magnitude)
                       {'name': 'obs_median', 'data': np.median(observations_t_numpy), 'type': 'scalar', 'step': self.total_step},
                      ]


        return observations_t_numpy, obs_t_tensor, data_for_tb

    async def act(self, **kwargs):
        timing = self.__participant['timing']
        current_round = self.__participant['timing']['current_round']
        round_duration = self.__participant['timing']['duration']
        next_round = (current_round[0]+round_duration, current_round[1]+round_duration)
        last_settle = self.__participant['timing']['last_settle']
        next_settle = self.__participant['timing']['next_settle']
        ts_obs = next_settle
        ts_act = next_settle

        observations_t_numpy, obs_t_tensor, data_for_tb = await self.pre_process_obs(ts_obs)

        shared_inputs = {}
        shared_inputs['observations'] = obs_t_tensor

        if self.share_actor_critic:
            pi_dict, V_t = await self._query_actor_critic(shared_inputs, current_round, next_round)
        else:
            #actor stuff
            pi_dict = await self._query_actor(shared_inputs, current_round, next_round)
            # critic stuff
            V_t = await self._query_critic(shared_inputs, current_round, next_round)


        if self.teacher and np.random.random() < self.teacher_sampling_rate:
            next_generation, next_load = await self.__participant['read_profile'](ts_obs)
            target_action, dist_action, log_prob, entropy = await pretend_greedy_policy(next_load=next_load,
                                                                                  next_generation=next_generation,
                                                                                  distribution=build_multivar(pi_dict, self.ppo_actor_dist, self.actions),
                                                                                  actions=self.actions)

            data_for_tb.append({'name': 'teacher_used', 'data': 1.0, 'type': 'scalar', 'step': self.total_step})

        else:
            target_action, log_prob, dist_action, entropy = await self.__sample_pi(pi_dict)

            data_for_tb.append({'name': 'teacher_used', 'data': 0, 'type': 'scalar', 'step': self.total_step})
        data_for_tb.append({'name': 'entropy', 'data': entropy, 'type': 'scalar', 'step': self.total_step})


        # lets log the stuff needed for the replay buffer
        self.observations_buffer[current_round[1]] = observations_t_numpy
        self.actions_buffer[current_round[1]] = dist_action
        self.pi_buffer[current_round[1]] = tf.squeeze(pi_dict).numpy().tolist()
        self.log_prob_buffer[current_round[1]] = log_prob
        self.value_buffer[current_round[1]] = V_t

        self.value_history.append(V_t)
        self.observations_history.append(observations_t_numpy)

        current_generation, current_load = await self.__participant['read_profile'](current_round)
        net_load_current = current_load - current_generation
        if 'storage' in self.__participant:
            storage_schedule = await self.__participant['storage']['check_schedule'](current_round)
            net_load_current = net_load_current + storage_schedule[current_round]['energy_scheduled']

        self.net_load_history.append(net_load_current)

        actions = await self.decode_actions(target_action, ts_act)

        tb_plotter(data_for_tb, self.summary_writer)

        # if self.track_metrics:
        #     await asyncio.gather(
        #         self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
        #         self.metrics.track('actions_dict', actions),
        #         self.metrics.track('next_settle_load', next_load),
        #         self.metrics.track('next_settle_generation', next_generation))
        #     if 'storage' in self.actions:
        #         await self.metrics.track('storage_soc', self.__participant['storage']['info']()['state_of_charge'])
        return actions

    async def decode_actions(self, target_action, ts_act):
        actions = dict()

        if 'price' in target_action:
            price = target_action['price']
            price = round(price, 4)
        else:
            price = 0.111

        if 'quantity' in target_action:
            quantity = int(target_action['quantity'])
        else:
            quantity = 0

        if 'storage' in target_action:
            storage = int(target_action['storage'])

        if quantity > 0:
            actions['bids'] = {
                str(ts_act): {
                    'quantity': quantity,
                    'price': price
                }
            }
        elif quantity < 0:
            actions['asks'] = {
                'solar': {
                    str(ts_act): {
                        'quantity': -quantity,
                        'price': price
                    }
                }
            }

        if 'storage' in self.actions:
            actions['bess'] = {
                str(ts_act): storage
                }

        #log actions for later histogram plot
        for action in self.actions:
            self.actions_history[action].append(target_action[action])
        return actions

    async def step(self):
        next_actions = await self.act()
        await self.learn()
        if self.track_metrics:
            await self.metrics.save(10000)
        # print(next_actions)
        self.total_step += 1
        return next_actions

    async def end_of_generation_tasks(self):
        # self.episode_reward_history.append(self.episode_reward)
        episode_G = sum(self.rewards_history)
        # print(self.__participant['id'], 'episode reward:', episode_G)

        data_for_tb = [{'name':'Return', 'data':episode_G, 'type':'scalar', 'step':self.gen},
                       {'name': 'Episode Rewards', 'data': self.rewards_history, 'type': 'histogram', 'step':self.gen},
                       {'name':'Values', 'data':self.value_history, 'type':'histogram', 'step':self.gen}]
        for action in self.actions:
            data_for_tb.append({'name':action, 'data':self.actions_history[action], 'type':'histogram', 'step':self.gen})
            data_for_tb.append({'name':'corrected' + action, 'data':self.corrected_actions_history[action], 'type':'histogram', 'step':self.gen})

        day_length = 24 #ToDo: find a way to make this auto adjust....
        socs = np.array(self.observations_history)[:,-1]*100
        data_for_tb.append({'name': 'SoC_during_day', 'data': socs, 'type': 'pseudo3D', 'step': self.gen, 'buckets': day_length})

        net_load_history = self.net_load_history - np.amin(self.net_load_history)
        data_for_tb.append(
           {'name': 'Effective_Ned_load_during_day', 'data': net_load_history, 'type': 'pseudo3D', 'step': self.gen, 'buckets': day_length})

        # loop = asyncio.get_running_loop()
        # await loop.run_in_executor(None, tb_plotter, [data_for_tb, self.summary_writer])
        tb_plotter(data_for_tb, self.summary_writer)

        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.observations_buffer.clear()
        self.value_buffer.clear()
        self.actions_buffer.clear()
        self.pi_buffer.clear()
        self.log_prob_buffer.clear()
        self.rewards_buffer.clear()
        if self.share_actor_critic:
            if self.actor_critic_type == 'GRU':
                self.actor_critic_input_states_buffer.clear()
        else:
            if self.actor_type == 'GRU':
                self.actor_input_states_buffer.clear()
            if self.critic_type == 'GRU':
                self.critic_input_states_buffer.clear()

        if len(self.aux_losses) >0:
            self.aux_targets_buffer.clear()

        self.rewards_history.clear()
        self.value_history.clear()
        self.observations_history.clear()
        self.net_load_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()
            self.corrected_actions_history[action].clear()

        return True