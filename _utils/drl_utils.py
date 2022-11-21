import random
import tensorflow as tf
from random import randint
import numpy as np
from _utils import utils
from collections import OrderedDict, Counter
import itertools
import scipy.signal
from tensorflow import keras as k
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_probability as tfp

huber =tf.keras.losses.Huber(
    delta=1.0,
    name='huber_loss'
)

def _explained_variance(ypred, y): #ypred and y should both be 1dim arrays
    # from https://github.com/openai/baselines/blob/52255beda5f5c8760b0ae1f676aa656bb1a61f80/baselines/common/math_util.py
    # we want this to be as close to 1 as possible, this means our critic is good
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    delta_var = np.var(y - ypred)
    return np.nan if vary == 0 else 1 -  delta_var/ vary

def build_hidden_layer(signal, type='FFNN', num_hidden=32, name='Actor', initial_state=None, initializer=k.initializers.HeNormal()):

    initializer =k.initializers.Orthogonal(gain=tf.math.sqrt(2.0), seed=None)

    if type == 'FFNN':
        signal = k.layers.Dense(num_hidden,
                                         activation="tanh",
                                         kernel_initializer=initializer,
                                         name=name)(signal)
        return signal, None
    elif type == 'GRU':

        signal, last_state = k.layers.GRU(num_hidden,
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              kernel_initializer=initializer,
                              return_sequences=True, return_state=True,
                              name=name)(signal, initial_state=initial_state)
        return signal, last_state

    else:
        print('requested layer type (', type, ') not recognized, failed to build ', name)
        return False, False

def build_hidden(internal_signal, inputs, outputs, hidden_actor=[32,32,32], type='FFNN'):
    hidden_layer = 0
    initial_states_dummy = {}
    for num_hidden_neurons in hidden_actor:
        if type == 'GRU':
            initial_state = k.layers.Input(shape=num_hidden_neurons, name='GRU_' + str(hidden_layer) + '_initial_state')
            inputs['GRU_'+str(hidden_layer)+'_state'] = initial_state
            initial_states_dummy['GRU_'+str(hidden_layer)+'_state'] = tf.zeros((1,num_hidden_neurons))

        else:
            initial_state = None

        internal_signal, last_state = build_hidden_layer(internal_signal,
                                       type=type,
                                       num_hidden=num_hidden_neurons,
                                       initial_state=initial_state,
                                       name='Actor_hidden_' + str(hidden_layer))
        if type == 'GRU':
            outputs['GRU_'+str(hidden_layer)+'_state'] = last_state
        hidden_layer += 1

    return internal_signal, inputs, outputs, initial_states_dummy

def value_head(internal_signal, num_hidden=[], name='ValueHead'):
    if num_hidden != []:
        for nbr in range(len(num_hidden)):
            internal_signal, _ = build_hidden_layer(internal_signal,
                                                             type='FFNN',
                                                             num_hidden=num_hidden[nbr],
                                                             name=name+'_hidden_' + str(nbr))

    initializer = k.initializers.Orthogonal(gain=1.0)
    value = k.layers.Dense(1,
                           activation=None, #ToDo: test tanh vs None
                           kernel_initializer=initializer,
                           name=name)(internal_signal)
    return value

def actor_head(internal_signal, num_actions, num_hidden=[],beta_offset=False, name='ActorHead'):

    if num_hidden != []:
        for nbr in range(len(num_hidden)):
            internal_signal, _ = build_hidden_layer(internal_signal,
                                                             type='FFNN',
                                                             num_hidden=num_hidden[nbr],
                                                             name=name+'_hidden_' + str(nbr))

    policy_head_initializer = k.initializers.Orthogonal(gain=0.1, seed=None)
    concentrations = k.layers.Dense(2 * num_actions,
                                    activation=None,  # ToDo: test tanh vs None
                                    kernel_initializer=policy_head_initializer,
                                    bias_initializer=tf.keras.initializers.RandomUniform(minval=2.8, maxval=3.2),
                                    name=name+'concentrations')(internal_signal)
    bias = 1.0 if beta_offset else 0.0
    concentrations += bias
    concentrations = tf.math.softplus(concentrations)
    return concentrations

def build_shared_actor_critic(num_inputs=4, num_actions=2, hidden=[32, 32, 32], model_type='FFNN', aux_losses=[], beta_offset=True):

    inputs = {}
    outputs = {}

    shape = (num_inputs,) if model_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Input')
    inputs['observations'] = internal_signal

    internal_signal, inputs, outputs, initial_states_dummy = build_hidden(internal_signal, inputs, outputs,
                                                                          hidden, model_type)

    #policy head
    pi = actor_head(internal_signal, num_actions, beta_offset=beta_offset, name='Actor')
    outputs['pi'] = pi

    #value_head
    value = value_head(internal_signal, name='Value')
    outputs['value'] = value

    for aux_loss in aux_losses:
        aux_output = value_head(internal_signal, name=aux_loss,num_hidden=[])
        outputs[aux_loss] = aux_output

    shared_model = k.Model(inputs=inputs, outputs=outputs)

    actor_distrib = tfp.distributions.Beta

    out_dict = {'model': shared_model,
                'distribution': actor_distrib,
                'initial_states_dummy': initial_states_dummy}

    return out_dict

def build_actor(num_inputs=4, num_actions=3, hidden_actor=[32], actor_type='FFNN', aux_losses=[], beta_offset=True):
    inputs = {}
    outputs = {}

    shape = (num_inputs,) if actor_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Actor_Input')
    inputs['observations'] = internal_signal

    internal_signal,  inputs, outputs, initial_states_dummy = build_hidden(internal_signal, inputs, outputs, hidden_actor, actor_type)

    concentrations = actor_head(internal_signal, num_actions, beta_offset=beta_offset)
    outputs['pi'] = concentrations

    for aux_loss in aux_losses:
        aux_output = value_head(internal_signal, name=aux_loss)
        outputs[aux_loss] = aux_output

    actor_model = k.Model(inputs=inputs, outputs=outputs)

    actor_distrib = tfp.distributions.Beta

    out_dict={'model': actor_model,
              'distribution': actor_distrib,
              'initial_states_dummy': initial_states_dummy}

    return out_dict

def build_critic(num_inputs=4, hidden_critic=[32, 32, 32], critic_type='FFNN', aux_losses=[]):
    inputs = {}
    outputs = {}
    shape = (num_inputs,) if critic_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Critic_Input')
    inputs['observations'] = internal_signal

    internal_signal, inputs, outputs, initial_states_dummy = build_hidden(internal_signal, inputs, outputs, hidden_critic, critic_type)

    value = value_head(internal_signal, name='Value')
    outputs['value'] = value

    for aux_loss in aux_losses:
        aux_output = value_head(internal_signal, name=aux_loss)
        outputs[aux_loss] = aux_output

    critic_model = k.Model(inputs=inputs, outputs=outputs)

    critic_dict = {'model': critic_model,
                   'initial_states_dummy': initial_states_dummy}
    return critic_dict

def build_actor_critic_models(**kwargs):
    # needs to return a suitable actor ANN, ctor PDF function and critic ANN
    share_params =kwargs['share_params']
    if not share_params:
        actor_dict = build_actor(num_inputs=kwargs['num_inputs'],
                                  num_actions=kwargs['num_actions'],
                                  hidden_actor=kwargs['hidden_actor'],
                                    aux_losses = kwargs['aux_losses'],
                                 beta_offset=kwargs['beta_offset'],
                                  actor_type=kwargs['actor_type'])
        critic_dict = build_critic(num_inputs=kwargs['num_inputs'],
                                  hidden_critic=kwargs['hidden_critic'],
                                   aux_losses=kwargs['aux_losses'],
                                  critic_type=kwargs['critic_type'])

        return actor_dict, critic_dict

    else:
        actor_critic_dict = build_shared_actor_critic(num_inputs=kwargs['num_inputs'],
                                                      num_actions=kwargs['num_actions'],
                                                      aux_losses=kwargs['aux_losses'],
                                                      beta_offset=kwargs['beta_offset'],
                                                      hidden=kwargs['hidden_actor_critic'],
                                                      model_type=kwargs['actor_critic_type'])
        return actor_critic_dict

def calculate_aux_losses(theta_out, aux_loss_targets, burn_in=None):
    aux_losses = []
    for aux_loss in aux_loss_targets:
        aux_loss_target = aux_loss_targets[aux_loss]
        theta_out_aux_loss = theta_out.pop(aux_loss)
        if burn_in is not None:
            if burn_in > 0:  # discard unwanted stages
                aux_loss_target = aux_loss_target[:, burn_in:]
                theta_out_aux_loss = theta_out_aux_loss[:, burn_in:]

        lin_loss = aux_loss_target - theta_out_aux_loss
        square_loss = tf.square(lin_loss)
        loss = tf.reduce_mean(square_loss)
        aux_losses.append(loss)
    return aux_losses

def calculate_critic_loss(theta_critic_out, V_target, burn_in=None):
    Vs = theta_critic_out.pop('value')
    Vs = tf.squeeze(Vs, axis=-1)
    if burn_in is not None:
        if burn_in > 0:  # discard unwanted stages
            Vs = Vs[:, burn_in:]
            V_target = V_target[:, burn_in:]
    loss = tf.square(Vs - V_target)
    losses_critic = tf.reduce_mean(loss)

    return losses_critic

def calculate_ppo_loss(theta_actor_out, a_taken, log_probs_old, advantages,
                             actor_distribution, actionspace,
                             policy_clip_ratio=0.2, burn_in=None):
    pi = theta_actor_out.pop('pi')
    if burn_in is not None:
        if burn_in > 0:
            pi = pi[:, burn_in:, :]
            a_taken = a_taken[:, burn_in:, :]
            log_probs_old = log_probs_old[:, burn_in:, :]
            advantages = advantages[:, burn_in:]

    dist = build_multivar(pi, actor_distribution, actionspace)

    log_probs_new = dist.log_prob(a_taken)
    # check = tf.reduce_sum(log_probs_new).numpy()
    # if np.isnan(check) or np.isinf(check):
    #     probs = dist.prob(a_taken)
    #     print('shit')
    # This is how baselines does it
    if len(log_probs_new.shape) > 2:
        log_probs_new = tf.squeeze(log_probs_new, axis=-1)
    if len(log_probs_old.shape) > 2:
        log_probs_old = tf.squeeze(log_probs_old, axis=-1)

    ratio = tf.exp(log_probs_new - log_probs_old)  # pi(a|s) / pi_old(a|s)

    clipped_ratio = tf.clip_by_value(ratio, 1 - policy_clip_ratio, 1 + policy_clip_ratio)

    weighted_ratio = ratio * advantages
    weighted_clipped_ratio = clipped_ratio * advantages
    loss_actor = -tf.math.minimum(weighted_ratio, weighted_clipped_ratio)
    loss_actor = tf.math.reduce_mean(loss_actor)

    # clip fraction, see https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/ppo.py#L246 line 239ish
    clipped = tf.math.logical_or(tf.math.greater(ratio, 1 + policy_clip_ratio),
                                 tf.math.less(ratio, 1 - policy_clip_ratio))
    clipped = tf.cast(clipped, dtype=tf.float32)
    clip_frac = tf.math.reduce_mean(clipped)

    # PPO early stopping as implemented in baselines
    approx_kl = tf.math.reduce_mean(log_probs_old - log_probs_new)

    # collect entropy because why not. If this keeps growing we might have a too small memory and too smal batchsize
    entropy = dist.entropy()
    entropy = tf.reduce_mean(entropy)

    return loss_actor, approx_kl, entropy, clip_frac

def apply_gradients_to_model(model, gratient_tape, loss, g_grad_norm=None):
    actor_vars = model.trainable_variables
    actor_grads = gratient_tape.gradient(loss, actor_vars)
    g_norm = 0

    for grad in actor_grads:
        g_norm += tf.reduce_sum(tf.square(grad**2))
    g_norm = tf.sqrt(g_norm)

    if g_grad_norm is not None:
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, g_grad_norm)
    model.optimizer.apply_gradients(zip(actor_grads, actor_vars))

    return model, g_norm

def tb_plotter(data_list, summary_writer):
    with summary_writer.as_default():
        for entry in data_list:
            type = entry['type']
            name = entry['name']
            data = entry['data']
            step = entry['step']
            if type == 'scalar':
                tf.summary.scalar(name, data, step)
            elif type == 'histogram':
                if 'buckets' in entry:
                    buckets = entry['buckets']
                else:
                    buckets = None
                tf.summary.histogram(name, data, step, buckets=buckets)
            elif type == 'pseudo3D':
                if 'buckets' not in entry:
                    print('did not assign periodicity/buckets value,expect break')
                else:
                    periodicity = entry['buckets']

                to_be_discarded = data.shape[-1]%periodicity
                data = data[:-to_be_discarded]
                data = np.reshape(data, (-1,periodicity))
                data = np.average(data, axis=0)
                data = np.squeeze(data)

                pseudo_counts = []
                for t in range(periodicity):
                    count = int(data[t])
                    for _ in range(count):
                        pseudo_counts.append(t)
                tf.summary.histogram(name, pseudo_counts, step, buckets=buckets)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# this is a function to robustly pick the argmax in a random fashion if we happen to have several identically maximal values
def huber(x, epsilon=1e-10):
    x = tf.where(tf.math.greater(x, 1.0),
                             # essentially just huber function it so its bigger than 0
                             tf.abs(x),
                             tf.square(x))
    if epsilon > 0:
        x = tf.where(tf.math.greater(x, epsilon),
                     # essentially just huber function it so its bigger than 0
                     x,
                     epsilon)
    return x

def tf_shuffle_axis(value, axis=0, seed=None, name=None):
    perm = list(range(tf.rank(value)))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm))
    value = tf.transpose(value, perm=perm)
    return value

async def robust_argmax(tensor):
    max_value = tf.reduce_max(tensor)
    max_value_idxs = tf.where(tf.math.equal(max_value, tf.squeeze(tensor, axis=0)))
    random_max_value_idx = tf.random.shuffle(max_value_idxs)[0]
    return random_max_value_idx

async def smart_squeeze(x, remaining_dims=1): #X is a tensorflow tensor
    cardinality = len(x.get_shape())
    if cardinality <= remaining_dims:
        return x
    else:
        assert cardinality >= remaining_dims, "cannot squeeze below 0, pls check your dims goal"
        to_be_reduced = np.arange(cardinality - remaining_dims, dtype=int).tolist()
        x = tf.squeeze(x, axis=to_be_reduced)
        x = x.numpy().tolist()
        return x

class EarlyStopper:
    def __init__(self, patience=30, tolerance=1e-8):
        self.patience = patience
        self.tolerance = tolerance

        self.best_loss = np.inf
        self.iterations_not_improved = 0

        self.best_model = None

    def check_iteration(self, loss, model):
        stop_early = False

        if loss <= self.best_loss:
            self.best_loss = loss
            self.best_model = model
            self.iterations_not_improved = 0

        else:
            self.iterations_not_improved += 1

            if self.iterations_not_improved >= self.patience:
                stop_early = True
                model = self.best_model

        return stop_early, model

def normalize_buffer_entry(buffer, key): #taken from https://github.com/ray-project/ray/blob/70153f2d995c70167ac1f64b83c44a6029e290e6/rllib/utils/sgd.py standardized
    array = []
    for episode in buffer.keys():
        episode_array = [step[key] for step in buffer[episode]]
        array.extend(episode_array)

    mean = np.mean(array)
    std = np.std(array)

    for episode in buffer.keys():
        for t in range(len(buffer[episode])):
            buffer[episode][t][key] = (buffer[episode][t][key] - mean) / max(std + 1e-10, 1e-4)

    return buffer

def build_multivar(concentrations, dist, actions):
    # independent_dists = []
    # for action in args:
    #     c0 = args[action]['c0']
    #     c1 = args[action]['c1']
    #     dist_action = dist(c0, c1)
    args = {}
    c0s, c1s  = tf.split(concentrations, 2, -1)
    c0s = tf.split(c0s, len(actions), -1)
    #c0s = tf.concat(c0s, axis=-2)
    c1s = tf.split(c1s, len(actions), -1)
    #c1s = tf.concat(c1s, axis=-2)
    args['concentration0'] = tf.concat(c0s, axis=-1)
    args['concentration1'] = tf.concat(c1s, axis=-1)
    betas = dist(**args)
    # sample = betas.sample(1)
    multivar_dist = tfp.distributions.Independent(betas, reinterpreted_batch_ndims=1)
    # sample_dis = multivar_dist.sample(1)
    return multivar_dist

def assemble_subdict_batch(list_of_dicts, entries=None): #entries being a list of entries to include
    dict_of_lists = {}
    if entries is not None:
        list_of_dicts = [list_of_dicts[entry] for entry in entries]
    for dict in list_of_dicts:

        for key in dict:
            if key not in dict_of_lists.keys():
                dict_of_lists[key] = [dict[key]]
            else:
                dict_of_lists[key].append(dict[key])

    return dict_of_lists

#ToDo:
# in order to make this recurrent we'll need to: store states(to initialize)
# have a length argument for the trajectory we extract
class ExperienceReplay:
    def __init__(self, max_length=1e4, trajectory_length=1, action_types=None, multivariate=True):
        self.max_length = max_length
        self.buffer = {}  # a dict of lists, each entry is an episode which is itself a list of entries such as below
        self.last_episode = []
        self.action_types = action_types
        self.trajectory_length = trajectory_length  # trajectory length
        self.multivariate = multivariate

    def add_entry(self, episode=0, **kwargs):
        entry = {}
        for keyword in kwargs:
            entry[keyword] = kwargs[keyword]

        if episode not in self.buffer: #ToDo: we might need to change this for asynch stuff
            self.buffer[episode] = []
        self.buffer[episode].append(entry)

    def clear_buffer(self):
        self.buffer = {}

    async def generate_availale_indices(self):

        #get available indices
        available_indices = []
        for episode in self.buffer:
            for step in range(len(self.buffer[episode]) - self.trajectory_length):
                available_indices.append([episode, step])

        self.available_indices = available_indices
        return True

    def should_we_learn(self):
        buffer_length = 0
        for episode in self.buffer:
            buffer_length += len(self.buffer[episode])

        if buffer_length >= self.max_length:
            return True
        else:
            return False

    #ToDo: check the weighting math
    async def calculate_explained_variance(self):
        explained_variance_buffer = []
        num_entries = []
        for episode in self.buffer:
            V_theta = [step['values'] for step in self.buffer[episode]]
            G_traj = [step['returns'] for step in self.buffer[episode]]

            assert len(V_theta) == len(G_traj), "the number of values and returns is not equal, cannot calculate_explained_variance"
            explained_var_episode = _explained_variance(ypred=np.array(V_theta),
                                                y=np.array(G_traj))
            num_episode_entries = len(V_theta)
            num_entries.append(num_episode_entries)
            explained_variance_buffer.append(explained_var_episode)
        total_entries = sum(num_entries)
        mean_explained_variance = sum([(var*weight)/total_entries for [var, weight] in zip(explained_variance_buffer, num_entries)])

        return mean_explained_variance

    async def calculate_advantage(self, gamma=0.99, gae_lambda=0.95):
        for episode in self.buffer:
            # self.buffer = normalize_buffer_entry(self.buffer, key='rewards')

            r = [step['rewards'] for step in self.buffer[episode]]
            V = [step['values'] for step in self.buffer[episode]]
            G = [None for step in self.buffer[episode]]
            A = [None for step in self.buffer[episode]]
            # A.append(None)
            for t in reversed(range(len(r))):
                if t == len(r)-1: # last step
                    # G[t] = r[t] + V[t] #thats our bootstrap from the value estimate
                    # delta_t = r[t] - V[t]
                    G[t] = V[t]
                    delta_t = r[t] - V[t]
                    A[t] = delta_t
                else:
                    G[t] = r[t] + gamma*G[t+1]
                    delta_t = r[t] + gamma*V[t+1] - V[t]
                    A[t] = delta_t + gamma*gae_lambda*A[t+1]

            A = A[:len(r)]
            for t in range(len(r)):
                self.buffer[episode][t]['advantages'] = A[t]
                self.buffer[episode][t]['returns'] = G[t]
                self.buffer[episode][t]['v_target'] = A[t] + V[t]

        return True

    def _fetch_buffer_entry(self, batch_indices, key, subkeys=False, only_first_entry=False):
        #godl: trajectory_start:trajectory_start+self.trajectory_length
        # if subkeys: #for nested buffer entries
        #     fetched_entry = {}
        #     for subkey in subkeys:
        #         fetched_entry[subkey] = [self.buffer[sample_episode][trajectory_start][key][subkey]
        #                                       for [sample_episode, trajectory_start] in batch_indices]
        #
        # else:
        if self.trajectory_length <=1 or only_first_entry:
            fetched_entry = [self.buffer[sample_episode][trajectory_start][key]
                                          for [sample_episode, trajectory_start] in batch_indices]
        else:
            fetched_entry = []
            for [sample_episode, trajectory_start] in batch_indices:
                fetched_trajectory = [self.buffer[sample_episode][trajectory_start + step][key] for step in range(self.trajectory_length)]
                fetched_entry.append(fetched_trajectory)

        return fetched_entry

    def fetch_batch(self, batchsize=32, indices=None,
                    keys=['actions_taken', 'log_probs', 'observations', 'advantages', 'returns'],

                    ):

        np.random.shuffle(self.available_indices)
        batch_indices = self.available_indices[:batchsize]

        #ToDo: implement trajectories longer than 1, might be base on same code as DQN buffer

        batch = {}
        for key in keys:
            batch[key] = self._fetch_buffer_entry(batch_indices,
                                                  key,
                                                  only_first_entry= True if key in ['actor_states', 'critic_states', 'actor_critic_states'] else False)

        return batch