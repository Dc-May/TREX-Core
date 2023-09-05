# from _clients.participants.participants import Residential

import tenacity
from TREX_env._utils.sml_utils import read_flag_x_times
from TREX_Core._agent._utils.metrics import Metrics
import asyncio
# import serialize
from multiprocessing import shared_memory
import importlib
import numpy as np

#ToDo: make all actions learnable (ask price, ask quan, bid_price, bid_quan, battery)
#ToDo: make the agent not wait until reward is a number, just pass through!


class Trader:
    """
    Class: Trader
    This class implements the gym compatible trader that is used in tandem with EPYMARL TREXEnv.

    """
    def __init__(self, **kwargs):
        """
        Initializes the trader using the parameters in the TREX_Core config that was selected.
        In particular, this sets up the connections to the shared lists that are established in EPYMARL TREXEnv or any
        other process that seeks to interact with TREX. These lists need to be already initialized before the gym trader
        attempts to connect with them.

        params: kwargs -> dictionary created from the config json file in TREX_Core._configs
        """
        # Some util stuffies
        # print('GOT TO THE GYM_AGENT INIT')
        self.t_acts = 0 # number of actions taken
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        ##### Setup the shared memory names based on config #####
        #ToDo: find a way to add the env number here
        self.name = self.__participant['id']

        assert 'env_id' in kwargs, 'Expected to find env_id in kwargs, needed to connect to shared memory manager'
        env_info = kwargs['env_id']
        #make sure 'env_id', 'smm_hash', 'smm_address' and 'smm_port' in kwargs

        env_id = kwargs['env_id']
        self.action_list_name = self.name+'_' + str(env_id) +  '_actions'
        # print('Gym agent', self.name, 'action list name', self.action_list_name, flush=True)
        self.observation_list_name = self.name+'_' + str(env_id) +  '_obs'
        # print('Gym agent', self.name, 'observation list name', self.observation_list_name, flush=True)
        self.reward_list_name = self.name+'_' + str(env_id) +  '_reward'
        # print('Gym agent', self.name, 'reward list name', self.reward_list_name, flush=True)
        ''' 
        Shared lists get initialized on TREXENV side, so all that the agents have to do is connect to their respective 
        observation and action lists. Agents dont have to worry about making the actions pretty, they just have to send
        them into the buffer. 
        '''

        # self._check_sharedmemory()
        # self.shared_list_action = shared_memory.ShareableList(name=self.action_list_name)
        # self.shared_list_observation = shared_memory.ShareableList(name=self.observation_list_name)
        # self.shared_list_reward = shared_memory.ShareableList(name=self.reward_list_name)


        #find the right default behaviors from kwargs['default_behaviors']
        self.observation_variables = kwargs['observations']

        #ToDo - Daniel - Think about a nicer way of doing this
        #decode actions, load heuristics if necessary
        self.allowed_actions = kwargs['actions']
        self.a_t = {}
        for action in kwargs['actions']:
            self.a_t[action] = None
            # Deprecated
            if kwargs['actions'][action]['heuristic'] != 'learned':
                raise NotImplementedError('Only learned actions are supported in the gym agent. Please reassign ', action, 'to learned', flush=True)
            # Deprecated
            #     heuristic = kwargs['actions'][action]['heuristic']
            #     if 'price' == action:
            #         self.price_heuristic = PriceHeuristics(type=heuristic)
            #     elif 'quantity' == action:
            #         self.quantity_heuristic = QuantityHeuristics(type=heuristic)
            #     elif 'storage' == action:
            #         raise NotImplementedError
            #     else:
            #         raise NotImplementedError


        # TODO: Find out where the action space will be defined: I suspect its not here
        # Initialize the agent learning parameters for the agent (your choice)
        # self.bid_price = kwargs['bid_price'] if 'bid_price' in kwargs else None
        # self.ask_price = kwargs['ask_price'] if 'ask_price' in kwargs else None

        ####### Metrics tracking initialization ########
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        ###### Reward function intialization from config #######
        reward_function = kwargs['reward_function']
        if reward_function:
            self._rewards = importlib.import_module('TREX_Core._agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

       #  print('init done')

    @tenacity.retry(wait=tenacity.wait_fixed(3))
    def _check_sharedmemory(self):
        """
        This method checks if all the shared memory arrays have been create by the EPYMARL process before having them
        initialized by the gym_agents.
        """

        shared_action_list = shared_memory.ShareableList(name=self.action_list_name)

        shared_observation_list = shared_memory.ShareableList(name=self.observation_list_name)

        shared_reward_list = shared_memory.ShareableList(name=self.reward_list_name)

        return True

    def __init_metrics(self):
        import sqlalchemy
        '''
        Pretty self explanitory, this method resets the metric lists in 'agent_metrics' as well as zeroing the metrics dictionary. 
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)

        # if self.battery:
        #     self.metrics.add('battery_action', sqlalchemy.Integer)
        #     self.metrics.add('state_of_charge', sqlalchemy.Float)

    # Core Functions, learn and act, called from outside
    async def pre_process_obs(self):
        # print('entered preprocessing')

        # we need to make sure that the observation get put into the right order
        obs_t_dict = {key: None for key in self.observation_variables}

        if 'reward_time_lag' in self.observation_variables:
            # calculations for reward time offset
            n_rounds_act_to_r = (self.next_settle[0] - self.last_settle[0]) / self.round_duration
            n_rounds_obs_to_act = (self.next_settle[0] - self.next_settle[0]) / self.round_duration # this is a remainder from earlier timing considerations, kept in for future flexibility
            n_rounds_obs_to_r = n_rounds_obs_to_act + n_rounds_act_to_r
            n_rounds_current_to_r = (self.next_settle[0] - self.current_round[0]) / self.round_duration + n_rounds_obs_to_r
            obs_t_dict['reward_time_lag'] = n_rounds_current_to_r

        if "t_now" in self.observation_variables:
            obs_t_dict['t_now'] = self.current_round[0]
        if 'generation_now' or 'load_now' in self.observation_variables:
            gen_now, load_now = await self.__participant['read_profile'](self.current_round)
            if 'generation_now' in self.observation_variables:
                obs_t_dict['generation_now'] = gen_now
            if 'load_now' in self.observation_variables:
                obs_t_dict['load_now'] = load_now

        if "t_settle" in self.observation_variables:
            obs_t_dict['t_settle'] = self.next_settle[0]
        if 'generation_settle' or 'load_settle' in self.observation_variables:
            gen_settle, load_settle = await self.__participant['read_profile'](self.next_settle)
            if 'generation_settle' in self.observation_variables:
                obs_t_dict['generation_settle'] = gen_settle
            if 'load_settle' in self.observation_variables:
                obs_t_dict['load_settle'] = load_settle



        if "t_deliver" in self.observation_variables:
            obs_t_dict['t_deliver'] = self.next_settle[0] + self.round_duration
        if 'generation_deliver' or 'load_deliver' in self.observation_variables:
            next_deliver = (self.next_settle[0] + self.round_duration, self.next_settle[1] + self.round_duration)
            gen_deliver, load_deliver = await self.__participant['read_profile'](next_deliver)
            if 'generation_deliver' in self.observation_variables:
                obs_t_dict['generation_deliver'] = gen_deliver
            if 'load_deliver' in self.observation_variables:
                obs_t_dict['load_deliver'] = load_deliver

        if 'SoC' in self.observation_variables:
            storage_schedule = await self.__participant['storage']['check_schedule'](self.next_settle)
            soc = storage_schedule[self.next_settle]['projected_soc_end']
            obs_t_dict['SoC'] = soc

        # collect the settle stats if necessary
        settle_stats = self.__participant['market_info']['settle_stats']
        participant = self.__participant
        if 'settled_time' in settle_stats:
            ts_settle_stats = settle_stats['settled_time']
            ts_settle_stats = str(tuple(ts_settle_stats))

            if ts_settle_stats in participant['market_info']:
                grid_stats = participant['market_info'][ts_settle_stats]
                grid_sell_price = grid_stats['grid']['sell_price']
                grid_buy_price = grid_stats['grid']['buy_price']
                assert grid_buy_price >= grid_sell_price, 'grid buy price should be higher than grid sell price'
            else:
                print('grid stats not available for participant', participant['id'], 'at timestep', self.next_settle, flush=True)
                # raise ValueError('Grid stats not available')
        else:
            print('settle stats not available for participant', participant['id'], 'at timestep', self.next_settle, flush=True)
            # raise ValueError('Settle stats not available')

        for obs in self.observation_variables:
            if obs in settle_stats:
                o_t = self.__participant['market_info']['settle_stats'][obs]
                obs_t_dict[obs] = o_t
                #raise ValueError('Observation variable not available')

        # now we convert the dict into a list, so we maintain the order of the original observation_variables list
        obs_list = [obs_t_dict[obs] for obs in self.observation_variables]
        return obs_list

    async def act(self, **kwargs):
        """


        """

        '''
        actions are none so far
        ACTIONS ARE FOR THE NEXT settle!!!!!

        actions = {
            'bess': {
                time_interval: scheduled_qty
            },
            'bids': {
                time_interval: {
                    'quantity': qty,
                    'price': dollar_per_kWh
                }
            },
            'asks': {
                source:{
                     time_interval: {
                        'quantity': qty,
                        'price': dollar_per_kWh?
                     }
                 }
            }
        sources inclued: 'solar', 'bess'
        Actions in the shared list 
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        }
        '''
        # print("in agent.act")
        ##### Initialize the actions
        # print('entered act')
        # self.t_acts += 1
        # print('t_acts', self.t_acts, flush=True)

        actions = {}
        # TODO: these are going to have to go into the obs_creation method, waiting on daniel for these
        bid_price = 0.0
        bid_quantity = 0.0
        solar_ask_price = 0.0
        solar_ask_quantity = 0.0
        bess_ask_price = 0.0
        bees_ask_quantity = 0.0

        # Timing information
        timing = self.__participant['timing']
        self.current_round = self.__participant['timing']['current_round']
        self.round_duration = self.__participant['timing']['duration']
        self.next_round = (self.current_round[0]+self.round_duration, self.current_round[1]+self.round_duration)
        self.last_settle = self.__participant['timing']['last_settle']
        self.next_settle = self.__participant['timing']['next_settle']
        self.last_round = timing['last_round']

        obs_t = await self.pre_process_obs()
        # print('Agent Observations', obs_t, flush=True)

        #### Send rewards into reward buffer:
        reward = await self._rewards.calculate()

        #if we get rewards we pass obs, etc to GYM
        # this is not the optimal way of doing this but it is going to allow us to keep everything outside of gym clean
        #ToDO: all - look for better solutions
        '''
        #########################################################################
        it is here that we wait for the action values to be written from Gym
        #########################################################################
        '''

        # if reward is not None: Deprecated behavior
        await self.write_obs_to_sml(obs_t)
    #
        await self.write_r_to_sml(reward)
    #
        await self.read_action_from_sml()

        # await self.get_heuristic_actions(ts_act=ts_act) #Deprecated
        # wait for the actions to come from EPYMARL

        # actions come in with a set order, they will need to be split up

        action_dict_t = await self.decode_actions()
        #     }
        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', action_dict_t),
                # self.metrics.track('next_settle_load', load),
                # self.metrics.track('next_settle_generation', generation)
                )

            await self.metrics.save(10000)
        # print("gym agent action_dict_t", action_dict_t)
        return action_dict_t

    async def step(self):
        # actions must come in the following format:
        # actions = {
        #     'bess': {
        #         time_interval: scheduled_qty
        #     },
        #     'bids': {
        #         time_interval: {
        #             'quantity': qty,
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks' {
        #         source: {
        #             time_interval: {
        #                 'quantity': qty,
        #                 'price': dollar_per_kWh?
        #             }
        #         }
        #     }
        #
        next_actions = await self.act()
        return next_actions

    async def reset(self, **kwargs):
        return True

    async def decode_actions(self):
        """
        We decode the external actions into the action dict to be fed into the market.
        As we are expecting an external heuristic to be feeding the gym agent, we want:
        1. A bid price and quantity
        2. A solar ask price and quantity
        3. A storage action

        """
        actions = dict()

        # bids
        assert 'price_bid' in self.a_t, 'price_bid not in actions, must be a float supplied by external heuristic'
        price_bid = self.a_t['price_bid']
        # price_bid = round(price_bid, 4)
        assert 'quantity_bid' in self.a_t, 'quantity_bid not in actions, must be a float or int supplied by external heuristic'
        quantity_bid = self.a_t['quantity_bid']
        # quantity_bid = round(quantity_bid, 4)
        # print the quantity bid in case of error
        assert quantity_bid >= 0, 'quantity_bid must be greater equal 0, quantity_bid: {}'.format(quantity_bid)
        if quantity_bid > 0:
            actions['bids'] = { str(self.next_settle): {'quantity': quantity_bid, 'price': price_bid }}

        # asks, we will dump everything into the solar pool for now
        assert 'price_ask' in self.a_t, 'price_ask not in actions, must be a float supplied by external heuristic'
        price_ask = self.a_t['price_ask']
        # price_ask = round(price_ask, 4)
        assert 'quantity_ask' in self.a_t, 'quantity_ask not in actions, must be a float or int supplied by external heuristic'
        quantity_ask = self.a_t['quantity_ask']
        assert quantity_ask >= 0, 'quantity_ask must be greater equal 0, quantity_bid: {}'.format(quantity_bid)
        if quantity_ask > 0:
            # quantity_ask = round(quantity_ask, 4)
            actions['asks'] = {'solar': { str(self.next_settle): {'quantity': quantity_ask, 'price': price_ask }}}

        assert 'storage' in self.a_t, 'storage not in actions, must be a float or int supplied by external heuristic'
        storage = self.a_t['storage']
        storage = int(storage) #ToDo: bug steven to update the battery model to accept floats
        actions['bess'] = { str(self.next_settle): storage }

        return actions

    async def read_action_from_sml(self):
        """
        This method checks the action buffer flag and if the read flag is set, it reads the value in the buffer and stores
        them in a_t


        """
        # check the action flag
        sml_actions = shared_memory.ShareableList(name=self.action_list_name)
        while not sml_actions[0]: #check the flag, if it indicates ready to read then read actions. We would expect the flag to be true by now
            await asyncio.sleep(0.001)

        #now, that sml[0] is True, we read the actions
        for action in self.a_t:
            if action in self.allowed_actions and self.allowed_actions[action]['heuristic'] == 'learned':
                sml_action_index = list(self.a_t.keys()).index(action) + 1 #because we need to respect the flag!
                self.a_t[action] = sml_actions[sml_action_index]

        sml_actions[0] = False #set flag to false

    async def write_obs_to_sml(self, obs):
        """
        This method writes the values in the observations array to the observation buffer and then sets the flag for
        EPYMARL to read the values.

        """
        # obs will be an array
        # pack the values of the obs array into the shares list
        # FixMe: this could be brittle if execution speed becomes too fast!
        sml_obs = shared_memory.ShareableList(name=self.observation_list_name)

        while sml_obs[0]: #the flag should be false, indicating a ready to be written. If thats not the case we wait
            await asyncio.sleep(0.001) #wait for 1ms and then check again

        for e, item in enumerate(obs):
            # print(e, item)
            sml_obs[e+1] = item #so we respect the flag
        sml_obs[0] = True #setting flag to True, indicating ready to be read

    async def write_r_to_sml(self, reward):
        """
        This method writes the reward value into the rewards array and then sets the flag for EPYMARL to read the
        values.
        """

        # FixMe: this could be brittle if execution speed becomes too fast!
        sml_reward = shared_memory.ShareableList(name=self.reward_list_name)
        while sml_reward[0]: #the flag should be false, indicating a ready to be written. If thats not the case we wait
            await asyncio.sleep(0.001) #wait for 1ms and then check again

        sml_reward[1] = reward
        sml_reward[0] = True #setting flag to True, indicating ready to be read




