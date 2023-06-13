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
    async def pre_process_obs(self, ts_obs):
        # print('entered preprocessing')
        # ToDo: add histograms for observations
        self.obs_order = []
        data_for_tb = []

        obs_generation, obs_load = await self.__participant['read_profile'](ts_obs)

        observations_t = []
        if not hasattr(self, 'profile_stats'):
            # self.profile_stats = await self.__participant['get_profile_stats']()
            #ToDo: at some point reintroduce normalization here
            self.profile_stats = {}
            self.profile_stats['avg_generation'] = 0
            self.profile_stats['stddev_generation'] = 1

            self.profile_stats['avg_consumption'] = 0
            self.profile_stats['stddev_consumption'] = 1

        if 'generation' in self.observation_variables:
            self.obs_order.append('generation')

            if self.profile_stats:
                avg_generation = self.profile_stats['avg_generation'] #FixMe: (Daniel, Jan9th 2023) We need to add the scaling from the config here otherwise the mean will be wrong
                generation_scale = self.__participant['profile_params']['generation_scale']
                avg_generation = round(avg_generation*generation_scale, 4) #turn into W,
                obs_generation = round(obs_generation, 4)
                stddev_generation = self.profile_stats['stddev_generation']
                z_next_generation = (obs_generation - avg_generation) / max(stddev_generation, 1e-8)
                observations_t.append(z_next_generation)
            else:
                observations_t.append(obs_generation)

        if 'load' in self.observation_variables:
            self.obs_order.append('load')

            if self.profile_stats:
                avg_load = self.profile_stats['avg_consumption'] #FixMe: (Daniel, Jan9th 2023) We need to add the scaling from the config here otherwise the mean will be wrong
                load_scale = self.__participant['profile_params']['load_scale']
                avg_load = round(avg_load* load_scale, 4)   # turn into W
                obs_load = round(obs_load, 4)
                stddev_load = self.profile_stats['stddev_consumption']
                z_next_load = (obs_load - avg_load) / max(stddev_load, 1e-8)
                observations_t.append(z_next_load)
            else:
                observations_t.append(obs_load)

        #ToDo - Daniel & Steven - get these from special market
        settle_stats = self.__participant['market_info']['settle_stats']

        #get grid prices to normalize, if necessary
        price_in_obs = ['price' in obs for obs in self.observation_variables]
        if any(price_in_obs) and len(settle_stats) > 0:
            # FixMe: atm we do not know if it makes sense to normalize market price based on this?
            # FixMe: this also assumes we cannot bid/ask above grid sell/buy price
            participant = self.__participant
            if 'settled_time' in settle_stats:
                ts_settle_stats = settle_stats['settled_time']
                ts_settle_stats = str(tuple(ts_settle_stats))
            else:
                ts_settle_stats = None
                raise NotImplementedError('Settle stats not available')

            #FixMe: why cant we just do this?
            # assert ts_settle_stats in participant['market_info']
            if ts_settle_stats in participant['market_info']:
                grid_stats = participant['market_info'][ts_settle_stats]
                grid_sell_price = grid_stats['grid']['sell_price']
                grid_buy_price = grid_stats['grid']['buy_price']
                assert grid_buy_price >= grid_sell_price, 'grid buy price should be higher than grid sell price'
            else:
                grid_sell_price = 0.069
                grid_buy_price = 0.1449

            def normalize_price(price):
                return (price - grid_sell_price) / (grid_buy_price - grid_sell_price)



        for obs in self.observation_variables:
            #settle stats keys:
            # {'settled_time': [1433145600, 1433149200],
            # 'total_settled_quantity': 0, 'avg_settlement_quantity_sell': 0, 'avg_settlement_quantity_buy': 0,
            #  'avg_settlement_sell_price_kWh': 0.1449, 'avg_settlement_buy_price_kWh': 0.069,
            #  'min_ask_price': 0.1449, 'max_ask_price': 0.1449, 'avg_ask_price_kWh': 0.1449,
            #  'total_ask_quantity': 0, 'avg_ask_quantity': 0,
            #  'min_bid_price': 0.069, 'max_bid_price': 0.069, 'avg_bid_price_kWh': 0.069,
            #  'total_bid_quantity': 0, 'avg_bid_quantity': 0}

            if obs not in ['generation', 'load'] and obs in self.__participant['market_info']['settle_stats']:
                o_t = self.__participant['market_info']['settle_stats'][obs]
                if 'price' in obs:
                    o_t = normalize_price(o_t)

                observations_t.append(o_t)
                self.obs_order.append(obs)

        # if total_quantity_ls > 0:
        #    print(total_quantity_ls, 'Wh settled, at price of', settle_stats['weighted_avg_settlement_buy_price'], 'for buy and', settle_stats['weighted_avg_settlement_sell_price'], 'for sell')
        # ToDo - Daniel - there should be an inbuilt conversion for these formats

        timestamp = ts_obs[0]
        # dt = datetime.fromtimestamp(ts_obs[0])
        # dt_asdelta = dt - datetime.min
        # dt_seconds = dt_asdelta.total_seconds()

        ts_to_minutes = 1/60
        ts_to_hour = ts_to_minutes*(1/60)
        ts_to_day = ts_to_hour*(1/24)
        ts_to_week = ts_to_day * (1 / 7)
        ts_to_year = ts_to_day * (1 / 365)

        # ToDo - Daniel - get rid of ugly if loop
        if 'time_sin_hour' in self.observation_variables:
            self.obs_order.append('time_sin_hour')
            hour_in_day = timestamp *ts_to_hour
            time_sin_hour=np.sin(2 * np.pi *hour_in_day )
            observations_t.append(time_sin_hour)

        if 'time_cos_hour' in self.observation_variables:
            self.obs_order.append('time_cos_hour')
            hour_in_day = timestamp * ts_to_hour
            time_cos_hour =np.cos(2 * np.pi * hour_in_day)
            observations_t.append(time_cos_hour)

        if 'time_sin_day' in self.observation_variables:
            self.obs_order.append('time_sin_day')
            daytype = timestamp *ts_to_day
            time_sin_day=np.sin(2 * np.pi * daytype)
            observations_t.append(time_sin_day)

        if 'time_cos_day' in self.observation_variables:
            self.obs_order.append('time_cos_day')
            daytype = timestamp * ts_to_day
            time_cos_day=np.cos(2 * np.pi * daytype)
            observations_t.append(time_cos_day)

        if 'time_sin_dayinyear' in self.observation_variables:
            self.obs_order.append('time_sin_year')
            day_in_year = timestamp *ts_to_year
            time_sin_dayinyear = np.sin(2 * np.pi * day_in_year)
            observations_t.append(time_sin_dayinyear)

        if 'time_cos_dayinyear' in self.observation_variables:
            self.obs_order.append('time_cos_year')
            day_in_year = timestamp * ts_to_year
            time_cos_dayinyear=np.cos(2 * np.pi * day_in_year)
            observations_t.append(time_cos_dayinyear)

        if 'soc' in self.observation_variables:
            self.obs_order.append('soc')
            storage_schedule = await self.__participant['storage']['check_schedule'](ts_obs)
            soc = storage_schedule[ts_obs]['projected_soc_end']
            observations_t.append(soc)

        return observations_t

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

        #the timestep we observe vs the timestep we act on
        ts_obs = self.next_settle
        ts_act = self.next_settle #ToDo: All - discuss iff shifting battery to last settle makes sense

        #calculations for reward time offset
        n_rounds_act_to_r = (ts_act[0] - self.last_settle[0])/self.round_duration
        n_rounds_obs_to_act = (ts_obs[0] - ts_act[0])/self.round_duration
        n_rounds_obs_to_r = n_rounds_obs_to_act + n_rounds_act_to_r
        n_rounds_current_to_r = (ts_obs[0] - self.current_round[0])/self.round_duration + n_rounds_obs_to_r


        obs_t = await self.pre_process_obs(ts_obs)
        # print('Agent Observations', obs_t)

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

        action_dict_t = await self.decode_actions(ts_act)
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

    async def decode_actions(self, ts_act):
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
        price_bid = round(price_bid, 4)
        assert 'quantity_bid' in self.a_t, 'quantity_bid not in actions, must be a float or int supplied by external heuristic'
        quantity_bid = self.a_t['quantity_bid']
        quantity_bid = round(quantity_bid, 4)
        # print the quantity bid in case of error
        assert quantity_bid >= 0, 'quantity_bid must be greater equal 0, quantity_bid: {}'.format(quantity_bid)
        actions['bids'] = { str(ts_act): {'quantity': quantity_bid, 'price': price_bid }}

        # asks, we will dump everything into the solar pool for now
        assert 'price_ask' in self.a_t, 'price_ask not in actions, must be a float supplied by external heuristic'
        price_ask = self.a_t['price_ask']
        price_ask = round(price_ask, 4)
        assert 'quantity_ask' in self.a_t, 'quantity_ask not in actions, must be a float or int supplied by external heuristic'
        quantity_ask = self.a_t['quantity_ask']
        quantity_ask = round(quantity_ask, 4)
        assert quantity_ask >= 0, 'quantity_ask must be greater equal 0, quantity_bid: {}'.format(quantity_bid)
        actions['asks'] = {'solar': { str(ts_act): {'quantity': quantity_ask, 'price': price_ask }}}

        assert 'storage' in self.a_t, 'storage not in actions, must be a float or int supplied by external heuristic'
        storage = self.a_t['storage']
        storage = int(storage) #ToDo: bug steven to update the battery model to accept floats
        actions['bess'] = { str(ts_act): storage }

        return actions

    async def read_action_from_sml(self):
        """
        This method checks the action buffer flag and if the read flag is set, it reads the value in the buffer and stores
        them in a_t

        # TODO: write conversion into dictionary

        Bid related asks
        bid_price = self.shared_list_action[0]
        bid_quantity = self.shared_list_action[1]

        Solar related asks
        solar_ask_price = self.shared_list_action[2]
        solar_ask_quantity = self.shared_list_action[3]

        Bess related asks
        bess_ask_price = self.shared_list_action[4]
        bees_ask_quantity = self.shared_list_action[5]

        """
        # check the action flag
        sml_actions = shared_memory.ShareableList(name=self.action_list_name)
        flag = False
        while not flag: #wait for the flag to be set

            flag = sml_actions[0]

            if flag:
                #read the buffer
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
        for e, item in enumerate(obs):
            # print(e, item)
            sml_obs[e+1] = item #so we respect the flag
        sml_obs[0] = True #setting flag to true

    async def write_r_to_sml(self, reward):
        """
        This method writes the reward value into the rewards array and then sets the flag for EPYMARL to read the
        values.
        """

        # FixMe: this could be brittle if execution speed becomes too fast!
        sml_reward = shared_memory.ShareableList(name=self.reward_list_name)
        sml_reward[1] = reward
        sml_reward[0] = True #setting flag to




