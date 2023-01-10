# from _clients.participants.participants import Residential

import tenacity
from datetime import datetime

from TREX_Core._agent._utils.metrics import Metrics
from TREX_Core._agent._utils.heuristics import PriceHeuristics, QuantityHeuristics
import asyncio
from TREX_Core._utils import jkson as json
# import serialize
from multiprocessing import shared_memory
import importlib
import numpy as np


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
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        ##### Setup the shared memory names based on config #####
        self.name = self.__participant['id']
        self.action_list_name = self.name + "_actions"
        self.observation_list_name = self.name + "_obs"
        self.reward_list_name = self.name + "_reward"
        '''
        Shared lists get initialized on TREXENV side, so all that the agents have to do is connect to their respective 
        observation and action lists. Agents dont have to worry about making the actions pretty, they just have to send
        them into the buffer. 
        '''

        self._check_sharedmemory()
        self.shared_list_action = shared_memory.ShareableList(name=self.action_list_name)
        self.shared_list_observation = shared_memory.ShareableList(name=self.observation_list_name)
        self.shared_list_reward = shared_memory.ShareableList(name=self.reward_list_name)


        #find the right default behaviors from kwargs['default_behaviors']
        self.observation_variables = kwargs['observations']

        #ToDo - Daniel - Think about a nicer way of doing this
        #decode actions, load heuristics if necessary
        self.allowed_actions = kwargs['actions']
        self.a_t = {}
        for action in kwargs['actions']:
            self.a_t[action] = None
            if kwargs['actions'][action]['heuristic'] != 'learned':
                heuristic = kwargs['actions'][action]['heuristic']
                if 'price' == action:
                    self.price_heuristic = PriceHeuristics(type=heuristic)
                elif 'quantity' == action:
                    self.quantity_heuristic = QuantityHeuristics(type=heuristic)
                elif 'storage' == action:
                    raise NotImplementedError
                else:
                    raise NotImplementedError


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
            self.profile_stats = await self.__participant['get_profile_stats']()

        #Todo: Daniel - check scales and why its fucked
        if 'generation' in self.observation_variables:
            self.obs_order.append('generation')

            if self.profile_stats:
                avg_generation = self.profile_stats['avg_generation'] #FixMe: (Daniel, Jan9th 2023) We need to add the scaling from the config here otherwise the mean will be wrong
                generation_scale = self.__participant['profile_params']['generation_scale']
                avg_generation = round(avg_generation, 4) * generation_scale #turn into W,
                obs_generation = round(obs_generation, 4)
                stddev_generation = self.profile_stats['stddev_generation']
                z_next_generation = (obs_generation - avg_generation) / (stddev_generation+ 1e-8)
                observations_t.append(z_next_generation)
            else:
                observations_t.append(obs_generation)

        if 'load' in self.observation_variables:
            self.obs_order.append('load')

            if self.profile_stats:
                avg_load = self.profile_stats['avg_consumption'] #FixMe: (Daniel, Jan9th 2023) We need to add the scaling from the config here otherwise the mean will be wrong
                load_scale = self.__participant['profile_params']['load_scale']
                avg_load = round(avg_load, 4)  * load_scale # turn into W
                obs_load = round(obs_load, 4)
                stddev_load = self.profile_stats['stddev_consumption']
                z_next_load = (obs_load - avg_load) / (stddev_load + 1e-8)
                observations_t.append(z_next_load)
            else:
                observations_t.append(obs_load)

        #ToDo - Daniel & Steven - get these from special market
        settle_stats = self.__participant['market_info']['settle_stats']
        if 'avg_bid_price_ls' in self.observation_variables:
            avg_bid_price_ls = settle_stats['weighted_avg_settlement_buy_price'] if 'weighted_avg_settlement_buy_price' in settle_stats else 0.1449  #ToDo: replace with automatic reading of config
            observations_t.append(avg_bid_price_ls)
            self.obs_order.append('avg_bid_price_ls')
        if 'avg_ask_price_ls' in self.observation_variables:
            avg_ask_price_ls = settle_stats['weighted_avg_settlement_sell_price'] if 'weighted_avg_settlement_sell_price' in settle_stats else 0.069 #ToDo: replace with automatic reading of config
            observations_t.append(avg_ask_price_ls)
            self.obs_order.append('avg_ask_price_ls')
        if 'avg_bid_quantity_ls' in self.observation_variables:
            avg_bid_quantity_ls = settle_stats['avg_settlement_quantity_buy'] if 'avg_settlement_quantity_buy' in settle_stats else 0.0
            observations_t.append(avg_bid_quantity_ls)
            self.obs_order.append('avg_bid_quantity_ls')
        if 'avg_ask_quantity_ls' in self.observation_variables:
            avg_ask_quantity_ls = settle_stats['avg_settlement_quantity_sell'] if 'avg_settlement_quantity_sell' in settle_stats else 0.0
            observations_t.append(avg_ask_quantity_ls)
            self.obs_order.append('avg_bid_quantity_ls')
        if 'total_quantity_ls' in self.observation_variables:
            total_quantity_ls = settle_stats['total_settled_quantity'] if 'total_settled_quantity' in settle_stats else 0.0
            observations_t.append(total_quantity_ls)
            self.obs_order.append('total_quantity_ls')

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
        if reward is not None:
            await self.obs_to_shared_memory(obs_t)
        #
            await self.r_to_shared_memory(reward)
        #
            await self.read_action_values()

        await self.get_heuristic_actions(ts_act=ts_act)
        '''
        #########################################################################
        it is here that we wait for the action values to be written from epymarl
        #########################################################################
        '''
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

    async def get_heuristic_actions(self, ts_act):
        act_generation, act_load = await self.__participant['read_profile'](ts_act)
        # print('Gym netloads ts_act', act_generation, act_load)
        heuristic_info = {'load': act_load,
                          'generation': act_generation
                          }
        for action in self.allowed_actions:
            if self.allowed_actions[action]['heuristic'] !=  'learned':
                if action == 'price':
                    self.a_t[action] = self.price_heuristic.get_value(**heuristic_info)
                elif action == 'quantity':
                    self.a_t[action] = self.quantity_heuristic.get_value(**heuristic_info)
                elif action == 'storage':
                    raise NotImplementedError
                else:
                    print('did not recognize action key', action)
                    raise NotImplementedError

        # print('Gym self.a_t', self.a_t)
        return

    async def decode_actions(self, ts_act):
        """
        TODO: November 30, 2022: this method will be used to decode the actions that are received from epymarl.
        #one price, one quantity for now
        if quantity > 0:
            we ask --> we only need
            bid beomes quantiity = 0, price = 0
        else
            we bid
            ask becomes quantity = 0, price = 0
        """

        if 'price' in self.a_t:
            price = self.a_t['price']
            price = round(price, 4) if price is not None else 0.0
        else:
            price = 0.0

        if "quantity" in self.a_t:
            quantity = self.a_t['quantity']
            quantity = quantity if quantity is not None else 0
        else:
            quantity = 0

        actions = dict()
        # print(action_indices)
        # price = self.actions['price'][action_indices['price']]
        # quantity = self.actions['quantity'][action_indices['quantity']]

        if quantity >= 0:
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


        if "storage" in self.a_t:
            storage = self.a_t['storage']
            storage = int(storage) if storage != None else 0

            actions['bess'] = {
                str(ts_act): storage
            }

        return actions

    async def check_read_flag(self, shared_list):
        """
        This method checks the read flag in a shared list.
        Parameters:
            Shared_list -> shared list object to check, assumes that element 0 is the flag and that flag can be
                            intepreted as boolean
            returns ->  Boolean
        """
        try:
            if shared_list[0]:
                return True
            else:
                return False
        except:
            print(shared_list)

    async def read_action_values(self):
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
        shared_list_keys = ['flag', 'price', 'quantity' 'storage']
        flag = False
        while not flag:
            flag = await self.check_read_flag(self.shared_list_action)
            if flag:
                # ToDo: Daniel or Peter - reformat this to a dictionary so every actionn gets explicitly assigned its entry
                #read the buffer
                for action in shared_list_keys:
                    if action in self.allowed_actions:
                        if self.allowed_actions[action]['heuristic'] == 'learned':
                            key_idx = shared_list_keys.index(action)
                            self.a_t[action] = self.shared_list_action[key_idx]

                # print('actions', self.a_t[key])
                #reset the flag

        await self.write_flag(self.shared_list_action, False) #this sets flag to false for the next step?

    async def write_flag(self, shared_list, flag):
        """
        This method sets the flag
        Parameters:
            shared_list ->  shared list object to be modified
            flag -> boolean that indicates write 0 or 1. True sets 1
        """

        if flag:
            shared_list[0] = 1

        else:
            shared_list[0] = 0

    async def obs_to_shared_memory(self, obs):
        """
        This method writes the values in the observations array to the observation buffer and then sets the flag for
        EPYMARL to read the values.

        """

        # obs will be an array
        # pack the values of the obs array into the shares list
        for e, item in enumerate(obs):
            # print(e, item)
            self.shared_list_observation[e+1] = item

        #set the observation flat to written
        await self.write_flag(self.shared_list_observation,True)

    async def r_to_shared_memory(self, reward):
        """
        This method writes the reward value into the rewards array and then sets the flag for EPYMARL to read the
        values.
        """
        self.shared_list_reward[-1] = reward
        await self.write_flag(self.shared_list_reward, True)




