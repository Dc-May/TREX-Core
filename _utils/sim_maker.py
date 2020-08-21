import os
import filecmp
import shutil
import commentjson
import gzip
import random
from _utils import launcher_maker
from _utils import db_utils
from _utils import jkson as json
import tenacity
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Column
from sqlalchemy_utils import database_exists, create_database, drop_database
import dataset
from packaging import version
import time

class Maker:
    def __init__(self, config, resume=False, **kwargs):
        self.configs = self.__get_config(config, resume, **kwargs)
        self.__config_version_valid = bool(version.parse(self.configs['version']) >= version.parse("3.6.0"))

        if not resume:
            r = tenacity.Retrying(
                wait=tenacity.wait_fixed(1))
            r.call(self.__make_sim_path)

    def __get_config(self, config_name:str, resume, **kwargs):
        config_file = '_simulations/_configs/' + config_name + '.json'
        with open(config_file) as f:
            config = commentjson.load(f)

        if 'name' in config['study'] and config['study']['name']:
            study_name = config['study']['name'].replace(' ', '_')
        else:
            study_name = config_name
        db_string = config['study']['output_db_location'] + '/' + study_name
        engine = create_engine(db_string)

        if resume:
            if 'db_string' in kwargs:
                db_string = kwargs['db_string']
            # look for existing db in db. if one exists, return it
            if database_exists(db_string):
                if engine.dialect.has_table(engine, 'configs'):
                    db = dataset.connect(db_string)
                    configs_table = db['configs']
                    configs = configs_table.find_one(id=0)['data']
                    configs['study']['resume'] = resume
                    return configs

        config['study']['name'] = study_name
        if 'output_database' not in config['study'] or not config['study']['output_database']:
            config['study']['output_database'] = db_string

        if database_exists(db_string):
            drop_database(db_string)
        if not database_exists(db_string):
            db_utils.create_db(db_string)

        self.__create_configs_table(db_string)
        db = dataset.connect(db_string)
        configs_table = db['configs']
        configs_table.insert({'id': 0, 'data': config})

        config['study']['resume'] = resume
        return config

    def __make_sim_path(self):
        output_path = self.configs['study']['sim_root'] + '_simulations/' + self.configs['study']['name'] + '/'
        print(output_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

    # Give starting time for simulation
    def __get_start_time(self, generation):
        import pytz
        import math
        from dateutil.parser import parse as timeparse
        #  TODO: NEED TO CHECK ALL DATABASES TO ENSURE THAT THE TIME RANGE ARE GOOD
        start_datetime = self.configs['study']['start_datetime']
        start_timezone = self.configs['study']['timezone']

        # If start_datetime is a single time, set that as start time
        if isinstance(start_datetime, str):
            start_time = pytz.timezone(start_timezone).localize(timeparse(start_datetime))
            return int(start_time.timestamp())

        # If start_datetime is formatted as a time step with beginning and end, choose either of these as a start time
        # If sequential is set then the startime will 
        if isinstance(start_datetime, (list, tuple)):
            if len(start_datetime) == 2:
                start_time_s = int(pytz.timezone(start_timezone).localize(timeparse(start_datetime[0])).timestamp())
                start_time_e = int(pytz.timezone(start_timezone).localize(timeparse(start_datetime[1])).timestamp())
                # This is the sequential startime code 
                if 'start_datetime_sequence' in self.configs['study']:
                    if self.configs['study']['start_datetime_sequence'] == 'sequential':
                        interval = int((start_time_e-start_time_s) / self.configs['study']['generations']/60)*60
                        start_time = range(start_time_s, start_time_e, interval)[generation]
                        return start_time
                start_time = random.choice(range(start_time_s, start_time_e, 60))
                return start_time
            else:
                 if 'start_datetime_sequence' in self.configs['study']:
                    if self.configs['study']['start_datetime_sequence'] == 'sequential':
                        multiplier = math.ceil(self.configs['study']['generations']/len(start_datetime))
                        start_time_readable = start_datetime*multiplier[generation]
                        start_time = pytz.timezone(start_timezone).localize(timeparse(start_time_readable))
                        return start_time
                 start_time = pytz.timezone(start_timezone).localize(timeparse(random.choice(start_datetime)))
                 return int(start_time.timestamp())

    def __make_sim_internal_directories(self, config=None):
        if not config:
            config = self.configs
        # make sim directories and shared settings files
        sim_path = self.configs['study']['sim_root'] + '_simulations/' + config['study']['name'] + '/'
        if not os.path.exists(sim_path):
            os.mkdir(sim_path)

        engine = create_engine(self.configs['study']['output_database'])
        if not engine.dialect.has_table(engine, 'metadata'):
            self.__create_metadata_table(self.configs['study']['output_database'])
        db = dataset.connect(self.configs['study']['output_database'])
        metadata_table = db['metadata']
        for generation in range(config['study']['generations']):
            # check if metadata is in table
            # if not, then add to table
            if not metadata_table.find_one(generation=generation):
                start_time = self.__get_start_time(generation)
                metadata = {
                    'start_timestamp': start_time,
                    'end_timestamp': int(start_time + self.configs['study']['days'] * 1440)
                }
                metadata_table.insert(dict(generation=generation, data=metadata))

    def __create_table(self, db_string, table):
        engine = create_engine(db_string)
        if not database_exists(engine.url):
            create_database(engine.url)
        table.create(engine, checkfirst=True)

    def __create_configs_table(self, db_string):
        table = sqlalchemy.Table(
            'configs',
            MetaData(),
            Column('id', sqlalchemy.Integer, primary_key=True),
            Column('data', sqlalchemy.JSON)
        )
        self.__create_table(db_string, table)

    def __create_metadata_table(self, db_string):
        table = sqlalchemy.Table(
            'metadata',
            MetaData(),
            Column('generation', sqlalchemy.Integer, primary_key=True),
            Column('data', sqlalchemy.JSON)
        )
        self.__create_table(db_string, table)
    
    def make_one(self, type:str, mode:str='', seq=0, skip_server=False, **kwargs):
        if not self.__config_version_valid:
            return []
        config = json.loads(json.dumps(self.configs))
        default_port = int(self.configs['server']['port']) if self.configs['server']['port'] else 3000
        config['server']['port'] = default_port + seq
        config['study']['type'] = type
        learning_participants = [participant for participant in config['participants'] if
                                 'learning' in config['participants'][participant]['trader'] and
                                 config['participants'][participant]['trader']['learning']]

        if type == 'baseline':
            if isinstance(config['study']['start_datetime'], str):
                config['study']['generations'] = 2
            config['market']['id'] = type
            config['market']['save_transactions'] = True
            for participant in config['participants']:
                config['participants'][participant]['trader'].update({
                    'learning': False,
                    'type': 'baseline_agent'
                })

        if type == 'training':
            config['market']['id'] = type
            config['market']['save_transactions'] = True

            if 'target' in kwargs:
                if not kwargs['target'] in config['participants']:
                    return []

                config['market']['id'] += '-' + kwargs['target']
                for participant in learning_participants:
                    config['participants'][participant]['trader']['learning'] = False
                config['participants'][kwargs['target']]['trader']['learning'] = True
            else:
                for participant in learning_participants:
                    config['participants'][participant]['trader']['learning'] = True

        if type == 'validation':
            config['market']['id'] = type
            config['market']['save_transactions'] = True

            for participant in config['participants']:
                config['participants'][participant]['trader']['learning'] = False

        self.__make_sim_internal_directories()
        lmaker = launcher_maker.Maker(config)
        server, market, sim_controller, participants = lmaker.make_launch_list()
        launch_sequence = market + sim_controller + participants
        if not skip_server:
            launch_sequence = server + launch_sequence
        return launch_sequence

    def launch_subprocess(self, args: list, delay=0):
        time.sleep(delay)

        import subprocess
        extension = args[0].split('.')[1]
        is_python = True if extension == 'py' else False

        if is_python:
            try:
                subprocess.run(['env/bin/python', args[0], *args[1]])
            except:
                subprocess.run(['venv/Scripts/python', args[0], *args[1]])
                # subprocess.run(['venv/Scripts/python', args[0], *args[1]])
            finally:
                subprocess.run(['python', args[0], *args[1]])
        else:
            subprocess.run([args[0], *args[1]])

    def launch(self, simulations, skip_servers=False):
        if not self.__config_version_valid:
            print('CONFIG NOT COMPATIBLE')
            return
        from multiprocessing import Pool

        launch_list = []
        seq = 0
        for sim in simulations:
            launch_list.extend(self.make_one(**sim, seq=seq))
            seq += 1

        pool_size = len(launch_list)
        pool = Pool(pool_size)
        pool.map(self.launch_subprocess, launch_list)
        pool.close()
