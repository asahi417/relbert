""" Configuration manager """
import os
import random
import json
import string
import logging
import shutil
from glob import glob
from typing import List

__all__ = 'Config'


class Config:

    def __init__(self,
                 export_dir: str,
                 config_name: str = 'trainer_config',
                 checkpoint_name: str = None,
                 checkpoint_path: str = None,
                 **kwargs):
        if checkpoint_path is not None:
            with open('{}/prompter_config.json'.format(checkpoint_path), 'r') as f:
                self.config = json.load(f)
            self.cache_dir = checkpoint_path
            iters = [i.split('prompt.')[-1].replace('.json', '') for i in glob('{}/prompt.*.json')]
            if len(iters) == 0:
                self.last_iter = 0
            else:
                self.last_iter = max(iters) + 1
        else:
            self.config = kwargs
            logging.info('hyperparameters')
            for k, v in self.config.items():
                logging.info('\t * {}: {}'.format(k, v))
            ex_configs = {i: self.safe_open(i) for i in glob('{}/*/{}.json'.format(export_dir, config_name))}
            taken_name = [os.path.basename(i.replace('/{}.json'.format(config_name), '')) for i in ex_configs.keys()]
            same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
            if len(same_config) != 0:
                input('\ncheckpoint already exists: {}\n enter to overwrite >>>'.format(same_config[0]))
                for _p, _ in same_config:
                    shutil.rmtree(os.path.dirname(_p))
                    taken_name.pop(taken_name.index(os.path.basename(os.path.dirname(_p))))

            if checkpoint_name is not None:
                assert checkpoint_name not in taken_name, '{} is taken, use different name'.format(checkpoint_name)
                self.cache_dir = '{}/{}'.format(export_dir, checkpoint_name)
            else:
                self.cache_dir = '{}/{}'.format(export_dir, self.get_random_string(taken_name))
            self.__cache_init(config_name)
            self.last_iter = 0
        self.__dict__.update(self.config)

    def __cache_init(self, config_name):
        if not os.path.exists('{}/{}.json'.format(self.cache_dir, config_name)):
            os.makedirs(self.cache_dir, exist_ok=True)
            with open('{}/{}.json'.format(self.cache_dir, config_name), 'w') as f:
                json.dump(self.config, f)

    @staticmethod
    def get_random_string(exclude: List = None, length: int = 6):
        while True:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
            if exclude is None:
                break
            elif tmp not in exclude:
                break
        return tmp

    @staticmethod
    def safe_open(_file):
        with open(_file, 'r') as f:
            return json.load(f)
