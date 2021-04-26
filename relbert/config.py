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
                 export: str = None,
                 config_name: str = 'trainer_config',
                 checkpoint_path: str = None,
                 **kwargs):
        if checkpoint_path is not None:
            with open('{}/prompter_config.json'.format(checkpoint_path), 'r') as f:
                self.config = json.load(f)
            self.cache_dir = checkpoint_path
            iters = [int(i.split('prompt.')[-1].replace('.json', '')) for i in
                     glob('{}/prompt.*.json'.format(checkpoint_path))]
            if len(iters) == 0:
                self.last_iter = 0
            else:
                self.last_iter = max(iters) + 1
        else:
            assert export, '`export` is required'
            assert not os.path.exists(export), '{} is taken, use different name'.format(export)
            self.config = kwargs
            logging.info('hyperparameters')
            for k, v in self.config.items():
                logging.info('\t * {}: {}'.format(k, str(v)[:min(100, len(str(v)))]))
            configs = {i: self.safe_open(i) for i in glob('{}/*/{}.json'.format(os.path.dirname(export), config_name))}
            configs = list(filter(lambda x: x[1] == self.config, configs.items()))
            if len(configs) != 0:
                input('\ncheckpoint with same config already exists: {}\n enter to overwrite >>>'.format(configs[0]))
                for _p, _ in configs:
                    shutil.rmtree(os.path.dirname(_p))
            self.cache_dir = export
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
