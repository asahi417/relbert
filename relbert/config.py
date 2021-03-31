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

    def __init__(self, export_dir: str, **kwargs):
        self.config = kwargs
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info('\t * {}: {}'.format(k, v))
        ex_configs = {i: self.safe_open(i) for i in glob('{}/*/trainer_config.json'.format(export_dir))}
        same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
        if len(same_config) != 0:
            input('\ncheckpoint already exists: {}\n enter to overwrite >>>'.format(same_config[0]))
            for _p, _ in same_config:
                shutil.rmtree(os.path.dirname(_p))
        self.cache_dir = '{}/{}'.format(export_dir, self.get_random_string(
            [os.path.basename(i.replace('/trainer_config.json', '')) for i in ex_configs.keys()]
        ))
        self.__dict__.update(self.config)
        self.__cache_init()

    def __cache_init(self):
        if not os.path.exists('{}/trainer_config.json'.format(self.cache_dir)):
            os.makedirs(self.cache_dir, exist_ok=True)
            with open('{}/trainer_config.json'.format(self.cache_dir), 'w') as f:
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
