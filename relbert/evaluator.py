import os
import logging

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .lm import RelBERT
from .config import Config
from .util import get_linear_schedule_with_warmup, triplet_loss, fix_seed
from .data import get_analogy_data


def evaluate(model,
             max_length: int = 64,
             template_type: str = 'a',
             mode: str = 'mask',
             test_type: str = 'analogy',
             cache_dir: str = None,
             batch: int = 64):

        if test_type == 'analogy':
            data = {d: get_analogy_data(d, cache_dir=cache_dir) for d in ['sat', 'u2', 'u4', 'google', 'bats']}
        else:
            raise ValueError('unknown test_type: {}'.format(test_type))
        lm = RelBERT(model, max_length=max_length)
        if lm.is_trained:



