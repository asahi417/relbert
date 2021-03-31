import os
import logging

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .lm import RelBERT
from .config import Config
from .util import get_linear_schedule_with_warmup, triplet_loss, fix_seed
from .data import get_analogy_data


def cos_similarity(a_, b_):
    inner = sum(list(map(lambda x: x[0] * x[1], zip(a_, b_))))
    norm_a = sum(list(map(lambda x: x * x, a_))) ** 0.5
    norm_b = sum(list(map(lambda x: x * x, b_))) ** 0.5
    if norm_b * norm_a == 0:
        return -100
    return inner / (norm_b * norm_a)


def diff(list_a, list_b):
    assert len(list_a) == len(list_b)
    return list(map(lambda x: x[0] - x[1], zip(list_a, list_b)))


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

    lm = RelBERT(model, max_length=max_length, mode=mode, template_type=template_type)
    # lm.mode
    # lm.template_type


    for k, (val, test) in data.items():







