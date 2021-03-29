import random
import os
import tarfile
import zipfile
import requests

import gdown
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


home_dir = '{}/.cache/relbert'.format(os.path.expanduser('~'))


def wget(url, cache_dir: str, gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    path = _wget(url, cache_dir, gdrive_filename=gdrive_filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    # return path


def _wget(url: str, cache_dir, gdrive_filename: str = None):
    """ get data from web """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        return gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return '{}/{}'.format(cache_dir, filename)


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=None, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        current_step += 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if num_training_steps is None:
            return 1
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def triplet_loss(tensor_positive_0, tensor_positive_1, tensor_negative, margin: int = 1, in_batch_negative: bool = True):
    """ Compute contrastive triplet loss with in batch augmentation which enables to propagate error on quadratic
    of batch size. """
    distance_positive = torch.sum((tensor_positive_0 - tensor_positive_1) ** 2, -1) ** 0.5

    # the first tensor as an anchor
    distance_negative = torch.sum((tensor_positive_0 - tensor_negative) ** 2, -1) ** 0.5
    loss = torch.sum(torch.clip(distance_positive - distance_negative - margin, min=0))
    # loss = torch.sum(torch.clip(distance_positive - distance_negative - mse_margin, min=0))

    # the second tensor as an anchor
    distance_negative = torch.sum((tensor_positive_1 - tensor_negative) ** 2, -1) ** 0.5
    loss += torch.sum(torch.clip(distance_positive - distance_negative - margin, min=0))
    # loss += torch.sum(torch.clip(distance_positive - distance_negative - mse_margin, min=0))

    if in_batch_negative:
        # No elements in single batch share same relation type, so here we construct negative sample within batch
        # by regarding positive sample from other entries as its negative. The original negative is the hard
        # negatives from same relation type and the in batch negative is easy negative from other relation types.
        distance_negative_batch = torch.sum((tensor_positive_0.unsqueeze(-1).permute(0, 2, 1) -
                                             tensor_positive_1.unsqueeze(-1).permute(2, 0, 1)) ** 2, -1) ** 0.5
        distance_positive_batch = distance_positive.unsqueeze(-1)
        loss_batch = torch.clip(distance_positive_batch - distance_negative_batch - margin, min=0)
        loss += torch.sum(loss_batch)
    return loss
