import random
import os
import tarfile
import zipfile
import requests
from typing import Dict
from itertools import combinations
from random import randint

import gdown
import numpy as np
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR


home_dir = '{}/.cache/relbert'.format(os.path.expanduser('~'))
module_output_dir = './relbert_output'


def load_language_model(model_name, cache_dir: str = None):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except ValueError:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    try:
        config = transformers.AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    except ValueError:
        config = transformers.AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    try:
        model = transformers.AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    except ValueError:
        model = transformers.AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir, local_files_only=True)
    return tokenizer, model, config


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


def triplet_loss(tensor_positive_0, tensor_positive_1, tensor_negative,
                 tensor_positive_parent=None, tensor_negative_parent=None,
                 margin: int = 1, in_batch_negative: bool = True):
    """ Compute contrastive triplet loss with in batch augmentation which enables to propagate error on quadratic
    of batch size. """
    loss = 0
    n_backward = 0
    boundary = 0

    # the main contrastive loss
    distance_positive = torch.sum((tensor_positive_0 - tensor_positive_1) ** 2, -1) ** 0.5
    for tensor_positive in [tensor_positive_0, tensor_positive_1]:
        distance_negative = torch.sum((tensor_positive - tensor_negative) ** 2, -1) ** 0.5
        loss += torch.sum(torch.clip(distance_positive - distance_negative - margin, min=boundary))
        n_backward += len(distance_positive)

    if in_batch_negative:
        # No elements in single batch share same relation type, so here we construct negative sample within batch
        # by regarding positive sample from other entries as its negative. The original negative is the hard
        # negatives from same relation type and the in batch negative is easy negative from other relation types.
        distance_negative_batch = torch.sum((tensor_positive_0.unsqueeze(-1).permute(0, 2, 1) -
                                             tensor_positive_1.unsqueeze(-1).permute(2, 0, 1)) ** 2, -1) ** 0.5
        distance_positive_batch = distance_positive.unsqueeze(-1)
        loss_batch = torch.clip(distance_positive_batch - distance_negative_batch - margin, min=boundary)
        loss += torch.sum(loss_batch)
        n_backward += len(loss_batch)

    if tensor_positive_parent is not None and tensor_negative_parent is not None:
        # contrastive loss of the parent class
        for tensor_positive in [tensor_positive_0, tensor_positive_1]:
            distance_positive = torch.sum((tensor_positive - tensor_positive_parent) ** 2, -1) ** 0.5
            distance_negative = torch.sum((tensor_positive - tensor_negative_parent) ** 2, -1) ** 0.5
            loss += torch.sum(torch.clip(distance_positive - distance_negative - margin, min=boundary))
            n_backward += len(distance_positive)
    # loss = loss
    return loss


def rand_sample(_list):
    return _list[randint(0, len(_list) - 1)]


class Dataset(torch.utils.data.Dataset):
    """ Dataset loader for triplet loss. """
    float_tensors = ['attention_mask']

    def __init__(self,
                 positive_samples: Dict,
                 negative_samples: Dict = None,
                 pairwise_input: bool = True,
                 relation_structure: Dict = None):
        if negative_samples is not None:
            assert positive_samples.keys() == negative_samples.keys()
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.pairwise_input = pairwise_input
        self.relation_structure = relation_structure
        self.positive_pattern_id = None
        if self.pairwise_input:
            self.keys = sorted(list(positive_samples.keys()))
            self.positive_pattern_id = {k: list(combinations(range(len(self.positive_samples[k])), 2))
                                        for k in self.keys}
        else:
            self.keys = sorted(list(self.positive_samples.keys()))
            assert all(len(self.positive_samples[k]) == 1 for k in self.keys)
            assert self.negative_samples is None

    def __len__(self):
        return len(self.keys)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        # relation type for positive sample
        relation_type = self.keys[idx]
        if self.pairwise_input:
            # sampling pair from the relation type for anchor positive sample
            a, b = rand_sample(self.positive_pattern_id[relation_type])
            positive_a = self.positive_samples[relation_type][a]
            positive_b = self.positive_samples[relation_type][b]
            tensor_positive_a = {k: self.to_tensor(k, v) for k, v in positive_a.items()}
            tensor_positive_b = {k: self.to_tensor(k, v) for k, v in positive_b.items()}

            # sampling negative from the relation type
            negative_list = self.negative_samples[relation_type]
            tensor_negative = {k: self.to_tensor(k, v) for k, v in rand_sample(negative_list).items()}

            if self.relation_structure is not None:
                # sampling relation type that shares same parent class with the positive sample
                parent_relation = [k for k, v in self.relation_structure.items() if relation_type in v]
                assert len(parent_relation) == 1
                relation_positive = rand_sample(self.relation_structure[parent_relation[0]])
                # sampling positive from the relation type
                positive_parent = rand_sample(self.positive_samples[relation_positive])
                tensor_positive_parent = {k: self.to_tensor(k, v) for k, v in positive_parent.items()}

                # sampling relation type from different parent class (negative)
                parent_relation_n = rand_sample([k for k in self.relation_structure.keys() if k != parent_relation[0]])
                relation_negative = rand_sample(self.relation_structure[parent_relation_n])
                # sample individual entry from the relation
                negative_parent = rand_sample(self.positive_samples[relation_negative])
                tensor_negative_parent = {k: self.to_tensor(k, v) for k, v in negative_parent.items()}

                return {'positive_a': tensor_positive_a, 'positive_b': tensor_positive_b, 'negative': tensor_negative,
                        'positive_parent': tensor_positive_parent, 'negative_parent': tensor_negative_parent}
            else:
                return {'positive_a': tensor_positive_a, 'positive_b': tensor_positive_b, 'negative': tensor_negative}
        else:
            # deterministic sampling for prediction
            positive_a = self.positive_samples[relation_type][0]
            return {k: self.to_tensor(k, v) for k, v in positive_a.items()}

