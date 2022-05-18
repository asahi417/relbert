""" RelBERT fine-tuning with NCE loss """
import os
import logging
import random
import json
from itertools import chain
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .lm import RelBERT, Dataset
from .data import get_training_data
from .util import fix_seed


def stack_sum(_list):
    if len(_list) == 0:
        return 0
    return torch.sum(torch.stack(_list))


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


class Trainer:

    def __init__(self,
                 export: str,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 mode: str = 'average_no_mask',
                 data: str = 'semeval2012',
                 template_mode: str = 'manual',
                 template: str = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>",
                 loss_function: str = 'nce_rank',
                 temperature_nce_type: str = 'linear',
                 temperature_nce_constant: float = 1.0,
                 temperature_nce_min: float = 0.1,
                 temperature_nce_max: float = 10.0,
                 epoch: int = 1,
                 batch: int = 64,
                 batch_positive_ratio: float = 0.3,
                 lr: float = 0.00002,
                 lr_decay: bool = False,
                 lr_warmup: int = 100,
                 weight_decay: float = 0,
                 random_seed: int = 0,
                 exclude_relation: List or str = None):
        assert not os.path.exists(export), f'{export} is taken, use different name'
        # config
        self.config = dict(
            model=model,
            max_length=max_length,
            mode=mode,
            data=data,
            template_mode=template_mode,
            template=template,
            loss_function=loss_function,
            temperature_nce_constant=temperature_nce_constant,
            temperature_nce_rank={'min': temperature_nce_min, 'max': temperature_nce_max, 'type': temperature_nce_type},
            epoch=epoch,
            batch=batch,
            batch_positive_ratio=batch_positive_ratio,
            lr=lr,
            lr_decay=lr_decay,
            lr_warmup=lr_warmup,
            weight_decay=weight_decay,
            random_seed=random_seed,
            exclude_relation=exclude_relation,
        )
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info('\t * {}: {}'.format(k, str(v)[:min(100, len(str(v)))]))
        self.model = RelBERT(
            model=self.config['model'],
            max_length=self.config['max_length'],
            mode=self.config['mode'],
            template_mode=self.config['template_mode'],
            template=self.config['template'])
        self.model.train()
        assert not self.model.is_trained, '{} is already trained'.format(model)
        
        self.export_dir = export
        if not os.path.exists(f'{self.export_dir}/trainer_config.json'):
            os.makedirs(self.export_dir, exist_ok=True)
            with open(f'{self.export_dir}/trainer_config.json', 'w') as f:
                json.dump(self.config, f)
        self.device = self.model.device
        self.parallel = self.model.parallel
        fix_seed(self.config['random_seed'])
        # get dataset
        self.data = get_training_data(data_name=self.config['data'], exclude_relation=self.config['exclude_relation'])
        self.model_parameters = list(self.model.model.named_parameters())

        # setup optimizer
        if self.config['weight_decay'] is not None or self.config['weight_decay'] != 0:
            no_decay = ["bias", "LayerNorm.weight"]
            self.model_parameters = [
                {"params": [p for n, p in self.model_parameters if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.config['weight_decay']},
                {"params": [p for n, p in self.model_parameters if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0}
            ]
        self.optimizer = torch.optim.AdamW(self.model_parameters, lr=self.config['lr'])
        
        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['lr_warmup'],
            num_training_steps=self.config['epoch'] if self.config['lr_decay'] else None)

    def train(self, epoch_save: int = 1):
        """ Train model.

        Parameters
        ----------
        epoch_save : int
            Epoch to run validation eg) Every 100000 epoch, it will save model weight as default.
        """
        encoded_pairs_dict = self.model.encode_word_pairs(list(chain(*[p + n for p, n in self.data.values()])))
        loader_dict = {}
        batch_size_positive = int(self.config['batch'] * self.config['batch_positive_ratio'])
        batch_size_negative = self.config['batch'] - batch_size_positive
        logging.info(f'batch size: positive ({batch_size_positive}), negative ({batch_size_negative})')

        for k, (pairs_p, pairs_n) in self.data.items():
            loader_dict[k] = {
                'positive': torch.utils.data.DataLoader(
                    Dataset([encoded_pairs_dict['__'.join(k)] for k in pairs_p], return_ranking=True), num_workers=0,
                    batch_size=batch_size_positive, shuffle=True, drop_last=True),
                'negative': torch.utils.data.DataLoader(
                    Dataset([encoded_pairs_dict['__'.join(k)] for k in pairs_n], return_ranking=False), num_workers=0,
                    batch_size=batch_size_negative, shuffle=True, drop_last=True)
            }
        logging.info('start model training')
        relation_keys = list(self.data.keys())
        cos_2d = nn.CosineSimilarity(dim=1)
        cos_1d = nn.CosineSimilarity(dim=0)
        for e in range(self.config['epoch']):  # loop over the epoch
            total_loss = []
            random.shuffle(relation_keys)
            for n, relation_key in enumerate(relation_keys):
                loader_p = iter(loader_dict[relation_key]['positive'])
                loader_n = iter(loader_dict[relation_key]['negative'])
                while True:
                    try:
                        self.optimizer.zero_grad()
                        x_p = next(loader_p)
                        x_n = next(loader_n)
                        x = {k: torch.concat([x_p[k], x_n[k]]) for k in x_n.keys()}
                        embedding = self.model.to_embedding(x)
                        embedding_p = embedding[:batch_size_positive]
                        embedding_n = embedding[batch_size_positive:]
                        loss = []
                        if self.config['loss_function'] == 'nce_rank':
                            rank = x_p.pop('ranking').cpu().tolist()
                            rank_map = {r: 1 + n for n, r in enumerate(sorted(rank))}
                            rank = [rank_map[r] for r in rank]
                            for i in range(batch_size_positive):
                                assert type(rank[i]) == int, rank[i]
                                tau = self.get_rank_temperature(rank[i], batch_size_positive)
                                deno_n = torch.sum(torch.exp(cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / tau))
                                dist = torch.exp(cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / tau)
                                # input([d for n, d in enumerate(dist) if rank[n] <= rank[i]])
                                # input([d for n, d in enumerate(dist) if rank[n] > rank[i]])
                                nume_p = stack_sum([d for n, d in enumerate(dist) if rank[n] >= rank[i]])
                                deno_p = stack_sum([d for n, d in enumerate(dist) if rank[n] < rank[i]])
                                loss.append(- torch.log(nume_p / (deno_p + deno_n)))
                        elif self.config['loss_function'] == 'nce_logout':
                            for i in range(batch_size_positive):
                                deno_n = torch.sum(torch.exp(
                                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / self.config['temperature_nce_constant']))
                                for p in range(batch_size_positive):
                                    logit_p = torch.exp(
                                        cos_1d(embedding_p[i], embedding_p[p]) / self.config['temperature_nce_constant'])
                                    loss.append(- torch.log(logit_p/(logit_p + deno_n)))
                        elif self.config['loss_function'] == 'nce_login':
                            for i in range(batch_size_positive):
                                deno_n = torch.sum(torch.exp(
                                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / self.config['temperature_nce_constant']))
                                logit_p = torch.sum(torch.exp(
                                    cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / self.config['temperature_nce_constant']))
                                loss.append(- torch.log(logit_p/(logit_p + deno_n)))
                        else:
                            raise ValueError(f"unknown loss function {self.config['loss_function']}")
                        loss = stack_sum(loss)
                        loss.backward()
                        total_loss.append(loss.cpu().item())
                        self.optimizer.step()
                        self.scheduler.step()
                    except StopIteration:
                        break

            mean_loss = round(sum(total_loss)/len(total_loss), 3)
            lr = round(self.optimizer.param_groups[0]['lr'], 5)
            logging.info(f"[epoch {e + 1}/{self.config['epoch']}] average loss: {mean_loss}, lr: {lr}")
            if (e + 1) % epoch_save == 0 and (e + 1) != 0:
                self.save(e)

        self.save(self.config['epoch'] - 1)
        logging.info('complete training: model ckpt was saved at {}'.format(self.export_dir))

    def save(self, current_epoch):
        cache_dir = '{}/epoch_{}'.format(self.export_dir, current_epoch + 1)
        os.makedirs(cache_dir, exist_ok=True)
        self.model.save(cache_dir)

    def get_rank_temperature(self, i, n):
        assert i <= n, f"{i}, {n}"
        if self.config['temperature_nce_rank']['type'] == 'linear':
            _min = self.config['temperature_nce_rank']['min']
            _max = self.config['temperature_nce_rank']['max']
            return (_min - _max) / (1 - n) * (i - 1) + _min
        raise ValueError(f"unknown type: {self.config['temperature_nce_rank']['type']}")
