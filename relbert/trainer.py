""" Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """
import os
import logging
import random
import json
import string
from typing import List
from glob import glob
from itertools import product, combinations
from typing import Dict

import torch
from torch import nn

from .lm import RelBERT
from .data import get_training_data, get_contrastive_data
from .util import get_linear_schedule_with_warmup, triplet_loss, fix_seed, Dataset


class Trainer:

    def __init__(self,
                 export: str,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 mode: str = 'average_no_mask',
                 data: str = 'semeval2012',
                 n_sample: int = 10,
                 template_type: str = 'a',
                 custom_template: str = None,
                 nce_loss: bool = True,
                 softmax_loss: bool = True,
                 in_batch_negative: bool = True,
                 mse_margin: float = 1,
                 epoch: int = 1,
                 batch: int = 64,
                 lr: float = 0.00002,
                 lr_decay: bool = False,
                 lr_warmup: int = 100,
                 weight_decay: float = 0,
                 gradient_accumulation_steps: int = 1,
                 random_seed: int = 0,
                 exclude_relation=None):
        assert not os.path.exists(export), f'{export} is taken, use different name'
        # config
        self.config = dict(
            model=model,
            max_length=max_length,
            mode=mode,
            data=data,
            n_sample=n_sample,
            custom_template=custom_template,
            template=template_type,
            softmax_loss=softmax_loss,
            in_batch_negative=in_batch_negative,
            nce_loss=nce_loss,
            mse_margin=mse_margin,
            epoch=epoch,
            lr_warmup=lr_warmup,
            batch=batch,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            random_seed=random_seed,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info('\t * {}: {}'.format(k, str(v)[:min(100, len(str(v)))]))
        self.model = RelBERT(
            model=self.config['model'],
            max_length=self.config['max_length'],
            mode=self.config['mode'],
            template_type=self.config['template_type'],
            custom_template=self.config['custom_template'])
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
        self.all_positive, self.all_negative, self.relation_structure = get_training_data(
            data_name=self.config['data'], n_sample=self.config['n_sample'], exclude_relation=exclude_relation
        )

        # calculate the number of trial to cover all combination in batch
        n_pos = min(len(i) for i in self.all_positive.values())
        n_neg = min(len(i) for i in self.all_negative.values())
        self.n_trial = len(list(product(combinations(range(n_pos), 2), range(n_neg))))

        self.model_parameters = list(self.model.model.named_parameters())

        if self.config['softmax_loss']:
            logging.info('add linear layer for softmax_loss')
            self.linear = nn.Linear(self.model.hidden_size * 3, 1)  # three way feature
            self.linear.weight.data.normal_(std=0.02)
            self.discriminative_loss = nn.BCELoss()
            self.linear.to(self.device)
            self.model_parameters += list(self.linear.named_parameters())
            if self.parallel:
                self.linear = torch.nn.DataParallel(self.linear)

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

    def save(self, current_epoch):
        cache_dir = '{}/epoch_{}'.format(self.export_dir, current_epoch + 1)
        os.makedirs(cache_dir, exist_ok=True)
        self.model.save(cache_dir)

    def preprocess(self, positive_samples, negative_samples: Dict = None, relation_structure: Dict = None):
        return self.model.preprocess(positive_samples, negative_samples, relation_structure)

    def model_output(self, encode):
        return self.model.to_embedding(encode)        

    def train(self, num_workers: int = 1, epoch_save: int = 1):
        """ Train model.

        Parameters
        ----------
        num_workers : int
            Workers for DataLoader.
        epoch_save : int
            Epoch to run validation eg) Every 100000 epoch, it will save model weight as default.
        """
        param = self.preprocess(self.all_positive, self.all_negative)
        logging.info('start model training')
        batch_index = list(range(self.n_trial))
        global_step = 0

        for e in range(self.config['epoch']):  # loop over the epoch
            random.shuffle(batch_index)
            for n, bi in enumerate(batch_index):
                dataset = Dataset(deterministic_index=bi, **param)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.config['batch'], shuffle=True, num_workers=num_workers, drop_last=True)
                mean_loss, global_step = self.train_single_epoch(loader, global_step=global_step)
                inst_lr = self.optimizer.param_groups[0]['lr']
                logging.info('[epoch {}/{}, batch_id {}/{}] average loss: {}, lr: {}'.format(
                    e, self.config['epoch'], n, self.n_trial, round(mean_loss, 3), inst_lr))
            if (e + 1) % epoch_save == 0 and (e + 1) != 0:
                self.save(e)

        self.save(e)
        logging.info('complete training: model ckpt was saved at {}'.format(self.export_dir))

    def train_single_epoch(self, data_loader, global_step: int):
        total_loss = 0
        step_in_epoch = len(data_loader)
        for x in data_loader:
            global_step += 1

            self.optimizer.zero_grad()

            encode = {k: torch.cat([x['positive_a'][k], x['positive_b'][k], x['negative'][k]])
                      for k in x['positive_a'].keys()}
            embedding = self.model_output(encode)
            v_anchor, v_positive, v_negative = embedding.chunk(3)

            # contrastive loss
            loss = triplet_loss(v_anchor, v_positive, v_negative,
                                margin=self.config['mse_margin'],
                                in_batch_negative=self.config['in_batch_negative'],
                                linear=self.linear, device=self.device)

            # backward: calculate gradient
            loss.backward()
            inst_loss = loss.cpu().item()

            # aggregate average loss over epoch
            total_loss += inst_loss

            self.optimizer.step()
            self.scheduler.step()

        return total_loss / step_in_epoch, global_step
