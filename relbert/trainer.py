""" RelBERT fine-tuning with NCE loss """
import os
import logging
import random
import json
import statistics
from itertools import chain
from typing import List
from tqdm import tqdm
from glob import glob
from os.path import join as pj
from distutils.dir_util import copy_tree

import torch
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from .lm import RelBERT, Dataset
from .util import fix_seed, NCELoss


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
                 data: str = 'relbert/semeval2012_relational_similarity',
                 split: str = 'train',
                 split_eval: str = 'validation',
                 template_mode: str = 'manual',
                 template: str = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>",
                 loss_function: str = 'nce_rank',
                 classification_loss: bool = False,
                 temperature_nce_type: str = 'linear',
                 temperature_nce_constant: float = 1.0,
                 temperature_nce_min: float = 0.1,
                 temperature_nce_max: float = 10.0,
                 epoch: int = 1,
                 batch: int = 64,
                 gradient_accumulation: int = 4,
                 n_sample: int = 640,
                 lr: float = 0.00002,
                 lr_decay: bool = False,
                 lr_warmup: int = 100,
                 weight_decay: float = 0,
                 random_seed: int = 0,
                 exclude_relation: List or str = None,
                 fix_epoch: bool = False,
                 relation_level: str or List = None):
        assert not os.path.exists(export), f'{export} is taken, use different name'
        # config
        self.config = dict(
            model=model,
            max_length=max_length,
            mode=mode,
            data=data,
            split=split,
            split_eval=split_eval,
            template_mode=template_mode,
            template=template,
            loss_function=loss_function,
            classification_loss=classification_loss,
            temperature_nce_constant=temperature_nce_constant,
            temperature_nce_rank={'min': temperature_nce_min, 'max': temperature_nce_max, 'type': temperature_nce_type},
            epoch=epoch,
            batch=batch,
            lr=lr,
            lr_decay=lr_decay,
            lr_warmup=lr_warmup,
            weight_decay=weight_decay,
            random_seed=random_seed,
            exclude_relation=exclude_relation,
            n_sample=n_sample,
            gradient_accumulation=gradient_accumulation,
            relation_level=relation_level
        )
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info(f'\t * {k}: {str(v)[:min(100, len(str(v)))]}')
        self.model = RelBERT(
            model=self.config['model'],
            max_length=self.config['max_length'],
            mode=self.config['mode'],
            template_mode=self.config['template_mode'],
            template=self.config['template'])
        self.model.train()
        assert not self.model.is_trained, f'{model} is already trained'
        
        self.export_dir = export
        if not os.path.exists(pj(self.export_dir, 'trainer_config.json')):
            os.makedirs(self.export_dir, exist_ok=True)
            with open(pj(self.export_dir, 'trainer_config.json'), 'w') as f:
                json.dump(self.config, f)
        self.device = self.model.device
        self.parallel = self.model.parallel
        fix_seed(self.config['random_seed'])
        # get dataset
        self.data = load_dataset(self.config['data'], split=self.config['split'])
        self.data_eval = load_dataset(self.config['data'], split=self.config['split_eval'])

        if self.config['relation_level'] is not None:
            relation_level = [relation_level] if type(relation_level) is str else relation_level
            self.data = self.data.filter(lambda _x: _x["level"] in relation_level)
            self.data_eval = self.data_eval.filter(lambda _x: _x["level"] in relation_level)

        if self.config['exclude_relation'] is not None:
            self.data = self.data.filter(lambda x: x['relation_type'] not in self.config['exclude_relation'])
            self.data_eval = self.data_eval.filter(lambda x: x['relation_type'] not in self.config['exclude_relation'])
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
        self.fix_epoch = fix_epoch

        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['lr_warmup'],
            num_training_steps=self.config['epoch'] if self.config['lr_decay'] else None)

    def compute_loss(self, encoded_pairs_dict_eval, batch_size: int=None):
        with torch.no_grad():
            nce_loss = NCELoss(
                loss_function=self.config['loss_function'],
                temperature_nce_constant=self.config['temperature_nce_constant'],
                temperature_nce_rank=self.config['temperature_nce_rank'],
                classification_loss=False,
                hidden_size=self.model.hidden_size,
                device=self.device,
                parallel=self.parallel
            )
            total_loss = []
            loader_dict = {}
            for example in self.data_eval:
                pairs_p = example['positives']
                pairs_n = example['negatives']
                if 'level' in example:
                    k = f"{example['relation_type']}/{example['level']}"
                else:
                    k = example['relation_type']
                if k in loader_dict:
                    _index = 1
                    while True:
                        _k = f"{k}/{_index}"
                        if _k not in loader_dict:
                            k = _k
                            break
                        _index += 1

                dataset_p = Dataset([encoded_pairs_dict_eval['__'.join(__k)] for __k in pairs_p], return_ranking=True)
                dataset_n = Dataset([encoded_pairs_dict_eval['__'.join(__k)] for __k in pairs_n], return_ranking=False)
                loader_dict[k] = {
                    'positive': torch.utils.data.DataLoader(dataset_p, num_workers=0, batch_size=len(pairs_p)),
                    'negative': torch.utils.data.DataLoader(dataset_n, num_workers=0, batch_size=len(pairs_n))
                }
            relation_keys = list(loader_dict.keys())
            for n, relation_key in tqdm(list(enumerate(relation_keys))):
                # data loader will return full instances
                x_p = next(iter(loader_dict[relation_key]['positive']))
                x_n = next(iter(loader_dict[relation_key]['negative']))
                # data loader will return full instances
                x = {k: torch.concat([x_p[k], x_n[k]]) for k in x_n.keys()}
                embedding = self.model.to_embedding(x, batch_size=batch_size)
                batch_size_positive = len(x_p['input_ids'])
                embedding_p = embedding[:batch_size_positive]
                embedding_n = embedding[batch_size_positive:]
                rank = x_p.pop('ranking').cpu().tolist()
                loss = nce_loss(embedding_p, embedding_n, rank).cpu().tolist()
                total_loss.append(loss)
        return statistics.mean(total_loss)

    def cap_tensor(self, _tensor): return _tensor[:min(len(_tensor), self.config['n_sample'])]

    def train(self, epoch_save: int = 1):
        """ Train model.

        Parameters
        ----------
        epoch_save : int
            Epoch to run validation eg) Every 100000 epoch, it will save model weight as default.
        """
        encoded_pairs_dict = self.model.encode_word_pairs(
            list(chain(*(self.data['positives'] + self.data['negatives']))),
            parallel=False
        )
        encoded_pairs_dict_eval = self.model.encode_word_pairs(
            list(chain(*(self.data_eval['positives'] + self.data_eval['negatives']))),
            parallel=False
        )
        loader_dict = {}
        for example in self.data:
            pairs_p = example['positives']
            pairs_n = example['negatives']
            # if self.config['loss_function'] == "triplet" and len(pairs_p) < 2:
            #     continue

            if 'level' in example:
                k = f"{example['relation_type']}/{example['level']}"
            else:
                k = example['relation_type']
            dataset_p = Dataset([encoded_pairs_dict['__'.join(__k)] for __k in pairs_p], return_ranking=True)
            dataset_n = Dataset([encoded_pairs_dict['__'.join(__k)] for __k in pairs_n], return_ranking=False)
            if k in loader_dict:
                _index = 1
                while True:
                    _k = f"{k}/{_index}"
                    if _k not in loader_dict:
                        k = _k
                        break
                    _index += 1

            loader_dict[k] = {
                'positive': torch.utils.data.DataLoader(dataset_p, num_workers=0, batch_size=len(pairs_p)),
                'negative': torch.utils.data.DataLoader(dataset_n, num_workers=0, batch_size=len(pairs_n))
            }
        relation_keys = list(loader_dict.keys())
        logging.info(f'start model training: {len(relation_keys)} relations')
        nce_loss = NCELoss(
            loss_function=self.config['loss_function'],
            temperature_nce_constant=self.config['temperature_nce_constant'],
            temperature_nce_rank=self.config['temperature_nce_rank'],
            classification_loss=self.config['classification_loss'],
            hidden_size=self.model.hidden_size,
            device=self.device,
            parallel=self.parallel
        )

        for e in range(self.config['epoch']):  # loop over the epoch
            total_loss = []
            random.shuffle(relation_keys)
            for n, relation_key in tqdm(list(enumerate(relation_keys))):
                self.optimizer.zero_grad()
                # data loader will return full instances
                x_p = next(iter(loader_dict[relation_key]['positive']))
                x_n = next(iter(loader_dict[relation_key]['negative']))
                # data loader will return full instances
                x = {k: self.cap_tensor(torch.concat([x_p[k], x_n[k]])) for k in x_n.keys()}
                embedding = self.model.to_embedding(x, batch_size=self.config['batch'])
                batch_size_positive = len(x_p['input_ids'])
                embedding_p = embedding[:batch_size_positive]
                embedding_n = embedding[batch_size_positive:]
                rank = x_p.pop('ranking').cpu().tolist()
                loss = nce_loss(embedding_p, embedding_n, rank)
                loss.backward()
                total_loss.append(loss.cpu().item())
                if (n + 1) % self.config['gradient_accumulation'] != 0:
                    continue
                self.optimizer.step()
                self.scheduler.step()
                mean_loss = round(sum(total_loss) / len(total_loss), 3)
                lr = round(self.optimizer.param_groups[0]['lr'], 10)
                logging.info(f"\t[step {n}/{len(relation_keys)}] average loss: {mean_loss}, lr: {lr}")

            mean_loss = round(sum(total_loss)/len(total_loss), 3)
            lr = round(self.optimizer.param_groups[0]['lr'], 5)
            logging.info(f"[epoch {e + 1}/{self.config['epoch']} complete] average loss: {mean_loss}, lr: {lr}")

            if (e + 1) % epoch_save == 0 and (e + 1) != 0:
                if self.fix_epoch:
                    self.save(e)
                else:
                    v_loss = self.compute_loss(encoded_pairs_dict_eval)
                    self.save(e, v_loss)
        v_loss = self.compute_loss(encoded_pairs_dict_eval)
        self.save(self.config['epoch'] - 1, v_loss)
        logging.info(f'complete training: model ckpt was saved at {self.export_dir}')
        if self.fix_epoch:
            best_ckpt = pj(self.export_dir, f'epoch_{self.config["epoch"]}')
        else:
            # choose the best model
            ckpt_loss = []
            for i in glob(pj(self.export_dir, 'epoch_*')):
                with open(pj(i, 'validation_loss.json')) as f:
                    loss = json.load(f)["loss"]
                ckpt_loss.append([i, loss])
            best_ckpt, loss = sorted(ckpt_loss, key=lambda _x: _x[1])[0]
        copy_tree(best_ckpt, pj(self.export_dir, 'best_model'))

    def save(self, current_epoch, v_loss=None):
        cache_dir = pj(self.export_dir, f'epoch_{current_epoch + 1}')
        os.makedirs(cache_dir, exist_ok=True)
        self.model.save(cache_dir)
        with open(pj(cache_dir, 'trainer_config.json'), 'w') as f:
            config = self.config.copy()
            config['epoch'] = current_epoch + 1
            json.dump(config, f)
        with open(pj(cache_dir, 'validation_loss.json'), 'w') as f:
            result = {
                "split": self.config["split_eval"],
                "loss": v_loss,
                "data": self.config["data"],
                "exclude_relation": self.config["exclude_relation"],
                "relation_level": self.config["relation_level"],
            }
            json.dump(result, f)
