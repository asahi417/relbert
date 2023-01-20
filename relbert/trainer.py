""" Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """
import os
import logging
import random
import json
from typing import Dict, List
from itertools import product, combinations

import torch
from datasets import load_dataset

from .list_keeper import ListKeeper
from .lm import RelBERT
from .util import triplet_loss, fix_seed
from .data import get_training_data

DEFAULT_TEMPLATE = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"


class Dataset(torch.utils.data.Dataset):
    """ Dataset loader for triplet loss. """

    float_tensors = ['attention_mask']

    def __init__(self,
                 positive_samples: Dict,
                 negative_samples: Dict = None,
                 pairwise_input: bool = True,
                 relation_structure: Dict = None,
                 deterministic_index: int = None):
        if negative_samples is not None:
            assert positive_samples.keys() == negative_samples.keys()
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.pairwise_input = pairwise_input
        self.relation_structure = relation_structure
        self.pattern_id = None
        self.deterministic_index = deterministic_index
        if self.pairwise_input:
            self.keys = sorted(list(positive_samples.keys()))
            self.pattern_id = {k: list(product(
                list(combinations(range(len(self.positive_samples[k])), 2)), list(range(len(self.negative_samples[k])))
            )) for k in self.keys}
        else:
            self.keys = sorted(list(self.positive_samples.keys()))
            assert all(len(self.positive_samples[k]) == 1 for k in self.keys)
            assert self.negative_samples is None

    @staticmethod
    def rand_sample(_list):
        return _list[random.randint(0, len(_list) - 1)]

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
            if self.deterministic_index:
                (a, b), n = self.pattern_id[relation_type][self.deterministic_index]
            else:
                (a, b), n = self.rand_sample(self.pattern_id[relation_type])
            positive_a = self.positive_samples[relation_type][a]
            positive_b = self.positive_samples[relation_type][b]
            negative = self.negative_samples[relation_type][n]
            tensor_positive_a = {k: self.to_tensor(k, v) for k, v in positive_a.items()}
            tensor_positive_b = {k: self.to_tensor(k, v) for k, v in positive_b.items()}
            tensor_negative = {k: self.to_tensor(k, v) for k, v in negative.items()}
            output = {'positive_a': tensor_positive_a, 'positive_b': tensor_positive_b, 'negative': tensor_negative}
            if self.relation_structure is not None:

                # sampling relation type that shares same parent class with the positive sample
                parent_relation = [k for k, v in self.relation_structure.items() if relation_type in v]
                assert len(parent_relation) == 1
                relation_positive = self.rand_sample(self.relation_structure[parent_relation[0]])

                # sampling positive from the relation type
                positive_parent = self.rand_sample(self.positive_samples[relation_positive])
                output['positive_parent'] = {k: self.to_tensor(k, v) for k, v in positive_parent.items()}

                # sampling relation type from different parent class (negative)
                parent_relation_n = self.rand_sample([k for k in self.relation_structure.keys() if k != parent_relation[0]])
                relation_negative = self.rand_sample(self.relation_structure[parent_relation_n])

                # sample individual entry from the relation
                negative_parent = self.rand_sample(self.positive_samples[relation_negative])
                output['negative_parent'] = {k: self.to_tensor(k, v) for k, v in negative_parent.items()}
            return output
        else:
            # deterministic sampling for prediction
            positive_a = self.positive_samples[relation_type][0]
            return {k: self.to_tensor(k, v) for k, v in positive_a.items()}


class Trainer:
    """ Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """

    def __init__(self,
                 output_dir: str,
                 template: str,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 epoch: int = 1,
                 batch: int = 64,
                 random_seed: int = 0,
                 gradient_accumulation: int = 1,
                 lr: float = 0.00002,
                 lr_warmup: int = 10,
                 n_sample: int = 10,
                 aggregation_mode: str = 'average_no_mask',
                 data: str = 'relbert/semeval2012_relational_similarity',
                 exclude_relation: List or str = None,
                 split: str = 'train',
                 loss_function: str = 'triplet',
                 classification_loss: bool = True,
                 loss_function_config: Dict = None):

        # load language model
        self.model = RelBERT(model=model, max_length=max_length, aggregation_mode=aggregation_mode, template=template)
        assert not self.model.is_trained, f'{model} is already trained'
        self.model.train()
        self.hidden_size = self.model.model_config.hidden_size
        
        # config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.config = dict(
            template=template,
            model=model,
            max_length=max_length,
            epoch=epoch,
            batch=batch,
            random_seed=random_seed,
            gradient_accumulation=gradient_accumulation,
            lr=lr,
            lr_warmup=lr_warmup,
            n_sample=n_sample,
            aggregation_mode=aggregation_mode,
            data=data,
            exclude_relation=exclude_relation,
            split=split,
            loss_function=loss_function,
            classification_loss=classification_loss,
            loss_function_config=loss_function_config
        )
        fix_seed(self.config['random_seed'])

        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler(f'{self.output_dir}/training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)

        # # get dataset
        # self.all_positive, self.all_negative, self.relation_structure = get_training_data()
        # #
        # # check
        # all_positive_1, all_negative_1, relation_structure_1 = get_training_data()
        #
        # data = load_dataset(self.config['data'], split=self.config['split'])
        # all_positive_2 = {i['relation_type']: [tuple(_i) for _i in i['positives']] for i in data}
        # all_negative_2 = {i['relation_type']: [tuple(_i) for _i in i['negatives']] for i in data}
        # parent = list(set([i.split("/")[0] for i in all_negative_2.keys()]))
        # relation_structure_2 = {p: [i for i in all_positive_2.keys() if p == i.split("/")[0]] for p in parent}
        #
        # assert all_positive_1.keys() == all_positive_2.keys()
        # print(relation_structure_1)
        # print(relation_structure_2)
        # assert relation_structure_1 == relation_structure_2
        # for k in all_positive_1.keys():
        #     print(all_positive_1[k], all_positive_2[k])
        #     assert all_positive_1[k] == all_positive_2[k]
        #     assert all_negative_1 == all_negative_2
        #
        # input()

        data = load_dataset(self.config['data'], split=self.config['split'])
        self.all_positive = {i['relation_type']: [tuple(_i) for _i in i['positives']] for i in data}
        self.all_negative = {i['relation_type']: [tuple(_i) for _i in i['negatives']] for i in data}
        assert self.all_positive.keys() == self.all_negative.keys(), \
            f"{self.all_positive.keys()} != {self.all_negative.keys()}"
        if self.config['exclude_relation'] is not None:
            self.all_positive = {k: v for k, v in self.all_positive.items() if k not in self.config['exclude_relation']}
            self.all_negative = {k: v for k, v in self.all_negative.items() if k not in self.config['exclude_relation']}
        parent = list(set([i.split("/")[0] for i in self.all_negative.keys()]))
        self.relation_structure = {p: [i for i in self.all_positive.keys() if p == i.split("/")[0]] for p in parent}

        # calculate the number of trial to cover all combination in batch
        n_pos = min(len(i) for i in self.all_positive.values())
        n_neg = min(len(i) for i in self.all_negative.values())
        self.n_trial = len(list(product(combinations(range(n_pos), 2), range(n_neg))))

        # classification loss
        model_parameters = list(self.model.model.named_parameters())
        self.linear = None
        if self.config['classification_loss']:
            logging.info('add linear layer for softmax_loss')
            self.linear = torch.nn.Linear(self.hidden_size * 3, 1)  # three way feature
            self.linear.weight.data.normal_(std=0.02)
            self.linear.to(self.model.device)
            model_parameters += list(self.linear.named_parameters())
            if self.model.parallel:
                self.linear = torch.nn.DataParallel(self.linear)
        self.optimizer = torch.optim.AdamW(
            [{"params": [p for n, p in model_parameters], "weight_decay": 0}], lr=self.config['lr'])

        # scheduler
        def lr_lambda(current_step: int):
            current_step += 1
            if current_step < self.config['lr_warmup']:
                return float(current_step) / float(max(1, self.config['lr_warmup']))
            return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)

    def process_data(self):
        if type(self.all_positive) is not dict:
            assert self.relation_structure is None
            assert self.all_negative is None
            assert len(self.all_positive) > 0, len(self.all_positive)
            assert type(self.all_positive) is list and all(type(i) is tuple for i in self.all_positive)
            self.all_positive = {k: [v] for k, v in enumerate(self.all_positive)}
        key = list(self.all_positive.keys())
        positive_samples_list = ListKeeper([self.all_positive[k] for k in key])
        logging.debug(f'{len(self.all_positive)} positive data to encode')

        negative_sample_list = None
        if self.all_negative is not None:
            assert len(self.all_negative) > 0, len(self.all_negative)
            assert self.all_negative.keys() == self.all_positive.keys()
            negative_sample_list = ListKeeper([self.all_negative[k] for k in key])
            logging.debug(f'{len(self.all_negative)} positive data to encode')

        positive_embedding_dict = self.model.encode_word_pairs(positive_samples_list.flatten_list)
        positive_embedding = [positive_embedding_dict[f"{a}__{b}"] for a, b in positive_samples_list.flatten_list]
        positive_embedding = positive_samples_list.restore_structure(positive_embedding)
        positive_embedding = {key[n]: v for n, v in enumerate(positive_embedding)}
        negative_embedding = None
        if negative_sample_list is not None:
            negative_embedding_dict = self.model.encode_word_pairs(negative_sample_list.flatten_list)
            negative_embedding = [negative_embedding_dict[f"{a}__{b}"] for a, b in negative_sample_list.flatten_list]
            negative_embedding = negative_sample_list.restore_structure(negative_embedding)
            negative_embedding = {key[n]: v for n, v in enumerate(negative_embedding)}
        return positive_embedding, negative_embedding

    def train(self, epoch_save: int = 1):
        """ Train model. """
        positive_embedding, negative_embedding = self.process_data()
        batch_index = list(range(self.n_trial))
        global_step = 0

        for e in range(self.config['epoch']):  # loop over the epoch
            random.shuffle(batch_index)
            for n, bi in enumerate(batch_index):
                dataset = Dataset(deterministic_index=bi, pairwise_input=True, positive_samples=positive_embedding,
                                  negative_samples=negative_embedding, relation_structure=self.relation_structure)
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch'], shuffle=True, drop_last=True)
                mean_loss, global_step = self.train_single_epoch(loader, global_step)
                logging.info(f"[epoch {e + 1}/{self.config['epoch']}, batch_id {n}/{self.n_trial}], "
                             f"loss: {round(mean_loss, 3)}, lr: {self.optimizer.param_groups[0]['lr']}")
            if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != self.config['epoch']:
                self.model.save(f'{self.output_dir}/epoch_{e + 1}')

        self.model.save(f'{self.output_dir}/model')
        with open(f"{self.output_dir}/model/finetuning_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        logging.info(f'complete training: model ckpt was saved at {self.output_dir}')

    def train_single_epoch(self, data_loader, global_step: int, total_loss: int = 0):

        loss = None
        for n, x in enumerate(data_loader):
            self.optimizer.zero_grad()
            global_step += 1
            encode = {k: torch.cat([
                x['positive_a'][k],
                x['positive_b'][k],
                x['negative'][k],
                x['positive_parent'][k],
                x['negative_parent'][k]]) for k in x['positive_a'].keys()}
            embedding = self.model.to_embedding(encode)
            v_anchor, v_positive, v_negative, v_positive_hc, v_negative_hc = embedding.chunk(5)

            if self.config['loss_function'] == 'triplet':
                loss = triplet_loss(
                    tensor_anchor=v_anchor,
                    tensor_positive=v_positive,
                    tensor_negative=v_negative,
                    tensor_positive_parent=v_positive_hc,
                    tensor_negative_parent=v_negative_hc,
                    margin=self.config['loss_function_config']['mse_margin'],
                    linear=self.linear,
                    device=self.model.device)
            else:
                raise ValueError(f"unknown loss function: {self.config['loss_function']}")

            if (n + 1) % self.config['gradient_accumulation'] != 0:
                continue

            loss.backward()
            total_loss += loss.cpu().item()
            self.optimizer.step()
            self.scheduler.step()
            loss = None

        if loss is not None:
            loss.backward()
            total_loss += loss.cpu().item()
            self.optimizer.step()
            self.scheduler.step()

        return total_loss / len(data_loader), global_step
