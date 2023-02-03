""" Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """
import os
import logging
import random
import json
from typing import Dict, List
from itertools import combinations, chain
from tqdm import tqdm
import torch
from datasets import load_dataset

from .list_keeper import ListKeeper
from .lm import RelBERT
from .util import fix_seed, loss_triplet, loss_nce


DEFAULT_TEMPLATE = "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"


def to_tensor(name, data):
    if name in ['attention_mask']:
        return torch.tensor(data, dtype=torch.float32)
    return torch.tensor(data, dtype=torch.long)


def rand_sample(_list):
    return _list[random.randint(0, len(_list) - 1)]


class Trainer:
    """ Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """

    def __init__(self,
                 output_dir: str,
                 template: str = None,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 epoch: int = 1,
                 batch: int = 64,
                 random_seed: int = 0,
                 lr: float = 0.00002,
                 lr_warmup: int = 10,
                 aggregation_mode: str = 'average_no_mask',
                 data: str = 'relbert/semeval2012_relational_similarity',
                 data_name: str = None,
                 exclude_relation: List or str = None,
                 split: str = 'train',
                 split_valid: str = 'validation',
                 loss_function: str = 'triplet',
                 classification_loss: bool = True,
                 loss_function_config: Dict = None,
                 parallel_preprocess: bool = True,
                 augment_negative_by_positive: bool = False):

        # load language model
        self.model = RelBERT(model=model, max_length=max_length, aggregation_mode=aggregation_mode, template=template)
        self.hidden_size = self.model.model_config.hidden_size
        self.parallel_preprocess = parallel_preprocess
        
        # config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler(f'{self.output_dir}/training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)

        self.config = dict(
            template=template,
            model=model,
            max_length=max_length,
            epoch=epoch,
            batch=batch,
            random_seed=random_seed,
            lr=lr,
            lr_warmup=lr_warmup,
            aggregation_mode=aggregation_mode,
            data=data,
            data_name=data_name,
            exclude_relation=exclude_relation,
            split=split,
            split_valid=split_valid,
            loss_function=loss_function,
            classification_loss=classification_loss,
            loss_function_config=loss_function_config,
            augment_negative_by_positive=augment_negative_by_positive
        )
        fix_seed(self.config['random_seed'], self.model.device == 'cuda')

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

    def train(self, epoch_save: int = 1):
        assert not self.model.is_trained, f'model is already trained'
        self.model.train()
        positive_encode, negative_encode, relation_structure = self.process_data(self.config['split'])
        if self.config['loss_function'] == 'triplet':
            self._train_triplet(positive_encode, negative_encode, relation_structure, epoch_save)
        elif self.config['loss_function'] in ['nce', 'iloob']:
            self._train_nce(positive_encode, negative_encode, relation_structure, epoch_save)
        else:
            raise ValueError(f"unknown loss function {self.config['loss_function']}")
        self.model.save(f'{self.output_dir}/model')
        with open(f"{self.output_dir}/model/finetuning_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        logging.info(f'complete training: model ckpt was saved at {self.output_dir}')

    def _train_nce(self, positive_encode, negative_encode, relation_structure, epoch_save):
        features = positive_encode[list(positive_encode.keys())[0]][0].keys()
        positive_encode = {k: {_k: [x[_k] for x in v] for _k in features} for k, v in positive_encode.items()}
        if negative_encode is not None:
            negative_encode = {k: {_k: [x[_k] for x in v] for _k in features} for k, v in negative_encode.items()}
        if self.config['augment_negative_by_positive']:
            if negative_encode is not None:
                negative_encode = {
                    k: {_k: list(chain(*[v[_k]] + [b[_k] for a, b in positive_encode.items() if a != k])) for
                        _k in features} for k, v in negative_encode.items()}
            else:
                negative_encode = {
                    k: {_k: list(chain(*[b[_k] for a, b in positive_encode.items() if a != k])) for
                        _k in features} for k in positive_encode.keys()}
        assert negative_encode is not None, 'negative encode is None (use -a to augment negative by positive)'

        # add parent relation types
        if relation_structure is not None:
            for k, v in relation_structure.items():
                positive_encode[k] = {_k: sorted(list(chain(*[positive_encode[_v][_k] for _v in v]))) for _k in features}
                n_list = sorted(list(chain(*[_v for _k, _v in relation_structure.items() if _k != k])))
                negative_encode[k] = {_k: sorted(list(chain(*[positive_encode[_v][_k] for _v in n_list]))) for _k in features}
        relation_types = sorted(list(positive_encode.keys()))

        for e in range(self.config['epoch']):  # loop over the epoch
            total_loss = []
            random.shuffle(relation_types)
            for n, r_type in enumerate(relation_types):

                self.optimizer.zero_grad()
                # embedding for positive samples
                pos = positive_encode[r_type]
                if len(pos['input_ids']) > self.config['loss_function_config']['num_positive']:
                    ids = list(range(len(pos['input_ids'])))
                    random.shuffle(ids)
                    pos = {k: [v[i] for i in ids[:self.config['loss_function_config']['num_positive']]] for k, v in
                           pos.items()}
                pos = {k: to_tensor(k, v) for k, v in pos.items()}
                positive_embedding = self.model.to_embedding(pos, batch_size=self.config['batch'])

                # embedding for negative samples
                neg = negative_encode[r_type]
                if len(neg['input_ids']) > self.config['loss_function_config']['num_negative']:
                    ids = list(range(len(neg['input_ids'])))
                    random.shuffle(ids)
                    neg = {k: [v[i] for i in ids[:self.config['loss_function_config']['num_negative']]] for k, v in
                           neg.items()}
                neg = {k: to_tensor(k, v) for k, v in neg.items()}
                negative_embedding = self.model.to_embedding(neg, batch_size=self.config['batch'])

                # loss computation
                loss = loss_nce(
                    tensor_positive=positive_embedding,
                    tensor_negative=negative_embedding,
                    temperature=self.config['loss_function_config']['temperature'],
                    info_loob=self.config['loss_function'] == 'iloob',
                    linear=self.linear,
                    device=self.model.device)
                loss.backward()
                total_loss.append(loss.cpu().item())
                self.optimizer.step()
                self.scheduler.step()
                # log
                logging.info(f"[epoch {e + 1}/{self.config['epoch']}, batch_id {n}/{len(relation_types)}], "
                             f"loss: {round(total_loss[-1], 3)}, lr: {self.optimizer.param_groups[0]['lr']}")

            if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != self.config['epoch']:
                logging.info(f"saving ckpt at `{self.output_dir}/epoch_{e + 1}`")
                self.model.save(f'{self.output_dir}/epoch_{e + 1}')
                with open(f"{self.output_dir}/epoch_{e + 1}/finetuning_config.json", 'w') as f:
                    json.dump(self.config, f, indent=2)

    def _train_triplet(self, positive_encode, negative_encode, relation_structure, epoch_save):

        def get_batch(_list):
            _index = list(range(len(_list)))
            _index = _index[::self.config['batch']] + [len(_list)]
            return [_list[_s:_e] for _s, _e in zip(_index[:-1], _index[1:])]

        def get_parent_sample(relation_type):
            # sampling relation type that shares same parent class with the positive sample
            parent_relation = [k for k, v in relation_structure.items() if relation_type in v]
            assert len(parent_relation) == 1
            relation_positive = rand_sample(relation_structure[parent_relation[0]])

            # sampling positive from the relation type
            positive_parent = rand_sample(positive_encode[relation_positive])

            # sampling relation type from different parent class (negative)
            parent_relation_n = rand_sample([k for k in relation_structure.keys() if k != parent_relation[0]])
            relation_negative = rand_sample(relation_structure[parent_relation_n])

            # sample individual entry from the relation
            negative_parent = rand_sample(positive_encode[relation_negative])
            return positive_parent, negative_parent

        if self.config['augment_negative_by_positive']:
            if negative_encode is None:
                negative_encode = {k: list(chain(*[v for _k, v in positive_encode.items() if k != _k])) for k in
                                   positive_encode.keys()}
            else:
                negative_encode = {
                    k: negative_encode[k] + list(chain(*[v for _k, v in positive_encode.items() if k != _k])) for k in
                    positive_encode.keys()}
        assert negative_encode is not None

        features = positive_encode[list(positive_encode.keys())[0]][0].keys()
        positive_pairs = {k: list(combinations(positive_encode[k], 2)) for k in sorted(positive_encode.keys())}
        n_iter_per_epoch = max(len(v) for k, v in positive_pairs.items())
        n_iter_per_epoch_neg = max(len(v) for k, v in negative_encode.items())
        relation_types = sorted(list(positive_encode.keys()))
        for e in range(self.config['epoch']):  # loop over the epoch
            pbar = tqdm(total=n_iter_per_epoch * n_iter_per_epoch_neg)
            random.shuffle(relation_types)
            for v in positive_pairs.values():
                random.shuffle(v)
            for v in negative_encode.values():
                random.shuffle(v)

            total_loss = []
            for i_pos in range(n_iter_per_epoch):
                for i_neg in range(n_iter_per_epoch_neg):
                    pbar.update(1)
                    _positive_pairs = {k: v[i_pos % len(v)] for k, v in positive_pairs.items()}
                    _negative_encode = {k: v[i_neg % len(v)] for k, v in negative_encode.items()}
                    self.optimizer.zero_grad()
                    loss = None
                    for batch_relations in get_batch(relation_types):

                        a = {h: to_tensor(h, [_positive_pairs[x][0][h] for x in batch_relations]) for h in features}
                        p = {h: to_tensor(h, [_positive_pairs[x][1][h] for x in batch_relations]) for h in features}
                        n = {h: to_tensor(h, [_negative_encode[x][h] for x in batch_relations]) for h in features}
                        v_a = self.model.to_embedding(a)
                        v_p = self.model.to_embedding(p)
                        v_n = self.model.to_embedding(n)
                        v_p_par = None
                        v_n_par = None
                        if relation_structure is not None:
                            tmp = [get_parent_sample(x) for x in batch_relations]
                            p_par = {h: to_tensor(h, [x[0][h] for x in tmp]) for h in features}
                            n_par = {h: to_tensor(h, [x[1][h] for x in tmp]) for h in features}
                            v_p_par = self.model.to_embedding(p_par)
                            v_n_par = self.model.to_embedding(n_par)
                        loss = loss_triplet(
                            tensor_anchor=v_a,
                            tensor_positive=v_p,
                            tensor_negative=v_n,
                            tensor_positive_parent=v_p_par,
                            tensor_negative_parent=v_n_par,
                            margin=self.config['loss_function_config']['mse_margin'],
                            linear=self.linear,
                            device=self.model.device)
                        total_loss.append(loss.cpu().item())

                    assert loss is not None
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                mean_loss = sum(total_loss) / len(total_loss)

                # log
                logging.info(
                    f"[epoch {e + 1}/{self.config['epoch']}, batch_id {pbar.n+1}/{pbar.total}], "
                    f"loss: {round(mean_loss, 3)}, lr: {self.optimizer.param_groups[0]['lr']}")

            if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != self.config['epoch']:
                self.model.save(f'{self.output_dir}/epoch_{e + 1}')
                with open(f"{self.output_dir}/epoch_{e + 1}/finetuning_config.json", 'w') as f:
                    json.dump(self.config, f, indent=2)

    def process_data(self, split):
        # raw data
        print(self.config['data_name'], self.config['data'])
        input()
        data = load_dataset(self.config['data'], self.config['data_name'], split=split)
        all_positive = {i['relation_type']: [tuple(_i) for _i in i['positives']] for i in data}
        all_negative = {i['relation_type']: [tuple(_i) for _i in i['negatives']] for i in data}
        assert all_positive.keys() == all_negative.keys(), f"{all_positive.keys()} != {all_negative.keys()}"
        if self.config['exclude_relation'] is not None:
            all_positive = {k: v for k, v in all_positive.items() if k not in self.config['exclude_relation']}
            all_negative = {k: v for k, v in all_negative.items() if k not in self.config['exclude_relation']}
        key = sorted(list(all_positive.keys()))
        logging.info(f'{len(key)} relations exist')

        # relation structure
        if all("/" not in i for i in key):
            relation_structure = None
            logging.info("no relation hierarchy is provided")
        else:
            parent = list(set([i.split("/")[0] for i in key]))
            relation_structure = {p: [i for i in key if p == i.split("/")[0]] for p in sorted(parent)}
            logging.info(f"relation_structure: {relation_structure}")

        # flatten pairs to encode them efficiently
        def _encode(pairs):
            sample_list = ListKeeper([pairs[k] for k in key])
            if len(sample_list.flatten_list) == 0:
                return None
            sample_dict = self.model.encode_word_pairs(sample_list.flatten_list, parallel=self.parallel_preprocess)
            e = [sample_dict[f"{a}__{b}"] for a, b in sample_list.flatten_list]
            e = sample_list.restore_structure(e)
            return {key[n]: v for n, v in enumerate(e)}

        positive_encode = _encode(all_positive)
        negative_encode = _encode(all_negative)
        assert len(positive_encode) >= self.config['batch']
        return positive_encode, negative_encode, relation_structure
