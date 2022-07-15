import json
import logging
import os
import statistics
from itertools import chain, product
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from torch import nn
from .lm import RelBERT, Dataset
from .data import get_lexical_relation_data, get_analogy_data, get_training_data
from .util import fix_seed

__all__ = ['evaluate_classification', 'evaluate_analogy']


def cosine_similarity(a, b, zero_vector_mask: float = -100):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    if norm_b * norm_a == 0:
        return zero_vector_mask
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/(norm_a * norm_b)


def euclidean_distance(a, b):
    return sum(map(lambda x: (x[0] - x[1])**2, zip(a, b))) ** 0.5


class RelationClassification:

    def __init__(self,
                 dataset,
                 shared_config,
                 label_dict,
                 target_relation=None,
                 default_config: bool = False,
                 config=None):
        self.dataset = dataset
        self.label_dict = label_dict
        self.target_relation = target_relation
        if default_config:
            self.configs = [{'random_state': 0}]
        elif config is not None:
            self.configs = [config]
        else:
            learning_rate_init = [0.001, 0.0001, 0.00001]
            hidden_layer_sizes = [100, 150, 200]
            self.configs = [{
                'random_state': 0, 'learning_rate_init': i[0], 'hidden_layer_sizes': i[1]} for i in
                            list(product(learning_rate_init, hidden_layer_sizes))]
        self.shared_config = shared_config

    def run_test(self, clf, x, y, per_class_metric: bool = False):
        """ run evaluation on valid or test set """
        y_pred = clf.predict(x)
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y, y_pred, average='macro')
        p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y, y_pred, average='micro')
        accuracy = sum([a == b for a, b in zip(y, y_pred.tolist())]) / len(y_pred)
        tmp = {
            'accuracy': accuracy,
            'f1_macro': f_mac,
            'f1_micro': f_mic,
            'p_macro': p_mac,
            'p_micro': p_mic,
            'r_macro': r_mac,
            'r_micro': r_mic
        }
        if per_class_metric and self.target_relation is not None:
            for _l in self.target_relation:
                if _l in self.label_dict:
                    p, r, f, _ = precision_recall_fscore_support(y, y_pred, labels=[self.label_dict[_l]])
                    tmp['f1/{}'.format(_l)] = f[0]
                    tmp['p/{}'.format(_l)] = p[0]
                    tmp['r/{}'.format(_l)] = r[0]
        return tmp

    @property
    def config_indices(self):
        return list(range(len(self.configs)))

    def __call__(self, config_id):
        config = self.configs[config_id]
        report = self.shared_config.copy()
        # train
        x, y = self.dataset['train']
        clf = MLPClassifier(**config).fit(x, y)
        report.update({'classifier_config': clf.get_params()})
        # test
        x, y = self.dataset['test']
        tmp = self.run_test(clf, x, y, per_class_metric=True)
        tmp = {'test/{}'.format(k): v for k, v in tmp.items()}
        report.update(tmp)
        if 'val' in self.dataset:
            x, y = self.dataset['val']
            tmp = self.run_test(clf, x, y, per_class_metric=True)
            tmp = {'val/{}'.format(k): v for k, v in tmp.items()}
            report.update(tmp)
        return report


def evaluate_classification(relbert_ckpt: str = None,
                            max_length: int = 64,
                            batch_size: int = 512,
                            target_relation=None,
                            random_seed: int = 0,
                            config=None):
    fix_seed(random_seed)
    model = RelBERT(relbert_ckpt, max_length=max_length)
    assert model.is_trained, 'model is not trained'

    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(relbert_ckpt, data_name))
        label_dict = v.pop('label')
        dataset = {}
        for _k, _v in v.items():
            x_tuple = [tuple(_x) for _x in _v['x']]
            x = model.get_embedding(x_tuple, batch_size=batch_size)
            x_back = model.get_embedding([(b, a) for a, b in x_tuple], batch_size=batch_size)
            x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
            dataset[_k] = [x, _v['y']]
        shared_config = {
            'model': relbert_ckpt,
            'label_size': len(label_dict),
            'data': data_name,
            'template': model.template,
            'template_mode': model.template_mode,
            'mode': model.mode,
            'max_length': model.max_length}
        # grid serach
        if 'val' not in dataset:
            logging.info('run default config')
            evaluator = RelationClassification(dataset, shared_config, label_dict, target_relation=target_relation,
                                               default_config=True)
            report += [evaluator(0)]
        elif config is not None and data_name in config:
            logging.info('run with given config')
            evaluator = RelationClassification(dataset, shared_config, label_dict, target_relation=target_relation,
                                               config=config[data_name])
            report += [evaluator(0)]

        else:
            logging.info('run grid search')
            pool = Pool()
            evaluator = RelationClassification(dataset, shared_config, label_dict, target_relation=target_relation)
            report += pool.map(evaluator, evaluator.config_indices)
            pool.close()
    del model
    return report


def compute_loss(model, validation_data: str, exclude_relation, loss_function: str, batch_size: int,
                 temperature_nce_rank: Dict = None, temperature_nce_constant: float = None):

    def stack_sum(_list):
        if len(_list) == 0:
            return 0
        return torch.mean(torch.stack(_list))

    data = get_training_data(data_name=validation_data, exclude_relation=exclude_relation, return_validation_set=True)
    encoded_pairs_dict = model.encode_word_pairs(list(chain(*[p + n for p, n in data.values()])))
    loader_dict = {}
    for k, (pairs_p, pairs_n) in data.items():
        dataset_p = Dataset([encoded_pairs_dict['__'.join(k)] for k in pairs_p], return_ranking=True)
        dataset_n = Dataset([encoded_pairs_dict['__'.join(k)] for k in pairs_n], return_ranking=False)
        loader_dict[k] = {
            'positive': torch.utils.data.DataLoader(dataset_p, num_workers=0, batch_size=len(pairs_p)),
            'negative': torch.utils.data.DataLoader(dataset_n, num_workers=0, batch_size=len(pairs_n))
        }
    relation_keys = list(data.keys())

    cos_2d = nn.CosineSimilarity(dim=1)
    cos_1d = nn.CosineSimilarity(dim=0)
    total_loss = []
    for n, relation_key in tqdm(list(enumerate(relation_keys))):
        # data loader will return full instances
        x_p = next(iter(loader_dict[relation_key]['positive']))
        x_n = next(iter(loader_dict[relation_key]['negative']))
        # data loader will return full instances
        x = {k: torch.concat([x_p[k], x_n[k]]) for k in x_n.keys()}
        embedding = model.to_embedding(x, batch_size=batch_size)
        batch_size_positive = len(x_p['input_ids'])
        embedding_p = embedding[:batch_size_positive]
        embedding_n = embedding[batch_size_positive:]
        loss = []
        if loss_function == 'nce_rank':

            def get_rank_temperature(_i, n):
                assert temperature_nce_rank is not None
                assert _i <= n, f"{_i}, {n}"
                if temperature_nce_rank['type'] == 'linear':
                    _min = temperature_nce_rank['min']
                    _max = temperature_nce_rank['max']
                    return (_min - _max) / (1 - n) * (_i - 1) + _min
                raise ValueError(f"unknown type: {temperature_nce_rank['type']}")

            rank = x_p.pop('ranking').cpu().tolist()
            rank_map = {r: 1 + n for n, r in enumerate(sorted(rank))}
            rank = [rank_map[r] for r in rank]
            for i in range(batch_size_positive):
                assert type(rank[i]) == int, rank[i]
                tau = get_rank_temperature(rank[i], batch_size_positive)
                deno_n = torch.sum(torch.exp(cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / tau))
                dist = torch.exp(cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / tau)
                nume_p = stack_sum([d for n, d in enumerate(dist) if rank[n] >= rank[i]])
                deno_p = stack_sum([d for n, d in enumerate(dist) if rank[n] < rank[i]])
                loss.append(- torch.log(nume_p / (deno_p + deno_n)))
        elif loss_function == 'nce_logout':
            assert temperature_nce_constant is not None
            for i in range(batch_size_positive):
                deno_n = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / temperature_nce_constant))
                for p in range(batch_size_positive):
                    logit_p = torch.exp(
                        cos_1d(embedding_p[i], embedding_p[p]) / temperature_nce_constant)
                    loss.append(- torch.log(logit_p/(logit_p + deno_n)))
        elif loss_function == 'nce_login':
            assert temperature_nce_constant is not None
            for i in range(batch_size_positive):
                deno_n = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / temperature_nce_constant))
                logit_p = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / temperature_nce_constant))
                loss.append(- torch.log(logit_p/(logit_p + deno_n)))
        else:
            raise ValueError(f"unknown loss function {loss_function}")
        total_loss.append(stack_sum(loss).cpu().item())
    total_loss = statistics.mean(total_loss)
    return total_loss


def evaluate_analogy(relbert_ckpt: str = None,
                     max_length: int = 64,
                     batch_size: int = 64,
                     distance_function: str = 'cosine_similarity',
                     validation_data: str = 'semeval2012',
                     return_validation_loss: bool = False,
                     exclude_relation=None):
    model = RelBERT(relbert_ckpt, max_length=max_length)
    assert model.is_trained, 'model is not trained'
    model.eval()
    result = []
    with torch.no_grad():
        if return_validation_loss:
            assert os.path.exists(f"{relbert_ckpt}/trainer_config.json")
            with open(f"{relbert_ckpt}/trainer_config.json") as f:
                trainer_config = json.load(f)
            loss_function = trainer_config['loss_function']
            temperature_nce_rank = trainer_config['temperature_nce_rank']
            temperature_nce_constant = trainer_config['temperature_nce_constant']
            validation_loss = compute_loss(
                model, validation_data, exclude_relation, loss_function, batch_size,
                temperature_nce_rank, temperature_nce_constant)
        else:
            validation_loss = None

        # Analogy test
        data = get_analogy_data()
        for d, (val, test) in data.items():
            logging.info('\t * data: {}'.format(d))
            # preprocess data
            all_pairs = [tuple(i) for i in list(chain(*[[o['stem']] + o['choice'] for o in val + test]))]
            embeddings = model.get_embedding(all_pairs, batch_size=batch_size)
            assert len(embeddings) == len(all_pairs), f"{len(embeddings)} != {len(all_pairs)}"
            embeddings_dict = {str(tuple(k_)): v for k_, v in zip(all_pairs, embeddings)}

            def prediction(_data):
                accuracy = []
                for single_data in _data:
                    v_stem = embeddings_dict[str(tuple(single_data['stem']))]
                    v_choice = [embeddings_dict[str(tuple(c))] for c in single_data['choice']]
                    if distance_function == "cosine_similarity":
                        sims = [cosine_similarity(v_stem, v) for v in v_choice]
                    elif distance_function == "euclidean_distance":
                        sims = [euclidean_distance(v_stem, v) for v in v_choice]
                    else:
                        raise ValueError(f'unknown distance function {distance_function}')
                    pred = sims.index(max(sims))
                    if sims[pred] == -100:
                        raise ValueError('failed to compute similarity')
                    accuracy.append(single_data['answer'] == pred)
                return sum(accuracy) / len(accuracy)

            # get prediction
            acc_val = prediction(val)
            acc_test = prediction(test)
            acc = prediction(val+test)
            result.append({
                'accuracy/valid': acc_val,
                'accuracy/test': acc_test,
                'accuracy/full': acc,
                'data': d,
                'model': relbert_ckpt,
                'template': model.template,
                'template_mode': model.template_mode,
                'mode': model.mode,
                'max_length': model.max_length,
                "distance_function": distance_function,
                "validation_loss": validation_loss,
                "validation_data": validation_data,
                "validation_data/exclude_relation": exclude_relation
            })
            logging.info(str(result[-1]))
    del model
    return result



