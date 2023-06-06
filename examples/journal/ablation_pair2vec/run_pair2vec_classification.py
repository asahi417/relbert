import logging
import tqdm
import tarfile
import requests
import os
import json
from itertools import product
from multiprocessing import Pool

import numpy as np
from gensim.models import KeyedVectors
from datasets import load_dataset

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

pbar = tqdm.tqdm()


def run_test(clf, x, y):
    """ run evaluation on valid or test set """
    y_pred = clf.predict(x)
    f_mac = f1_score(y, y_pred, average='macro')
    f_mic = f1_score(y, y_pred, average='micro')
    accuracy = sum([a == b for a, b in zip(y, y_pred.tolist())]) / len(y_pred)
    return accuracy, f_mac, f_mic


class Evaluate:

    def __init__(self, dataset, shared_config, default_config: bool = False):
        self.dataset = dataset
        if default_config:
            self.configs = [{'random_state': 0}]
        else:
            learning_rate_init = [0.001, 0.0001, 0.00001]
            hidden_layer_sizes = [100, 150, 200]
            self.configs = [{
                'random_state': 0, 'learning_rate_init': i[0],
                'hidden_layer_sizes': i[1]} for i in list(product(learning_rate_init, hidden_layer_sizes))]
        self.shared_config = shared_config

    @property
    def config_indices(self):
        return list(range(len(self.configs)))

    def __call__(self, config_id):
        pbar.update(1)
        config = self.configs[config_id]
        # train
        x, y = self.dataset['train']
        clf = MLPClassifier(**config).fit(x, y)
        # test
        x, y = self.dataset['test']
        t_accuracy, t_f_mac, t_f_mic = run_test(clf, x, y)
        report = self.shared_config.copy()
        report.update({'metric/test/accuracy': t_accuracy,
                       'metric/test/f1_macro': t_f_mac,
                       'metric/test/f1_micro': t_f_mic,
                       'classifier_config': clf.get_params()})
        if 'val' in self.dataset:
            x, y = self.dataset['val']
            v_accuracy, v_f_mac, v_f_mic = run_test(clf, x, y)
            report.update({'metric/val/accuracy': v_accuracy,
                           'metric/val/f1_macro': v_f_mac,
                           'metric/val/f1_micro': v_f_mic})
        return report


if __name__ == '__main__':
    # load pair2vec
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)
    path = f'{cache_dir}/pair2vec.fasttext.bin'
    if not os.path.exists(path):
        url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/pair2vec.fasttext.bin.tar.gz'
        os.makedirs(cache_dir, exist_ok=True)
        with open(f'{cache_dir}/{os.path.basename(url)}', "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    model = KeyedVectors.load_word2vec_format(path, binary=True)

    def get_vector(a, b):
        try:
            fr = model['__'.join([a, b]).lower().replace(' ', '_')]
        except KeyError:
            fr = np.zeros(model.vector_size)
        try:
            bw = model['__'.join([b, a]).lower().replace(' ', '_')]
        except KeyError:
            bw = np.zeros(model.vector_size)
        return np.append(fr, bw)

    # prepare input data
    report = []
    for data_name in ['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']:
        v = load_dataset("relbert/lexical_relation_classification", data_name)
        oov = {}
        dataset = {}
        label2id = None
        for _k in v:
            _v = v[_k]
            if label2id is None:
                label2id = {q: w for w, q in enumerate(sorted(list(set(_v['relation']))))}
            # initialize zero vector for OOV
            dataset[_k] = [
                [get_vector(a, b) for a, b in zip(_v['head'], _v['tail'])],
                [label2id[q] for q in _v['relation']]
            ]
            oov[_k] = len([i for i in dataset[_k][0] if i.sum() == 0])
        shared_config = {'data': data_name, 'oov': oov}

        # grid search
        if 'val' not in dataset:
            evaluator = Evaluate(dataset, shared_config, default_config=True)
            tmp_report = evaluator(0)
        else:
            pool = Pool()
            evaluator = Evaluate(dataset, shared_config)
            tmp_report = pool.map(evaluator, evaluator.config_indices)
            pool.close()
        tmp_report = [tmp_report] if type(tmp_report) is not list else tmp_report
        report += tmp_report
    del model

    with open("pair2vec.jsonl", "w") as f:
        f.write('\n'.join([json.dumps(i) for i in report]))
