""" get SemEval2012 task 2 dataset """
import os
import json
import re
from glob import glob
from itertools import chain, product

import truecase
from .util import wget, home_dir


def get_training_data(data_name: str, n_sample: int = 10, return_score: bool = False, cache_dir: str = None):
    """ Get RelBERT training data
    - SemEval 2012 task 2 dataset (case sensitive)
    - BATS (lowercased/truecased)

    Parameters
    ----------
    data_name : str
    cache_dir : str
    n_sample : int
        Sample size of positive/negative.
    return_score : bool
        To return prototypical score.

    Returns
    -------
    all_positive, all_negative, all_relation_type : dict
        Dictionary with relation type as key.
    """
    cache_dir = cache_dir if cache_dir is not None else home_dir
    cache_dir = '{}/data'.format(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if data_name == 'semeval2012':
        path_answer = '{}/Phase2Answers'.format(cache_dir)
        path_scale = '{}/Phase2AnswersScaled'.format(cache_dir)
        url = 'https://drive.google.com/u/0/uc?id=0BzcZKTSeYL8VYWtHVmxUR3FyUmc&export=download'
        filename = 'SemEval-2012-Platinum-Ratings.tar.gz'
        if not (os.path.exists(path_scale) and os.path.exists(path_answer)):
            wget(url, gdrive_filename=filename, cache_dir=cache_dir)
        files_answer = [os.path.basename(i) for i in glob('{}/*.txt'.format(path_answer))]
        files_scale = [os.path.basename(i) for i in glob('{}/*.txt'.format(path_scale))]
        assert files_answer == files_scale, 'files are not matched: {} vs {}'.format(files_scale, files_answer)
        all_positive = {}
        all_negative = {}
        all_relation_type = {}
        for i in files_scale:
            relation_id = i.split('-')[-1].replace('.txt', '')
            with open('{}/{}'.format(path_answer, i), 'r') as f:
                lines_answer = [l.replace('"', '').split('\t') for l in f.read().split('\n')
                                if not l.startswith('#') and len(l)]
                relation_type = list(set(list(zip(*lines_answer))[-1]))
                assert len(relation_type) == 1, relation_type
                relation_type = relation_type[0]
            with open('{}/{}'.format(path_scale, i), 'r') as f:
                lines_scale = [[float(l[:5]), l[6:].replace('"', '')] for l in f.read().split('\n')
                               if not l.startswith('#') and len(l)]
                lines_scale = sorted(lines_scale, key=lambda x: x[0])
                if return_score:
                    all_negative[relation_id] = list(filter(lambda x: x[0] < 0, lines_scale[:n_sample]))
                    all_positive[relation_id] = list(filter(lambda x: x[0] > 0, lines_scale[-n_sample:]))
                else:
                    all_negative[relation_id] = [tuple(i.split(':')) for i in
                                                 list(zip(*list(filter(lambda x: x[0] < 0, lines_scale[:n_sample]))))[1]]
                    all_positive[relation_id] = [tuple(i.split(':')) for i in
                                                 list(zip(*list(filter(lambda x: x[0] > 0, lines_scale[-n_sample:]))))[1]]

            all_relation_type[relation_id] = relation_type
        parent = list(set([i[:-1] for i in all_relation_type.keys()]))
        relation_structure = {p: [i for i in all_relation_type.keys() if p == i[:-1]] for p in parent}
        return all_positive, all_negative, relation_structure
    elif data_name in ['bats']:
        url = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/BATS_3.0.zip'
        path = '{}/BATS_3.0'.format(cache_dir)
        if not os.path.exists(path):
            wget(url, cache_dir=cache_dir)
        relation_structure = {}
        all_positive = {}
        for _file in glob('{}/*'.format(path)):
            if not os.path.isdir(_file):
                continue
            relation_type = os.path.basename(_file).split('_')[0]
            relation_structure[relation_type] = []
            for c in glob('{}/*.txt'.format(_file)):
                relation_type_c = os.path.basename(c).split(' ')[0]
                relation_structure[relation_type].append(relation_type_c)

                def flatten_pair(_pair):
                    a, b = _pair
                    a, b = a.replace('_', ' '), b.replace('_', ' ')
                    # if data_name == 'bats_truecase':
                    #
                    #     a = truecase.get_true_case('A {}'.format(a))[2:].rstrip()
                    #     a = re.sub(r'\A\s', '', a)
                    #     if 'reorganise' in b:
                    #         print(b)
                    #     b = truecase.get_true_case('A {}'.format(b))[2:].rstrip()
                    #     if 'reorganise' in b:
                    #         print(b)
                    #         input()
                    #     b = re.sub(r'\A\s', '', b)
                    return list(product(a.split('/'), b.split('/')))

                with open(c, 'r') as f_read:
                    pairs = list(chain(*[flatten_pair(i.split('\t')) for i in f_read.read().split('\n') if len(i)]))
                all_positive[relation_type_c] = pairs

        def get_parent(i):
            return [k for k, v in relation_structure.items() if i in v][0]

        all_negative = {k: list(chain(*[v for _k, v in all_positive.items()
                                        if get_parent(_k) == get_parent(k) and _k != k])) for k in all_positive.keys()}
        return all_positive, all_negative, relation_structure


def get_analogy_data(data_name: str, cache_dir: str = None):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = cache_dir if cache_dir is not None else home_dir
    cache_dir = '{}/data'.format(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    root_url_analogy = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0'
    assert data_name in ['sat', 'u2', 'u4', 'google', 'bats'], 'unknown data_iterator: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        wget('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val_set, test_set

