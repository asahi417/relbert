""" get SemEval2012 task 2 dataset """
import json
import os
import tarfile
import zipfile
import gzip
import requests

from glob import glob
from itertools import chain
from typing import List

import gdown


semeval_relations = {
    1: "Class Inclusion",  # Hypernym
    2: "Part-Whole",  # Meronym, Substance Meronym
    3: "Similar",  # Synonym, Co-hypornym
    4: "Contrast",  # Antonym
    5: "Attribute",  # Attribute, Event
    6: "Non Attribute",
    7: "Case Relation",
    8: "Cause-Purpose",
    9: "Space-Time",
    10: "Representation"
}

home_dir = '{}/.cache/relbert'.format(os.path.expanduser('~'))


def wget(url, cache_dir: str = './cache', gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        return gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    path = '{}/{}'.format(cache_dir, filename)

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
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)


def get_training_data(data_name: str = 'semeval2012', exclude_relation: List or str = None,
                      return_validation_set: bool = False, top_n: int = 10):
    """ Get RelBERT training data
    - SemEval 2012 task 2 dataset (case sensitive)

    Parameters
    ----------
    data_name : str
    exclude_relation : str

    Returns
    -------
    pairs: dictionary of list (positive pairs, negative pairs)
    {'1b': [[0.6, ('office', 'desk'), ..], [[-0.1, ('aaa', 'bbb'), ...]]
    """
    cache_dir = f'{home_dir}/data'
    os.makedirs(cache_dir, exist_ok=True)
    remove_relation = None
    if exclude_relation is not None:
        exclude_relation = [exclude_relation] if type(exclude_relation) is str else exclude_relation
        remove_relation = [k for k, v in semeval_relations.items() if v in exclude_relation]

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
        positives = {}
        negatives = {}
        all_relation_type = {}
        positives_score = {}
        # score_range = [90.0, 88.7]  # the absolute value of max/min prototypicality rating
        for i in files_scale:
            relation_id = i.split('-')[-1].replace('.txt', '')
            if remove_relation and int(relation_id[:-1]) in remove_relation:
                continue
            with open('{}/{}'.format(path_answer, i), 'r') as f:
                lines_answer = [_l.replace('"', '').split('\t') for _l in f.read().split('\n')
                                if not _l.startswith('#') and len(_l)]
                relation_type = list(set(list(zip(*lines_answer))[-1]))
                assert len(relation_type) == 1, relation_type
                relation_type = relation_type[0]
            with open('{}/{}'.format(path_scale, i), 'r') as f:
                # list of tuple [score, ("a", "b")]
                scales = [[float(_l[:5]), _l[6:].replace('"', '')] for _l in f.read().split('\n')
                          if not _l.startswith('#') and len(_l)]
                scales = sorted(scales, key=lambda _x: _x[0])
                # positive pairs are in the reverse order of prototypicality score
                positive_pairs = [[s, tuple(p.split(':'))] for s, p in filter(lambda _x: _x[0] > 0, scales)]
                positive_pairs = sorted(positive_pairs, key=lambda x:  x[0], reverse=True)
                if return_validation_set:
                    positive_pairs = positive_pairs[min(top_n, len(positive_pairs)):]
                    if len(positive_pairs) == 0:
                        continue
                else:
                    positive_pairs = positive_pairs[:min(top_n, len(positive_pairs))]
                positives_score[relation_id] = positive_pairs
                positives[relation_id] = list(list(zip(*positive_pairs))[1])
                negatives[relation_id] = [tuple(p.split(':')) for s, p in filter(lambda _x: _x[0] < 0, scales)]
            all_relation_type[relation_id] = relation_type

        # consider positive from other relation as negative
        for k in positives.keys():
            negatives[k] += list(chain(*[_v for _k, _v in positives.items() if _k != k]))
        pairs = {k: [positives[k], negatives[k]] for k in positives.keys()}
        parent = list(set([i[:-1] for i in all_relation_type.keys()]))
        relation_structure = {p: [i for i in all_relation_type.keys() if p == i[:-1]] for p in parent}
        for k, v in relation_structure.items():
            positive = list(chain(*[positives_score[_v] for _v in v]))
            positive = list(list(zip(*sorted(positive, key=lambda x: x[0], reverse=True)))[1])
            negative = []
            for _k, _v in relation_structure.items():
                if _k != k:
                    negative += list(chain(*[positives[__v] for __v in _v]))
            pairs[k] = [positive, negative]
        return pairs
    else:
        raise ValueError('unknown data: {}'.format(data_name))


def get_analogy_data(cache_dir: str = None):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = cache_dir if cache_dir is not None else home_dir
    cache_dir = '{}/data'.format(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset_with_prediction.zip'
    if not os.path.exists('{}/analogy_test_dataset_with_prediction'.format(cache_dir)):
        wget(root_url_analogy, cache_dir)
    data = {}
    for d in ['bats', 'sat', 'u2', 'u4', 'google']:
        with open('{}/analogy_test_dataset_with_prediction/{}/test.jsonl'.format(cache_dir, d), 'r') as f:
            test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
        with open('{}/analogy_test_dataset_with_prediction/{}/valid.jsonl'.format(cache_dir, d), 'r') as f:
            val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
        data[d] = (val_set, test_set)
    return data


def get_lexical_relation_data(cache_dir: str = None):
    cache_dir = cache_dir if cache_dir is not None else home_dir
    cache_dir = '{}/data'.format(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/lexical_relation_dataset.zip'
    if not os.path.exists('{}/lexical_relation_dataset'.format(cache_dir)):
        wget(root_url_analogy, cache_dir)
    full_data = {}
    for i in glob('{}/lexical_relation_dataset/*'.format(cache_dir)):
        if not os.path.isdir(i):
            continue
        full_data[os.path.basename(i)] = {}
        label = {}
        for t in glob('{}/*tsv'.format(i)):
            with open(t) as f:
                data = [line.split('\t') for line in f.read().split('\n') if len(line) > 0]
            x = [d[:2] for d in data]
            y = [d[-1] for d in data]
            for _y in y:
                if _y not in label:
                    label[_y] = len(label)
            y = [label[_y] for _y in y]
            full_data[os.path.basename(i)][os.path.basename(t).replace('.tsv', '')] = {'x': x, 'y': y}
        full_data[os.path.basename(i)]['label'] = label
    return full_data


if __name__ == '__main__':
    for _n in [10, 15, 20]:
        _data = get_training_data(return_validation_set=False, top_n=_n)
        print(sum([len(_data[k][0]) for k in _data.keys()]))
        _data = get_training_data(return_validation_set=True, top_n=_n)
        print(sum([len(_data[k][0]) for k in _data.keys()]))
