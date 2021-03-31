""" get SemEval2012 task 2 dataset """
import os
import json
from glob import glob
from .util import wget, home_dir

URL = 'https://drive.google.com/u/0/uc?id=0BzcZKTSeYL8VYWtHVmxUR3FyUmc&export=download'
FILENAME = 'SemEval-2012-Platinum-Ratings.tar.gz'


def get_semeval_data(n_sample: int = 10, return_score: bool = False, cache_dir: str = None):
    """ Get SemEval 2012 task 2 dataset

    Parameters
    ----------
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

    path_answer = '{}/Phase2Answers'.format(cache_dir)
    path_scale = '{}/Phase2AnswersScaled'.format(cache_dir)
    if not (os.path.exists(path_scale) and os.path.exists(path_answer)):
        wget(URL, gdrive_filename=FILENAME, cache_dir=cache_dir)
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

