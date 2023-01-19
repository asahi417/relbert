""" get SemEval2012 task 2 dataset """
import os
import json
# from datasets import load_dataset
from glob import glob

from relbert.util import wget, home_dir

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


def get_training_data(data_name: str = 'semeval2012',
                      n_sample: int = 10,
                      cache_dir: str = None,
                      validation_set: bool = False,
                      exclude_relation: str = None):
    """ Get RelBERT training data
    - SemEval 2012 task 2 dataset (case sensitive)

    Parameters
    ----------
    data_name : str
    cache_dir : str
    n_sample : int
        Sample size of positive/negative.
    validation_set : bool
        To return get the validation set

    Returns
    -------
    all_positive, all_negative, all_relation_type : dict
        Dictionary with relation type as key.
    """
    v_rate = 0.2
    n_sample_max = 10
    assert n_sample <= n_sample_max
    cache_dir = cache_dir if cache_dir is not None else home_dir
    cache_dir = '{}/data'.format(home_dir)

    os.makedirs(cache_dir, exist_ok=True)
    remove_relation = None
    if exclude_relation:
        remove_relation = [k for k, v in semeval_relations.items() if exclude_relation == v]

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
            if remove_relation and int(relation_id[:-1]) in remove_relation:
                continue
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
                _negative = [tuple(i.split(':')) for i in
                             list(zip(*list(filter(lambda x: x[0] < 0, lines_scale[:n_sample_max]))))[1]]
                _positive = [tuple(i.split(':')) for i in
                             list(zip(*list(filter(lambda x: x[0] > 0, lines_scale[-n_sample_max:]))))[1]]
                v_negative = _negative[::int(len(_negative) * (1 - v_rate))]
                v_positive = _positive[::int(len(_positive) * (1 - v_rate))]
                t_negative = [i for i in _negative if i not in v_negative]
                t_positive = [i for i in _positive if i not in v_positive]
                if validation_set:
                    all_negative[relation_id] = v_negative
                    all_positive[relation_id] = v_positive
                else:
                    all_negative[relation_id] = t_negative[:n_sample]
                    all_positive[relation_id] = t_positive[-n_sample:]

            all_relation_type[relation_id] = relation_type
        parent = list(set([i[:-1] for i in all_relation_type.keys()]))
        relation_structure = {p: [i for i in all_relation_type.keys() if p == i[:-1]] for p in parent}
        return all_positive, all_negative, relation_structure
    else:
        raise ValueError('unknown data: {}'.format(data_name))


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
