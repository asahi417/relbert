"""

"""
from copy import deepcopy

import pandas as pd
from relbert.data import get_analogy_data, get_lexical_relation_data, get_training_data


def freq(_list, prefix=None):
    def _get(_x):
        if prefix:
            return _x[prefix]
        return _x
    f_dict = {}
    for e in _list:
        if _get(e) in f_dict:
            f_dict[_get(e)] += 1
        else:
            f_dict[_get(e)] = 1
    return f_dict


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

data_freq = {}
data = get_lexical_relation_data()
for k, v in data.items():
    label = {v: k for k, v in v['label'].items()}
    data_freq[k] = {label[k]: v for k, v in freq(v['test']['y']).items()}
relations_in_train = ['Meronym', 'Antonym', 'Synonym', 'Attribute', 'Hypernym', 'Co-hypornym', 'Substance Meronym']
shared_relation = {
    'Random': ['random', 'RANDOM', 'false'],
    'Meronym': ['PartOf', 'PART_OF', 'mero', 'MadeOf'],
    'Event': ['event'],
    'Substance Meronym': ['HasA'],
    'Antonym': ['ANT', 'Antonym'],
    'Synonym': ['SYN', 'Synonym'],
    'Hypernym': ['HYPER', 'hyper', 'hypo', 'IsA'],
    'Co-hypornym': ['COORD', 'coord', 'sibl'],
    'Attribute': ['attri', 'HasProperty']
}

data_freq_ = deepcopy(data_freq)
for k, v in data_freq.items():
    for _k, _v in v.items():
        for __k, __v in shared_relation.items():
            if _k in __v:
                data_freq_[k][__k] = data_freq_[k].pop(_k)

df = pd.DataFrame(data_freq_)
df.to_csv('examples/analysis_data/data_stats.csv')
# relations = list(df.index)
# relations_in = [i for i in relations if i in relations_in_train]
# relations_out = [i for i in relations if i not in relations_in_train]
# print(df.T[relations_in].T)
# print(df.T[relations_out].T)

# data_freq = {}
# a_data = get_analogy_data()
# _, test = a_data['bats']
# test = [i['prefix'].split('BATS_3.0/')[-1].split('/')[0] for i in test]
# data_freq['bats'] = freq(test)
# _, test = a_data['google']
# data_freq['google'] = freq(test, 'prefix')

