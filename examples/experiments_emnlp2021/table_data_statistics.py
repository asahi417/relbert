""" Export Latex Table for statistics of lexical relation classification dataset """
from copy import deepcopy

import pandas as pd
from relbert.data import get_lexical_relation_data
import warnings
warnings.filterwarnings("ignore")

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


df_all = {}
for _type in ['train', 'test', 'val']:
    data_freq = {}
    data = get_lexical_relation_data()
    for k, v in data.items():
        if _type not in v.keys():
            data_freq[k] = {}
        else:
            label = {v: k for k, v in v['label'].items()}
            data_freq[k] = {label[k]: v for k, v in freq(v[_type]['y']).items()}
    relations_in_train = ['Meronym', 'Antonym', 'Synonym', 'Attribute', 'Hypernym', 'Co-hypornym', 'Substance Meronym']

    data_freq_ = deepcopy(data_freq)
    for k, v in data_freq.items():
        for _k, _v in v.items():
            for __k, __v in shared_relation.items():
                if _k in __v:
                    data_freq_[k][__k] = data_freq_[k].pop(_k)

    df_all[_type] = pd.DataFrame(data_freq_)
df_all['train'].applymap(lambda x: "{:,}".format(x))
a = df_all['train'].fillna(0).applymap(lambda x: "{:,}".format(round(x)))
b = df_all['val'].fillna(0).applymap(lambda x: "{:,}".format(round(x)))
c = df_all['test'].fillna(0).applymap(lambda x: "{:,}".format(round(x)))
c = a + '/' + b + '/' + c
c = c.applymap(lambda x: '-' if x == '0/0/0' else x)
c = c.applymap(lambda x: x[:-2] if x[-2:] == '/0' else x)
c = c.applymap(lambda x: x.replace('/0/', '/'))
c = c.T[['Random', 'Meronym', 'Event', 'Hypernym', 'Co-hypornym', 'Attribute', 'Substance Meronym', 'Antonym', 'Synonym']].T
# c.index = [i.replace('Antonym', 'ant').replace('Attribute', 'attr').replace('Co-hypornym', 'cohyp').
#                replace('Event', 'event').replace('Substance Meronym', 'subs').replace('Hypernym', 'hyp').replace('Meronym', 'mero').
#                replace('Random', 'rand').replace('Synonym', 'syn') for i in c.index]

c = c[['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']]


print(c.to_latex())
