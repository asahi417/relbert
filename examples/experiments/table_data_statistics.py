""" Export Latex Table for statistics of lexical relation classification """
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
for _type in ['train', 'test']:
    data_freq = {}
    data = get_lexical_relation_data()
    for k, v in data.items():
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
b = df_all['test'].fillna(0).applymap(lambda x: "{:,}".format(round(x)))
c = a + '/' + b
c = c.applymap(lambda x: '-' if x == '0/0' else x)
c.index = [i.replace('Antonym', 'ant').replace('Attribute', 'attr').replace('Co-hypornym', 'cohyp').
               replace('Event', 'event').replace('Hypernym', 'hyp').replace('Meronym', 'mero').
               replace('Random', 'rand').replace('Substance Meronym', 'subs').replace('Synonym', 'syn') for i in c.index]

c = c[['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']]
print(c.to_latex())
