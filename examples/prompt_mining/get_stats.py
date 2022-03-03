""" bar plot number of sentence """
import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import chain
from relbert.data import get_training_data, semeval_relations

os.makedirs('output', exist_ok=True)
path_to_corpus = 'cache/filtered_wiki.txt'

# get relation structure
all_positive, all_negative, relation_structure = get_training_data()
structure = {}
for k, v in all_positive.items():
    parent = [_k for _k, _v in relation_structure.items() if k in _v][0]
    for word_a, word_b in v:
        key = '{}-{}'.format(word_a, word_b)
        if key not in structure.keys():
            structure[key] = [[parent, k]]
        else:
            structure[key] += [[parent, k]]

# load main data
with open(path_to_corpus) as f:
    original_data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    data = {}
    for n, t in tqdm(list(enumerate(original_data))):
        t['relation_type'] = []
        for word_a, word_b in t['word_pairs']:
            key = '{}-{}'.format(word_a, word_b)
            t['relation_type'] += structure[key]
            if key not in data:
                data[key] = [n]
            else:
                data[key] += [n]

# get frequency
df_freq = pd.DataFrame([(k, len(v)) for k, v in sorted(data.items(), key=lambda x: len(x[1]), reverse=True)],
                       columns=['word pair', 'freq'])
df_freq['relation_type_parent'] = [list(set([semeval_relations[int(r[0])] for r in structure[k]])) for k in df_freq['word pair']]
df_freq['relation_type_child'] = [list(set([r[1] for r in structure[k]])) for k in df_freq['word pair']]

# plot frequency per data
ax = np.log(df_freq['freq']).plot.bar(legend=False)
ax.set_ylabel("Frequency (log scale)")
ax.axes.xaxis.set_visible(False)
fig = ax.get_figure()
plt.tight_layout()
fig.savefig('output/freq.per_relation.png')
fig.clear()
df_freq.to_csv('output/freq.csv')

# plot frequency per parent relation
unique_relation = list(set(list(chain(*df_freq['relation_type_parent'].values))))
freq = {r: df_freq[[r in i for i in df_freq['relation_type_parent']]]['freq'].sum() for r in unique_relation}
ax = pd.DataFrame(freq.values(), index=freq.keys()).sort_values(by=[0], ascending=False).plot.bar(legend=False)
ax.set_ylabel("Frequency")
fig = ax.get_figure()
plt.tight_layout()
fig.savefig('output/freq.per_parent_relation.png')
fig.clear()


