import os
import json
from glob import glob
from random import shuffle, seed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from gensim.models import KeyedVectors

sample_size = 100000
# load gensim model
model = KeyedVectors.load_word2vec_format("data/relbert_embedding.bin", binary=True)
concept_net_processed_file_dir = 'data/conceptnet'


def get_term(arg):
    return arg.split('/en/')[-1].split('/')[0]


stats = {}
for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
    relation_type = os.path.basename(i).replace('.jsonl', '')
    with open(i) as f:
        tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]
    stats[relation_type] = len(tmp)
print('Raw')
print(stats)
stats = {k: int(v * sample_size/sum(stats.values())) for k, v in stats.values()}
print('Down sampled')
print(stats)

embeddings = []
relation_types = []
for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
    relation_type = os.path.basename(i).replace('.jsonl', '')
    print('relation type: {}'.format(relation_type))
    with open(i) as f:
        tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]

    # down sample
    seed(0)
    shuffle(tmp)
    tmp = tmp[:stats[relation_type]]

    # get embedding
    for _tmp in tmp:
        key = '{}__{}'.format(get_term(_tmp['arg1']), get_term(_tmp['arg2']))
        embeddings.append(model[key])
        relation_types.append(relation_type)

relation_type_dict = {n: v for n, v in enumerate(sorted(list(set(relation_types))))}
data = np.stack(embeddings)  # data x dimension

# dimension reduction
embedding_2d = UMAP().fit_transform(data)
print(embedding_2d.shape)
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=[sns.color_palette()[x] for x in keys)])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24)