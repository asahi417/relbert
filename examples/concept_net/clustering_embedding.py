import os
import json
from glob import glob
from random import shuffle, seed

import hdbscan
import numpy as np
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

cluster_info = {}
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
    embeddings = []
    keys = []
    for _tmp in tmp:
        key = '{}__{}'.format(get_term(_tmp['arg1']), get_term(_tmp['arg2']))
        keys.append(key)
        embeddings.append(model[key])
    data = np.stack(embeddings)  # data x dimension
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(data)
    print('{} clusters'.format(clusterer.labels_.max()))
    cluster_info[relation_type] = {k: i for i, k in zip(clusterer.labels_, keys) if i != -1}
