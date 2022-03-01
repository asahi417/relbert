import os
import json
from glob import glob
from random import shuffle, seed

import hdbscan
import numpy as np
from gensim.models import KeyedVectors


def get_term(arg):
    return arg.split('/en/')[-1].split('/')[0]


def load_embedding(top_n: int = 20, max_sample_size: int = 1000):
    concept_net_processed_file_dir = 'data/conceptnet'
    _data = {}
    for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
        _relation_type = os.path.basename(i).replace('.jsonl', '').replace('cache_', '')
        if _relation_type == 'None':
            continue
        with open(i) as f_reader:
            tmp = [json.loads(t) for t in f_reader.read().split('\n') if len(t) > 0]
            tmp = [(get_term(i['arg1']), get_term(i['arg2'])) for i in tmp]
            tmp = [i for i in tmp if '_' not in i[0] and '_' not in i[1] and i[0] != i[1]]
        _data[_relation_type] = tmp
    _top_types = [_a for _a, _b in sorted(_data.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]]
    _size = {k: min(max_sample_size, len(__v)) for k, __v in _data.items()}
    return _top_types, _size, _data


if not os.path.exists('data/conceptnet_clusters.json'):

    top_types, size, data = load_embedding(10, 1000)
    # load gensim model
    print('Collect embeddings')
    model = KeyedVectors.load_word2vec_format("data/relbert_embedding.bin", binary=True)

    cluster_info = {}
    for relation_type, v in data.items():
        if relation_type not in top_types:
            continue
        # down sample
        seed(0)
        shuffle(v)
        v = v[:size[relation_type]]

        # get embedding
        embeddings = []
        keys = []
        for a, b in v:
            key = '{}__{}'.format(a, b)
            keys.append(key)
            embeddings.append(model[key])
        data = np.stack(embeddings)  # data x dimension

        # clustering
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(data)
        if clusterer.labels_.max() == -1:
            continue

        print('{} clusters'.format(clusterer.labels_.max()))
        cluster_info[relation_type] = {k: int(i) for i, k in zip(clusterer.labels_, keys) if i != -1}

    cluster = {}
    for k, v in cluster_info.items():
        k = k.replace('cache_', '')
        _cluster = {}
        for _k, _v in v.items():
            if _v in _cluster:
                _cluster[_v].append(_k)
            else:
                _cluster[_v] = [_k]
        cluster[k] = _cluster

    with open('data/conceptnet_clusters.json', 'w') as f:
        json.dump(cluster, f)

else:
    with open('data/conceptnet_clusters.json') as f:
        cluster = json.load(f)

for n, (k, v) in enumerate(cluster.items()):
    print('RELATION [{}/{}] : {}'.format(n + 1, len(cluster), k))
    for _k, _v in v.items():
        seed(0)
        shuffle(_v)
        print('* cluster {}'.format(_k))
        for pair in _v[:min(10, len(_v))]:
            print('\t - {}'.format(pair.split('__')))
