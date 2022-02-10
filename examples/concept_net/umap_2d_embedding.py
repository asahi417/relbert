"""
pip install umap-learn


"""
import os
import json
from glob import glob
from random import shuffle, seed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from gensim.models import KeyedVectors

if not os.path.exists('data/conceptnet_2d_embeddings.npy') or \
        not os.path.exists('data/conceptnet_2d_embeddings.relation_type.txt'):
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
    stats = {k: int(v * sample_size/sum(stats.values())) for k, v in stats.items()}
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

    relation_type_dict = {v: n for n, v in enumerate(sorted(list(set(relation_types))))}
    data = np.stack(embeddings)  # data x dimension

    # dimension reduction
    print('UMAP training'.format(data.shape))
    embedding_2d = UMAP().fit_transform(data)
    np.save('data/conceptnet_2d_embeddings.npy', embedding_2d)
    with open('data/conceptnet_2d_embeddings.relation_type.txt', 'w') as f:
        f.write('\n'.join(relation_types))
else:
    embedding_2d = np.load('data/conceptnet_2d_embeddings.npy')
    with open('data/conceptnet_2d_embeddings.relation_type.txt') as f:
        relation_types = [i for i in f.read().split('\n') if len(i) > 0]
# print(embedding_2d.shape)
# plt.scatter(
#     embedding_2d[:, 0],
#     embedding_2d[:, 1],
#     c=[sns.color_palette()[relation_type_dict[x]] for x in relation_types)]
#
# )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Penguin dataset', fontsize=24)