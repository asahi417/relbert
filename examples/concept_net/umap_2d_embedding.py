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


def get_term(arg):
    return arg.split('/en/')[-1].split('/')[0]


if not os.path.exists('data/conceptnet_2d_embeddings.npy') or \
        not os.path.exists('data/conceptnet_2d_embeddings.relation_type.txt'):

    top_n = 10
    max_sample_size = 1000
    sample_size = 100000
    concept_net_processed_file_dir = 'data/conceptnet'
    data = {}
    for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
        relation_type = os.path.basename(i).replace('.jsonl', '').replace('cache_', '')
        if relation_type == 'None':
            continue
        with open(i) as f:
            tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]
            tmp = [(get_term(i['arg1']), get_term(i['arg2'])) for i in tmp]
            tmp = [i for i in tmp if '_' not in i[0] and '_' not in i[1]]
        data[relation_type] = tmp
    top_types = [a for a, b in sorted(data.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]]
    size = {k: min(max_sample_size, len(v)) for k, v in data.items()}

    # load gensim model
    print('Collect embeddings')
    model = KeyedVectors.load_word2vec_format("data/relbert_embedding.bin", binary=True)

    embeddings = []
    relation_types = []
    for relation_type, v in data.items():
        if relation_type not in top_types:
            continue
        # down sample
        seed(0)
        shuffle(v)
        v = v[:size[relation_type]]

        # get embedding
        for a, b in v:
            embeddings.append(model['{}__{}'.format(a, b)])
            relation_types.append(relation_type)

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

print('Embedding shape: {}'.format(embedding_2d.shape))
assert len(embedding_2d) == len(relation_types)

print('Plot')
unique_relation_types = sorted(list(set(relation_types)))
relation_type_dict = {v: n for n, v in enumerate(unique_relation_types)}

plt.figure()
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    s=2,
    c=[relation_type_dict[i] for i in relation_types],
    cmap=sns.color_palette('Spectral', len(relation_type_dict), as_cmap=True)
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('2-d Embedding Space', fontsize=12)
plt.legend(handles=scatter.legend_elements(num=len(relation_type_dict))[0],
           labels=unique_relation_types,
           bbox_to_anchor=(1.04, 1),
           borderaxespad=0)
plt.savefig('data/conceptnet_2d_embeddings.png', bbox_inches='tight')
