""" """
import os
import json
import argparse
from random import shuffle, seed
from tqdm import tqdm
from itertools import chain

import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from relbert import RelBERT
from gensim.models import KeyedVectors
from umap import UMAP

parser = argparse.ArgumentParser(description='Visualize RelBERT Embedding')
parser.add_argument('-m', '--model', help='language model', default="relbert/relbert-roberta-large", type=str)
parser.add_argument('-b', '--batch', help='', default=256, type=int)
parser.add_argument('-c', '--chunk', help='', default=512, type=int)
parser.add_argument('--raw', help='Process raw ConceptNet', action='store_true')
opt = parser.parse_args()

gensim_file = f"data/{os.path.basename(opt.model)}"

cluster_file = f"data/{os.path.basename(opt.model)}.cluster"
embedding_file = f"data/{os.path.basename(opt.model)}.embedding"
figure_file = f"data/{os.path.basename(opt.model)}.figure.png"

#################
# GET EMBEDDING #
#################
if not os.path.exists(f'{gensim_file}.bin'):
    word_pairs = list(chain(*[i['positives'] for i in load_dataset("relbert/conceptnet_high_confidence", split="full")]))
    model = RelBERT(opt.model, max_length=128)
    chunk_start, chunk_end = 0, opt.chunk
    pbar = tqdm(total=len(word_pairs))

    print(f'generate gensim file `{gensim_file}.txt` for {len(word_pairs)} pairs with {opt.model}')
    with open(f'{gensim_file}.txt', 'w', encoding='utf-8') as f:
        f.write(str(len(word_pairs)) + " " + str(model.model_config.hidden_size) + "\n")
        while chunk_start != chunk_end:
            vector = model.get_embedding(word_pairs[chunk_start:chunk_end], batch_size=opt.batch)
            for token_i, token_j in word_pairs[chunk_start:chunk_end]:
                f.write('__'.join([token_i.replace(' ', '_'), token_j.replace(' ', '_')]) + " ")
                f.write(' '.join([str(y) for y in vector[n]]) + "\n")
            chunk_start = chunk_end
            chunk_end = min(chunk_end + opt.chunk, len(word_pairs))
            pbar.update(chunk_end - chunk_start)

    print('convert to binary file')
    model = KeyedVectors.load_word2vec_format(f'{gensim_file}.txt')
    model.wv.save_word2vec_format(f'{gensim_file}.bin', binary=True)
    os.remove(f'{gensim_file}.txt')

###############
# GET CLUSTER #
###############
if not os.path.exists(f'{cluster_file}.json'):
    model = KeyedVectors.load_word2vec_format(f'{gensim_file}.bin', binary=True)

    cluster_info = {}
    for i in load_dataset("relbert/conceptnet_high_confidence", split="full"):

        # get embedding
        embeddings, keys = [], []
        for a, b in i['positives']:
            key = f'{a.replace("_", " ")}__{b.replace("_", " ")}'
            keys.append(key)
            embeddings.append(model[key])

        # clustering
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(np.stack(embeddings))  # data x dimension
        if clusterer.labels_.max() == -1:
            continue

        print(f'{clusterer.labels_.max()} clusters')
        cluster_info[i['relation_type']] = {k: int(i) for i, k in zip(clusterer.labels_, keys) if i != -1}

    cluster = {}
    for k, v in cluster_info.items():
        _cluster = {}
        for _k, _v in v.items():
            if _v in _cluster:
                _cluster[_v].append(_k)
            else:
                _cluster[_v] = [_k]
        cluster[k] = _cluster

    with open(f'{cluster_file}.json', 'w') as f:
        json.dump(cluster, f)

    pd.DataFrame(cluster).sort_index().to_csv(f'{cluster_file}.csv')

    for n, (k, v) in enumerate(cluster.items()):
        print(f'RELATION [{n+1}/{len(cluster)}] : {k}')
        for _k, _v in v.items():
            seed(0)
            shuffle(_v)
            print(f'* cluster {_k}')
            for pair in _v[:min(10, len(_v))]:
                print(f"\t - {pair.split('__')}")


################
# 2d embedding #
################
with open(f'{cluster_file}.json') as f:
    cluster = json.load(f)

if not os.path.exists(f'{embedding_file}.npy') or not os.path.exists(f'{embedding_file}.txt'):

    # load gensim model
    print('Collect embeddings')
    model = KeyedVectors.load_word2vec_format(f'{gensim_file}.bin', binary=True)

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
            embeddings.append(model[f'{a}__{b}'])
            relation_types.append(relation_type)

    data = np.stack(embeddings)  # data x dimension

    # dimension reduction
    print(f'UMAP training {data.shape}')
    embedding_2d = UMAP().fit_transform(data)
    np.save(f'{embedding_file}.npy', embedding_2d)
    with open(f'{embedding_file}.txt', 'w') as f:
        f.write('\n'.join(relation_types))
else:
    embedding_2d = np.load(f'{embedding_file}.npy')
    with open(f'{embedding_file}.txt') as f:
        relation_types = [i for i in f.read().split('\n') if len(i) > 0]

print(f'Embedding shape: {embedding_2d.shape}')
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
plt.savefig(figure_file, bbox_inches='tight')
