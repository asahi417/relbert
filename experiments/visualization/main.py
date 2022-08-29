"""
pip install umap-learn
pip install hdbscan
pip install seaborn

export MODEL_ALIAS=relbert/relbert-roberta-large-semeval2012-average-prompt-d-nce
python main.py

export MODEL_ALIAS=relbert/relbert-roberta-large-semeval2012-average-no-mask-prompt-d-triplet
python main.py
"""

import os
from tqdm import tqdm
import json
from random import shuffle, seed
from glob import glob

import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from relbert import RelBERT
from gensim.models import KeyedVectors
from umap import UMAP

MODEL_ALIAS = os.getenv("MODEL_ALIAS", "relbert/relbert-roberta-large-semeval2012-average-prompt-d-nce")
# MODEL_ALIAS = os.getenv("MODEL_ALIAS", "relbert/relbert-roberta-large-semeval2012-average-no-mask-prompt-d-triplet")

concept_net_processed_file_dir = './data/conceptnet'
gensim_file = f"data/{os.path.basename(MODEL_ALIAS)}"
cluster_file = f"data/{os.path.basename(MODEL_ALIAS)}.cluster"
embedding_file = f"data/{os.path.basename(MODEL_ALIAS)}.embedding"
figure_file = f"data/{os.path.basename(MODEL_ALIAS)}.figure.png"


def get_term(arg):
    return arg.split('/en/')[-1].split('/')[0]


######################
# PROCESS CONCEPTNET #
######################
if len(glob(f'{concept_net_processed_file_dir}/*.jsonl')) == 0:
    os.makedirs(concept_net_processed_file_dir, exist_ok=True)
    dataset = load_dataset("conceptnet5", "conceptnet5", split="train")
    dataset = dataset.filter(lambda example: example['lang'] == 'en')
    dataset = dataset.sort('rel')

    cur_relation_type = None
    f = None
    for i in tqdm(dataset):
        if cur_relation_type is None or cur_relation_type != i['rel']:
            cur_relation_type = i['rel']
            if f is not None:
                f.close()
            _file = f'{concept_net_processed_file_dir}/cache_{os.path.basename(cur_relation_type)}.jsonl'
            f = open(_file, 'w')
        f.write(json.dumps({
            'rel': i['rel'],
            'arg1': i['arg1'],
            'arg2': i['arg2'],
            'sentence': i['sentence']
        }) + '\n')
    f.close()
    # get statistics
    table = {}
    for i in glob(f'{concept_net_processed_file_dir}/*.jsonl'):
        r_type = os.path.basename(i).replace('.jsonl', '')
        with open(i) as f:
            data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
        table[r_type] = len(data)
    print(json.dumps(table, indent=4))
    with open(f'data/conceptnet_stats.csv', 'w') as f:
        json.dump(table, f)


#################
# GET EMBEDDING #
#################
if not os.path.exists(f'{gensim_file}.bin'):
    BATCH = int(os.getenv("BATCH", "1024"))
    CHUNK = int(os.getenv("CHUNK", "10240"))
    model = RelBERT(MODEL_ALIAS, max_length=128)
    word_pairs = []
    for i in glob(f'{concept_net_processed_file_dir}/*.jsonl'):
        with open(i) as f:
            tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]
        word_pairs += [(get_term(t['arg1']), get_term(t['arg2'])) for t in tmp]
    # remove duplicate
    word_pairs = [t for t in (set(tuple(i) for i in word_pairs))]

    print(f'found {len(word_pairs)} word pairs')

    # cache embeddings
    chunk_start = 0
    chunk_end = CHUNK
    pbar = tqdm(total=len(word_pairs))
    print(f'generate gensim file `{gensim_file}.txt`')
    with open(f'{gensim_file}.txt', 'w', encoding='utf-8') as f:
        f.write(str(len(word_pairs)) + " " + str(model.model.config.hidden_size) + "\n")
        while True:
            if chunk_start == chunk_end:
                break
            word_pairs_chunk = word_pairs[chunk_start:chunk_end]
            vector = model.get_embedding(word_pairs_chunk, batch_size=BATCH)
            for n, (token_i, token_j) in enumerate(word_pairs_chunk):
                token_i, token_j = token_i.replace(' ', '_'), token_j.replace(' ', '_')
                f.write('__'.join([token_i, token_j]))
                for y in vector[n]:
                    f.write(' ' + str(y))
                f.write("\n")

            chunk_start = chunk_end
            chunk_end = min(chunk_end + CHUNK, len(word_pairs))
            pbar.update(chunk_end - chunk_start)

    print('Convert to binary file')
    model = KeyedVectors.load_word2vec_format(f'{gensim_file}.txt')
    model.wv.save_word2vec_format(f'{gensim_file}.bin', binary=True)
    os.remove(f'{gensim_file}.txt')

###############
# GET CLUSTER #
###############
relations_to_exclude = ["EtymologicallyRelatedTo", "DerivedFrom", "RelatedTo"]


def load_embedding(top_n: int = 20, max_sample_size: int = 1000):
    _data = {}
    for _i in glob(f'{concept_net_processed_file_dir}/*.jsonl'):
        _relation_type = os.path.basename(_i).replace('.jsonl', '').replace('cache_', '')
        if _relation_type == 'None':
            continue
        with open(_i) as f_reader:
            _tmp = [json.loads(t) for t in f_reader.read().split('\n') if len(t) > 0]
            _tmp = [(get_term(__i['arg1']), get_term(__i['arg2'])) for __i in _tmp]
            _tmp = [__i for __i in _tmp if '_' not in __i[0] and '_' not in __i[1] and __i[0] != __i[1]]
        _data[_relation_type] = _tmp
    _top_types = [_a for _a, _b in sorted(_data.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]]
    _size = {k: min(max_sample_size, len(__v)) for k, __v in _data.items()}
    return _top_types, _size, _data


if not os.path.exists(f'{cluster_file}.json'):

    top_types, size, data = load_embedding(10, 1000)
    # load gensim model
    print('Collect embeddings')
    model = KeyedVectors.load_word2vec_format(f'{gensim_file}.bin', binary=True)

    cluster_info = {}
    for relation_type, v in data.items():
        if relation_type not in top_types:
            continue
        if relation_type in relations_to_exclude:
            continue
        # down sample
        seed(0)
        shuffle(v)
        v = v[:size[relation_type]]

        # get embedding
        embeddings = []
        keys = []
        for a, b in v:
            key = f'{a}__{b}'
            keys.append(key)
            embeddings.append(model[key])
        data = np.stack(embeddings)  # data x dimension

        # clustering
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(data)
        if clusterer.labels_.max() == -1:
            continue

        print(f'{clusterer.labels_.max()} clusters')
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

    top_types, size, data = load_embedding(10, 1000)

    # load gensim model
    print('Collect embeddings')
    model = KeyedVectors.load_word2vec_format(f'{gensim_file}.bin', binary=True)

    embeddings = []
    relation_types = []
    for relation_type, v in data.items():
        if relation_type not in top_types:
            continue
        if relation_type in relations_to_exclude:
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
