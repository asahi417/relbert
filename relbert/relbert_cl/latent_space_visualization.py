import os
import logging
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
from sklearn.manifold import TSNE

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Visualize RelBERT Embedding')
    parser.add_argument('-m', '--model', help='language model', default="relbert/relbert-roberta-large", type=str)
    parser.add_argument('-b', '--batch', default=256, type=int)
    parser.add_argument('-c', '--chunk', default=512, type=int)
    parser.add_argument('-d', '--data', default="relbert/conceptnet_relational_similarity", type=str)
    parser.add_argument('-n', '--name', default=None, type=str)
    parser.add_argument('-s', '--split', default="test", type=str)
    parser.add_argument('-o', '--output-dir', default="cluster_visualization_cache", type=str)
    parser.add_argument('--max-length', default=128, type=int)
    opt = parser.parse_args()

    # cache files
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gensim_file = f"{output_dir}/{os.path.basename(opt.model)}"
    cluster_file = f"{output_dir}/{os.path.basename(opt.model)}.cluster.csv"
    embedding_file = f"{output_dir}/{os.path.basename(opt.model)}.embedding"
    figure_file = f"{output_dir}/{os.path.basename(opt.model)}.figure.png"

    data = load_dataset(opt.data, opt.name, split=opt.split)
    vocab = {f'{a.replace(" ", "_")}__{b.replace(" ", "_")}': [a, b] for a, b in chain(*[i['positives'] for i in data])}
    #########################################
    # GET EMBEDDING AND SAVE AS GENSIM FILE #
    #########################################
    if not os.path.exists(f'{gensim_file}.bin'):
        logging.info('# GET EMBEDDING AND SAVE AS GENSIM FILE #')
        model = RelBERT(opt.model, max_length=opt.max_length)
        chunk_start, chunk_end = 0, opt.chunk
        pbar = tqdm(total=len(vocab))

        logging.info(f'generate gensim file `{gensim_file}.txt` for {len(vocab)} pairs with {opt.model}')
        with open(f'{gensim_file}.txt', 'w', encoding='utf-8') as f:
            f.write(str(len(vocab)) + " " + str(model.model_config.hidden_size) + "\n")
            keys = sorted(vocab.keys())
            while chunk_start != chunk_end:
                vector = model.get_embedding([vocab[k] for k in keys[chunk_start:chunk_end]], batch_size=opt.batch)
                f.write('\n'.join([f"{k} {' '.join([str(y) for y in vector[n]])}" for n, k in enumerate(keys[chunk_start:chunk_end])]) + "\n")
                chunk_start = chunk_end
                chunk_end = min(chunk_end + opt.chunk, len(vocab))
                pbar.update(chunk_end - chunk_start)
        logging.info('convert to binary file')
        model = KeyedVectors.load_word2vec_format(f'{gensim_file}.txt')
        model.wv.save_word2vec_format(f'{gensim_file}.bin', binary=True)
        os.remove(f'{gensim_file}.txt')

    ###############
    # GET CLUSTER #
    ###############
    if not os.path.exists(cluster_file):
        logging.info('# GET CLUSTER WITHIN SAME RELATION TO EXPLORE FINE-GRAINED RELATION CLASSES #')
        model = KeyedVectors.load_word2vec_format(f'{gensim_file}.bin', binary=True)
        cluster_info = {}
        for i in data:  # loop over all relations

            # get embedding
            keys = [f'{a.replace(" ", "_")}__{b.replace(" ", "_")}' for a, b in i['positives']]
            embeddings = [model[k] for k in keys]

            # clustering
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(np.stack(embeddings))  # data x dimension
            if clusterer.labels_.max() == -1:
                continue
            logging.info(f'relation {i["relation_type"]}: {clusterer.labels_.max()} clusters')
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

        # save
        pd.DataFrame(cluster).sort_index().to_csv(cluster_file)

        # print result
        for n, (k, v) in enumerate(cluster.items()):
            logging.info(f'RELATION [{n+1}/{len(cluster)}] : {k}')
            for _k, _v in v.items():
                seed(0)
                shuffle(_v)
                logging.info(f'* cluster {_k}')
                for pair in _v[:min(10, len(_v))]:
                    logging.info(f"\t - {pair.split('__')}")

    #######################
    # DIMENSION REDUCTION #
    #######################
    if not os.path.exists(f'{embedding_file}.npy') or not os.path.exists(f'{embedding_file}.txt'):
        logging.info('# DIMENSION REDUCTION #')

        # load gensim model
        logging.info('Collect embeddings')
        model = KeyedVectors.load_word2vec_format(f'{gensim_file}.bin', binary=True)
        relation_types = []
        embeddings = []
        for i in data:  # loop over all relations
            relation_types += [i['relation_type']] * len(i['positives'])
            embeddings += [model[f'{a.replace(" ", "_")}__{b.replace(" ", "_")}'] for a, b in i['positives']]
        data = np.stack(embeddings)  # data x dimension

        # dimension reduction
        logging.info(f'Dimension reduction: {data.shape}')
        embedding_2d = TSNE(n_components=2, random_state=0).fit_transform(data)
        np.save(f'{embedding_file}.npy', embedding_2d)
        with open(f'{embedding_file}.txt', 'w') as f:
            f.write('\n'.join(relation_types))
    else:
        embedding_2d = np.load(f'{embedding_file}.npy')
        with open(f'{embedding_file}.txt') as f:
            relation_types = [i for i in f.read().split('\n') if len(i) > 0]

    logging.info(f'Embedding shape: {embedding_2d.shape}')
    assert len(embedding_2d) == len(relation_types)

    logging.info('# PLOT #')
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


if __name__ == '__main__':
    main()
