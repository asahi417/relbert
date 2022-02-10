import os
import json
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from gensim.models import KeyedVectors

# load gensim model
model = KeyedVectors.load_word2vec_format("data/relbert_embedding.bin", binary=True)
concept_net_processed_file_dir = 'data/conceptnet'

umap_training_n = 100000

def get_term(arg):
    return arg.split('/en/')[-1].split('/')[0]


for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
    relation_type = os.path.basename(i).replace('.jsonl', '')
    print('relation type: {}'.format(relation_type))
    with open(i) as f:
        tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]

    embeddings = []
    keys = []
    for _tmp in tmp:
        key = '{}__{}'.format(get_term(_tmp['arg1']), get_term(_tmp['arg2']))
        keys.append(key)
        embeddings.append(model[key])
    # data x dimension
    data = np.stack(embeddings)
    embedding_2d = UMAP().fit_transform(data)
    print(embedding_2d.shape)

    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Penguin dataset', fontsize=24)