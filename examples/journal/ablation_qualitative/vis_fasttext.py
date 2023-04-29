import zipfile
import requests
import os
from random import shuffle, seed

import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from datasets import load_dataset
from gensim.models import fasttext


pretty_name = {
    'concept:automobilemakerdealersincity': "automobilemakerDealersInCity",
    'concept:automobilemakerdealersincountry': 'automobilemakerDealersInCountry',
    'concept:geopoliticallocationresidenceofpersion': "geopoliticalLocationOfPerson",
    'concept:politicianusendorsespoliticianus': 'politicianEndorsesPolitician',
    'concept:producedby': 'producedBy',
    'concept:teamcoach': 'teamCoach',
}

# load fasttext
def load_model():
    os.makedirs('./cache', exist_ok=True)
    path = './cache/crawl-300d-2M-subword.bin'
    if not os.path.exists(path):
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
        filename = os.path.basename(url)
        _path = f"./cache/{filename}"
        with open(_path, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        with zipfile.ZipFile(_path, 'r') as zip_ref:
            zip_ref.extractall("./cache")
        os.remove(_path)
    return fasttext.load_facebook_model(path)

# load dataset
model = None
os.makedirs("fasttext_vis", exist_ok=True)
for t in ["conceptnet_relational_similarity", "t_rex_relational_similarity", "nell_relational_similarity"]:
    if t == "t_rex_relational_similarity":
        data = load_dataset(f"relbert/{t}", "filter_unified.min_entity_4_max_predicate_10", split="test")
    else:
        data = load_dataset(f"relbert/{t}", split="test")

    if not os.path.exists(f"./fasttext_vis/{t}.cluster.csv"):
        ###############
        # GET CLUSTER #
        ###############
        print('# GET CLUSTER WITHIN SAME RELATION TO EXPLORE FINE-GRAINED RELATION CLASSES #')
        cluster_info = {}
        for i in data:  # loop over all relations

            # get embedding
            if model is None:
                model = load_model()
            embeddings = [model[a] - model[b] for a, b in i['positives']]
            keys = [f"{a}__{b}" for a, b in i['positives']]

            # clustering
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(np.stack(embeddings))  # data x dimension
            if clusterer.labels_.max() == -1:
                continue
            print(f'relation {i["relation_type"]}: {clusterer.labels_.max()} clusters')
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
        pd.DataFrame(cluster).sort_index().to_csv(f"./fasttext_vis/{t}.cluster.csv")

        # print result
        for n, (k, v) in enumerate(cluster.items()):
            print(f'RELATION [{n+1}/{len(cluster)}] : {k}')
            for _k, _v in v.items():
                seed(0)
                shuffle(_v)
                print(f'* cluster {_k}')
                for pair in _v[:min(10, len(_v))]:
                    print(f"\t - {pair.split('__')}")

    relation_types = []
    for i in data:  # loop over all relations
        if t == "nell_relational_similarity":
            relation_types += [pretty_name[i['relation_type']]] * len(i['positives'])
        else:
            relation_types += [i['relation_type']] * len(i['positives'])
    if not os.path.exists(f"./fasttext_vis/{t}.npy"):
        #######################
        # DIMENSION REDUCTION #
        #######################
        print('# DIMENSION REDUCTION #')

        # load gensim model
        print('Collect embeddings')
        embeddings = []
        for i in data:  # loop over all relations
            if model is None:
                model = load_model()

            embeddings += [model[a] - model[b] for a, b in i['positives']]
        data = np.stack(embeddings)  # data x dimension

        # dimension reduction
        print(f'Dimension reduction: {data.shape}')
        embedding_2d = TSNE(n_components=2, random_state=0).fit_transform(data)
        np.save(f"./fasttext_vis/{t}.npy", embedding_2d)
        with open(f"./fasttext_vis/{t}.txt", 'w') as f:
            f.write('\n'.join(relation_types))

    embedding_2d = np.load(f"./fasttext_vis/{t}.npy")
    assert len(embedding_2d) == len(relation_types)
    print(f'Embedding shape: {embedding_2d.shape}')

    print('# PLOT #')
    unique_relation_types = sorted(list(set(relation_types)))
    relation_type_dict = {v: n for n, v in enumerate(unique_relation_types)}

    plt.figure()
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=12 if t == "t_rex_relational_similarity" else 8,
        c=[relation_type_dict[i] for i in relation_types],
        cmap=sns.color_palette('Spectral', len(relation_type_dict), as_cmap=True)
    )
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('2-d Embedding Space', fontsize=12)
    plt.legend(handles=scatter.legend_elements(num=len(relation_type_dict))[0],
               labels=unique_relation_types,
               bbox_to_anchor=(1.04, 1),
               borderaxespad=0)
    plt.savefig(f"./fasttext_vis/{t}.png", bbox_inches='tight', dpi=600)

