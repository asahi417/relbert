"""
# Relation Mapping Problem
Relation mapping `M` is the set of bijective map in between two sets of terms (`A` and `B`):
```
[set `A`]: ("solar system", "sun", "planet", "mass", "attracts", "revolves", "gravity")
[set `B`]: ("atom", "nucleus", "electron", "charge", "attracts", "revolves", "electromagnetism")

[Relation Mapping `M`]
* "solar system"   -> "atom"
* "sun"            -> "nucleus"
* "planet"         -> "electron"
* "mass"           -> "charge"
* "attracts"       -> "attracts"
* "revolves"       -> "revolves"
* "gravity"        -> "electromagnetism"
```

***[Relation Mapping Problem](https://www.jair.org/index.php/jair/article/view/10583)*** is the task to identify the mapping `M` given the sets of terms `A` and `B`.

## Dataset
The [dataset file](./data.jsonl) is a jsonline where each line is a json data containing following data.

- `source`: A list of terms, which is the source of the relation mapping from.
- `target_random`: A list of terms, where we want to find a mapping from `source` to.
- `target`: A correctly ordered `target_random` that aligns with the `source`.

Given `source` and `target_random`, the task is to predict the correct order of `target_random` so that it matches `target`.
In average 7 terms are in the set, so the total number of possible order is 5040.


## Approach
As an approach to solve the relation mapping with RelBERT (or relation embedding model in general), we can follow something like this.
In a permutation `P:=[(a_1, b_1), ..., (a_7, b_7)]`, we compute a relation embedding of each word pair `(a_i, b_i)`
and intuitively the permutation is valid if all the word pairs hold same relation, meaning their relation embeddings are
close each other. So we can somehow calculate coherence of the relation embeddings and choose the most coherent permutation.
"""

import os
import json
import logging
from statistics import mean
from itertools import permutations
from os.path import join as pj
from tqdm import tqdm
import gc

from numpy import dot
from numpy.linalg import norm
from datasets import load_dataset

from ..lm import RelBERT


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-4)


def evaluate_relation_mapping(relbert_ckpt: str, batch_size: int = 512, cache_embedding_dir: str = 'embeddings',
                              dataset: str = "relbert/scientific_and_creative_analogy"):
    # data
    data = [i for i in load_dataset(dataset)["test"]]
    # compute embedding
    model = None
    os.makedirs(pj(cache_embedding_dir, relbert_ckpt.replace("/", "_")), exist_ok=True)
    logging.info('COMPUTE EMBEDDING')
    for data_id, _data in enumerate(data):
        logging.info(f'[{relbert_ckpt}]: {data_id}/{len(data)}')
        cache_file = pj(cache_embedding_dir, relbert_ckpt.replace("/", "_"), f'vector.{data_id}.json')
        embedding_dict = {}
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                embedding_dict = json.load(f)
        inputs = []
        inputs_id = []
        for _type in ['source', 'target']:
            for x, y in permutations(_data[_type], 2):
                _id = f'{x}__{y}'
                if _id not in embedding_dict:
                    inputs_id.append(_id)
                    inputs.append([x, y])
        if len(inputs) == 0:
            continue
        if model is None:
            model = RelBERT(relbert_ckpt)
        vector = model.get_embedding(inputs, batch_size=batch_size)
        embedding_dict.update({i: v for i, v in zip(inputs_id, vector)})
        logging.info(f'update cache file {cache_file}')
        with open(cache_file, 'w') as f_writer:
            json.dump(embedding_dict, f_writer)

    # solve relation mapping
    accuracy = []
    sims_full = []
    perms_full = []

    logging.info('SOLVING RELATION MAPPING')
    for data_id, _data in enumerate(data):
        logging.info(f'[{relbert_ckpt}]: {data_id}/{len(data)}')
        cache_embedding = pj(cache_embedding_dir, relbert_ckpt.replace("/", "_"), f'vector.{data_id}.json')

        logging.info(f"loading {cache_embedding}")
        with open(cache_embedding) as f:
            embedding_dict = json.load(f)
        # similarity
        sim = {}
        cache_sim = pj(cache_embedding_dir, relbert_ckpt.replace("/", "_"), f'sim.{data_id}.json')
        if os.path.exists(cache_sim):
            with open(cache_sim) as f:
                sim = json.load(f)

        source = _data['source']
        target = _data['target']
        logging.info(f"[number] source: {len(source)}, target: {len(target)}")
        perms = []

        for n, tmp_target in tqdm(list(enumerate(permutations(target, len(target))))):
            list_sim = []
            for id_x in range(len(target)):
                _list_sim = []
                for id_y in range(len(target)):
                    if id_x == id_y:
                        continue
                    _id = f'{source[id_x]}__{source[id_y]} || {tmp_target[id_x]}__{tmp_target[id_y]}'
                    if _id not in sim:
                        sim[_id] = cosine_similarity(
                            embedding_dict[f'{source[id_x]}__{source[id_y]}'],
                            embedding_dict[f'{tmp_target[id_x]}__{tmp_target[id_y]}']
                        )
                        with open(cache_sim, 'w') as f_writer:
                            json.dump(sim, f_writer)
                    _list_sim.append(sim[_id])
                list_sim.append(max(_list_sim))
            perms.append({'target': tmp_target, 'similarity_mean': mean(list_sim)})
        sims_full.extend([{'pair': k, 'sim': v, 'data_id': data_id} for k, v in sim.items()])
        pred = sorted(perms, key=lambda _x: _x['similarity_mean'], reverse=True)
        accuracy.append(mean([int(t == p) for t, p in zip(target, pred[0]['target'])]))
        tmp = [i for i in perms if list(i['target']) == target]
        assert len(tmp) == 1, perms
        perms_full.append({
            'source': source,
            'true': target,
            'pred': pred[0]['target'],
            'alignment_match': list(pred[0]['target']) == target,
            'accuracy': mean([int(t == p) for t, p in zip(target, pred[0]['target'])]),
            'similarity': pred[0]['similarity_mean'],
            'similarity_true': tmp[0]['similarity_mean']
        })
        del embedding_dict
        # del
        gc.collect()
    mean_accuracy = mean(accuracy)
    logging.info(f'Accuracy: {mean_accuracy}')
    del model
    return mean_accuracy, sims_full, perms_full

