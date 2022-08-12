import os
import json
import argparse
import logging
from statistics import mean
from itertools import permutations
from os.path import join as pj
from tqdm import tqdm

import pandas as pd
from numpy import dot
from numpy.linalg import norm
from relbert import RelBERT
from datasets import load_dataset


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-4)


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy/relation classification')
parser.add_argument('-m', '--model', help='model name', default="relbert/relbert-roberta-large", type=str)
parser.add_argument('-b', '--batch-size', help='batch', default=512, type=str)
parser.add_argument('-a', '--aggregation', help='mean or max', default='max', type=str)
opt = parser.parse_args()

# data
data = [i for i in load_dataset("relbert/relation_mapping")["test"]]

# compute embedding
model = None
model_alias = os.path.basename(opt.model)
os.makedirs(pj('embeddings', model_alias), exist_ok=True)
logging.info('COMPUTE EMBEDDING')
for data_id, _data in enumerate(data):
    logging.info(f'[{model_alias}]: {data_id}/{len(data)}')
    cache_file = pj('embeddings', model_alias, f'vector.{data_id}.json')
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
        model = RelBERT(opt.model)
    vector = model.get_embedding(inputs, batch_size=opt.batch_size)
    embedding_dict.update({i: v for i, v in zip(inputs_id, vector)})
    logging.info(f'update cache file {cache_file}')
    with open(cache_file, 'w') as f_writer:
        json.dump(embedding_dict, f_writer)

# solve relation mapping
os.makedirs('output', exist_ok=True)
accuracy = []
sims_full = []
perms_full = []

logging.info('SOLVING RELATION MAPPING')
for data_id, _data in enumerate(data):
    logging.info(f'[{model_alias}]: {data_id}/{len(data)}')
    cache_embedding = pj('embeddings', model_alias, f'vector.{data_id}.json')
    with open(cache_embedding) as f:
        embedding_dict = json.load(f)
    # similarity
    sim = {}
    cache_sim = pj('embeddings', model_alias, f'sim.{data_id}.json')
    if os.path.exists(cache_sim):
        with open(cache_sim) as f:
            sim = json.load(f)

    source = _data['source']
    target = _data['target']
    perms = []
    for n, tmp_target in tqdm(list(enumerate(permutations(target, len(target))))):
        list_sim = []
        for id_x, id_y in permutations(range(len(target)), 2):
            _id = f'{source[id_x]}__{source[id_y]} || {tmp_target[id_x]}__{tmp_target[id_y]}'
            if _id not in sim:
                sim[_id] = cosine_similarity(
                    embedding_dict[f'{source[id_x]}__{source[id_y]}'],
                    embedding_dict[f'{tmp_target[id_x]}__{tmp_target[id_y]}']
                )
                with open(cache_sim, 'w') as f_writer:
                    json.dump(sim, f_writer)
            list_sim.append(sim[_id])
        perms.append({'target': tmp_target, 'similarity_mean': mean(list_sim), 'similarity_max': max(list_sim)})
    sims_full.extend([{'pair': k, 'sim': v, 'data_id': data_id} for k, v in sim.items()])
    pred = sorted(perms, key=lambda _x: _x[f'similarity_{opt.aggregation}'], reverse=True)
    accuracy.extend([t == p for t, p in zip(target, pred[0]['target'])])
    tmp = [i for i in perms if list(i['target']) == target]
    assert len(tmp) == 1, perms
    perms_full.append({
        'source': source,
        'true': target,
        'pred': pred[0]['target'],
        'accuracy': list(pred[0]['target']) == target,
        'similarity': pred[0][f'similarity_{opt.aggregation}'],
        'similarity_true': tmp[0][f'similarity_{opt.aggregation}']
    })
logging.info(f'Accuracy: {100*mean(accuracy)}')
result_file = pj("output", "result.json")
result = {}
if os.path.exists(result_file):
    with open(result_file) as f:
        result = json.load(f)
result[f'{model_alias}/{opt.aggregation}'] = mean(accuracy)
with open(result_file, 'w') as f:
    json.dump(result, f)
pd.DataFrame(sims_full).to_csv(pj('output', f'stats.sim.{model_alias}.{opt.aggregation}.csv'))
pd.DataFrame(perms_full).to_csv(pj('output', f'stats.breakdown.{model_alias}.{opt.aggregation}.csv'))
