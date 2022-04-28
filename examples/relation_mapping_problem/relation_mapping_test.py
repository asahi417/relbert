"""
pip install munkres
"""
import os
import json
from itertools import permutations

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

from munkres import Munkres
from relbert import RelBERT


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-4)


def mean(_list):
    return sum(_list)/len(_list)


def compute_score(_data, cache_file, model_name, anchor_fixed=False):
    # compute each assignment score
    model = None
    get_embedding = None
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            embedding_dict = json.load(f)
    else:
        embedding_dict = {}
        if model_name == 'relbert':
            model = RelBERT('asahi417/relbert-roberta-large')
            def get_embedding(a, b): return model.get_embedding(a, b)
        elif model_name in ['fasttext', 'fasttext_cc']:
            model = get_word_embedding_model(model_name)
            def get_embedding(a, b): return (model[a] - model[b]).tolist()
        else:
            raise ValueError(f'unknown model {model_name}')

    def embedding(word_pairs):
        _id = '__'.join(word_pairs)
        if _id in embedding_dict:
            return embedding_dict[_id]
        else:
            assert model is not None
            vector = get_embedding(*word_pairs)
            embedding_dict[_id] = vector
            with open(cache_file, 'w') as f_writer:
                json.dump(embedding_dict, f_writer)
            return vector

    def get_score(_query, _option):
        return 1 - cosine_similarity(embedding(_query), embedding(_option))

    true_pairs = {a: b for a, b in zip(_data['source'], _data['target'])}
    assignments = {}
    scores = {}
    anchor = {}
    # source, target = _data['source'][0], true_pairs[_data['source'][0]]
    iters = zip(_data['source'][0:1], _data['target'][0:1]) if anchor_fixed else product(_data['source'], _data['target_random'])
    for source, target in iters:
    # for source, target in product(_data['source'], _data['target_random']):
        candidate_s = [i for i in _data['source'] if i != source]
        candidate_t = [i for i in _data['target_random'] if i != target]
        matrix = np.array([[get_score([source, target], [_s, _t]) for _t in candidate_t] for _s in candidate_s])
        assignment = Munkres().compute(matrix.copy())
        assignments[f'{source}__{target}'] = {candidate_s[a]: candidate_t[b] for a, b in assignment}
        assignments[f'{source}__{target}'].update({source: target})
        anchor[f'{source}__{target}'] = [source, target]
        scores[f'{source}__{target}'] = float(np.sum([matrix[a][b] for a, b in assignment]))
    key = sorted(scores.items(), key=lambda x: x[1])[0][0]
    prediction = [assignments[key][i] for i in _data['source']]
    intermediate = []
    for pair, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        s, t = anchor[pair]
        intermediate.append([
            f'{s}__{true_pairs[s]}' == f'{s}__{t}', f'{s}__{t}', score, f'{s}__{true_pairs[s]}', scores[f'{s}__{true_pairs[s]}']
        ])
    intermediate = pd.DataFrame(intermediate, columns=['Correct', 'pair', 'score', 'pair (true target)', 'score (true target)'])
    return prediction, anchor[key], intermediate


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
    accuracy_full = {}
    for m in ['relbert', 'fasttext_cc']:
        accuracy = []
        for data_id, _data in enumerate(data):
            print(f'[{m}]: {data_id}/{len(data)}')
            with open(f'embeddings/{m}.vector.{data_id}.json') as f:
                embedding_dict = json.load(f)
            source = _data['source']
            target = _data['target']
            perms = []
            for n, tmp_target in enumerate(permutations(target, len(target))):
                sim = {}
                for id_x, id_y in permutations(range(len(target)), 2):
                    _sim = cosine_similarity(
                        embedding_dict[f'{source[id_x]}__{source[id_y]}'],
                        embedding_dict[f'{tmp_target[id_x]}__{tmp_target[id_y]}']
                    )
                    sim[f'{source[id_x]}__{source[id_y]} || {tmp_target[id_x]}__{tmp_target[id_y]}'] = _sim
                perms.append({'target': tmp_target, 'similarity_mean': mean(sim.values()), 'similarity_all': sim})
            pred = sorted(perms, key=lambda _x: _x['similarity_mean'], reverse=True)
            accuracy.append(target == pred[0]['target'])
        accuracy_full[m] = mean(accuracy)
    print(json.dumps(accuracy_full, indent=4))
