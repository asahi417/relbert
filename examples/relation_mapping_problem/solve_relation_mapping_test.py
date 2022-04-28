"""
pip install munkres
"""
import os
import json
from itertools import product

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

from munkres import Munkres
from relbert import RelBERT

from word_embedding import get_word_embedding_model

os.makedirs('output', exist_ok=True)

# get data
with open('data.jsonl') as f_reader:
    data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]


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
    os.makedirs('embeddings', exist_ok=True)
    accuracy_all = []
    accuracy_breakdown = []
    intermediate_output = []

    # for model_type in ['relbert']:
    for model_type in ['fasttext_cc', 'relbert']:
        accuracy = []
        accuracy_anchor = []
        for data_id, i in enumerate(data):
            source2target = {a: b for a, b in zip(i['source'], i['target'])}
            compute_score(i, cache_file=f'embeddings/{model_type}.vector.{data_id}.json', model_name=model_type)
            p, (a_s, a_t), inter = compute_score(i, cache_file=f'embeddings/{model_type}.vector.{data_id}.json', model_name=model_type)
            inter['data'] = data_id
            inter['model'] = model_type
            intermediate_output.append(inter)
            accuracy += [int(a == b) for a, b in zip(p, i['target'])]
            accuracy_anchor.append(int(source2target[a_s] == a_t))
            accuracy_breakdown.append({'model_type': model_type, 'data_id': data_id,
                                       'accuracy': mean([int(a == b) for a, b in zip(p, i['target'])])})
        accuracy_all.append(
            {'model_type': model_type, 'accuracy': mean(accuracy) * 100, 'accuracy_anchor': mean(accuracy_anchor) * 100})
    pd.concat(intermediate_output).to_csv('output/intermediate_output.csv')
    pd.DataFrame(accuracy_all).to_csv('output/accuracy.csv')
    pd.DataFrame(accuracy_breakdown).to_csv('output/accuracy.breakdown.csv')
    print('\nAccuracy')
    for v in accuracy_all:
        print('\t Model: {}'.format(v['model_type']))
        print('\t\t * accuracy: {}'.format(v['accuracy']))
        print('\t\t * accuracy: {} (anchor)'.format(v['accuracy_anchor']))


