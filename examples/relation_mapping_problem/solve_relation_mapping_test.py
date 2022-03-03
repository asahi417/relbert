"""
pip install munkres
"""
import os
import json
from itertools import product

import numpy as np
import pandas as pd
from munkres import Munkres
from relbert import AnalogyScore, RelBERT, cosine_similarity, euclidean_distance

os.makedirs('cache/test', exist_ok=True)


def compute_score(source_list, target_list, cache_file='tmp.json'):
    # compute each assignment score
    size = len(source_list)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            embedding_dict = json.load(f)
    else:
        embedding_dict = {}

    relbert_model = RelBERT('asahi417/relbert-roberta-large')

    def get_score(_query, _options, __id):
        if __id in embedding_dict:
            vector = embedding_dict[__id]
        else:
            vector = relbert_model.get_embedding([_query] + _options, batch_size=1024)
            embedding_dict[__id] = vector
        return [euclidean_distance(vector[0], _v) for _v in vector[1:]]

    model_input = {}
    for n, (source_n, target_n) in enumerate(product(range(size), range(size))):
        print('\t compute score: {}/{}'.format(n + 1, size*size))
        query = [source_list[source_n], target_list[target_n]]
        options = []
        for source_pair_n in range(size):
            if source_n == source_pair_n:
                continue
            for target_pair_n in range(size):
                if target_n == target_pair_n:
                    continue
                options.append([source_list[source_pair_n], target_list[target_pair_n]])
        model_input['{}-{}'.format(source_n, target_n)] = get_score(query, options, n)
    with open(cache_file, 'w') as f:
        json.dump(embedding_dict, f)

    # get overall score for each assignment pattern
    scores = {}
    assignments = {}
    m = Munkres()
    size = size - 1
    for n, (source_n, target_n) in enumerate(product(range(size), range(size))):
        # scores for all possible pairs apart from the reference
        score = model_input['{}-{}'.format(source_n, target_n)]
        matrix = 1 - np.array([score[size * i:size * (1 + i)] for i in range(size)])

        # compute the cheapest assignments and get the overall cost by summing up each cost
        best_assignment = m.compute(matrix.copy())
        scores['{}-{}'.format(source_n, target_n)] = np.sum([matrix[a][b] for a, b in best_assignment])
        # get the best assignment's pairs
        best_assignment = [[a if a < source_n else a + 1, b if b < target_n else b + 1] for a, b in best_assignment]
        best_assignment = best_assignment + [[source_n, target_n]]
        assignments['{}-{}'.format(source_n, target_n)] = sorted(best_assignment, key=lambda st: st[0])
    # find the reference with the cheapest assignment
    best_assignment_key = sorted(scores.items(), key=lambda kv: kv[1])[0][0]
    best_assignment = assignments[best_assignment_key]
    # get the final order of the target term
    target_order = [i[1] for i in best_assignment]
    target_list_fixed = [target_list[i] for i in target_order]
    return target_list_fixed


if __name__ == '__main__':
    logger = open('./report.txt', 'w')
    # get data
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]

    accuracy = {'relbert': []}
    for data_id, i in enumerate(data):
        tmp_result = [i['source'], i['target']]
        print('Processing [RelBERT] {}/{}'.format(data_id + 1, len(data)))
        pred = compute_score(
            i['source'], i['target_random'], cache_file='cache/test/relbert.roberta_large.{}.json'.format(data_id))
        tmp_result.append(pred)
        accuracy['relbert'] += [int(a == b) for a, b in zip(pred, i['target'])]

    print('\nAccuracy')
    logger.write('ACCURACY\n')
    for k, v in accuracy.items():
        if len(v) > 0:
            print('\t{}: {}'.format(k, sum(v)/len(v) * 100))
            logger.write('\t{}: {}\n'.format(k, sum(v)/len(v) * 100))

