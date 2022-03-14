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


def compute_score(source_list, target_list, cache_file='tmp.json', distance_function='cos',
                  transformation: str = None):
    # compute each assignment score
    size = len(source_list)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            embedding_dict = json.load(f)
        relbert_model = None
    else:
        embedding_dict = {}
        relbert_model = RelBERT('asahi417/relbert-roberta-large')

    def get_score(_query, _options, __id):
        __id = str(__id)
        if __id in embedding_dict:
            vector = embedding_dict[__id]
        else:
            assert relbert_model is not None, '{} not in {}'.format(__id, embedding_dict.keys())
            vector = relbert_model.get_embedding([_query] + _options, batch_size=1024)
            embedding_dict[__id] = vector
        if distance_function == 'l2':
            return [euclidean_distance(vector[0], _v) for _v in vector[1:]]
        elif distance_function == 'cos':
            return [1 - cosine_similarity(vector[0], _v) for _v in vector[1:]]
        else:
            raise ValueError('unknown ditance {}'.format(distance_function))

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
    if relbert_model is not None:
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
        matrix = np.array([score[size * i:size * (1 + i)] for i in range(size)])

        if transformation == 'log':
            matrix = np.log(matrix)
        elif transformation == 'rank':
            matrix = matrix.argsort().argsort()
        elif transformation is None:
            pass
        else:
            raise ValueError('unknown transformation: {}'.format(transformation))

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
    anchor = [int(i) for i in best_assignment_key.split('-')]
    # get the final order of the target term
    target_order = [i[1] for i in best_assignment]
    target_list_fixed = [target_list[i] for i in target_order]
    return target_list_fixed, anchor


if __name__ == '__main__':
    logger = open('./report_test.txt', 'w')
    os.makedirs('embeddings', exist_ok=True)

    # get data
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]

    accuracy_all = {}
    for d in ['cos', 'l2']:
        for t in [None, 'rank', 'log']:
            accuracy = []
            accuracy_anchor = []
            for data_id, i in enumerate(data):
                tmp_result = [i['source'], i['target']]

                pred, (anchor_a, anchor_b) = compute_score(
                    i['source'], i['target_random'], cache_file='embeddings/relbert.roberta_large.{}.json'.format(data_id),
                    distance_function=d, transformation=t
                )
                tmp_result.append(pred)
                accuracy += [int(a == b) for a, b in zip(pred, i['target'])]
                accuracy_anchor.append(int(i['target_random'][anchor_a] == i['target'][anchor_b]))
            accuracy_all['relbert/{}/{}'.format(d, t)] = {
                'accuracy': sum(accuracy)/len(accuracy) * 100,
                'accuracy_anchor': sum(accuracy_anchor) / len(accuracy_anchor) * 100
            }

    print('\nAccuracy')
    logger.write('ACCURACY\n')
    logger.write(json.dumps(accuracy_all))
    for k, v in accuracy_all.items():
        print('\t Model: {}'.format(k))
        print('\t\t * accuracy: {}'.format(v['accuracy']))
        print('\t\t * accuracy: {} (anchor)'.format(v['accuracy_anchor']))

    logger.close()
