"""
pip install munkres
"""
import os
import json
from itertools import product

import numpy as np
import pandas as pd
from munkres import Munkres
from relbert import AnalogyScore, RelBERT, cosine_similarity

os.makedirs('cache', exist_ok=True)


def compute_score(source_list,
                  target_list,
                  cache_file='tmp.json',
                  model='bert-large-cased',
                  model_type: str = 'analogy_score'):
    # analogy score configuration
    configs = {
        "roberta-large": {"weight_head": 0.2, "weight_tail": 0.2, "template": 'as-what-same',
                          "positive_permutation": 4, "negative_permutation": 10, "weight_negative": 0.2},
        "gpt2-xl": {"weight_head": -0.4, "weight_tail": 0.2, "template": 'rel-same',
                    "positive_permutation": 2, "negative_permutation": 0, "weight_negative": 0.8},
        "bert-large-cased": {"weight_head": -0.2, "weight_tail": -0.4, "template": 'what-is-to',
                             "positive_permutation": 4, "negative_permutation": 4, "weight_negative": 0.2}
    }

    # compute each assignment score
    size = len(source_list)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            model_input = json.load(f)
    else:
        if model_type == 'analogy_score':
            config = configs[model]
            scorer = AnalogyScore(model=model)

            def get_score(_query, _options):
                return scorer.analogy_score(query_word_pair=query, option_word_pairs=options, batch_size=1024, **config)

        elif model_type == 'relbert':
            relbert_model = RelBERT('asahi417/relbert-roberta-large')

            def get_score(_query, _options):
                vector = relbert_model.get_embedding([_query] + _options, batch_size=1024)
                return [cosine_similarity(vector[0], _v) for _v in vector[1:]]
        else:
            raise ValueError('unknown type: {}'.format(model_type))
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
            score = get_score(query, options)
            model_input['{}-{}'.format(source_n, target_n)] = score
        with open(cache_file, 'w') as f:
            json.dump(model_input, f)

    # get overall score for each assignment pattern
    scores = {}
    assignments = {}
    m = Munkres()
    size = size - 1
    for n, (source_n, target_n) in enumerate(product(range(size), range(size))):
        # scores for all possible pairs apart from the reference
        score = model_input['{}-{}'.format(source_n, target_n)]
        # matrix of assignment cost: source x target
        matrix = -1 * np.array([score[size * i:size * (1 + i)] for i in range(size)])
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

    accuracy = {'analogy_score (roberta)': [], 'relbert': []}
    dfs = []
    for data_id, i in enumerate(data):

        tmp_result = [i['source'], i['target']]
        print('Processing [Analogy Score] {}/{}'.format(data_id + 1, len(data)))
        pred = compute_score(
            i['source'], i['target_random'], model_type='analogy_score', model='roberta-large',
            cache_file='cache/analogy_score.roberta_large.{}.json'.format(data_id))
        accuracy['analogy_score (roberta)'] += [int(a == b) for a, b in zip(pred, i['target'])]
        tmp_result.append(pred)
        print('Processing [RelBERT] {}/{}'.format(data_id + 1, len(data)))
        pred = compute_score(
            i['source'], i['target_random'], model_type='relbert',
            cache_file='cache/relbert.roberta_large.{}.json'.format(data_id))
        tmp_result.append(pred)
        df = pd.DataFrame(tmp_result, index=[
            'source/{}'.format(data_id),
            'target/{}'.format(data_id),
            'pred_analogy_score/{}'.format(data_id), 'pred_relbert/{}'.format(data_id)])
        logger.write(' * data: {}\n{} \n\n'.format(data_id + 1, df.to_string(header=False)))
        accuracy['relbert'] += [int(a == b) for a, b in zip(pred, i['target'])]
        dfs.append(df)

    print('\nAccuracy')
    logger.write('ACCURACY\n')
    for k, v in accuracy.items():
        if len(v) > 0:
            print('\t{}: {}'.format(k, sum(v)/len(v) * 100))
            logger.write('\t{}: {}\n'.format(k, sum(v)/len(v) * 100))

    logger.close()
    dfs = pd.concat(dfs)
    dfs.to_csv('result.csv')
