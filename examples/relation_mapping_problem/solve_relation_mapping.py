import os
import json
from itertools import permutations
from relbert import AnalogyScore, RelBERT

os.makedirs('cache', exist_ok=True)


def analogy_score(source_list, target_list, cache_file='tmp.json', model='bert-large-cased', aggregation=min):
    configs = {
        "roberta-large": {"weight_head": 0.2, "weight_tail": 0.2, "template": 'as-what-same',
                          "positive_permutation": 4, "negative_permutation": 10, "weight_negative": 0.2},
        "gpt2-xl": {"weight_head": -0.4, "weight_tail": 0.2, "template": 'rel-same',
                    "positive_permutation": 2, "negative_permutation": 0, "weight_negative": 0.8},
        "bert-large-cased": {"weight_head": -0.2, "weight_tail": -0.4, "template": 'what-is-to',
                             "positive_permutation": 4, "negative_permutation": 4, "weight_negative": 0.2}
    }
    config = configs[model]
    scorer = AnalogyScore(model=model)
    perms = list(permutations(target_list))
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            scores = json.load(f)
    else:
        scores = {}
        for n, candidate in enumerate(perms):
            print('\t permutation: {}/{}'.format(n + 1, len(perms)))
            score = scorer.analogy_score(query_word_pair=[source_list[0], candidate[0]],
                                         option_word_pairs=[[source_list[i+1], candidate[i+1]] for i in range(len(source_list) - 1)],
                                         **config)
            scores[str(n)] = score
        with open(cache_file, 'w') as f:
            json.dump(scores, f)
    score = [(k, aggregation(v)) for k, v in scores.items()]
    pred = int(sorted(score, key=lambda x: x[1], reverse=True)[0][0])
    pred_permutation = perms[pred]
    return pred_permutation


if __name__ == '__main__':

    # get data
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]

    for data_id, i in enumerate(data):
        print('Processing {}/{}'.format(data_id + 1, len(data)))
        pred_as_ro = analogy_score(i['source'], i['target_random'],
                                   model='roberta-large',
                                   cache_file='cache/analogy_score.roberta_large.{}.json'.format(data_id))
        # print(pred_as_ro)
        # prediction = analogy_score(i['source'], i['target_random'],
        #                            model='gpt2-xl',
        #                            cache_file='cache/analogy_score.gpt2_Xl.{}.json'.format(data_id))
        # prediction = analogy_score(i['source'], i['target_random'],
        #                            model='bert-large-cased',
        #                            cache_file='cache/analogy_score.bert_large_cased.{}.json'.format(data_id))
        print(pred_as_ro)
        print(i['target'])
