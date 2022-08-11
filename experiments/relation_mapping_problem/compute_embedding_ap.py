import os
import json
from itertools import permutations
from tqdm import tqdm
from os.path import join as pj
from relbert import AnalogyScore
from datasets import load_dataset

data = [i for i in load_dataset("relbert/relation_mapping")["test"]]
ap_score_config = {
        "roberta-large": {"weight_head": 0.2, "weight_tail": 0.2, "template": 'as-what-same',
                          "positive_permutation": 4, "negative_permutation": 10, "weight_negative": 0.2},
        "gpt2-xl": {"weight_head": -0.4, "weight_tail": 0.2, "template": 'rel-same',
                    "positive_permutation": 2, "negative_permutation": 0, "weight_negative": 0.8},
        "bert-large-cased": {"weight_head": -0.2, "weight_tail": -0.4, "template": 'what-is-to',
                             "positive_permutation": 4, "negative_permutation": 4, "weight_negative": 0.2}
    }


def ap_score(model_name):
    _scorer = AnalogyScore(model=model_name)

    def get_score(query, options):
        return _scorer.analogy_score(
            query_word_pair=query,
            option_word_pairs=options,
            batch_size=1024,
            **ap_score_config[model_name])
    return get_score


if __name__ == '__main__':

    os.makedirs(pj('embeddings', 'ap_score'), exist_ok=True)
    # get data

    for k in ap_score_config.keys():
        scorer = ap_score(k)
        for data_id, _data in enumerate(data):
            print(f'[{k}]: {data_id}/{len(data)}')
            cache_file = pj('embeddings', 'ap_score', f'ap_score.{k}.{data_id}.json')
            if os.path.exists(cache_file):
                continue
            score_dict = {}
            source = _data['source']
            target = _data['target']
            for x, y in tqdm(list(permutations(source, 2))):
                options = [list(o) for o in permutations(target, 2)]
                scores = scorer(query=[x, y], options=options)
                assert len(scores) == len(options), f'{len(scores)} != {len(options)}'
                for s, (o_x, o_y) in zip(scores, options):
                    _id = f'{x}_{y}__{o_x}_{y}'
                    score_dict[_id] = s
            with open(cache_file, 'w') as f_writer:
                json.dump(score_dict, f_writer)
