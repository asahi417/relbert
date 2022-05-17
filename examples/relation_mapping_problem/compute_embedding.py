import os
import json
from itertools import permutations
from tqdm import tqdm
from relbert import RelBERT, AnalogyScore
from util_word_embedding import get_word_embedding_model

ap_score_config = {
        "roberta-large": {"weight_head": 0.2, "weight_tail": 0.2, "template": 'as-what-same',
                          "positive_permutation": 4, "negative_permutation": 10, "weight_negative": 0.2},
        "gpt2-xl": {"weight_head": -0.4, "weight_tail": 0.2, "template": 'rel-same',
                    "positive_permutation": 2, "negative_permutation": 0, "weight_negative": 0.8},
        "bert-large-cased": {"weight_head": -0.2, "weight_tail": -0.4, "template": 'what-is-to',
                             "positive_permutation": 4, "negative_permutation": 4, "weight_negative": 0.2}
    }


def embedding_model(model_name):
    if model_name == 'relbert':
        model = RelBERT('asahi417/relbert-roberta-large')
        def get_embedding(a, b): return model.get_embedding([a, b])
    elif model_name in ['fasttext', 'fasttext_cc']:
        model = get_word_embedding_model(model_name)
        def get_embedding(a, b): return (model[a] - model[b]).tolist()
    else:
        raise ValueError(f'unknown model {model_name}')
    return get_embedding


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

    os.makedirs('embeddings', exist_ok=True)
    # get data
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
    # for m in ['relbert', 'fasttext_cc']:
    #     embeder = embedding_model(m)
    #     for data_id, _data in enumerate(data):
    #         print(f'[{m}]: {data_id}/{len(data)}')
    #         cache_file = f'embeddings/{m}.vector.{data_id}.json'
    #         embedding_dict = {}
    #         if os.path.exists(cache_file):
    #             with open(cache_file) as f:
    #                 embedding_dict = json.load(f)
    #         for _type in ['source', 'target']:
    #             for x, y in permutations(_data[_type], 2):
    #                 _id = f'{x}__{y}'
    #                 if _id not in embedding_dict:
    #                     vector = embeder(x, y)
    #                     embedding_dict[_id] = vector
    #                     with open(cache_file, 'w') as f_writer:
    #                         json.dump(embedding_dict, f_writer)

    for k in ap_score_config.keys():
        scorer = ap_score(k)
        for data_id, _data in enumerate(data):
            print(f'[{k}]: {data_id}/{len(data)}')
            cache_file = f'embeddings/ap_score.{k}.{data_id}.json'
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
