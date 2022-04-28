import os
import json
from itertools import permutations

from relbert import RelBERT
from word_embedding import get_word_embedding_model


def embedding_model(model_name):
    if model_name == 'relbert':
        model = RelBERT('asahi417/relbert-roberta-large')
        def get_embedding(a, b): return model.get_embedding(a, b)
    elif model_name in ['fasttext', 'fasttext_cc']:
        model = get_word_embedding_model(model_name)
        def get_embedding(a, b): return (model[a] - model[b]).tolist()
    else:
        raise ValueError(f'unknown model {model_name}')
    return get_embedding


if __name__ == '__main__':

    os.makedirs('output', exist_ok=True)
    # get data
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
    for m in ['relbert', 'fasttext_cc']:
        embeder = embedding_model(m)
        for data_id, _data in enumerate(data):
            print(f'[{m}]: {data_id}/{len(data)}')
            cache_file = f'embeddings/{m}.vector.{data_id}.json'
            embedding_dict = {}
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    embedding_dict = json.load(f)
            for _type in ['source', 'target']:

                for x, y in list(permutations(_data[_type], 2)):
                    _id = f'{x}__{y}'
                    if _id not in embedding_dict:
                        vector = embeder(x, y)
                        embedding_dict[_id] = vector
                        with open(cache_file, 'w') as f_writer:
                            json.dump(embedding_dict, f_writer)
