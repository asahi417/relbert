import os
import json
import argparse
from itertools import permutations
from os.path import join as pj
from relbert import RelBERT
from datasets import load_dataset

data = [i for i in load_dataset("relbert/relation_mapping")["test"]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RelBERT evaluation on analogy/relation classification')
    parser.add_argument('-m', '--model', help='model name', default="relbert/relbert-roberta-large", type=str)
    parser.add_argument('-b', '--batch', help='batch', default=64, type=str)
    opt = parser.parse_args()

    model_alias = os.path.basename(opt.model)
    model = RelBERT(opt.model)
    os.makedirs(pj('embeddings', model_alias), exist_ok=True)

    for data_id, _data in enumerate(data):
        print(f'[{model_alias}]: {data_id}/{len(data)}')
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
        vector = model.get_embedding(inputs, batch_size=opt.batch_size)
        embedding_dict.update({i: v for i, v in zip(inputs_id, vector)})
        with open(cache_file, 'w') as f_writer:
            json.dump(embedding_dict, f_writer)
