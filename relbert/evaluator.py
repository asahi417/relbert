import os
import logging
from itertools import chain
from typing import List, Dict

import pandas as pd

from .lm import RelBERT
from .data import get_analogy_data


def cos_similarity(a_, b_):
    inner = sum(list(map(lambda x: x[0] * x[1], zip(a_, b_))))
    norm_a = sum(list(map(lambda x: x * x, a_))) ** 0.5
    norm_b = sum(list(map(lambda x: x * x, b_))) ** 0.5
    if norm_b * norm_a == 0:
        return -100
    return inner / (norm_b * norm_a)


def evaluate(model: List,
             max_length: int = 64,
             template_type: str = 'a',
             mode: str = 'mask',
             test_type: str = 'analogy',
             export_file: str = './eval.csv',
             cache_dir: str = None,
             batch: int = 64,
             num_worker: int = 1,
             shared_config: Dict = None):
    logging.info('{} checkpoints'.format(len(model)))
    result = []
    for n, i in enumerate(model):
        logging.info('\t * checkpoint {}/{}: {}'.format(n + 1, len(model), i))
        result += _evaluate(i, max_length, template_type, mode, test_type, cache_dir, batch, num_worker)
    if shared_config is not None:
        for tmp in result:
            tmp.update(shared_config)
    df = pd.DataFrame(result)
    logging.info('result\n {}'.format(df))
    os.makedirs(os.path.dirname(export_file), exist_ok=True)
    if os.path.exists(export_file):
        df_tmp = pd.read_csv(export_file, index_col=0)
        df = pd.concat([df_tmp, df])
        df = df.drop_duplicates()
    df.to_csv(export_file)
    logging.info('exported to {}'.format(export_file))


def _evaluate(model,
              max_length: int = 64,
              template_type: str = 'a',
              mode: str = 'mask',
              test_type: str = 'analogy',
              cache_dir: str = None,
              batch: int = 64,
              num_worker: int = 1):
    if test_type == 'analogy':
        data = {d: get_analogy_data(d, cache_dir=cache_dir) for d in ['sat', 'u2', 'u4', 'google', 'bats']}
    else:
        raise ValueError('unknown test_type: {}'.format(test_type))

    lm = RelBERT(model, max_length=max_length, mode=mode, template_type=template_type)
    result = []
    for k, (val, test) in data.items():
        logging.debug('\t * data: {}'.format(k))
        all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in val + test]))
        all_pairs = [tuple(v) for v in all_pairs]
        embeddings = lm.get_embedding(all_pairs, batch_size=batch, num_worker=num_worker)
        assert len(embeddings) == len(all_pairs)
        embedding_dict = {str(k): v for k, v in zip(all_pairs, embeddings)}

        def prediction(_data):
            accuracy = []
            for single_data in _data:
                v_stem = embedding_dict[str(tuple(single_data['stem']))]
                v_choice = [embedding_dict[str(tuple(c))] for c in single_data['choice']]
                sims = [cos_similarity(v_stem, v) for v in v_choice]
                pred = sims.index(max(sims))
                if sims[pred] == -100:
                    raise ValueError('failed to compute similarity')
                accuracy.append(single_data['answer'] == pred)
            return sum(accuracy)/len(accuracy)

        acc_val = prediction(val)
        acc_test = prediction(test)
        acc = (acc_val * len(val) + acc_test * len(test))/(len(val) + len(test))
        result.append({
            'accuracy_valid': acc_val, 'accuracy_test': acc_test, 'accuracy_full': acc,
            'model': model, 'mode': lm.mode, 'template_type': lm.template_type,
            'data': k
        })
        logging.info(str(result[-1]))
    return result








