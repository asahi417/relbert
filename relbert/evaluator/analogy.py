import os
import logging
from itertools import chain, product, combinations
from typing import List, Dict
from tqdm import tqdm

import pandas as pd
import torch

from relbert.lm import RelBERT
from relbert.data import get_analogy_data, get_training_data
from relbert.util import Dataset, triplet_loss


def evaluate(model: List,
             export_file: str,
             max_length: int = 64,
             template_type: str = None,
             mode: str = 'average',
             cache_dir: str = None,
             batch: int = 64,
             num_worker: int = 1,
             validation_data: str = 'semeval2012',
             mse_margin: int = 1,
             in_batch_negative: bool = True,
             shared_config: Dict = None):
    if not export_file.endswith('.csv'):
        export_file += '.csv'
    logging.info('{} checkpoints'.format(len(model)))
    result = []
    for n, i in enumerate(model):
        logging.info('\t * checkpoint {}/{}: {}'.format(n + 1, len(model), i))
        tmp_result = _evaluate(
            i, max_length, template_type, mode, cache_dir, batch, num_worker,
            validation_data, mse_margin, in_batch_negative
        )
        result += tmp_result
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
              template_type: str = None,
              mode: str = 'mask',
              cache_dir: str = None,
              batch: int = 64,
              num_worker: int = 1,
              validation_data: str = 'semeval2012',
              mse_margin: int = 1,
              in_batch_negative: bool = False):
    lm = RelBERT(model, max_length=max_length, mode=mode, template_type=template_type)
    lm.eval()

    result = []
    with torch.no_grad():

        # Loss value on validation set
        all_positive, all_negative, _ = get_training_data(validation_data, validation_set=True, cache_dir=cache_dir)
        # calculate the number of trial to cover all combination in batch
        n_pos = min(len(i) for i in all_positive.values())
        n_neg = min(len(i) for i in all_negative.values())
        n_trial = len(list(product(combinations(range(n_pos), 2), range(n_neg))))
        batch_index = list(range(n_trial))
        param = lm.preprocess(all_positive, all_negative)
        total_loss = 0
        size = 0
        for n, bi in enumerate(tqdm(batch_index)):
            data_loader = torch.utils.data.DataLoader(
                Dataset(deterministic_index=bi, **param), batch_size=batch, num_workers=num_worker)
            for x in data_loader:
                encode = {k: torch.cat([x['positive_a'][k], x['positive_b'][k], x['negative'][k]]) for k in
                          x['positive_a'].keys()}
                embedding = lm.to_embedding(encode)
                v_anchor, v_positive, v_negative = embedding.chunk(3)
                loss = triplet_loss(
                    v_anchor, v_positive, v_negative, margin=mse_margin, in_batch_negative=in_batch_negative)
                total_loss += loss.cpu().item()
                size += len(v_anchor)
        valid_loss = total_loss / size
        logging.info('valid loss: {}'.format(valid_loss))

        # Analogy test
        data = {d: get_analogy_data(d, cache_dir=cache_dir) for d in ['bats', 'sat', 'u2', 'u4', 'google']}
        for d, (val, test) in data.items():
            logging.info('\t * data: {}'.format(d))
            # preprocess data
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in val + test]))
            all_pairs = [tuple(i) for i in all_pairs]
            data_ = lm.preprocess(all_pairs, pairwise_input=False)
            batch = len(all_pairs) if batch is None else batch
            data_loader = torch.utils.data.DataLoader(Dataset(**data_), num_workers=num_worker, batch_size=batch)
            # get embedding
            embeddings = []
            for encode in data_loader:
                embeddings += lm.to_embedding(encode).cpu().tolist()
            assert len(embeddings) == len(all_pairs)
            embeddings = {str(k_): v for k_, v in zip(all_pairs, embeddings)}

            def cos_similarity(a_, b_):
                inner = sum(list(map(lambda y: y[0] * y[1], zip(a_, b_))))
                norm_a = sum(list(map(lambda y: y * y, a_))) ** 0.5
                norm_b = sum(list(map(lambda y: y * y, b_))) ** 0.5
                if norm_b * norm_a == 0:
                    return -100
                return inner / (norm_b * norm_a)

            def prediction(_data):
                accuracy = []
                for single_data in _data:
                    v_stem = embeddings[str(tuple(single_data['stem']))]
                    v_choice = [embeddings[str(tuple(c))] for c in single_data['choice']]
                    sims = [cos_similarity(v_stem, v) for v in v_choice]
                    pred = sims.index(max(sims))
                    if sims[pred] == -100:
                        raise ValueError('failed to compute similarity')
                    accuracy.append(single_data['answer'] == pred)
                return sum(accuracy) / len(accuracy)

            # get prediction
            acc_val = prediction(val)
            acc_test = prediction(test)
            acc = (acc_val * len(val) + acc_test * len(test)) / (len(val) + len(test))
            result.append({
                'analogy_accuracy_valid': acc_val,
                'analogy_accuracy_test': acc_test,
                'analogy_accuracy_full': acc,
                'analogy_data': d,
                'valid_loss': valid_loss, 'validation_data': validation_data,
                'model': model, 'mode': lm.mode,
                'template_type': template_type
            })
            logging.info(str(result[-1]))
    return result

