import json
import os
import statistics
import logging
from itertools import chain
from tqdm import tqdm
from os.path import join as pj
from typing import Dict, List

import torch
from datasets import load_dataset

from ..util import NCELoss
from ..lm import RelBERT, Dataset


def compute_loss(model,
                 validation_data: str,
                 exclude_relation,
                 loss_function: str,
                 batch_size: int,
                 relation_level: str or List = None,
                 temperature_nce_rank: Dict = None,
                 temperature_nce_constant: float = None,
                 split: List or str = 'validation'):
    nce_loss = NCELoss(loss_function, temperature_nce_constant, temperature_nce_rank)
    assert type(split) is str or type(split) is list, f'unknown type for split {split} ({type(split)})'
    split = [split] if type(split) is str else split
    total_loss = []
    for s in split:
        logging.info(f'computing loss for the split {s}')
        data = load_dataset(validation_data, split=s)
        if relation_level is not None:
            relation_level = [relation_level] if type(relation_level) is str else relation_level
            data = data.filter(lambda _x: _x["level"] in relation_level)
        if exclude_relation is not None:
            data = data.filter(lambda _x: _x['relation_type'] not in exclude_relation)
        encoded_pairs_dict = model.encode_word_pairs(
            list(chain(*(data["positives"] + data["negatives"])))
        )
        loader_dict = {}
        for example in data:
            pairs_p = example['positives']
            pairs_n = example['negatives']
            k = example['relation_type']
            dataset_p = Dataset([encoded_pairs_dict['__'.join(k)] for k in pairs_p], return_ranking=True)
            dataset_n = Dataset([encoded_pairs_dict['__'.join(k)] for k in pairs_n], return_ranking=False)
            loader_dict[k] = {
                'positive': torch.utils.data.DataLoader(dataset_p, num_workers=0, batch_size=len(pairs_p)),
                'negative': torch.utils.data.DataLoader(dataset_n, num_workers=0, batch_size=len(pairs_n))
            }

        for n, relation_key in tqdm(list(enumerate(data['relation_type']))):
            # data loader will return full instances
            x_p = next(iter(loader_dict[relation_key]['positive']))
            x_n = next(iter(loader_dict[relation_key]['negative']))
            # data loader will return full instances
            x = {k: torch.concat([x_p[k], x_n[k]]) for k in x_n.keys()}
            embedding = model.to_embedding(x, batch_size=batch_size)
            batch_size_positive = len(x_p['input_ids'])
            embedding_p = embedding[:batch_size_positive]
            embedding_n = embedding[batch_size_positive:]
            rank = x_p.pop('ranking').cpu().tolist()
            loss = nce_loss(embedding_p, embedding_n, rank).cpu().tolist()
            total_loss.append(loss)
    total_loss = statistics.mean(total_loss)
    return total_loss


def evaluate_validation_loss(validation_data: str,
                             relation_level: str or List = None,
                             relbert_ckpt: str = None,
                             max_length: int = 64,
                             batch_size: int = 64,
                             split: List or str = 'validation',
                             exclude_relation=None):
    model = RelBERT(relbert_ckpt, max_length=max_length)
    assert model.is_trained, 'model is not trained'
    model.eval()
    with torch.no_grad():
        assert os.path.exists(pj(relbert_ckpt, "trainer_config.json"))
        with open(pj(relbert_ckpt, "trainer_config.json")) as f:
            trainer_config = json.load(f)
        loss_function = trainer_config['loss_function']
        temperature_nce_rank = trainer_config['temperature_nce_rank']
        temperature_nce_constant = trainer_config['temperature_nce_constant']
        validation_loss = compute_loss(
            model=model,
            validation_data=validation_data,
            exclude_relation=exclude_relation,
            loss_function=loss_function,
            batch_size=batch_size,
            temperature_nce_rank=temperature_nce_rank,
            temperature_nce_constant=temperature_nce_constant,
            split=split,
            relation_level=relation_level
        )
    split_alias = split if type(split) is str else '_'.join(sorted(split))
    result = {
        'loss': validation_loss,
        'data': validation_data,
        'split': split_alias,
        'exclude_relation': exclude_relation
    }
    if relation_level is not None:
        result['relation_level'] = relation_level
    logging.info(str(result))
    del model
    return result
