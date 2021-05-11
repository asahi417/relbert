import logging
from itertools import chain, product, combinations
from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from .lm import RelBERT
from .data import get_lexical_relation_data, get_analogy_data, get_training_data
from .util import Dataset, triplet_loss

__all__ = ['evaluate_classification', 'evaluate_analogy']


def evaluate_classification(
        relbert_ckpt: str = None,
        batch_size: int = 512,
        target_relation=None,
        cache_dir: str = None):

    model = RelBERT(relbert_ckpt)
    data = get_lexical_relation_data(cache_dir)
    result = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(relbert_ckpt, data_name))
        label_dict = v.pop('label')
        x_tuple = [tuple(_x) for _x in v['train']['x']]
        x = model.get_embedding(x_tuple, batch_size=batch_size)
        x_back = model.get_embedding([(b, a) for a, b in x_tuple], batch_size=batch_size)
        x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, v['train']['y'])

        report_tmp = {'model': relbert_ckpt, 'label_size': len(label_dict), 'data': data_name}
        for prefix in ['test', 'val']:
            if prefix not in v:
                continue
            logging.info('\t run {}'.format(prefix))
            x_tuple = [tuple(_x) for _x in v[prefix]['x']]
            x = model.get_embedding(x_tuple, batch_size=batch_size)
            x_back = model.get_embedding([(b, a) for a, b in x_tuple], batch_size=batch_size)
            x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
            y_pred = clf.predict(x)

            p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(v[prefix]['y'], y_pred, average='macro')
            p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(v[prefix]['y'], y_pred, average='micro')

            accuracy = sum([a == b for a, b in zip(v[prefix]['y'], y_pred.tolist())])/len(y_pred)
            report_tmp.update(
                {'accuracy/{}'.format(prefix): accuracy,
                 'f1_macro/{}'.format(prefix): f_mac,
                 'f1_micro/{}'.format(prefix): f_mic,
                 'p_macro/{}'.format(prefix): p_mac,
                 'p_micro/{}'.format(prefix): p_mic,
                 'r_macro/{}'.format(prefix): r_mac,
                 'r_micro/{}'.format(prefix): r_mic,
                 'data_size/{}'.format(prefix): len(y_pred)}
            )
            if target_relation and prefix == 'test':
                for _l in target_relation:
                    if _l not in label_dict:
                        continue
                    p, r, f, _ = precision_recall_fscore_support(v[prefix]['y'], y_pred, labels=[label_dict[_l]])
                    report_tmp['f1/{}/{}'.format(prefix, _l)] = f[0]
                    report_tmp['p/{}/{}'.format(prefix, _l)] = p[0]
                    report_tmp['r/{}/{}'.format(prefix, _l)] = r[0]
        logging.info('\t accuracy: \n{}'.format(report_tmp))
        result.append(report_tmp)
    del model
    return result


def evaluate_analogy(
        relbert_ckpt: str = None,
        batch_size: int = 64,
        max_length: int = 64,
        template_type: str = None,
        mode: str = 'mask',
        cache_dir: str = None,
        validation_data: str = 'semeval2012'):
    model = RelBERT(relbert_ckpt, max_length=max_length, mode=mode, template_type=template_type)
    model.eval()

    result = []
    with torch.no_grad():

        # Loss value on validation set
        all_positive, all_negative, _ = get_training_data(validation_data, validation_set=True, cache_dir=cache_dir)
        # calculate the number of trial to cover all combination in batch
        n_pos = min(len(i) for i in all_positive.values())
        n_neg = min(len(i) for i in all_negative.values())
        n_trial = len(list(product(combinations(range(n_pos), 2), range(n_neg))))
        batch_index = list(range(n_trial))
        param = model.preprocess(all_positive, all_negative)
        total_loss = 0
        size = 0
        for n, bi in enumerate(tqdm(batch_index)):
            data_loader = torch.utils.data.DataLoader(
                Dataset(deterministic_index=bi, **param), batch_size=batch_size)
            for x in data_loader:
                encode = {k: torch.cat([x['positive_a'][k], x['positive_b'][k], x['negative'][k]]) for k in
                          x['positive_a'].keys()}
                embedding = model.to_embedding(encode)
                v_anchor, v_positive, v_negative = embedding.chunk(3)
                loss = triplet_loss(v_anchor, v_positive, v_negative, in_batch_negative=True)
                total_loss += loss.cpu().item()
                size += len(v_anchor)
        valid_loss = total_loss / size
        logging.info('valid loss: {}'.format(valid_loss))

        # Analogy test
        data = get_analogy_data(cache_dir)
        for d, (val, test) in data.items():
            logging.info('\t * data: {}'.format(d))
            # preprocess data
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in val + test]))
            all_pairs = [tuple(i) for i in all_pairs]
            data_ = model.preprocess(all_pairs, pairwise_input=False)
            batch = len(all_pairs) if batch_size is None else batch_size
            data_loader = torch.utils.data.DataLoader(Dataset(**data_), batch_size=batch)
            # get embedding
            embeddings = []
            for encode in data_loader:
                embeddings += model.to_embedding(encode).cpu().tolist()
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
                'accuracy/valid': acc_val,
                'accuracy/test': acc_test,
                'accuracy/full': acc,
                'data': d,
                'validation_loss': valid_loss, 'validation_data': validation_data,
                'model': relbert_ckpt, 'mode': model.mode, 'template_type': template_type
            })
            logging.info(str(result[-1]))
    del model
    return result

