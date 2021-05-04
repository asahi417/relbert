import logging
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from .. import RelBERT
from ..data import get_lexical_relation_data


def evaluate(relbert_ckpt: str = None, batch_size: int = 512, both_direction: bool = True,
             target_relation=None):

    model = RelBERT(relbert_ckpt)
    model_name = relbert_ckpt
    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(model_name, data_name))
        label_dict = v.pop('label')
        x_tuple = [tuple(_x) for _x in v['train']['x']]
        x = model.get_embedding(x_tuple, batch_size=batch_size)
        if both_direction:
            x_back = model.get_embedding([(b, a) for a, b in x_tuple], batch_size=batch_size)
            x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, v['train']['y'])

        report_tmp = {'model': model_name, 'both_direction': both_direction, 'label_size': len(label_dict), 'data': data_name}
        for prefix in ['test', 'val']:
            if prefix not in v:
                continue
            logging.info('\t run {}'.format(prefix))
            x_tuple = [tuple(_x) for _x in v[prefix]['x']]
            x = model.get_embedding(x_tuple, batch_size=batch_size)
            if both_direction:
                x_back = model.get_embedding([(b, a) for a, b in x_tuple], batch_size=batch_size)
                x = [np.concatenate([a, b]) for a, b in zip(x, x_back)]
            y_pred = clf.predict(x)
            f_mac = f1_score(v[prefix]['y'], y_pred, average='macro')
            f_mic = f1_score(v[prefix]['y'], y_pred, average='micro')
            accuracy = sum([a == b for a, b in zip(v[prefix]['y'], y_pred.tolist())])/len(y_pred)
            report_tmp.update(
                {'accuracy/{}'.format(prefix): accuracy,
                 'f1_macro/{}'.format(prefix): f_mac,
                 'f1_micro/{}'.format(prefix): f_mic,
                 'data_size/{}'.format(prefix): len(y_pred)}
            )
            if target_relation and prefix == 'test':
                for _l in target_relation:
                    if _l not in label_dict:
                        continue
                    _y_true = [i if i == label_dict[_l] else 0 for i in v[prefix]['y']]
                    _y_pred = [i if i == label_dict[_l] else 0 for i in y_pred.tolist()]
                    report_tmp['accuracy/{}/{}'.format(prefix, _l)] = \
                        sum([a == b for a, b in zip(_y_true, _y_pred)]) / len(y_pred)

        logging.info('\t accuracy: \n{}'.format(report_tmp))
        report.append(report_tmp)
    del model
    return report

