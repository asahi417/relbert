import os
import logging
import pandas as pd
from sklearn.neural_network import MLPClassifier

from relbert import RelBERT
from relbert.data import get_lexical_relation_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

if not os.path.exists('./cache/global_vocab.txt'):
    raise ValueError('run `word_embedding_baseline.py` first to get shared vocab.')

with open('./cache/global_vocab.txt') as f:
    global_vocab = {i for i in f.read().split('\n') if len(i) > 0}


def main(embedding_model: str, global_vocab):
    model = get_word_embedding_model(embedding_model)
    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(embedding_model, data_name))
        label_dict = v.pop('label')
        x = [diff(a, b, model) if a in global_vocab and b in global_vocab else None for a, b in v['train']['x']]
        y = [t for n, t in enumerate(v['train']['y']) if x[n] is not None]
        x = [t for n, t in enumerate(x) if x[n] is not None]
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, y)

        logging.info('\t run validation')
        x = [diff(a, b, model) if a in global_vocab and b in global_vocab else None for a, b in v['test']['x']]
        oov = len([_x for _x in x if _x is None])
        y = [t for n, t in enumerate(v['test']['y']) if x[n] is not None]
        x = [t for n, t in enumerate(x) if x[n] is not None]
        accuracy = clf.score(x, y)
        report_tmp = {'model': embedding_model, 'accuracy': accuracy, 'label_size': len(label_dict), 'oov': oov,
                      'test_total': len(x), 'data': data_name}
        logging.info('\t accuracy: \n{}'.format(report_tmp))
        report.append(report_tmp)
    del model
    return report

