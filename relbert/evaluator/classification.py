import logging
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from .. import RelBERT
from ..data import get_lexical_relation_data


def evaluate(relbert_ckpt: str = None, batch_size: int = 512):

    model = RelBERT(relbert_ckpt)
    model_name = relbert_ckpt
    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(model_name, data_name))
        label_dict = v.pop('label')
        print(v['train']['x'])
        x = model.get_embedding(v['train']['x'], batch_size=batch_size)
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, v['train']['y'])
        logging.info('\t run validation')
        x = model.get_embedding(v['test']['x'], batch_size=batch_size)
        y_pred = clf.predict(x)

        # accuracy
        accuracy = clf.score(x, v['test']['y'])
        f_mac = f1_score(v['test']['y'], y_pred, average='macro')
        f_mic = f1_score(v['test']['y'], y_pred, average='micro')

        report_tmp = {'model': model_name, 'accuracy': accuracy, 'f1_macro': f_mac, 'f1_micro': f_mic,
                      'label_size': len(label_dict), 'test_total': len(x), 'data': data_name}
        logging.info('\t accuracy: \n{}'.format(report_tmp))
        report.append(report_tmp)
    del model
    return report

