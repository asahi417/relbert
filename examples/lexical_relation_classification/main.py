"""
You may need additional libraries as below
```
pip install sklearn
pip install gensim==3.8.1
```
"""
import os
import logging

from relbert.util import wget
from relbert.data import get_lexical_relation_data

import pandas as pd
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_word_embedding_model(model_name: str = 'fasttext'):
    """ get word embedding model """
    os.makedirs('./cache', exist_ok=True)
    if model_name == 'w2v':
        path = './cache/GoogleNews-vectors-negative300.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            # get the embedding from mirroring site
            wget(url="https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/GoogleNews-vectors-negative300.bin.gz")
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'fasttext':
        path = './cache/wiki-news-300d-1M.vec'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
        model = KeyedVectors.load_word2vec_format(path)
    elif model_name == 'glove':
        path = './cache/glove.840B.300d.gensim.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://drive.google.com/u/0/uc?id=1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN&export=download',
                 gdrive_filename='glove.840B.300d.gensim.bin.tar.gz')
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'relative_init.fasttext.concat':
        path = './cache/relative_init.fasttext.concat.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://drive.google.com/u/0/uc?id=1EH0oywBo8OaNExyc5XTGIFhLvf8mZiBz&export=download',
                 gdrive_filename='relative_init.fasttext.concat.bin.tar.gz')
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'relative_init.fasttext.truecase.concat':
        path = './cache/relative_init.fasttext.truecase.concat.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url="https://drive.google.com/u/0/uc?id=1iUuCYM_UJ6FHI5yxg5UIGkXN4qqU5S3G&export=download",
                 gdrive_filename='relative_init.fasttext.truecase.concat.bin.tar.gz')
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        path = './cache/{}.bin'.format(model_name)
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/{}.bin.tar.gz'.format(model_name))
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model


def get_shared_vocab(model_list):
    cache = './cache/global_vocab.{}.txt'.format('.'.join(model_list))
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            return {i for i in f.read().split('\n') if len(i) > 0}
    shared_vocab = None
    for _m in model_list:
        model = get_word_embedding_model(_m)
        v = set(model.vocab.keys())
        if shared_vocab is None:
            shared_vocab = v
        else:
            shared_vocab = shared_vocab.intersection(v)
        del model
    with open(cache, 'w') as f:
        f.write('\n'.join(list(shared_vocab)))
    return shared_vocab


def diff(a, b, model):
    return model[a] - model[b]


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


if __name__ == '__main__':
    target_word_embedding = ['w2v', 'glove', 'fasttext']
    vocab = get_shared_vocab(target_word_embedding)
    logging.info('shared vocab has {} word'.format(len(vocab)))
    full_result = []
    for m in target_word_embedding:
        full_result += main('fasttext', vocab)
    df = pd.DataFrame(full_result)
    print(df)
    df.to_csv('./examples/lexical_relation_classification/result.csv')
