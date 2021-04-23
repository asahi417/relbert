"""
You may need additional libraries as below
```
pip install sklearn
pip install gensim==3.8.1
```
"""
import os
import logging
from glob import glob

import pandas as pd
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors

from relbert import RelBERT
from relbert.util import wget
from relbert.data import get_lexical_relation_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
batch_size = 512


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
    cache = './cache/global_vocab.txt'
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


def main(global_vocab, embedding_model: str = None, relbert_ckpt: str = None):

    if relbert_ckpt:
        model = RelBERT(relbert_ckpt)
        relbert_model = True
    else:
        model = get_word_embedding_model(embedding_model)
        relbert_model = False
    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(embedding_model, data_name))
        label_dict = v.pop('label')
        in_vocab_index = [a in global_vocab and b in global_vocab for a, b in v['train']['x']]
        if relbert_model:
            x = [(a, b) for (a, b), flag in zip(v['train']['x'], in_vocab_index) if flag]
            x = model.get_embedding(x, batch_size=batch_size)
        else:
            x = [diff(a, b, model) for (a, b), flag in zip(v['train']['x'], in_vocab_index) if flag]
        y = [y for y, flag in zip(v['train']['y'], in_vocab_index) if flag]
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, y)

        logging.info('\t run validation')
        in_vocab_index = [a in global_vocab and b in global_vocab for a, b in v['test']['x']]
        oov = len(in_vocab_index) - sum(in_vocab_index)
        if relbert_model:
            x = [(a, b) for (a, b), flag in zip(v['test']['x'], in_vocab_index) if flag]
            x = model.get_embedding(x, batch_size=batch_size)
        else:
            x = [diff(a, b, model) for (a, b), flag in zip(v['test']['x'], in_vocab_index) if flag]
        y = [y for y, flag in zip(v['test']['y'], in_vocab_index) if flag]
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
    out_csv = './examples/lexical_relation_classification/result.csv'
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv, index_col=0)
        done_list = df['model'].values
    else:
        df = None
        done_list = []
    full_result = []

    logging.info("RUN WORD-EMBEDDING BASELINE")
    for m in target_word_embedding:
        print(m, done_list)
        if m in done_list:
            continue
        full_result += main(vocab, embedding_model=m)

    pd.DataFrame(full_result).to_csv(out_csv)

    logging.info("RUN RELBERT")
    ckpts = glob('relbert_output/ckpt/*/epoch*')
    for m in ckpts:
        if m in done_list:
            continue
        full_result += main(vocab, relbert_ckpt=m)

    pd.DataFrame(full_result).to_csv(out_csv)



