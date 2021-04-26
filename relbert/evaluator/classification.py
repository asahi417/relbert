import os
import logging

import numpy as np
# from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors

from .. import RelBERT
from ..util import wget, home_dir
from ..data import get_lexical_relation_data
cache = '{}/relation_classification'.format(home_dir)


def get_word_embedding_model(model_name: str = 'fasttext'):
    """ get word embedding model """
    os.makedirs(cache, exist_ok=True)
    if model_name == 'w2v':
        path = '{}/GoogleNews-vectors-negative300.bin'.format(cache)
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            # get the embedding from mirroring site
            wget(url="https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/GoogleNews-vectors-negative300.bin.gz",
                 cache_dir=cache)
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'fasttext':
        path = './cache/wiki-news-300d-1M.vec'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                 cache_dir=cache)
        model = KeyedVectors.load_word2vec_format(path)
    elif model_name == 'glove':
        path = './cache/glove.840B.300d.gensim.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://drive.google.com/u/0/uc?id=1DbLuxwDlTRDbhBroOVgn2_fhVUQAVIqN&export=download',
                 gdrive_filename='glove.840B.300d.gensim.bin.tar.gz',
                 cache_dir=cache)
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'relative_init.fasttext.concat':
        path = './cache/relative_init.fasttext.concat.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://drive.google.com/u/0/uc?id=1EH0oywBo8OaNExyc5XTGIFhLvf8mZiBz&export=download',
                 gdrive_filename='relative_init.fasttext.concat.bin.tar.gz',
                 cache_dir=cache)
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name == 'relative_init.fasttext.truecase.concat':
        path = './cache/relative_init.fasttext.truecase.concat.bin'
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url="https://drive.google.com/u/0/uc?id=1iUuCYM_UJ6FHI5yxg5UIGkXN4qqU5S3G&export=download",
                 gdrive_filename='relative_init.fasttext.truecase.concat.bin.tar.gz',
                 cache_dir=cache)
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        path = './cache/{}.bin'.format(model_name)
        if not os.path.exists(path):
            logging.info('downloading {}'.format(model_name))
            wget(url='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/{}.bin.tar.gz'.format(model_name),
                 cache_dir=cache)
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model


def get_shared_vocab(model_list):
    _cache = '{}/global_vocab.txt'.format(cache)
    if os.path.exists(_cache):
        with open(_cache, 'r') as f:
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
    with open(_cache, 'w') as f:
        f.write('\n'.join(list(shared_vocab)))
    return shared_vocab


def diff(a, b, model, add_feature_set):
    vec_a = model[a]
    vec_b = model[b]
    feature = [vec_a, vec_b]
    if 'diff' in add_feature_set:
        feature.append(vec_a - vec_b)
    if 'dot' in add_feature_set:
        feature.append(vec_a * vec_b)
    return np.concatenate(feature)


def evaluate(global_vocab, embedding_model: str = None, relbert_ckpt: str = None, batch_size: int = 512,
             feature_set=None):

    if relbert_ckpt:
        model = RelBERT(relbert_ckpt)
        model_name = relbert_ckpt
        relbert_model = True
    else:
        model = get_word_embedding_model(embedding_model)
        model_name = embedding_model
        relbert_model = False
    data = get_lexical_relation_data()
    report = []
    for data_name, v in data.items():
        logging.info('train model with {} on {}'.format(model_name, data_name))
        label_dict = v.pop('label')
        in_vocab_index = [a in global_vocab and b in global_vocab for a, b in v['train']['x']]
        if relbert_model:
            x = [(a, b) for (a, b), flag in zip(v['train']['x'], in_vocab_index) if flag]
            x = model.get_embedding(x, batch_size=batch_size)
        else:
            x = [diff(a, b, model, feature_set) for (a, b), flag in zip(v['train']['x'], in_vocab_index) if flag]
        y = [y for y, flag in zip(v['train']['y'], in_vocab_index) if flag]
        logging.info('\t training data info: data size {}, label size {}'.format(len(x), len(label_dict)))
        clf = MLPClassifier().fit(x, y)
        # clf = LinearSVC().fit(x, y)

        logging.info('\t run validation')
        in_vocab_index = [a in global_vocab and b in global_vocab for a, b in v['test']['x']]
        oov = len(in_vocab_index) - sum(in_vocab_index)
        if relbert_model:
            x = [(a, b) for (a, b), flag in zip(v['test']['x'], in_vocab_index) if flag]
            x = model.get_embedding(x, batch_size=batch_size)
        else:
            x = [diff(a, b, model) for (a, b), flag in zip(v['test']['x'], in_vocab_index) if flag]
        y_true = [y for y, flag in zip(v['test']['y'], in_vocab_index) if flag]
        y_pred = clf.predict(x)
        # accuracy
        accuracy = clf.score(x, y_true)
        f_mac = f1_score(y_true, y_pred, average='macro')
        f_mic = f1_score(y_true, y_pred, average='micro')

        report_tmp = {'model': model_name, 'accuracy': accuracy, 'f1_macro': f_mac, 'f1_micro': f_mic,
                      'feature_set': feature_set,
                      'label_size': len(label_dict), 'oov': oov,
                      'test_total': len(x), 'data': data_name}
        logging.info('\t accuracy: \n{}'.format(report_tmp))
        report.append(report_tmp)
    del model
    return report

