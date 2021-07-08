""" Generate gensim embedding model for qualitative analysis on Table 4. """
import pickle
import os
import argparse
from gensim.models import KeyedVectors

from tqdm import tqdm
import relbert
from relbert.util import wget


def config(parser):
    parser.add_argument('-c', '--ckpt', help='checkpoint', default="asahi417/relbert-roberta-large", type=str)
    parser.add_argument('-b', '--batch', help='batch', default=2048, type=int)
    parser.add_argument('-e', '--export', help='export path', default='gensim_model.bin', type=str)
    return parser

argument_parser = argparse.ArgumentParser(description='Qualitative analysis')
argument_parser = config(argument_parser)
opt = argument_parser.parse_args()


if not os.path.exists(opt.export):
    # get embedding for the common word pairs and store it in gensim w2v format
    path_pair = './cache/common_word_pairs.pkl'

    if not os.path.exists(path_pair):
        wget(url='https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl',
             cache_dir='./cache')
    with open(path_pair, 'rb') as f:
        pair_data = pickle.load(f)
    pbar = tqdm(total=len(pair_data))
    model = relbert.RelBERT(opt.ckpt)
    chunk_size = 2 * opt.batch
    chunk_start = list(range(0, len(pair_data), chunk_size))
    chunk_end = chunk_start[1:] + [len(pair_data)]
    print('Start embedding extraction')
    with open(opt.export + '.txt', 'w', encoding='utf-8') as f:
        f.write(str(len(pair_data)) + " " + str(model.hidden_size) + "\n")
        for s, e in zip(chunk_start, chunk_end):
            vector = model.get_embedding(pair_data[s:e], batch_size=opt.batch)
            for n, (token_i, token_j) in enumerate(pair_data[s:e]):
                token_i, token_j = token_i.replace(' ', '_'), token_j.replace(' ', '_')
                f.write('__'.join([token_i, token_j]))
                for y in vector[n]:
                    f.write(' ' + str(y))
                f.write("\n")
            pbar.update(e-s)

    print('\nConvert to binary file')
    model = KeyedVectors.load_word2vec_format(opt.export + '.txt')
    model.wv.save_word2vec_format(opt.export, binary=True)
    print("new embeddings are available at `gensim_model.bin`")
    os.remove(opt.export + '.txt')
