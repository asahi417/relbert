import pickle
import os
from gensim.models import KeyedVectors

from tqdm import tqdm
import relbert
from relbert.util import wget

# get embedding for the common word pairs and store it in gensim w2v format
path_pair = './common_word_pairs.pkl'

if not os.path.exists(path_pair):
    wget('https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl')
with open(path_pair, 'rb') as f:
    pair_data = pickle.load(f)
pbar = tqdm(total=len(pair_data))
model = relbert.RelBERT('relbert_output/ckpt/roberta_custom_c/epoch_2')
batch_size = 512
chunk_size = 10 * batch_size
chunk_start = list(range(0, len(pair_data), chunk_size))
chunk_end = chunk_start[1:] + [len(pair_data)]
print('Start embedding extraction')
with open('gensim_model.txt', 'w', encoding='utf-8') as f:
    f.write(str(len(pair_data)) + " " + str(model.hidden_size) + "\n")
    for s, e in zip(chunk_start, chunk_end):
        vector = model.get_embedding(pair_data[s:e], batch_size=batch_size)
        for n, (token_i, token_j) in enumerate(pair_data[s:e]):
            f.write('__'.join([token_i, token_j]))
            for y in vector[n]:
                f.write(' ' + str(y))
            f.write("\n")
        pbar.update(e-s)

print('\nConvert to binary file')
model = KeyedVectors.load_word2vec_format('gensim_model.txt')
model.wv.save_word2vec_format('gensim_model.bin', binary=True)
print("new embeddings are available at `gensim_model.bin`")
os.remove('gensim_model.txt')
