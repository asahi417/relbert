import pickle
import os
import relbert
from relbert.util import wget

# get embedding for the common word pairs and store it in gensim w2v format
path_pair = './common_word_pairs.pkl'
if not os.path.exists(path_pair):
    wget('https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/common_word_pairs.pkl')
with open(path_pair, 'rb') as f:
    pair_data = pickle.load(f)
