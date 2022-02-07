""" Get relation embeddings from RelBERT and cache as Gensim model """
import os
from tqdm import tqdm
import json
from glob import glob
from relbert import RelBERT
from gensim.models import KeyedVectors


MODEL_ALIAS = os.getenv("MODEL_ALIAS", "asahi417/relbert-roberta-large")
GENSIM_FILE = os.getenv("GENSIM_FILE", "examples/concept_net/relbert_embedding")
BATCH = int(os.getenv("BATCH", "32"))
CHUNK = int(os.getenv("BATCH", "3200"))

concept_net_processed_file_dir = './examples/concept_net/data'
model = RelBERT(MODEL_ALIAS, max_length=128)

# get word_pairs
word_pairs = []
for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
    with open(i) as f:
        tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]
    word_pairs += [(os.path.basename(t['arg1']), os.path.basename(t['arg2'])) for t in tmp]
# remove single character
word_pairs = [(a, b) for a, b in word_pairs if len(a) != 1 and len(b) != 1]
# remove duplicate
word_pairs = [t for t in (set(tuple(i) for i in word_pairs))]

print('found {} word pairs'.format(len(word_pairs)))

# cache embeddings
chunk_start = 0
chunk_end = CHUNK
pbar = tqdm(total=len(word_pairs))
print('generate gensim file `{}.txt`'.format(GENSIM_FILE))
with open('{}.txt'.format(GENSIM_FILE), 'w', encoding='utf-8') as f:
    f.write(str(len(word_pairs)) + " " + str(model.hidden_size) + "\n")
    while True:
        if chunk_start == chunk_end:
            break
        word_pairs_chunk = word_pairs[chunk_start:chunk_end]
        vector = model.get_embedding(word_pairs_chunk, batch_size=BATCH)
        for n, (token_i, token_j) in enumerate(word_pairs_chunk):
            token_i, token_j = token_i.replace(' ', '_'), token_j.replace(' ', '_')
            f.write('__'.join([token_i, token_j]))
            for y in vector[n]:
                f.write(' ' + str(y))
            f.write("\n")

        chunk_start = chunk_end
        chunk_end = min(chunk_end + CHUNK, len(word_pairs))
        pbar.update(chunk_end - chunk_start)

print('Convert to binary file')
model = KeyedVectors.load_word2vec_format('{}.txt'.format(GENSIM_FILE))
model.wv.save_word2vec_format('{}.bin'.format(GENSIM_FILE), binary=True)
os.remove('{}.txt'.format(GENSIM_FILE))
