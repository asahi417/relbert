import os
import json
from glob import glob

import hdbscan
from gensim.models import KeyedVectors

# load gensim model
model = KeyedVectors.load_word2vec_format("data/relbert_embedding.bin", binary=True)
concept_net_processed_file_dir = 'data/conceptnet'


for i in glob('{}/*.jsonl'.format(concept_net_processed_file_dir)):
    relation_type = os.path.basename(i).replace('.jsonl', '')
    with open(i) as f:
        tmp = [json.loads(t) for t in f.read().split('\n') if len(t) > 0]

    for _tmp in tmp:
        if len(os.path.basename(_tmp['arg1'])) == 1 or len(os.path.basename(_tmp['arg2'])) == 1:
            input(_tmp)
        pair = '{}__{}'.format(os.path.basename(_tmp['arg1']), os.path.basename(_tmp['arg2']))
        embedding = model[pair]

    clusterer = hdbscan.HDBSCAN()
