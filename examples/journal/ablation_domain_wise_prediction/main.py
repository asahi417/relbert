import os
from statistics import mean

import pandas as pd
from datasets import load_dataset
from relbert import RelBERT

os.makedirs("results", exist_ok=True)

def cosine_similarity(a, b):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/(norm_a * norm_b)


data_names = {
    'u2': [],
    'u4': [],
    'google': [],
    'bats': [],
    'scan': []
}


model = RelBERT("relbert/relbert-roberta-large")

for i in data_names.keys():
    if not os.path.exists(f"results/{i}.csv"):

        data = load_dataset("relbert/analogy_questions", i, split='test')
        v_stem = model.get_embedding(data['stem'], batch_size=1024)

        choice_flat = []
        choice_index = []
        for n, c in enumerate(data['choice']):
            choice_flat += c
            choice_index += [n] * len(c)
        v_choice_flat = model.get_embedding(choice_flat, batch_size=1024)

        v_choice = [[v for v, c_i in zip(v_choice_flat, choice_index) if c_i == _i] for _i in range(len(data['choice']))]
        assert len(v_choice) == len(v_stem)
        sims = [[cosine_similarity(_c, s) for _c in c] for c, s in zip(v_choice, v_stem)]
        pred = [s.index(max(s)) for s in sims]
        acc = [int(a == p) for a, p in zip(data['answer'], pred)]
        df = pd.DataFrame([data['prefix'], acc], index=['prefix', 'accuracy']).T
        df.to_csv(f"results/{i}.csv")
