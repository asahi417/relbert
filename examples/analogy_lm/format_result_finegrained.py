import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset


os.makedirs('results/figures', exist_ok=True)
model_size = {
    # "roberta-base": [110, "RoBERTa"],
    # "roberta-large": [355, "RoBERTa"],
    "google/flan-t5-small": [60, "Flan-T5"],
    "t5-small": [60, "T5"],
    "gpt2": [124, "GPT-2"],
    'facebook/opt-125m': [125, "OPT"],
    "google/flan-t5-base": [220, "Flan-T5"],
    "t5-base": [220, "T5"],
    "facebook/opt-350m": [350, "OPT"],
    "gpt2-medium": [355, "GPT-2"],
    "google/flan-t5-large": [770, "Flan-T5"],
    "t5-large": [770, "T5"],
    "gpt2-large": [774, "GPT-2"],
    "facebook/opt-1.3b": [1300, "OPT"],
    "facebook/opt-iml-1.3b": [1300, "OPT-IML"],
    "facebook/opt-iml-max-1.3b": [1300, "OPT-IML"],
    "gpt2-xl": [1500, "GPT-2"],
    "google/flan-t5-xl": [3000, "Flan-T5"],
    "t5-3b": [3000, "T5"],
    "google/flan-t5-xxl": [11000, "Flan-T5"],
    "t5-11b": [11000, "T5"],
    "EleutherAI/gpt-neo-125M": [125, "GPT-J"],
    "EleutherAI/gpt-neo-1.3B": [1300, "GPT-J"],
    "EleutherAI/gpt-neo-2.7B": [2700, "GPT-J"],
    "EleutherAI/gpt-j-6B": [6000, "GPT-J"],
    "EleutherAI/gpt-neox-20b": [20000, "GPT-J"],
    "facebook/opt-30b": [30000, "OPT"],
    "facebook/opt-iml-30b": [30000, "OPT-IML"],
    "facebook/opt-iml-max-30b": [30000, "OPT-IML"],
    "relbert/flan-t5-small-analogy": [60, "Flan-T5 (FT)"],
    "relbert/flan-t5-base-analogy": [220, "Flan-T5 (FT)"],
    "relbert/flan-t5-large-analogy": [770, "Flan-T5 (FT)"],
    "relbert/flan-t5-xl-analogy": [3000, "Flan-T5 (FT)"]
}


output = []
for m in model_size:
    m = os.path.basename(m)
    for i in glob(f"results/breakdown/{m}*.csv"):
        df = pd.read_csv(i)

        data = '_'.join(i.split("_None")[0].split("_")[1:])
        if data == 'bats':
            df['prefix'] = df['prefix'].apply(lambda x: os.path.dirname(x.replace("./cache/BATS_3.0/", "")))
        elif data == 'google':
            df['prefix'] = df['prefix'].apply(lambda x: 'Morphological' if 'gram' in x else "Semantic")
        elif 'sat' in data:
            continue
        elif data == 'nell_relational_similarity':
            df['prefix'] = df['prefix'].apply(lambda x: x.replace("concept:", ""))

        for prefix, g in df.groupby("prefix"):
            output.append({'model': m, "data": data, 'accuracy': g['accuracy'].mean(), 'prefix': prefix})
df = pd.DataFrame(output)
for (data, prefix), g in df.groupby(["data", 'prefix']):
    g
    g = g.sort_values("accuracy", ascending=False)
    print(g['model'].values[0], g['accuracy'].values[0])
