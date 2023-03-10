import json
import os
from typing import List
from statistics import mean
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle, seed
from format_result import model_size

df_full_i = pd.read_csv('results/full_result.instruction.csv').sort_values(by=['model', 'data'])
df_full_p = pd.read_csv('results/full_result.prompt.csv').sort_values(by=['model', 'data'])
df_full_p = df_full_p[[i in df_full_i.model.unique() for i in df_full_p.model]]
df_full_p = df_full_p[[i in df_full_i.data.unique() for i in df_full_p.data]]
df_full_i['instruction'] = df_full_i.pop("accuracy").values
df_full_i['analogical statement'] = df_full_p['accuracy'].values
df_full_i.pop('prefix')
df_full_i = df_full_i[[i not in ['sat', 'sat_metaphor'] for i in df_full_i['data']]]
df_full_i['lm'] = [model_size[i][1] for i in df_full_i['model']]
df_full_i['Model Size'] = [model_size[i][0] / 1000 for i in df_full_i['model']]
df_full_i = df_full_i[[i > 1 for i in df_full_i['Model Size']]]
df_full_i.index = [f"{j} ({int(k) if int(k) == k else k}B)" for j, k in zip(df_full_i['lm'], df_full_i["Model Size"])]
df_full_i.pop("model")
df_full_i.pop("lm")
df_full_i.pop("Model Size")
os.makedirs('results/figures/instruction', exist_ok=True)

for data, g in df_full_i.groupby('data'):
    g.pop("data")
    g.plot.bar(rot=45)
    plt.tight_layout()
    plt.savefig(f"results/figures/instruction/{data}.png", bbox_inches="tight", dpi=600)

df_full_i["model"] = df_full_i.index
df_full_i.groupby("model").mean().plot.bar(rot=45)
plt.tight_layout()
plt.savefig(f"results/figures/instruction/average.png", bbox_inches="tight", dpi=600)
