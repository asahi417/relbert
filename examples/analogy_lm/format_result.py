import os
from math import log
import matplotlib.pyplot as plt
import pandas as pd

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
    "EleutherAI/gpt-j-6B": [6000, "GPT-J"],
    "google/flan-t5-xxl": [11000, "Flan-T5"],
    "t5-11b": [11000, "T5"],
    "EleutherAI/gpt-neox-20b": [20000, "GPT-J"],
    "facebook/opt-30b": [30000, "OPT"],
    "facebook/opt-iml-30b": [30000, "OPT-IML"],
    "facebook/opt-iml-max-30b": [30000, "OPT-IML"],
    "relbert/flan-t5-small-analogy": [60, "Flan-T5 (FT)"],
    "relbert/flan-t5-base-analogy": [220, "Flan-T5 (FT)"],
    "relbert/flan-t5-large-analogy": [770, "Flan-T5 (FT)"],
    "relbert/flan-t5-xl-analogy": [3000, "Flan-T5 (FT)"]
}

df = pd.read_csv('results/full_result.prompt.csv')
df['Accuracy'] = df.pop('accuracy')  # * 100
df = df[[i not in ['sat_metaphor', 'sat'] for i in df['data']]]
df = df[[i in model_size for i in df['model']]]
df.pop('prefix')
for data, g in df.groupby('data'):
    output = {}
    g['lm'] = [model_size[i][1] for i in g['model']]
    # g['Model Size (log)'] = [log(model_size[i][0]) for i in g['model']]
    g['Model Size'] = [model_size[i][0] * 1000000 for i in g['model']]
    out = g.pivot_table(index='Model Size', columns='lm', aggfunc='mean')
    out.columns = [i[1] for i in out.columns]
    out = out.reset_index()
    lms = ['GPT-2', 'GPT-J', 'OPT', 'OPT-IML', 'T5', 'Flan-T5', 'Flan-T5 (FT)']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray', 'black']
    styles = ['o-', 'o--', 'o:', 's-', 's--', 's:', '^-', '^--', '^:']

    tmp = out[['Model Size', lms[0]]].dropna().reset_index()
    tmp['Accuracy'] = tmp[lms[0]]
    # ax = tmp.plot(kind='scatter', y='Accuracy', x='Model Size', color=colors[0], label=lms[0], logx=True)
    ax = tmp.plot.line(y='Accuracy', x='Model Size', color=colors[0], style=styles[0], label=lms[0], logx=True)
    for n, c in enumerate(lms[1:]):
        tmp = out[['Model Size', c]].dropna().reset_index()
        tmp['Accuracy'] = tmp[c]
        # tmp.plot(kind='scatter', y='Accuracy', x='Model Size', ax=ax, color=colors[n+1], label=c, logx=True)
        tmp.plot.line(y='Accuracy', x='Model Size', ax=ax, color=colors[n+1], style=styles[n+1], label=c, logx=True)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/figures/curve.{data}.png", bbox_inches="tight", dpi=600)

