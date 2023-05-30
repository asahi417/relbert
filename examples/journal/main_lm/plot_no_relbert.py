import json
import os
from itertools import product
from statistics import mean
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle, seed
from datasets import load_dataset

os.makedirs('results/figures', exist_ok=True)
df_full = pd.read_csv('results/full_result.prompt.csv')
df_full['Accuracy'] = df_full.pop('accuracy') * 100
df_full = df_full[[i not in ['sat', 'sat_metaphor'] for i in df_full['data']]]
df_full.pop('prefix')

plt.rcParams.update({'font.size': 14})  # must set in top

random_guess = {}
relbert_result = {}
relbert_result_base = {}
fasttext_result = {}
data_names = {
    'sat_full': "SAT",
    'u2': "U2",
    'u4': "U4",
    'google': "Google",
    'bats': "BATS",
    'scan': "SCAN",
    "nell_relational_similarity": "NELL",
    "t_rex_relational_similarity": "T-REX",
    "conceptnet_relational_similarity": "ConceptNet",
}
for t in ["sat_full", "u2", "u4", "bats", "google", 'scan', 'nell_relational_similarity', 't_rex_relational_similarity',
          'conceptnet_relational_similarity']:
    # calculate random guess
    data = load_dataset("relbert/analogy_questions_private", t, split="test")
    random_guess[t] = mean([1 / len(i['choice']) for i in data]) * 100
    # load relbert result
    with open(f"results/relbert_prediction/{t}.json") as f:
        relbert_result[t] = mean(json.load(f)[f'{t}/test']) * 100
    with open(f"results/relbert_prediction_base/{t}.json") as f:
        relbert_result_base[t] = mean(json.load(f)[f'{t}/test']) * 100
    # load fasttext result
    with open(f"results/fasttext_prediction/{t}.json") as f:
        fasttext_result[t] = mean(json.load(f)["full"]) * 100
model_size = {
    "bert-large-cased": [335, "BERT"],
    "bert-base-cased": [110, "BERT"],
    "roberta-large": [335, "RoBERTa"],
    "roberta-base": [110, "RoBERTa"],
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
    "google/flan-ul2": [20000, "Flan-UL2"],
    "EleutherAI/gpt-neo-125M": [125, "GPT-J"],
    "EleutherAI/gpt-neo-1.3B": [1300, "GPT-J"],
    "EleutherAI/gpt-neo-2.7B": [2700, "GPT-J"],
    "EleutherAI/gpt-j-6B": [6000, "GPT-J"],
    "EleutherAI/gpt-neox-20b": [20000, "GPT-J"],
    "facebook/opt-30b": [30000, "OPT"],
    "facebook/opt-iml-30b": [30000, "OPT-IML"],
    "facebook/opt-iml-max-30b": [30000, "OPT-IML"]
}


def main(model_dict, lm_target):
    df = df_full[[i in model_dict for i in df_full['model']]]
    df['lm'] = [model_dict[i][1] for i in df['model']]
    df['Model Size'] = [model_dict[i][0] * 1000000 for i in df['model']]
    # single analogy
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    grids = list(product(range(3), range(3)))
    styles = ['o-', 'o--', 'o:', 's-', 's--', 's:', '^-', '^--', '^:', "X-", "X--", "X:"]
    for n, _data in enumerate(data_names.keys()):
        # for n, (_data, df_target) in enumerate(df.groupby('data')):
        df_target = df[df['data'] == _data]
        seed(1)
        colors = list(mpl.colormaps['tab20b'].colors)
        shuffle(colors)
        ax = axes[grids[n][0], grids[n][1]]
        if grids[n][0] != 2:
            xlabel = ''
        else:
            xlabel = 'Model Size'

        relbert_base_accuracy = None
        relbert_accuracy = None
        fasttext_accuracy = mean([v for k, v in fasttext_result.items() if k == _data])
        out = df_target.pivot_table(index='Model Size', columns='lm', aggfunc='mean')
        out.columns = [i[1] for i in out.columns]
        out = out.reset_index()
        tmp = out[['Model Size', lm_target[0]]].dropna().reset_index()
        tmp['Accuracy'] = tmp[lm_target[0]]

        # random guess
        r = mean([v for k, v in random_guess.items() if k in df_target['data'].unique()])
        df_rand = pd.DataFrame([{"Model Size": df_target['Model Size'].min(), "Accuracy": r},
                                {"Model Size": df_target['Model Size'].max(), "Accuracy": r}])
        df_rand.plot.line(y='Accuracy', x='Model Size', xlabel=xlabel, ax=ax, color='black', style='-', label="Random",
                          logx=True, grid=True)
        if fasttext_accuracy is not None:
            df_fasttext = pd.DataFrame(
                [{"Model Size": df_target['Model Size'].min(), "Accuracy": fasttext_accuracy},
                 {"Model Size": df_target['Model Size'].max(), "Accuracy": fasttext_accuracy}])
            df_fasttext.plot.line(y='Accuracy', x='Model Size', xlabel=xlabel, ax=ax, color=colors.pop(-1), style="--",
                                  label="FastText", logx=True, grid=True)

        if relbert_accuracy is not None:
            df_relbert = pd.DataFrame([
                {"Model Size": 335 * 1000000, "Accuracy": relbert_accuracy},
                {"Model Size": 110 * 1000000, "Accuracy": relbert_base_accuracy}
            ])
            df_relbert.plot.line(y='Accuracy', x='Model Size', xlabel=xlabel, ax=ax, color=colors.pop(-1), style="*:",
                                 label="RelBERT", logx=True, grid=True)

        for m, c in enumerate(lm_target):
            tmp = out[['Model Size', c]].dropna().reset_index()
            tmp['Accuracy'] = tmp[c]
            tmp.plot.line(y='Accuracy', x='Model Size', xlabel=xlabel, ax=ax, color=colors[m],
                          style="P" if len(tmp) == 1 else styles[m], label=c, logx=True, grid=True)

        if grids[n][0] == 0 and grids[n][1] == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend().remove()

        ax.title.set_text(data_names[_data])

    plt.tight_layout()
    plt.savefig(f"results/figures/main_no_relbert.curve.subplots.png", bbox_inches="tight", dpi=600)


if __name__ == '__main__':
    main(model_size, ["BERT", "RoBERTa", 'GPT-2', 'GPT-J', 'OPT', 'OPT-IML', 'T5', 'Flan-T5', "Flan-UL2"])
