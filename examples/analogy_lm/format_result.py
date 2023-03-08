import os
from typing import List
from statistics import mean
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle, seed
from datasets import load_dataset


os.makedirs('results/figures', exist_ok=True)
df_full = pd.read_csv('results/full_result.prompt.csv')
df_full['Accuracy'] = df_full.pop('accuracy')  # * 100
df_full = df_full[[i not in ['sat', 'sat_metaphor'] for i in df_full['data']]]
df_full.pop('prefix')

random_guess = {}
for t in ["scan", "sat_full", "u2", "u4", "bats", "google", "t_rex_relational_similarity", "nell_relational_similarity", "conceptnet_relational_similarity"]:
    data = load_dataset("relbert/analogy_questions", t, split="test")
    random_guess[t] = mean([1/len(i['choice']) for i in data])

relbert_accuracy = {
    'sat_full': 0.732620320855615,
    'u2': 0.6754385964912281,
    'u4': 0.6296296296296297,
    'google': 0.952,
    'bats': 0.8093385214007782,
    't_rex_relational_similarity': 0.644808743169399,
    'conceptnet_relational_similarity': 0.4748322147651007,
    'nell_relational_similarity': 0.6583333333333333,
    # "scan": 0.2469059405940594
}
# relbert_accuracy_triplet = {
#     'sat_full': 0.6818181818181818,
#     'sat': 0.6884272997032641,
#     'u2': 0.6798245614035088,
#     'u4': 0.6481481481481481,
#     'google': 0.922,
#     'bats': 0.7837687604224569,
#     't_rex_relational_similarity': 0.5300546448087432,
#     'conceptnet_relational_similarity': 0.3154362416107382,
#     'nell_relational_similarity': 0.6383333333333333
# }
model_size = {
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
    "facebook/opt-iml-max-30b": [30000, "OPT-IML"],
    "relbert/t5-small-analogy": [60, "T5 (FT)"],
    "relbert/t5-base-analogy": [220, "T5 (FT)"],
    "relbert/t5-large-analogy": [770, "T5 (FT)"],
    "relbert/t5-3b-analogy": [3000, "T5 (FT)"],
    "relbert/flan-t5-small-analogy": [60, "Flan-T5 (FT)"],
    "relbert/flan-t5-base-analogy": [220, "Flan-T5 (FT)"],
    "relbert/flan-t5-large-analogy": [770, "Flan-T5 (FT)"],
    "relbert/flan-t5-xl-analogy": [3000, "Flan-T5 (FT)"]
}
model_size_ft_data = {
    "relbert/flan-t5-small-analogy": [60, "SemEval"],
    "relbert/flan-t5-base-analogy": [220, "SemEval"],
    "relbert/flan-t5-large-analogy": [770, "SemEval"],
    "relbert/flan-t5-xl-analogy": [3000, "SemEval"],
    "relbert/flan-t5-small-analogy-t-rex": [60, "T-REX"],
    "relbert/flan-t5-base-analogy-t-rex": [220, "T-REX"],
    "relbert/flan-t5-large-analogy-t-rex": [770, "T-REX"],
    "relbert/flan-t5-xl-analogy-t-rex": [3000, "T-REX"],
    "relbert/flan-t5-small-analogy-nell": [60, "NELL"],
    "relbert/flan-t5-base-analogy-nell": [220, "NELL"],
    "relbert/flan-t5-large-analogy-nell": [770, "NELL"],
    "relbert/flan-t5-xl-analogy-nell": [3000, "NELL"],
    "relbert/flan-t5-small-analogy-conceptnet": [60, "ConceptNet"],
    "relbert/flan-t5-base-analogy-conceptnet": [220, "ConceptNet"],
    "relbert/flan-t5-large-analogy-conceptnet": [770, "ConceptNet"],
    "relbert/flan-t5-xl-analogy-conceptnet": [3000, "ConceptNet"],
}
model_size_ft_perm = {
    "relbert/flan-t5-small-analogy": [60, "Reverse Permutation"],
    "relbert/flan-t5-base-analogy": [220, "Reverse Permutation"],
    "relbert/flan-t5-large-analogy": [770, "Reverse Permutation"],
    "relbert/flan-t5-xl-analogy": [3000, "Reverse Permutation"],
    "relbert/flan-t5-small-analogy-permutation-domain": [60, "In-domain Permutation"],
    "relbert/flan-t5-base-analogy-permutation-domain": [220, "In-domain Permutation"],
    "relbert/flan-t5-large-analogy-permutation-domain": [770, "In-domain Permutation"],
    "relbert/flan-t5-xl-analogy-permutation-domain": [3000, "In-domain Permutation"],
    "relbert/flan-t5-small-analogy-permutation": [60, "Full Permutation"],
    "relbert/flan-t5-base-analogy-permutation": [220, "Full Permutation"],
    "relbert/flan-t5-large-analogy-permutation": [770, "Full Permutation"],
    "relbert/flan-t5-xl-analogy-permutation": [3000, "Full Permutation"],
}


def plot(df_target, path_to_save: str, lm_target: List, no_relbert: bool = False, legend_out: bool = False, r: float = None):

    styles = ['o-', 'o--', 'o:', 's-', 's--', 's:', '^-', '^--', '^:', "P-", "P--", "P:"]
    out = df_target.pivot_table(index='Model Size', columns='lm', aggfunc='mean')
    out.columns = [i[1] for i in out.columns]
    out = out.reset_index()
    tmp = out[['Model Size', lm_target[0]]].dropna().reset_index()
    tmp['Accuracy'] = tmp[lm_target[0]]

    # random guess
    if r is None:
        r = mean([v for k, v in random_guess.items() if k in df_target['data'].unique()])
    df_rand = pd.DataFrame([{"Model Size": df_target['Model Size'].min(), "Accuracy": r}, {"Model Size": df_target['Model Size'].max(), "Accuracy": r}])
    ax = df_rand.plot.line(y='Accuracy', x='Model Size', color='black', style='--', label="Random", logx=True)

    # colors = ['red', 'blue', 'green', 'orange', 'brown', 'purple', 'gray', 'pink', 'olive', 'darkgreen']
    colors = list(mpl.colormaps['tab20b'].colors)
    seed(1)
    shuffle(colors)
    c = colors.pop(0)
    if not no_relbert:
        # relbert result
        df_relbert = pd.DataFrame([{"Model Size": 340 * 1000000, "Accuracy": mean([v for k, v in relbert_accuracy.items() if k in df_target['data'].unique()])}])
        df_relbert.plot.line(y='Accuracy', x='Model Size', ax=ax, color=c, style="*", label="RelBERT", logx=True)

    for n, c in enumerate(lm_target):
        tmp = out[['Model Size', c]].dropna().reset_index()
        tmp['Accuracy'] = tmp[c]
        if len(tmp) == 1:
            tmp.plot.line(y='Accuracy', x='Model Size', ax=ax, color=colors[n], style=styles[n][0], label=c, logx=True)
        else:
            tmp.plot.line(y='Accuracy', x='Model Size', ax=ax, color=colors[n], style=styles[n], label=c, logx=True)
    plt.grid()

    if legend_out:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(path_to_save, bbox_inches="tight", dpi=600)


def main(model_dict, lm_target, prefix, legend_out=False):
    df = df_full[[i in model_dict for i in df_full['model']]]
    df['lm'] = [model_dict[i][1] for i in df['model']]
    df['Model Size'] = [model_dict[i][0] * 1000000 for i in df['model']]
    # average result
    plot(df, f"results/figures/{prefix}.curve.average.png", lm_target, legend_out=legend_out)
    # average over 5 analogies
    plot(
        df[[i in ['sat_full', 'u2', 'u4', 'bats', 'google'] for i in df['data']]],
        f"results/figures/{prefix}.curve.average_5.png",
        lm_target,
        legend_out=legend_out)
    # average over entities analogies
    plot(
        df[[i in ['nell_relational_similarity', 't_rex_relational_similarity'] for i in df['data']]],
        f"results/figures/{prefix}.curve.average_entity.png",
        lm_target,
        legend_out=legend_out)
    # single analogy
    for data, g in df.groupby('data'):
        plot(g, f"results/figures/{prefix}.curve.{data}.png", lm_target, legend_out=legend_out)


if __name__ == '__main__':
    main(model_size, ["RoBERTa", 'GPT-2', 'GPT-J', 'OPT', 'OPT-IML', 'T5', 'T5 (FT)', 'Flan-T5', 'Flan-T5 (FT)', "Flan-UL2"], "main", legend_out=True)
    main(model_size_ft_data, ['SemEval', 'T-REX', 'NELL', 'ConceptNet'], "data")
    main(model_size_ft_perm, ['Reverse Permutation', 'In-domain Permutation', 'Full Permutation'], "perm")
