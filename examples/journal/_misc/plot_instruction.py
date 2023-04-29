import os
import matplotlib.pyplot as plt
import pandas as pd

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


df_full_i = pd.read_csv('results/full_result.instruction.csv').sort_values(by=['model', 'data'])
df_full_p = pd.read_csv('../main_lm/results/full_result.prompt.csv').sort_values(by=['model', 'data'])
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
