import pandas as pd

data_names = {
    'sat_full': "SAT",
    'u2': "U2",
    'u4': "U4",
    'bats': "BATS",
    'google': "Google",
    'scan': "SCAN",
    "nell_relational_similarity": "NELL",
    "t_rex_relational_similarity": "T-REX",
    "conceptnet_relational_similarity": "ConceptNet",
}
model_names = {
    'bert-base-cased': ['BERT\textsubscript{BASE}', 110],
    'bert-large-cased': ['BERT\textsubscript{LARGE}', 335],
    'roberta-base': ['RoBERTa\textsubscript{BASE}', 110],
    'roberta-large': ['RoBERTa\textsubscript{LARGE}', 335],
    'gpt2': ['GPT-2\textsubscript{SMALL}', 124],
    'gpt2-medium': ['GPT-2\textsubscript{BASE}', 355],
    'gpt2-large': ['GPT-2\textsubscript{LARGE}', 774],
    'gpt2-xl': ['GPT-2\textsubscript{1.5B}', 1500],
    'EleutherAI/gpt-neo-125M': ['GPT-J\textsubscript{125M}', 125],
    'EleutherAI/gpt-neo-1.3B': ['GPT-J\textsubscript{1.3B}', 1300],
    'EleutherAI/gpt-neo-2.7B': ['GPT-J\textsubscript{2.7B}', 2700],
    'EleutherAI/gpt-j-6B': ['GPT-J\textsubscript{6B}', 6000],
    'EleutherAI/gpt-neox-20b': ['GPT-J\textsubscript{20B}', 20000],
    'facebook/opt-125m': ['OPT\textsubscript{125M}', 125],
    'facebook/opt-350m': ['OPT\textsubscript{350M}', 350],
    'facebook/opt-1.3b': ['OPT\textsubscript{1.3B}', 1300],
    'facebook/opt-30b': ['OPT\textsubscript{30B}', 30000],
    'facebook/opt-iml-1.3b': ['OPT-IML\textsubscript{1.3B}', 1300],
    'facebook/opt-iml-30b': ['OPT-IML\textsubscript{30B}', 30000],
    'facebook/opt-iml-max-1.3b': ['OPT-IML-MAX\textsubscript{1.3B}', 1300],
    'facebook/opt-iml-max-30b': ['OPT-IML\textsubscript{30B}', 30000],
    't5-small': ['T5\textsubscript{SMALL}', 60],
    't5-base': ['T5\textsubscript{BASE}', 220],
    't5-large': ['T5\textsubscript{LARGE}', 770],
    't5-3b': ['T5\textsubscript{3B}', 3000],
    't5-11b': ['T5\textsubscript{11B}', 11000],
    'google/flan-t5-small': ['Flan-T5\textsubscript{SMALL}', 60],
    'google/flan-t5-base': ['Flan-T5\textsubscript{BASE}', 220],
    'google/flan-t5-large': ['Flan-T5\textsubscript{LARGE}', 770],
    'google/flan-t5-xl': ['Flan-T5\textsubscript{3B}', 3000],
    'google/flan-t5-xxl': ['Flan-T5\textsubscript{11B}', 11000],
    'google/flan-ul2': ['Flan-UL2\textsubscript{20B}', 20000]
}

df_lms = pd.read_csv("../analogy_lm/results/full_result.prompt.csv")
df_lms.pop("prefix")
df_lms['accuracy'] = (100 * df_lms['accuracy']).round(1)
df_lms = df_lms[[i in data_names for i in df_lms['data']]]
df_lms = df_lms[[i in model_names for i in df_lms['model']]]
df_lms['data'] = [data_names[i] for i in df_lms['data']]
df_lms['model'] = [model_names[i][0] for i in df_lms['model']]
df_lms['model'] = [model_names[i][0] for i in df_lms['model']]

pv = df_lms.pivot_table(index="model", columns="data", values="accuracy")
pv = pv[list(data_names.values())]
pv = pv.T[list(model_names.values())].T
pv['Average'] = pv.mean(axis=1).round(1)
pv = pv.round(1)
pv.columns.name = "Model"
pv.index.name = None
print(pv.to_latex(escape=False))


df = pd.DataFrame([{v[0]: k for k, v in model_names.items()}]).T
df.columns = ["Name on HuggingFace"]
df.columns.name = "Model"
print(df.to_latex(escape=False))
