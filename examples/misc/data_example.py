import pandas as pd
from datasets import load_dataset


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

samples = []
for k, v in data_names.items():
    data = load_dataset("relbert/analogy_questions_private", k, split='test')
    df = data.to_pandas()

    def get_sample(_df):
        return _df['stem'].values[0].tolist() + _df['choice'].values[0][_df['answer'].values[0]].tolist()

    samples += [{"data": v, "prefix": p, "sample": get_sample(g.head(1))} for p, g in df.groupby('prefix')]
df = pd.DataFrame(samples)
df.to_csv("analogy_samples.csv", index=False)