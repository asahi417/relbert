from itertools import chain
from statistics import mean
import pandas as pd
from datasets import load_dataset


def pretty(num): return "{:,}".format(num)


data_names = {
    'bats': "BATS",
    'google': "Google",
    'sat_full': "SAT",
    'u2': "U2",
    'u4': "U4",
    "t_rex_relational_similarity": "T-REX",
    "conceptnet_relational_similarity": "ConceptNet",
    "nell_relational_similarity": "NELL",
    'scan': "SCAN"
}

out = []
for data_name in data_names.keys():
    data = load_dataset("relbert/analogy_questions", data_name)
    out.append({
        "Dataset": data_names[data_name],
        "Average Number of Choices": round(mean(list(chain(*[[len(x['choice']) for x in data[i]] for i in data])))),
        "Number of Questions": "/".join([pretty(len(data[i])) if i in data else "-" for i in ['validation', 'test']]),
        "Number of Types": len(set(list(chain(*[data[i]["prefix"] for i in data])))),
    })
df = pd.DataFrame(out)
print(df.to_latex(index=False))