from statistics import mean
from datasets import load_dataset
import pandas as pd


def pretty(num): return str(round(num, 1))

data = [
    ["relbert/semeval2012_relational_similarity", "Relational Similarity", "True", "Semantic Relation"],
    ["relbert/conceptnet_relational_similarity", "ConceptNet", "False", "Commonsense Reasoning"],
    ["relbert/t_rex_relational_similarity", "T-REX", "False", "Encyclopedic Knowledge"],
    ["relbert/nell_relational_similarity", "NELL-One", "False", "Encyclopedic Knowledge"]
]
out = []
for d, name, parent, _type in data:

    if d == "relbert/t_rex_relational_similarity":
        x = load_dataset(d, "filter_unified.min_entity_4_max_predicate_10")
    else:
        x = load_dataset(d)
    if "test" in x:
        r = set(x['train']['relation_type'])
        r.update(set(x['validation']['relation_type']))
        r.intersection(set(x['test']['relation_type']))
        ratio = pretty(len(r.intersection(set(x["test"]["relation_type"])))/len(r) * 100)
    else:
        ratio = "-"
    out.append({
        "Dataset": name,
        "Number of Relations": " / ".join([str(len(x[split])) if split in x else "-" for split in ['train', 'validation', 'test']]),
        "Average Number of Triples": " / ".join([pretty(mean([len(i) for i in x[split]['positives']])) if split in x else "-" for split in ['train', 'validation', 'test']]),
        "Hierarchy": parent,
        "Type": _type,
        "Test/Train Overlap": ratio
    })
df = pd.DataFrame(out)
print(df.T.to_latex(index=False))