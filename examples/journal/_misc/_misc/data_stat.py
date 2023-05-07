from statistics import mean
from itertools import chain
from datasets import load_dataset
import pandas as pd


def pretty(num): return str(round(num, 1))

data = [
    ["relbert/semeval2012_relational_similarity", "Relational Similarity", "True", "Concepts"],
    ["relbert/conceptnet_relational_similarity", "ConceptNet", "False", "Concepts"],
    ["relbert/t_rex_relational_similarity", "T-REX", "False", "Named entities"],
    ["relbert/nell_relational_similarity", "NELL-One", "False", "Named entities"]
]
out = []
for d, name, parent, _type in data:
    x = load_dataset(d)
    if d == "relbert/semeval2012_relational_similarity":
        x = load_dataset(d)
        num = {}
        for _s in ['train', 'validation']:
            num_pos = [len(i['positives']) for i in x[_s]]
            tmp = {k: len(list(chain(*[_x['positives'] for _x in x[_s] if _x['relation_type'].startswith(k)]))) for k in set([i.split("/")[0] for i in x[_s]['relation_type']])}
            num_pos += list(tmp.values())
            num[_s] = num_pos

        out.append({
            "Dataset": name,
            "#relations": f"{len(num['train'])}/{len(num['validation'])}/-",
            '#unique positive examples': "/".join([str(len(set([".".join(i) for i in chain(*list(x[split]['positives']))]))) for split in ['train', 'validation']])+"/-",
            "Average #positive examples per relation": f"{round(mean(num['train']), 1)}/{round(mean(num['validation']), 1)}/-",
            "Relation Hierarchy": parent,
            "Domain": _type,
            "Test/Train Overlap": '-'
        })
    else:
        r = set(x['train']['relation_type'])
        r.update(set(x['validation']['relation_type']))
        r.intersection(set(x['test']['relation_type']))
        ratio = pretty(len(r.intersection(set(x["test"]["relation_type"])))/len(r) * 100)
        for _s in ['train', 'validation']:
            print(sum([len(i['positives']) for i in x[_s]]), _s, d)
        out.append({
            "Dataset": name,
            "#relations": "/".join([str(len(x[split])) for split in ['train', 'validation', 'test']]),
            '#unique positive examples': "/".join([str(len(set([".".join(i) for i in chain(*list(x[split]['positives']))]))) for split in ['train', 'validation', 'test']]),
            "Average #positive examples per relation": "/".join([pretty(mean([len(i) for i in x[split]['positives']])) for split in ['train', 'validation', 'test']]),
            "Relation Hierarchy": parent,
            "Domain": _type,
            "Test/Train Overlap": ratio
        })
df = pd.DataFrame(out)
df.index = df.pop("Dataset")
print(df.T.to_latex())