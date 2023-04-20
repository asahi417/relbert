import json
import os
from glob import glob
from pprint import pprint

import pandas as pd

from relbert import evaluate_classification

pretty_name = {
    "part_whole": "Meronym",
    "class_inclusion": "Hypernym",
    "attribute": "Attribute",
    "contrast": "Antonym",
    "similar": "Synonym",
    'random': 'Random',
    'attri': "Attribute",
    'ANT': 'Antonym',
    'RANDOM': 'Random',
    'Antonym': 'Antonym',
    'false': 'Random',
    'Synonym': 'Synonym',
    'SYN': 'Synonym',
    'mero': "Meronym",
    'HYPER': "Hypernym",
    'IsA': "Hypernym",
    'hyper': "Hypernym",
    'event': "Event",
    'hypo': "Co-hyponym",
    'COORD': "Co-hyponym",
    'sibl': "Co-hyponym",
    'coord': "Co-hyponym",
    'PART_OF': "Meronym",
    'PartOf': "Meronym",
    'MadeOf': "Meronym",
    'HasProperty': "Attribute",
    'HasA': "Possession",
    "Full": "Full"
}
full_result = []
if os.path.exists("result.jsonl"):
    with open("result.jsonl") as f:
        full_result += [json.loads(i) for i in f.read().split("\n") if len(i) > 0]

done_experiment = [i["target"] for i in full_result]
for i in glob("relbert_output/ckpt/exclusion*"):
    experiment_type = os.path.basename(i)
    if experiment_type in done_experiment:
        continue
    accuracy = []
    for x in glob(f"{i}/*/analogy.forward.json"):
        with open(x) as f:
            a = json.load(f)["semeval2012_relational_similarity/validation"]
        accuracy.append({"accuracy": a, "model": os.path.dirname(x)})
    if len(accuracy) != 10:
        continue
    best_model = sorted(accuracy, key=lambda _x: _x["accuracy"], reverse=True)[0]['model']
    out = evaluate_classification(relbert_ckpt=best_model, max_length=64, batch_size=512)
    out.update({"target": experiment_type})
    full_result.append(out)
    pprint(full_result[-1])

if "Full" not in done_experiment:
    out = evaluate_classification(relbert_ckpt="relbert/relbert-roberta-base-nce-a-semeval2012", max_length=64, batch_size=512)
    out.update({"target": "Full"})
    full_result.append(out)

with open("result.jsonl", "w") as f:
    f.write("\n".join([json.dumps(i) for i in full_result]))

output = []
for i in full_result:
    for data in ["BLESS", "CogALexV", "EVALution", "K&H+N", "ROOT09"]:
        target_relation = pretty_name[i['target'].replace("exclusion_", "")]
        f1_breakdown = {os.path.basename(k): v for k, v in i[f'lexical_relation_classification/{data}'].items() if k.startswith('test/f1')}
        f1_breakdown = {pretty_name[k] if not k.startswith('f1_') else k: v for k, v in f1_breakdown.items()}
        f1_breakdown.update({"target": target_relation, "data": data})
        output.append(f1_breakdown)
df = pd.DataFrame(output)
for data, g in df.groupby("data"):
    g.pop("data")
    g.index = g.pop("target")
    g.index.name = None
    g = (g[[i for i in g.columns if not i.startswith("f1_")] + [i for i in g.columns if i.startswith("f1_")]] * 100).round(1)
    g = g.sort_index()
    g = g.T
    g = g[[i for i in g.columns if i != "Full"] + ["Full"]]
    g = g.dropna()
    # g = g.fillna("-")

    print()
    print()
    print(data)
    print(g.to_latex())
    input()

