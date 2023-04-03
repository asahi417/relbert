import json
import os
from glob import glob
from pprint import pprint
from relbert import evaluate_classification

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

with open("result.jsonl", "w") as f:
    f.write("\n".join([json.dumps(i) for i in full_result]))

for i in full_result:
    i['lexical_relation_classification/BLESS']

REL_ID="part_whole"
REL_ID="class_inclusion"

#	Attribute
REL_ID="attribute"

#	Contrast
REL_ID="contrast"

#	Similar
REL_ID="similar"