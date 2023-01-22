import json
import os
from glob import glob
import pandas as pd

out = []
for i in glob("relbert_output/ckpt/*/*/model/analogy.json"):
    with open(i) as f:
        tmp = json.load(f)
    with open(f"{os.path.dirname(i)}/loss.json") as f:
        tmp['loss'] = json.load(f)['loss']
    with open(f"{os.path.dirname(i)}/classification.json") as f:
        result = json.load(f)
        for n in ["BLESS", "CogALexV", "EVALution", "K&H+N", "ROOT09"]:
            tmp["classification/{n}/f1_macro"] = result[f"lexical_relation_classification/{n}"]["test/f1_macro"]
            tmp["classification/{n}/f1_micro"] = result[f"lexical_relation_classification/{n}"]["test/f1_micro"]
    out.append(tmp)

df = pd.DataFrame(out)
df.pop("distance_function")
df.pop("aggregation")
df.pop("template")
df = df.sort_values(by=["loss"])
df.to_csv("examples/summary.csv", index=False)
