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
    out.append(tmp)

df = pd.DataFrame(out)
df.pop("distance_function")
df.pop("aggregation")
df.pop("template")
df = df.sort_values(by=["loss"])
df.to_csv("examples/summary.csv", index=False)