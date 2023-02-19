import os
import json
import pandas as pd
from glob import glob

root_dir = 'relbert_output/ckpt'

output = []
for i in glob(f"{root_dir}/*"):
    for t in glob(f"{i}/template-*"):
        for k in glob(f"{t}/*/analogy.forward.json"):
            with open(k) as f:
                data = json.load(f)
            data['template'] = os.path.basename(t)
            data['model'] = os.path.basename(os.path.dirname(k))
            data['data'] = os.path.basename(os.path.basename(i))
            data['path'] = os.path.dirname(k)
            output.append(data)
df = pd.DataFrame(output)

for (data, template), g in df.groupby(["data", "template"]):
    print(data, template)
    data = data.replace("nce_combined.", "").split(".")
    g['objective'] = sum(g[f'{i}/validation'] for i in data) / len(data)
    result = g.sort_values(by='objective', ascending=False).head(1)
    print(result['objective'].values[0], result['model'].values[0])
    print(result[[c for c in result.columns if c.endswith('test')]].T)
    print()
    input()

