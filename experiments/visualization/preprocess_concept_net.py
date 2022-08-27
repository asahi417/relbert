import os
import json
from tqdm import tqdm
from glob import glob
from datasets import load_dataset

export_dir = './data/conceptnet'
os.makedirs(export_dir, exist_ok=True)
dataset = load_dataset("conceptnet5", "conceptnet5", split="train")
dataset = dataset.filter(lambda example: example['lang'] == 'en')
dataset = dataset.sort('rel')

cur_relation_type = None
f = None
for i in tqdm(dataset):
    if cur_relation_type is None or cur_relation_type != i['rel']:
        cur_relation_type = i['rel']
        if f is not None:
            f.close()
        f = open(f'{export_dir}/cache_{os.path.basename(cur_relation_type)}.jsonl', 'w')
    f.write(json.dumps({
        'rel': i['rel'],
        'arg1': i['arg1'],
        'arg2': i['arg2'],
        'sentence': i['sentence']
    }) + '\n')
f.close()

# get statistics
table = {}
for i in glob(f'{export_dir}/*.jsonl'):
    r_type = os.path.basename(i).replace('.jsonl', '')
    with open(i) as f:
        data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    table[r_type] = len(data)
print(json.dumps(table, indent=4))
with open(f'data/conceptnet_stats.csv', 'w') as f:
    json.dump(table, f)
