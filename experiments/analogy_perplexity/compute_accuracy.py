import os
import argparse
import json
from statistics import mean
from glob import glob
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', '--output-dir', help='output dir', default='output', type=str)
parser.add_argument('-e', '--export-file', help='export file', default='output/accuracy.csv', type=str)
opt = parser.parse_args()
full_result = []
for i in glob(f"{opt.output_dir}/*.json"):
    basename = os.path.basename(i)
    _, model, data, prompt, _ = basename.split('.')
    with open(i) as f:
        ppl = json.load(f)
    result = {"model": model, 'data': data, 'prompt': prompt}
    for _split, list_ppl in ppl.items():
        accuracy = []
        for single_entry in list_ppl:
            prediction = single_entry['ppl'].index(min(single_entry['ppl']))
            accuracy.append(int(int(single_entry['answer']) == prediction))
        result[f'accuracy ({_split})'] = mean(accuracy) * 100
    full_result.append(result)
df = pd.DataFrame(full_result)
df.to_csv(opt.export_file, index=False)
# with open(opt.export_file, 'w') as f:
#     f.write()




