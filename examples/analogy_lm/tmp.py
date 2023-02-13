import json
from glob import glob


data_input = {}
for i in glob("results/scores/gpt2_*.prompt.json"):
    data = i.split('gpt2_')[1].split('.prompt.json')[0]
    with open(i) as f:
        scores = json.load(f)
    data_input[data] = [s['input'] for s in scores]


for i in glob("results/scores/*.prompt.json"):
    data = i.split('_')[-1].split('.prompt.json')[0]
    with open(i) as f:
        scores = json.load(f)
        flag = True
        for s in scores:

            if s['output'] != "":
                flag = False
                continue

            if 'what' in s['input']:
                flag = False
                continue

        if flag:
            print(i)
            data_input[data] = [s['input'] for s in scores]
            for a, b in zip(scores, data_input[data]):
                a['input'] = b
            with open(i, 'w') as f_write:
                json.dump(scores, f_write)


