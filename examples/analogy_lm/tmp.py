import json
from glob import glob


data_input = {}
for i in glob("results/scores/gpt2_*.prompt.json"):
    data = "_".join(i.split('gpt2_')[1:])
    # print(data)
    with open(i) as f:
        scores = json.load(f)
    data_input[data] = [s['input'] for s in scores]


for i in glob("results/scores/*.prompt.json"):
    data = "_".join(i.split('_')[1:])
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
            # data_input[data] = [s['input'] for s in scores]
            # print(scores)
            print(i)
            # input()
            for a, b in zip(scores, data_input[data]):
                a['input'] = b
            # print(scores)
            # input('save')
            with open(i, 'w') as f_write:
                json.dump(scores, f_write)


