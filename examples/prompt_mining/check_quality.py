import json
import math


with open('cache/template.score.jsonl') as f:
    data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    for d in data:
        score = d['scores']['score']
        d['score_mean'] = sum(score)/len(score)
        var = sum((l - d['score_mean']) ** 2 for l in score) / len(score)
        d['score_std'] = math.sqrt(var)
        d['score_lowest'] = min(score)
        d['score_highest'] = max(score)

data_mean = sorted(data, key=lambda x: x['score_mean'])
data_std = sorted(data, key=lambda x: x['score_std'])
data_low = sorted(data, key=lambda x: x['score_lowest'])
data_high = sorted(data, key=lambda x: x['score_highest'], reverse=True)

for i in range(10):
    print(f"top {i}")
    print(f"\t mean: {data_mean[i]['template']}")
    print(f"\t std : {data_std[i]['template']}")
    print(f"\t low : {data_low[i]['template']}")
    print(f"\t high: {data_high[i]['template']}")

for i in range(10):
    print(f"worst {i}")
    print(f"\t mean: {data_mean[-1 -i]['template']}")
    print(f"\t std : {data_std[-1 -i]['template']}")
    print(f"\t low : {data_low[-1 -i]['template']}")
    print(f"\t high: {data_high[-1 -i]['template']}")

with open('cache/template.mean.top10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_mean[:10]]))
with open('cache/template.std.top10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_std[:10]]))
with open('cache/template.low.top10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_low[:10]]))
with open('cache/template.high.top10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_high[:10]]))

with open('cache/template.mean.bottom10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_mean[-10:]]))
with open('cache/template.std.bottom10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_std[-10:]]))
with open('cache/template.low.bottom10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_low[-10:]]))
with open('cache/template.high.bottom10.csv', 'w') as f:
    f.write('\n'.join([i['template'] for i in data_high[-10:]]))
