""" Preprocess ConceptNet
"product": 519,
"capital": 459,
"MannerOf": 12715,
"PartOf": 13077,
"genus": 2937,
"influencedBy": 1273,
"knownFor": 607,
"SimilarTo": 30280,
"EtymologicallyRelatedTo": 32075,
"CapableOf": 22677,
"CausesDesire": 4688,
"HasA": 5545,
"EtymologicallyDerivedFrom": 71,
"HasContext": 232935,
"IsA": 230137,
"HasProperty": 8433,
"genre": 3824,
"leader": 84,
"HasLastSubevent": 2874,
"NotCapableOf": 329,
"HasPrerequisite": 22710,
"RelatedTo": 1703582,
"occupation": 1043,
"FormOf": 378859,
"DerivedFrom": 325374,
"SymbolOf": 4,
"Antonym": 19066,
"HasFirstSubevent": 3347,
"Entails": 405,
"field": 643,
"Synonym": 222156,
"UsedFor": 39790,
"DefinedAs": 2173,
"MotivatedByGoal": 9489,
"CreatedBy": 263,
"None": 19066,
"HasSubevent": 25238,
"language": 916,
"AtLocation": 27797,
"LocatedNear": 49,
"InstanceOf": 1480,
"NotHasProperty": 327,
"Desires": 3170,
"NotDesires": 2886,
"Causes": 16801,
"MadeOf": 545,
"ReceivesAction": 6037,
"DistinctFrom": 3315
"""
import os
import json
from tqdm import tqdm
from glob import glob
from datasets import load_dataset

export_dir = './conceptnet_dataset'
os.makedirs(export_dir, exist_ok=True)
dataset = load_dataset("conceptnet5", "conceptnet5", split="train")
dataset = dataset.filter(lambda example: example['lang'] == 'en')
dataset = dataset.filter(lambda example: example['rel'] != 'None')
dataset = dataset.sort('rel')
relations = list(set(dataset["rel"]))

stats = {}
for r in tqdm(relations):
    _dataset = dataset.filter(lambda example: example['rel'] == r)
    stats[r] = len(_dataset)
    with open(f"{export_dir}/cache_{os.path.basename(r)}.jsonl", 'w') as f:
        f.write('\n'.join([
            json.dumps({'rel': i['rel'], 'arg1': i['arg1'], 'arg2': i['arg2'], 'sentence': i['sentence']})
            for i in _dataset]))

cur_relation_type = None
f = None
for i in tqdm(dataset):
    if cur_relation_type is None or cur_relation_type != i['rel']:
        cur_relation_type = i['rel']
        if f is not None:
            f.close()
        f = open(f"{export_dir}/cache_{os.path.basename(cur_relation_type)}.jsonl", 'w')
    f.write(json.dumps({
        'rel': i['rel'],
        'arg1': i['arg1'],
        'arg2': i['arg2'],
        'sentence': i['sentence']
    }) + '\n')
if f is not None:
    f.close()

# get statistics
table = {}
for i in glob('{}/*.jsonl'.format(export_dir)):
    r_type = os.path.basename(i).replace('.jsonl', '')
    with open(i) as f:
        data = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    table[r_type] = len(data)
print(json.dumps(table, indent=4))
