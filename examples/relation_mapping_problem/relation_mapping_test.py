import os
import json
from itertools import permutations
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-4)


def mean(_list):
    return sum(_list)/len(_list)


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    with open('data.jsonl') as f_reader:
        data = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]

    accuracy_full = {}
    for m in ['relbert', 'fasttext_cc']:
        accuracy = []
        for data_id, _data in enumerate(data):
            print(f'[{m}]: {data_id}/{len(data)}')
            with open(f'embeddings/{m}.vector.{data_id}.json') as f:
                embedding_dict = json.load(f)
            sim_file = f'embeddings/{m}.sim.{data_id}.json'
            sim = {}
            if os.path.exists(sim_file):
                with open(sim_file) as f:
                    sim = json.load(f)

            source = _data['source']
            target = _data['target']
            perms = []
            for n, tmp_target in tqdm(list(enumerate(permutations(target, len(target))))):
                list_sim = []
                for id_x, id_y in permutations(range(len(target)), 2):
                    _id = f'{source[id_x]}__{source[id_y]} || {tmp_target[id_x]}__{tmp_target[id_y]}'
                    if _id not in sim:
                        sim[_id] = cosine_similarity(
                            embedding_dict[f'{source[id_x]}__{source[id_y]}'],
                            embedding_dict[f'{tmp_target[id_x]}__{tmp_target[id_y]}']
                        )
                        with open(sim_file, 'w') as f_writer:
                            json.dump(sim, f_writer)
                    list_sim.append(sim[_id])
                perms.append({'target': tmp_target, 'similarity_mean': mean(list_sim)})
            pred = sorted(perms, key=lambda _x: _x['similarity_mean'], reverse=True)
            accuracy.extend([t == p for t, p in zip(target, pred[0]['target'])])
        accuracy_full[m] = mean(accuracy)
    print(json.dumps(accuracy_full, indent=4))
