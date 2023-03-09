""" Solve multi choice analogy task by word embedding model """
import json
import zipfile
import requests
import os
from statistics import mean
from datasets import load_dataset
from gensim.models import fasttext


def cos_similarity(a_, b_):
    inner = (a_ * b_).sum()
    norm_a = (a_ * a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    return inner / (norm_b * norm_a)


# load fasttext
os.makedirs('./cache_fasttext', exist_ok=True)
path = './cache/crawl-300d-2M-subword.bin'
if not os.path.exists(path):
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    _path = f"{cache_dir}/{filename}"
    assert path.endswith('.zip')
    with open(_path, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    with zipfile.ZipFile(_path, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    os.remove(_path)
model = fasttext.load_facebook_model(path)

# load dataset
for t in ["scan", "sat_full", "u2", "u4", "bats", "google", "t_rex_relational_similarity", "nell_relational_similarity", "conceptnet_relational_similarity"]:
    path = f"results/fasttext_prediction/{t}.json"
    if os.path.exists(path):
        continue
    os.makedirs("results/fasttext_prediction", exist_ok=True)
    data = load_dataset("relbert/analogy_questions", t, split="test")
    accuracy = []

    for i in data:
        q = model[i['stem'][0]] - model[i['stem'][1]]
        c = [model[x] - model[y] for x, y in i['choice']]
        sim = [cos_similarity(_c, q) for _c in c]
        pred = sim.index(max(sim))
        accuracy.append(pred == i['answer'])

    print(t, mean(accuracy))
    with open(path, "w") as f:
        json.dump({"full": accuracy, "mean": mean(accuracy)}, f)
