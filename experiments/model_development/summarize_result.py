import json
import os
import requests

import pandas as pd


MODEL = "roberta-large"
METHODS = ["average", "mask", "average-no-mask"]
# METHODS = ["average", "average-no-mask"]
LOSS = ["nce", "triplet"]
DATA = ["semeval2012"]
PROMPT = ["a", "b", "c", "d"]
# PROMPT = ["a", "b", "c", "d", "e"]

TMP_DIR = 'metric_files'
EXPORT_DIR = 'output'


def download(filename, url):
    print(f'download {url}')
    try:
        with open(f'{TMP_DIR}/{filename}') as f_reader:
            json.load(f_reader)
    except Exception:
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(f'{TMP_DIR}/{filename}', "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
    with open(f'{TMP_DIR}/{filename}') as f_reader:
        tmp = json.load(f_reader)
    return tmp


def get_result():
    output = []
    for l in LOSS:
        for d in DATA:
            for p in PROMPT:
                for m in METHODS:
                    v_loss = f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/validation_loss.json"
                    result = download(
                        f"analogy-{MODEL}-{d}-{m}-{p}-{l}.json",
                        f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/analogy.json"
                    )
                    result.update({
                        "loss": l,
                        "data": d,
                        "prompt": p,
                        "method": m,
                        "loss_value": download(
                            f"loss-{MODEL}-{d}-{m}-{p}-{l}.json",
                            v_loss)['validation_loss']})
                    output.append(result)
    return pd.DataFrame(output)


full_output = get_result()
full_output.to_csv('./summary.csv')
