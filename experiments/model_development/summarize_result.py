import json
import os
import requests

import pandas as pd


MODEL = "roberta-large"
METHODS = ["average", "mask", "average-no-mask"]
LOSS = ["nce", "triplet", 'loob']
DATA = ["semeval2012-v2", "semeval2012", "conceptnet-hc"]
PROMPT = ["a", "b", "c", "d", "e"]

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
            if l != 'nce' and d != 'semeval2012':
                continue
            for p in PROMPT:
                for m in METHODS:
                    v_loss = f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/validation_loss.json"
                    v_loss_cn = f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/validation_loss.conceptnet_high_confidence.json"
                    result = {k: v for k, v in download(
                        f"analogy-{MODEL}-{d}-{m}-{p}-{l}.json",
                        f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/analogy.json"
                    ).items() if 'valid' not in k}
                    result.update({
                        "loss": l,
                        "data": d,
                        "prompt": p,
                        "method": m,
                        "loss_value": download(
                            f"loss-{MODEL}-{d}-{m}-{p}-{l}.json",
                            v_loss)['validation_loss']
                    })
                    result.update({k: v['test/f1_micro'] for k, v in download(
                        f"classification-{MODEL}-{d}-{m}-{p}-{l}.json",
                        f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/classification.json"
                    ).items()})
                    metric = download(
                        f"relation_mapping-{MODEL}-{d}-{m}-{p}-{l}.json",
                        f"https://huggingface.co/relbert/relbert-{MODEL}-{d}-{m}-prompt-{p}-{l}/raw/main/relation_mapping.json"
                    )
                    result.update({'relation_mapping_accuracy': metric['accuracy']})
                    if d != 'conceptnet-hc' and l != 'triplet':
                        result.update(
                            {"loss_value_conceptnet (validation)": download(
                                f"loss-{MODEL}-{d}-{m}-{p}-{l}.cn.json",
                                v_loss_cn)['validation_loss']}
                        )
                        result.update(
                            {"loss_value_conceptnet (train)": download(
                                f"loss-{MODEL}-{d}-{m}-{p}-{l}.cn.json",
                                v_loss_cn)['train_loss']}
                        )
                        result.update(
                            {"loss_value_conceptnet (all)": download(
                                f"loss-{MODEL}-{d}-{m}-{p}-{l}.cn.json",
                                v_loss_cn)['train_validation_loss']}
                        )
                    output.append(result)
    return pd.DataFrame(output)


full_output = get_result()
full_output.to_csv('summary.csv')
