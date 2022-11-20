import json
import os
import requests

import pandas as pd


MODEL = "roberta-large"
METHODS = ["average", "mask", "average-no-mask"]
LOSS = ["nce", "triplet", 'loob']
DATA = ["semeval2012", "conceptnet-hc"]
PROMPT = ["a", "b", "c", "d", "e"]
VALIDATE = [True, False]
CLASSIFICATION = [True, False]

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
                    for c in CLASSIFICATION:
                        for v in VALIDATE:
                            suffix = "-classification" if c else ""
                            suffix += "-conceptnet-validated" if v else ""
                            v_loss = f"https://huggingface.co/relbert/{MODEL}-{d}-{m}-prompt-{p}-{l}{suffix}/raw/main/validation_loss.json"
                            try:
                                loss = download(
                                    f"loss-{MODEL}-{d}-{m}-{p}-{l}{suffix}.json",
                                    v_loss)
                                result = {k: v for k, v in download(
                                    f"analogy-{MODEL}-{d}-{m}-{p}-{l}{suffix}.json",
                                    f"https://huggingface.co/relbert/{MODEL}-{d}-{m}-prompt-{p}-{l}{suffix}/raw/main/analogy.json"
                                ).items() if 'valid' not in k}
                                result.update({
                                    "loss": l,
                                    "data": d,
                                    "prompt": p,
                                    "method": m,
                                    "classification_loss": c,
                                    "validation": "conceptnet" if v else None,
                                    "loss_value": loss['loss'],
                                })
                                result.update({k: v['test/f1_micro'] for k, v in download(
                                    f"classification-{MODEL}-{d}-{m}-{p}-{l}{suffix}.json",
                                    f"https://huggingface.co/relbert/{MODEL}-{d}-{m}-prompt-{p}-{l}{suffix}/raw/main/classification.json"
                                ).items()})
                                metric = download(
                                    f"relation_mapping-{MODEL}-{d}-{m}-{p}-{l}{suffix}.json",
                                    f"https://huggingface.co/relbert/{MODEL}-{d}-{m}-prompt-{p}-{l}{suffix}/raw/main/relation_mapping.json"
                                )
                                result.update({'relation_mapping_accuracy': metric['accuracy']})
                                output.append(result)
                            except Exception:
                                print(f"error:{v_loss}")
    return pd.DataFrame(output)


full_output = get_result()
full_output.to_csv('summary.csv')