import json
import os
import requests

import pandas as pd


# MODEL = "roberta-base"
# # METHODS = ["average", "mask", "average-no-mask"]
# METHODS = ["average", "mask"]
# # LOSS = ["nce", "triplet", 'loob']
# LOSS = ["nce", "loob"]
# # DATA = ["semeval2012-v3", "semeval2012-v4", "semeval2012-v5"]
# DATA = ["semeval2012-v4", "semeval2012-v5"]
# PROMPT = ["a", "b", "c", "d", "e"]
# # CLASSIFICATION = [True, False]
# CLASSIFICATION = [False]
#
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
    loss = 'nce'
    language_model = 'roberta-base'
    for data in ['semeval2012-v4', 'semeval2012-v5']:
        for aggregate in ['average', 'mask']:
            for prompt in ['a', 'b', 'c', 'd', 'e']:
                for seed in range(3):
                    model = f'{language_model}-{data}-{aggregate}-prompt-{prompt}-{loss}-{seed}'
                    try:
                        result = {"prompt": prompt, "method": aggregate, 'data': data, 'seed': seed}
                        result.update({'loss': download(
                            f"loss-{model}.json",
                            f"https://huggingface.co/relbert/{model}/raw/main/validation_loss.json")['loss']})
                        result.update({k: v for k, v in download(
                            f"analogy-{model}.json",
                            f"https://huggingface.co/relbert/{model}/raw/main/analogy.json"
                        ).items() if 'valid' not in k})
                        result.update({k: v['test/f1_micro'] for k, v in download(
                            f"classification-{model}.json",
                            f"https://huggingface.co/relbert/{model}/raw/main/classification.json"
                        ).items()})
                        result.update({'relation_mapping_accuracy': download(
                            f"relation_mapping-{model}.json",
                            f"https://huggingface.co/relbert/{model}/raw/main/relation_mapping.json"
                        )['accuracy']})
                        output.append(result)
                    except Exception:
                        print(model)
    df = pd.DataFrame(output)
    df.pop('distance_function')
    return df


full_output = get_result()
os.makedirs('summary', exist_ok=True)
full_output.to_csv('summary/roberta_base.parameter_search.csv', index=False)

