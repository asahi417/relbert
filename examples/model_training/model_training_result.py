import json
import os
import requests

import pandas as pd

TMP_DIR = 'metric_files'


def download(filename, url):
    try:
        with open(f'{TMP_DIR}/{filename}') as f_reader:
            return json.load(f_reader)
    except Exception:
        print(f'download {url}')
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(f'{TMP_DIR}/{filename}', "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
    with open(f'{TMP_DIR}/{filename}') as f_reader:
        return json.load(f_reader)


def get_result(language_model: str = 'roberta-large', random_seed: int = 0):
    output = []
    for loss in ['nce', 'triplet', "iloob"]:
        for prompt in ['a', 'b', 'c', 'd', 'e']:
            model = f'relbert-{language_model}-{loss}-{prompt}-{random_seed}'

            config = download(
                f"config-{model}.json",
                f"https://huggingface.co/relbert/{model}/raw/main/finetuning_config.json")
            result = {"template_id": prompt, "model": config['model'], 'loss_function': config['loss_function']}
            # result['validation_loss'] = download(
            #     f"loss-{model}.json",
            #     f"https://huggingface.co/relbert/{model}/raw/main/loss.json")['loss']

            tmp_result = {}
            for _type in ['forward', 'reverse', 'bidirection']:
                tmp_result[_type] = download(
                    f"analogy-validation-{model}.{_type}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/analogy_relation_dataset.{_type}.json"
                )[f"{config['data']}/validation"]
            tmp_result['mean'] = (tmp_result["forward"] + tmp_result["reverse"]) / 2
            for k, v in tmp_result.items():
                result[f"validation_analogy.{k}"] = v

            _type = 'forward'
            result.update({f"{k}": v for k, v in download(
                f"analogy-{model}.json",
                f"https://huggingface.co/relbert/{model}/raw/main/analogy.{_type}.json"
            ).items() if 'test' in k or k == "sat_full"})

            result.update({os.path.basename(k): v['test/f1_micro'] for k, v in download(
                f"classification-{model}.json",
                f"https://huggingface.co/relbert/{model}/raw/main/classification.json"
            ).items()})

            result.update({'relation_mapping_accuracy': download(
                f"relation_mapping-{model}.json",
                f"https://huggingface.co/relbert/{model}/raw/main/relation_mapping.json"
            )['accuracy']})

            # for _type in ['forward', 'reverse', 'bidirection']:
            #     result.update({f"{k}.{_type}": v for k, v in download(
            #         f"analogy-{model}.json",
            #         f"https://huggingface.co/relbert/{model}/raw/main/analogy.{_type}.json"
            #     ).items() if 'test' in k or k == "sat_full"})

            output.append(result)
    df = pd.DataFrame(output)
    return df


full_output = get_result()
full_output.to_csv('examples/model_training/result.csv', index=False)
