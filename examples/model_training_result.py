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
    for loss in ['nce', 'triplet']:
        for prompt in ['a', 'b', 'c', 'd', 'e']:
            model = f'relbert-{language_model}-{loss}-{prompt}-{random_seed}'
            try:
                config = download(
                    f"config-{model}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/finetuning_config.json")

                for _type in ['forward', 'reverse', 'bidirection']:
                    download(
                        f"analogy-validation-{model}.{_type}.json",
                        f"https://huggingface.co/relbert/{model}/raw/main/analogy_relation_dataset.{_type}.json"
                    )[f"{config['data']}/test"]

                loss_value = download(
                    f"loss-{model}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/loss.json")['loss']
                relation_classification = {k: v['test/f1_micro'] for k, v in download(
                    f"classification-{model}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/classification.json"
                ).items()}
                analogy = {k: v for k, v in download(
                    f"analogy-{model}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/analogy.json"
                ).items() if 'valid' not in k}


                result = {'template': config['template'], "template_id": prompt, "model": config['model']}
                result.update({'loss': })
                result.update({k: v for k, v in download(
                    f"analogy-{model}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/analogy.json"
                ).items() if 'valid' not in k})
                result.update()
                result.update({'relation_mapping_accuracy': download(
                    f"relation_mapping-{model}.json",
                    f"https://huggingface.co/relbert/{model}/raw/main/relation_mapping.json"
                )['accuracy']})
                output.append(result)
            except Exception:
                print(model)
    df = pd.DataFrame(output)
    print(df.columns)
    df.pop('distance_function')
    return df


full_output = get_result()
os.makedirs('summary', exist_ok=True)
full_output.to_csv('summary/roberta_base.parameter_search.csv', index=False)
