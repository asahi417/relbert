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


def get_result(language_model: str = 'roberta-large', data: str = 'semeval2012', loss='nce'):
    output = []
    for prompt in ['a', 'b', 'c', 'd', 'e']:
        model = f'relbert-{language_model}-{loss}-{prompt}-{data}'

        config = download(
            f"config-{model}.json",
            f"https://huggingface.co/relbert/{model}/raw/main/finetuning_config.json")
        result = {"data": data, "template_id": prompt, "model": config['model'], 'loss_function': config['loss_function']}

        for _type in ['forward']:
        # for _type in ['forward', 'reverse', 'bidirection']:
            result.update({f"{k}.{_type}": v for k, v in download(
                f"analogy-{model}.json",
                f"https://huggingface.co/relbert/{model}/raw/main/analogy.{_type}.json"
            ).items()})

        result.update({os.path.basename(k): v['test/f1_micro'] for k, v in download(
            f"classification-{model}.json",
            f"https://huggingface.co/relbert/{model}/raw/main/classification.json"
        ).items()})

        result.update({'relation_mapping_accuracy': download(
            f"relation_mapping-{model}.json",
            f"https://huggingface.co/relbert/{model}/raw/main/relation_mapping.json"
        )['accuracy']})

        output.append(result)
    return output


if __name__ == '__main__':
    full_output = []
    full_output += get_result(data='semeval2012', loss='triplet')
    full_output += get_result(data='semeval2012', loss='iloob')
    full_output += get_result(data='semeval2012')
    full_output += get_result(data='t-rex')
    full_output += get_result(data='conceptnet')
    full_output += get_result(data='nell')
    full_output += get_result(data='semeval2012-nell')
    full_output += get_result(data='semeval2012-t-rex')
    full_output += get_result(data='semeval2012-nell-t-rex')

    df = pd.DataFrame(full_output)
    df.to_csv('result.csv', index=False)
