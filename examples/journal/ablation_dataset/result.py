import json
import os
import requests
from typing import List
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


def get_result(language_model: str = 'roberta-large', data_list: str or List = 'semeval2012', loss: str = 'nce', aggregate: str = None):
    output = []
    data_list = [data_list] if type(data_list) is str else data_list
    data = "-".join(data_list)
    model = f'relbert-{language_model}-{loss}-{data}'
    if aggregate is not None:
        model = f'relbert-{language_model}-{loss}-{data}-{aggregate}'

    config = download(
        f"config-{model}.json",
        f"https://huggingface.co/relbert/{model}/raw/main/finetuning_config.json")
    result = {"data": data, "model": config['model'], 'loss_function': config['loss_function'], "aggregate": aggregate if aggregate is not None else "average_no_mask"}

    for _type in ['forward']:
        result.update({f"{k}.{_type}": v for k, v in download(
            f"analogy-{model}.json",
            f"https://huggingface.co/relbert/{model}/raw/main/analogy.{_type}.json"
        ).items()})

    result.update({os.path.basename(k): v['test/f1_micro'] for k, v in download(
        f"classification-{model}.json",
        f"https://huggingface.co/relbert/{model}/raw/main/classification.json"
    ).items()})

    # result.update({'relation_mapping_accuracy': download(
    #     f"relation_mapping-{model}.json",
    #     f"https://huggingface.co/relbert/{model}/raw/main/relation_mapping.json"
    # )['accuracy']})

    output.append(result)
    return output


if __name__ == '__main__':
    full_output = []
    full_output += get_result(language_model="roberta-base", data_list='semeval2012-0')
    full_output += get_result(language_model="roberta-base", data_list='conceptnet')
    full_output += get_result(language_model="roberta-base", data_list='t-rex')
    full_output += get_result(language_model="roberta-base", data_list='nell')
    df = pd.DataFrame(full_output)
    df.to_csv('result.csv', index=False)
    df = df[[c for c in df.columns if 'valid' not in c]]
    df.pop("sat/test.forward")
    df = df[df['data'] == 't-rex']
    df.pop("data")
    df.pop("model")
    df.pop("loss_function")
    df.pop("aggregate")
    print((df.T * 100).round(1))
    print((df[[c for c in df.columns if 'test' in c]].mean(1)* 100).round(1))
    print((df[[c for c in df.columns if c[0].isupper()]].mean(1) * 100).round(1))

