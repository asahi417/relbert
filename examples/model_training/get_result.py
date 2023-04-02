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
    for prompt in ['a', 'b', 'c', 'd', 'e']:
        model = f'relbert-{language_model}-{loss}-{prompt}-{data}'
        if aggregate is not None:
            model = f'relbert-{language_model}-{loss}-{prompt}-{data}-{aggregate}'

        config = download(
            f"config-{model}.json",
            f"https://huggingface.co/relbert/{model}/raw/main/finetuning_config.json")
        result = {"data": data, "template_id": prompt, "model": config['model'], 'loss_function': config['loss_function'], "aggregate": aggregate if aggregate is not None else "average_no_mask"}

        for _type in ['forward']:
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
    df_tmp = pd.DataFrame(output)
    tmp = sum([df_tmp[f"{d.replace('-', '_')}_relational_similarity/validation.forward"] for d in data_list]).values.tolist()
    best_config = tmp.index(max(tmp))
    return [df_tmp.T[best_config].T.to_dict()]


if __name__ == '__main__':
    full_output = []
    full_output += get_result(language_model="bert-base", data_list='semeval2012')
    full_output += get_result(language_model="albert-base", data_list='semeval2012')
    full_output += get_result(language_model="roberta-base", data_list='semeval2012')
    full_output += get_result(language_model="roberta-base", data_list='semeval2012', aggregate="mask")
    full_output += get_result(language_model="roberta-base", data_list='semeval2012', aggregate="average")
    full_output += get_result(data_list='semeval2012', loss='triplet')
    full_output += get_result(data_list='semeval2012', loss='iloob')
    full_output += get_result(data_list='semeval2012')
    full_output += get_result(language_model="roberta-base", data_list='t-rex')
    full_output += get_result(language_model="roberta-base", data_list='conceptnet')
    full_output += get_result(language_model="roberta-base", data_list='nell')

    df = pd.DataFrame(full_output)
    print(df)
    df.to_csv('result.csv', index=False)
