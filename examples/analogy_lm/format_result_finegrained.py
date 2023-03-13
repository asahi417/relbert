import json
import os
from glob import glob
from statistics import mean
import pandas as pd
from format_result import plot, model_size

os.makedirs('results/figures/detail', exist_ok=True)


if __name__ == '__main__':
    output = []
    for m_full in model_size:
        for i in glob(f"results/breakdown/{os.path.basename(m_full)}*.csv"):
            df = pd.read_csv(i)
            data = '_'.join(i.split("_None")[0].split("_")[1:])
            if 'sat' in data:
                continue
            if data == 'bats':
                df['prefix'] = df['prefix'].apply(lambda x: os.path.dirname(x.replace("./cache/BATS_3.0/", "")).split("_")[-1])
            elif data == 'google':
                df['prefix'] = df['prefix'].apply(lambda x: 'Morphological' if 'gram' in x else "Semantic")
            elif data == 'nell_relational_similarity':
                df['prefix'] = df['prefix'].apply(lambda x: x.replace("concept:", ""))

            with open(f"results/relbert_prediction/{data}.json") as f:
                df['relbert'] = json.load(f)[f'{data}/test']
            with open(f"results/relbert_prediction_base/{data}.json") as f:
                df['relbert_base'] = json.load(f)[f'{data}/test']

            with open(f"results/fasttext_prediction/{data}.json") as f:
                df['fasttext'] = json.load(f)["full"]

            for prefix, g in df.groupby("prefix"):
                if len(g) >= 10:
                    output.append({
                        'model': m_full,
                        "data": data,
                        'accuracy': g['accuracy'].mean(),
                        'prefix': prefix,
                        "random": mean([1/len(i.split("], [")) for i in df['choice']]),
                        "relbert": g['relbert'].mean(),
                        "relbert_base": g['relbert_base'].mean(),
                        "fasttext": g['fasttext'].mean(),
                    })

    df = pd.DataFrame(output)
    for (data, prefix), g in df.groupby(["data", 'prefix']):
        print(data, prefix)
        g['lm'] = [model_size[i][1] for i in g['model']]
        g['Model Size'] = [model_size[i][0] * 1000000 for i in g['model']]
        plot(g,
             f"results/figures/detail/{data}.{prefix.replace(' ', '_').replace('/', '_').replace(':', '_')}.png",
             ['RoBERTa', 'GPT-2', 'GPT-J', 'OPT', 'OPT-IML', 'T5', 'T5 (FT)', 'Flan-T5', "Flan-T5 (FT)", "Flan-UL2"],
             relbert_accuracy=g.pop("relbert").values[0],
             relbert_base_accuracy=g.pop("relbert_base").values[0],
             fasttext_accuracy=g.pop("fasttext").values[0],
             r=g.pop("random").values[0],
             legend_out=True)

