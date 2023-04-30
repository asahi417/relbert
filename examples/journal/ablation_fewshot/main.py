import os
from itertools import permutations
from random import shuffle, seed
from time import time
from statistics import mean
import pandas as pd
from datasets import load_dataset
from lmppl import EncoderDecoderLM

model_name = "google/flan-t5-xxl"
batch_size = 1
data_analogy = load_dataset('relbert/analogy_questions', 'sat_full', split='test')
data_fewshot = load_dataset('relbert/semeval2012_relational_similarity', split='train')


def get_prompt(k: int = 5, random_seed: int = 42):
    index_list = list(range(len(data_fewshot)))
    seed(random_seed)
    shuffle(index_list)
    index_list = index_list[:k]
    sample = data_fewshot.select(index_list)
    sample_list = []
    for i in sample:
        index_list = list(permutations(range(len(i['positives'])), 2))
        shuffle(index_list)
        a, b = index_list[0]
        sample_list.append(i['positives'][a] + i['positives'][b])
    return "\n".join([f"{a} is to {b} what {c} is to {d}" for a, b, c, d in sample_list])


def solve_analogy(prompt_prefix: str, scorer):
    feature_list = []
    input_list = []
    output_list = []
    for n, i in enumerate(data_analogy):
        input_list += [f"{prompt_prefix}\n{i['stem'][0]} is to {i['stem'][1]} what"] * len(i['choice'])
        output_list += [f"{x} is to {y}" for x, y in i['choice']]
        feature_list += [[n, i['answer'], o] for o in range(len(i['choice']))]
    ppl = scorer.get_perplexity(input_texts=input_list, output_texts=output_list, batch=batch_size)
    return pd.DataFrame([{"ppl": p, "input": i, "output": o, "index": d[0], "answer": d[1], "choice": d[2]} for p, i, o, d in zip(ppl, input_list, output_list, feature_list)])


if __name__ == '__main__':
    export_dir = 'results'
    score_model = None
    os.makedirs(export_dir, exist_ok=True)
    output = []
    for k_shot, n_seed in zip([10, 5, 1], [5, 5, 5]):
        for s in range(n_seed):
            print(f"[{k_shot}-shot]: seed {s}")
            ppl_file = f"{export_dir}/ppl.{os.path.basename(model_name)}.{k_shot}.{s}.csv"
            if not os.path.exists(ppl_file):
                if score_model is None:
                    score_model = EncoderDecoderLM(model_name)
                p = get_prompt(k_shot, s)
                start = time()
                ppl_df = solve_analogy(p, score_model)
                elapsed = time() - start
                ppl_df.to_csv(ppl_file, index=False)
                print(f"\t {elapsed} seconds")
            df = pd.read_csv(ppl_file)
            accuracy = mean([int(g.sort_values('ppl')['choice'].values[0] == g['answer'].values[0]) for _, g in df.groupby("index")])
            output.append({"k": k_shot, "seed": s, "accuracy": accuracy})
    df = pd.DataFrame(output)
    print(df)
    df.to_csv(f"{export_dir}/result.csv", index=False)

