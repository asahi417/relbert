from itertools import permutations
import os
from random import shuffle, seed

import pandas as pd
from datasets import load_dataset
from lmppl import EncoderDecoderLM

model_name = "google/flan-t5-xxl"
# model_name = "google/flan-t5-small"
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
        feature_list += [[n, i['answer']] for _ in range(len(i['choice']))]
    ppl = scorer.get_perplexity(input_texts=input_list, output_texts=output_list, batch=batch_size)
    return pd.DataFrame([{"ppl": p, "input": i, "output": o, "index": d[0], "answer": d[1]} for p, i, o, d in zip(ppl, input_list, output_list, feature_list)])


if __name__ == '__main__':
    export_dir = 'results'
    score_model = None
    os.makedirs(export_dir, exist_ok=True)
    for n_seed, k_shot in zip([1, 5], [3, 3]):
        for s in range(n_seed):
            print(f"[{k_shot}-shot]: seed {n_seed}")
            ppl_file = f"{export_dir}/ppl.{model_name}.{k_shot}.{s}.csv"
            if not os.path.exists(ppl_file):
                if score_model is None:
                    score_model = EncoderDecoderLM(model_name)
                p = get_prompt(k_shot, s)
                ppl_df = solve_analogy(p, score_model)

