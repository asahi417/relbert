""" Solving Metaphor Detection via Prompting """
import json
import logging
import os
from typing import List
import lmppl
from datasets import load_dataset

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
all_datasets = [
    'sat_full',
    # 'u2', 'u4', 'google', 'bats',
    # 't_rex_relational_similarity', 'conceptnet_relational_similarity', 'nell_relational_similarity', 'scan'
]


def get_input(query_pair: List, candidate_pairs: List):
    text = "Which one of the following is an analogy?\n"
    text += "\n".join([f'{n + 1}) {query_pair[0]} is to {query_pair[1]} what {c} is to {d}' for n, (c, d) in
                       enumerate(candidate_pairs)])
    text += "\nThe correct answer is"
    return [f"{text} {n + 1}." for n in range(len(candidate_pairs))]


def get_ppl(model, data_name):
    scorer = lmppl.OpenAI(OPENAI_API_KEY, model=model)
    dataset = load_dataset("relbert/analogy_questions", data_name, split="test")
    dataset_prompt = [get_input(x['stem'], x['choice']) for x in dataset]
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    scores = scorer.get_perplexity(dataset_flat)
    scores = [{"input": f"{d}", "output": "", "score": s, "index": i} for s, d, i in zip(scores, dataset_flat, dataset_index)]
    scores_aligned = [(i, [s['score'] for s in scores if s['index'] == i]) for i in sorted(list(set(dataset_index)))]
    df_tmp = dataset.to_pandas()
    df_tmp['choice'] = [[_i.tolist() for _i in i] for i in df_tmp['choice']]
    df_tmp['prediction'] = [i.index(min(i)) for _, i in scores_aligned]
    df_tmp['accuracy'] = df_tmp['prediction'] == df_tmp['answer']
    return df_tmp, scores_aligned


if __name__ == '__main__':
    os.makedirs('results/scores', exist_ok=True)
    os.makedirs('results/breakdown', exist_ok=True)

    # compute perplexity
    for target_model in ['davinci']:
        for target_data in all_datasets:
            scores_file = f"results/scores/{target_model}_{target_data}_None.instruction.json"
            breakdown_file = f"results/breakdown/{target_model}_{target_data}_None.instruction.csv"
            if not os.path.exists(scores_file):
                logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data}`")
                _df_tmp, _scores_aligned = get_ppl(target_model, target_data)
                with open(scores_file, 'w') as f:
                    json.dump(_scores_aligned, f)
                _df_tmp.to_csv(breakdown_file, index=False)
