""" Solving Analogy via Instruction Prompting
Given an analogy question, consisting of a query pair and a set of candidate pairs, we convert each candidate to a
sentence by the following instruction.
```
Which one of the following is an analogy?
1) A is to B what a_1 is to b_1
2) A is to B what a_2 is to b_2
3) A is to B what a_3 is to b_3
```
For recurrent LMs, we add `The correct answer is` at the end of the instruction. Output is the correct analogy
statement (eg. `A is to B what a_2 is to b_2`).

We won't run the instruction approach on the new analogy questions (trex/conceptnet/nell) as they can be
too long for some models.
"""
import json
import logging
import os
import gc
from typing import List
import torch
import lmppl
import pandas as pd
from datasets import load_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
template = "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>"
instruction_header = "Which one of the following is an analogy?"
instruction_footer = "The correct answer is"
analogy_types = [
    ['sat_metaphor', '0'],
    ['sat_metaphor', '1'],
    ['sat_metaphor', '2'],
    ['sat_metaphor', None],
    ['sat', None],
    ['sat_full', None],
    ['u2', None],
    ['u4', None],
    ['google', None],
    ['bats', None],
]
language_models = {
    "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, 8],  # 11B
    "google/flan-t5-xl": [lmppl.EncoderDecoderLM, 16],  # 3B
    "google/flan-t5-large": [lmppl.EncoderDecoderLM, 256],  # 770M
    "google/flan-t5-base": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "google/flan-t5-small": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "facebook/opt-iml-max-1.3b": [lmppl.LM, 32],  # 1.3B
    "facebook/opt-iml-1.3b": [lmppl.LM, 32],  # 1.3B
    "facebook/opt-1.3b": [lmppl.LM, 32],  # 1.3B
    "facebook/opt-350m": [lmppl.LM, 128],  # 350M
    "facebook/opt-125m": [lmppl.LM, 256],  # 125M
    "t5-11b": [lmppl.EncoderDecoderLM, 4],  # 11B
    "t5-3b": [lmppl.EncoderDecoderLM, 16],  # 3B
    "t5-large": [lmppl.EncoderDecoderLM, 128],  # 770M
    "t5-base": [lmppl.EncoderDecoderLM, 512],  # 220M
    "t5-small": [lmppl.EncoderDecoderLM, 512],  # 60M
    "EleutherAI/gpt-j-6B": [lmppl.LM, 16],  # 6B
    "EleutherAI/gpt-neo-2.7B": [lmppl.LM, 8],  # 2.7B
    "EleutherAI/gpt-neo-1.3B": [lmppl.LM, 32],  # 1.3B
    "EleutherAI/gpt-neo-125M": [lmppl.LM, 64],  # 125M
    "gpt2-xl": [lmppl.LM, 32],  # 1.5B
    "gpt2-large": [lmppl.LM, 128],  # 774M
    "gpt2-medium": [lmppl.LM, 256],  # 355M
    "gpt2": [lmppl.LM, 512],  # 124M
}

# Add Large Models
language_models.update({
    "EleutherAI/gpt-neox-20b": [lmppl.LM, 1],  # 20B
    "facebook/opt-iml-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-iml-max-30b": [lmppl.LM, 1],  # 30B
    "facebook/opt-30b": [lmppl.LM, 1],  # 30B
    "google/flan-ul2": [lmppl.EncoderDecoderLM, 1],  # 20B
})


def get_input(query_pair: List, candidate_pairs: List, encoder_decoder: bool):
    tmp = template.replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    candidate_tmp = [tmp.replace('<subj-b>', a).replace('<obj-b>', b) for a, b in candidate_pairs]
    body = "\n".join([f"{n+1}) {t}" for n, t in enumerate(candidate_tmp)])
    body = f"{instruction_header}\n{body}"
    if not encoder_decoder:
        body = f"{body}\n{instruction_footer}"
    return [[body, t] for t in candidate_tmp]


def analogy_solver(scoring_model, data_name, batch_size: int, scores_texts, data_prefix: str):

    is_encdec = type(scoring_model) is lmppl.EncoderDecoderLM
    # dataset setup
    dataset = load_dataset('relbert/analogy_questions', data_name, split='test')
    if data_prefix is not None:
        dataset = dataset.filter(lambda x: x['prefix'] == data_prefix)
        assert len(dataset) > 0

    # prompt data
    dataset_prompt = [get_input(x['stem'], x['choice'], encoder_decoder=is_encdec) for x in dataset]
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    if scores_texts is None:
        if is_encdec:
            scores = scoring_model.get_perplexity(
                input_texts=[x[0] for x in dataset_flat],
                output_texts=[x[1] for x in dataset_flat],
                batch=batch_size)
            scores_texts = [{"input": x[0], "output": x[1]} for x in dataset_flat]
        else:
            scores = scoring_model.get_perplexity(
                input_texts=[f"{x[0]} {x[1]}" for x in dataset_flat],
                batch=batch_size)
            scores_texts = [{"input": f"{x[0]} {x[1]}", "output": ""} for x in dataset_flat]
        for i, s in zip(scores_texts, scores):
            i['score'] = float(s)
    scores = [x['score'] for x in scores_texts]
    index_score = list(zip(dataset_index, scores))
    scores_aligned = [(i, [b for a, b in index_score if a == i]) for i in sorted(list(set(dataset_index)))]
    prediction = [i[1].index(min(i[1])) if len(set(i[1])) > 1 else None for i in scores_aligned]

    # compute accuracy
    df_tmp = dataset.to_pandas()
    df_tmp['choice'] = [[_i.tolist() for _i in i] for i in df_tmp['choice']]
    df_tmp['prediction'] = prediction
    df_tmp['accuracy'] = df_tmp['prediction'] == df_tmp['answer']
    return df_tmp, scores_texts


if __name__ == '__main__':
    os.makedirs('results/breakdown', exist_ok=True)
    os.makedirs('results/scores', exist_ok=True)

    results = []
    for target_model in language_models.keys():

        scorer = None
        lm_class, batch = language_models[target_model]

        for target_data, prefix in analogy_types:

            score_file = f"results/scores/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.json"
            breakdown_file = f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv"
            if not os.path.exists(breakdown_file):

                _scores_texts = None
                if os.path.exists(score_file):
                    with open(score_file) as f:
                        _scores_texts = json.load(f)

                if scorer is None:

                    # model setup
                    if lm_class is lmppl.MaskedLM:
                        scorer = lm_class(target_model, max_length=256)
                    else:
                        scorer = lm_class(target_model, device_map='auto', low_cpu_mem_usage=True)

                _df, _scores_texts = analogy_solver(scorer, target_data, batch_size=batch, data_prefix=prefix, scores_texts=_scores_texts)
                _df.to_csv(breakdown_file, index=False)

                if _scores_texts is not None:
                    with open(score_file, 'w') as f:
                        json.dump(_scores_texts, f)
            else:
                _df = pd.read_csv(breakdown_file)

            results.append(
                {
                    'accuracy': _df['accuracy'].mean(),
                    'model': target_model,
                    'prefix': prefix,
                    'data': target_data,
                }
            )
            print(target_data, prefix, target_model, _df['accuracy'].mean())
            assert _df['prediction'].isnull().sum() == 0, _df['prediction'].isnull().sum()

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv('results/full_result.instruction.csv', index=False)
    df = df[[i != "sat_metaphor" for i in df['data']]]
    df = df[[i != "sat" for i in df['data']]]
    df.groupby("model")['accuracy'].mean().to_csv('results/full_result.instruction.average.csv')

