import json
import logging
import os
import gc
from typing import List
import torch
import lmppl
import pandas as pd
from datasets import load_dataset
from itertools import product

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
template = "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>"
template_partial = "<subj> is to <obj>"
instruction = ["Which one of the following is an analogy?", "The correct answer is"]
cot = 'Answer the following question by reasoning step-by-step.'
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
    ['t_rex_relational_similarity', None],
    ['conceptnet_relational_similarity', None],
    ['nell_relational_similarity', None]
]

language_models = {
    "gpt2": [lmppl.LM, 64],  # 124M
    "gpt2-medium": [lmppl.LM, 64],  # 355M
    "gpt2-large": [lmppl.LM, 32],  # 774M
    "gpt2-xl": [lmppl.LM, 16],  # 1.5B
    "EleutherAI/gpt-j-6B": [lmppl.LM, 1],  # 6B
    "facebook/opt-125m": [lmppl.LM, 64],  # 125M
    "facebook/opt-350m": [lmppl.LM, 64],  # 350M
    "facebook/opt-1.3b": [lmppl.LM, 16],  # 1.3B
    "facebook/opt-iml-1.3b": [lmppl.LM, 16],  # 1.3B
    "facebook/opt-iml-max-1.3b": [lmppl.LM, 16],  # 1.3B
    "t5-small": [lmppl.EncoderDecoderLM, 64],  # 60M
    "t5-base": [lmppl.EncoderDecoderLM, 64],  # 220M
    "t5-large": [lmppl.EncoderDecoderLM, 32],  # 770M
    "t5-3b": [lmppl.EncoderDecoderLM, 16],  # 3B
    "t5-11b": [lmppl.EncoderDecoderLM, 1],  # 11B
    "google/flan-t5-small": [lmppl.EncoderDecoderLM, 64],  # 60M
    "google/flan-t5-base": [lmppl.EncoderDecoderLM, 64],  # 220M
    "google/flan-t5-large": [lmppl.EncoderDecoderLM, 32],  # 770M
    "google/flan-t5-xl": [lmppl.EncoderDecoderLM, 32],  # 3B
    "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, 1],  # 11B
    "google/switch-base-128": [lmppl.EncoderDecoderLM, 1],  # 220M
}
# language_models.update({
#     "roberta-base": [lmppl.MaskedLM],  # 110M
#     "roberta-large": [lmppl.MaskedLM],  # 355M
#     "microsoft/deberta-v3-xsmall": [lmppl.MaskedLM],  # 70M
#     "microsoft/deberta-v3-small": [lmppl.MaskedLM, 64],  # 142M
#     "microsoft/deberta-v3-base": [lmppl.MaskedLM, 64],  # 184M
#     "microsoft/deberta-v3-large": [lmppl.MaskedLM, 32],  # 434M
#     "microsoft/deberta-v2-xlarge": [lmppl.MaskedLM, 8],  # 900M
#     "microsoft/deberta-v2-xxlarge": [lmppl.MaskedLM, 4],  # 1.5B
# })
# language_models.update({
#     "EleutherAI/gpt-neox-20b": [lmppl.LM, 1],  # 20B
#     "facebook/opt-30b": [lmppl.LM, 1],  # 30B
#     "facebook/opt-iml-30b": [lmppl.LM, 1],  # 30B
#     "facebook/opt-iml-max-30b": [lmppl.LM, 1],  # 30B
#     "google/switch-large-128": [lmppl.EncoderDecoderLM, 2],  # 770M
# })


def get_input(query_pair: List, candidate_pairs: List, output_index: bool, use_cot: bool = False):
    tmp = template.replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    tmp = "\n".join([f"{n+1}) {tmp.replace('<subj-b>', a).replace('<obj-b>', b)}" for n, (a, b) in enumerate(candidate_pairs)])
    tmp = f"{instruction[0]}\n{tmp}\n{instruction[1]}"
    if use_cot:
        tmp = f"{cot}\n{tmp}"
    if output_index:
        return [[tmp, str(n+1)] for n, _ in enumerate(candidate_pairs)]
    return [[tmp, template_partial.replace('<subj>', a).replace('<obj>', b)] for a, b in candidate_pairs]


def analogy_solver(scoring_model, data_name, use_cot: bool, output_index: bool, batch_size: int, scores_texts=None, data_prefix: str = None):

    # dataset setup
    dataset = load_dataset('relbert/analogy_questions', data_name, split='test')
    if data_prefix is not None:
        dataset = dataset.filter(lambda x: x['prefix'] == data_prefix)
        assert len(dataset) > 0

    # prompt data
    dataset_prompt = [get_input(x['stem'], x['choice'], use_cot=use_cot, output_index=output_index) for x in dataset]
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    if scores_texts is None:
        if type(scoring_model) is lmppl.EncoderDecoderLM:
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
    df = dataset.to_pandas()
    df['choice'] = [[_i.tolist() for _i in i] for i in df['choice']]
    df['prediction'] = prediction
    df['accuracy'] = df['prediction'] == df['answer']
    return df, scores_texts


if __name__ == '__main__':
    os.makedirs('results/breakdown', exist_ok=True)
    os.makedirs('results/scores', exist_ok=True)

    results = []
    for target_model in language_models.keys():
        scorer = None
        lm_class, batch = language_models[target_model]

        for (target_data, prefix), _use_cot, _output_index in product(analogy_types, [True, False], [True, False]):

            score_file = f"results/scores/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.{_use_cot}.{_output_index}.json"
            breakdown_file = f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.{_use_cot}.{_output_index}..csv"
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

                _df, _scores_texts = analogy_solver(
                    scorer, target_data, batch_size=batch, data_prefix=prefix, scores_texts=_scores_texts, use_cot=_use_cot, output_index=_output_index)
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
                    'approach': 'prompt',
                    'prefix': prefix,
                    'data': target_data,
                    "cot": _use_cot,
                    "output_index": _output_index
                }
            )
            print(target_data, prefix, target_model, _df['accuracy'].mean())
            assert _df['prediction'].isnull().sum() == 0, _df['prediction'].isnull().sum()

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv('results/full_result.instruction.csv', index=False)
