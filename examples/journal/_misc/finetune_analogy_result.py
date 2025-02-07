""" Solving Analogy via Prompting
Given an analogy question, consisting of a query pair and a set of candidate pairs, we convert each candidate to a
sentence by the prompt `<subj-a> is to <obj-a> what <subj-b> is to <obj-b>`, where we compute perplexity and the one
with the lowest perplexity is regarded as the model prediction. Note that encoder-decoder models such as T5 use
slightly different prompts, where the input to the encoder is `<subj-a> is to <obj-a>` and the output to the decoder
is `<subj-b> is to <obj-b>`.
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
template_header = "<subj-a> is to <obj-a>"
template_join = "what"
template_footer = "<subj-b> is to <obj-b>"
analogy_types = [
    ['sat_full', None],
    ['u2', None],
    ['u4', None],
    ['google', None],
    ['bats', None],
    ['t_rex_relational_similarity', None],
    ['conceptnet_relational_similarity', None],
    ['nell_relational_similarity', None],
    ['scan', None]
]

# Add fine-tuned models
language_models = {
    "relbert/flan-t5-xl-analogy-conceptnet": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/flan-t5-large-analogy-conceptnet": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/flan-t5-base-analogy-conceptnet": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/flan-t5-small-analogy-conceptnet": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/flan-t5-xl-analogy-t-rex": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/flan-t5-large-analogy-t-rex": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/flan-t5-base-analogy-t-rex": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/flan-t5-small-analogy-t-rex": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/flan-t5-xl-analogy-nell": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/flan-t5-large-analogy-nell": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/flan-t5-base-analogy-nell": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/flan-t5-small-analogy-nell": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/flan-t5-xl-analogy": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/flan-t5-large-analogy": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/flan-t5-base-analogy": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/flan-t5-small-analogy": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/flan-t5-xl-analogy-permutation": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/flan-t5-large-analogy-permutation": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/flan-t5-base-analogy-permutation": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/flan-t5-small-analogy-permutation": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/flan-t5-xl-analogy-permutation-domain": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/flan-t5-large-analogy-permutation-domain": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/flan-t5-base-analogy-permutation-domain": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/flan-t5-small-analogy-permutation-domain": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/t5-3b-analogy": [lmppl.EncoderDecoderLM, 16],  # 3B
    "relbert/t5-large-analogy": [lmppl.EncoderDecoderLM, 256],  # 770M
    "relbert/t5-base-analogy": [lmppl.EncoderDecoderLM, 1024],  # 220M
    "relbert/t5-small-analogy": [lmppl.EncoderDecoderLM, 1024],  # 60M
    "relbert/opt-350m-analogy": [lmppl.LM, 128],  # 350M
    "relbert/opt-350m-analogy-permutation": [lmppl.LM, 128],  # 350M
    "relbert/opt-350m-analogy-permutation-domain": [lmppl.LM, 128],  # 350M
    "relbert/opt-125m-analogy": [lmppl.LM, 256],  # 125M
    "relbert/opt-125m-analogy-permutation": [lmppl.LM, 256],  # 125M
    "relbert/opt-125m-analogy-permutation-domain": [lmppl.LM, 256],  # 125M
}


def get_input(query_pair: List, candidate_pairs: List, encoder_decoder: bool = False):
    _template_header = template_header.replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    if encoder_decoder:
        return [[f"generate analogy: {_template_header}", template_footer.replace('<subj-b>', a).replace('<obj-b>', b)] for a, b in candidate_pairs]
    return [[_template_header, template_footer.replace('<subj-b>', a).replace('<obj-b>', b)] for a, b in candidate_pairs]


def analogy_solver(scoring_model, data_name, batch_size, scores_texts, data_prefix):

    # dataset setup
    dataset = load_dataset('relbert/analogy_questions_private', data_name, split='test')
    if data_prefix is not None:
        dataset = dataset.filter(lambda x: x['prefix'] == data_prefix)
        assert len(dataset) > 0

    # prompt data
    dataset_prompt = [get_input(x['stem'], x['choice'], encoder_decoder=type(scoring_model) is lmppl.EncoderDecoderLM) for x in dataset]
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # get scores
    if scores_texts is None:
        if lm_class is lmppl.EncoderDecoderLM:
            scores = scoring_model.get_perplexity(
                input_texts=[x[0] for x in dataset_flat],
                output_texts=[x[1] for x in dataset_flat],
                batch=batch_size
            )
            scores_texts = [{"input": x[0], "output": x[1]} for x in dataset_flat]
        else:
            scores = scoring_model.get_perplexity(
                input_texts=[f"{x[0]} {template_join} {x[1]}" for x in dataset_flat],
                batch=batch_size
            )
            scores_texts = [{"input": f"{x[0]} {template_join} {x[1]}", "output": ""} for x in dataset_flat]
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

            score_file = f"results/scores/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.json"
            breakdown_file = f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.csv"
            if not os.path.exists(breakdown_file):
                _scores_texts = None
                if os.path.exists(score_file):
                    with open(score_file) as f:
                        _scores_texts = json.load(f)

                if scorer is None:
                    # model setup
                    if lm_class is lmppl.MaskedLM:
                        scorer = lm_class(target_model, max_length=256)
                    elif lm_class is lmppl.LM:
                        scorer = lm_class(target_model, max_length=256, device_map='auto', low_cpu_mem_usage=True)
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
                    'data': target_data
                }
            )
            print(target_data, prefix, target_model, _df['accuracy'].mean())
            assert _df['prediction'].isnull().sum() == 0, _df['prediction'].isnull().sum()

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv('results/full_result.ft_analogy_models.prompt.csv', index=False)
    df = df[[i != "sat" for i in df['data']]]
    df.groupby("model")['accuracy'].mean().to_csv('results/full_result.ft_analogy_models.prompt.average.csv')

