"""
Which one of the following is an analogy?
1) `shove` is to `nudge` what `vex` is to `mutter`
2) `shove` is to `nudge` what `teach` is to `lecture`
3) `shove` is to `nudge` what `push` is to `fight`
4) `shove` is to `nudge` what `stare` is to `glance`
The correct answer is

Which one of the following is an analogy?
1) `beauty` is to `aesthete` what `pleasure` is to `hedonist`
2) `beauty` is to `aesthete` what `emotion` is to `demagogue`
3) `beauty` is to `aesthete` what `opinion` is to `sympathizer`
4) `beauty` is to `aesthete` what `seance` is to `medium`
5) `beauty` is to `aesthete` what `luxury` is to `ascetic`
The correct answer is

Which one of the following is an analogy?
1) `story` is to `building` what `crust` is to `sandwich`
2) `story` is to `building` what `shingle` is to `roof`
3) `story` is to `building` what `data` is to `file`
4) `story` is to `building` what `layer` is to `cake`
5) `story` is to `building` what `root` is to `plant`
The correct answer is

Which one of the following is an analogy?
1) ''story'' is to ''building'' what ''crust'' is to ''sandwich''
2) ''story'' is to ''building'' what ''shingle'' is to ''roof''
3) ''story'' is to ''building'' what ''data'' is to ''file''
4) ''story'' is to ''building'' what ''layer'' is to ''cake''
5) ''story'' is to ''building'' what ''root'' is to ''plant''
The correct answer is
"""
import json
import logging
import os
from typing import List

import torch
import lmppl
import pandas as pd
from datasets import load_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

instructions = {
    "instruction-a": ["Which one of the following is an analogy?", "The correct answer is"]
}
templates = {
        'template-a': ["<subj-a> is to <obj-a> what", "<subj-b> is to <obj-b>"],
        'template-b': ["<subj-a> is to <obj-a> as", "<subj-b> is to <obj-b>"],
    }


def get_input(query_pair: List,
              candidate_pairs: List,
              template_type: str = 'template-a',
              instruction_type: str = None,
              encoder_decoder: bool = False):
    template_header = templates[template_type][0].replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    if instruction_type is None:
        if encoder_decoder:
            template_header = ' '.join(template_header.split(' ')[:-1])  # remove the last word
            return [[f"generate analogy: {template_header}", templates[template_type][1].replace('<subj-b>', a).replace('<obj-b>', b)] for a, b in candidate_pairs]
        else:
            return [[template_header, templates[template_type][1].replace('<subj-b>', a).replace('<obj-b>', b)] for a, b in candidate_pairs]
    candidate_prompted = "\n".join([f"{n+1}) {template_header} {templates[template_type][1].replace('<subj-b>', a).replace('<obj-b>', b)}" for n, (a, b) in enumerate(candidate_pairs)])
    header, footer = instructions[instruction_type]
    return [[f"{header}\n{candidate_prompted}\n{footer}", f"{n + 1}"] for n in range(len(candidate_pairs))]


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
    # "roberta-base": [lmppl.MaskedLM, None],  # 110M
    # "roberta-large": [lmppl.MaskedLM, None],  # 355M
    # "microsoft/deberta-v3-xsmall": [lmppl.MaskedLM, None],  # 70M
    # "microsoft/deberta-v3-small": [lmppl.MaskedLM, None, 128],  # 142M
    # "microsoft/deberta-v3-base": [lmppl.MaskedLM, None, 128],  # 184M
    # "microsoft/deberta-v3-large": [lmppl.MaskedLM, None, 64],  # 434M
    # "microsoft/deberta-v2-xlarge": [lmppl.MaskedLM, None, 64],  # 900M
    # "microsoft/deberta-v2-xxlarge": [lmppl.MaskedLM, None, 32],  # 1.5B
    "gpt2": [lmppl.LM, None, 128],  # 124M
    "gpt2-medium": [lmppl.LM, None, 128],  # 355M
    "gpt2-large": [lmppl.LM, None, 64],  # 774M
    "gpt2-xl": [lmppl.LM, None, 16],  # 1.5B
    "facebook/opt-125m": [lmppl.LM, None, 128],  # 125M
    "facebook/opt-350m": [lmppl.LM, None, 128],  # 350M
    "facebook/opt-1.3b": [lmppl.LM, None, 16],  # 1.3B
    # "facebook/opt-30b": [lmppl.LM, torch.float16, 1],  # 30B
    "facebook/opt-iml-1.3b": [lmppl.LM, None, 16],  # 1.3B
    "facebook/opt-iml-max-1.3b": [lmppl.LM, None, 16],  # 1.3B
    # "facebook/opt-iml-30b": [lmppl.LM, torch.float16, 1],  # 30B
    "t5-small": [lmppl.EncoderDecoderLM, None, 128],  # 60M
    "t5-base": [lmppl.EncoderDecoderLM, None, 128],  # 220M
    "t5-large": [lmppl.EncoderDecoderLM, None, 64],  # 770M
    "t5-3b": [lmppl.EncoderDecoderLM, None, 16],  # 3B
    "t5-11b": [lmppl.EncoderDecoderLM, torch.float16, 1],  # 11B
    "google/flan-t5-small": [lmppl.EncoderDecoderLM, None, 128],  # 60M
    "google/flan-t5-base": [lmppl.EncoderDecoderLM, None, 128],  # 220M
    "google/flan-t5-large": [lmppl.EncoderDecoderLM, None, 64],  # 770M
    "google/flan-t5-xl": [lmppl.EncoderDecoderLM, None, 16],  # 3B
    "google/flan-t5-xxl": [lmppl.EncoderDecoderLM, torch.float16, 1],  # 11B
}


def analogy_solver(
        model,
        data_name,
        scores_texts=None,
        data_prefix: str = None,
        template_type: str = 'template-a',
        instruction_type: str = None):

    # dataset setup
    dataset = load_dataset('relbert/analogy_questions', data_name, split='test')
    if data_prefix is not None:
        dataset = dataset.filter(lambda x: x['prefix'] == data_prefix)
        assert len(dataset) > 0

    # prompt data
    dataset_prompt = [get_input(x['stem'], x['choice'], template_type, instruction_type) for x in dataset]
    dataset_index, dataset_flat = [], []
    for n, i in enumerate(dataset_prompt):
        dataset_flat += i
        dataset_index += [n] * len(i)

    # model setup
    lm_class, torch_type, batch = language_models[model]
    scorer = lm_class(model, max_length=256 if lm_class is lmppl.MaskedLM else None, torch_dtype=torch_type)

    # get scores
    if scores_texts is None:
        if lm_class is lmppl.EncoderDecoderLM:
            scores = scorer.get_perplexity(
                input_texts=[x[0] for x in dataset_flat],
                output_texts=[x[1] for x in dataset_flat],
                batch=batch
            )
            input_text = [{"input": x[0], "output": x[1]} for x in dataset_flat]
        else:
            scores = scorer.get_perplexity(
                input_texts=[f"{x[0]} {x[1]}" for x in dataset_flat],
                batch=batch
            )
            input_text = [{"input": f"{x[0]} {x[1]}", "output": ""} for x in dataset_flat]
        for i, s in zip(input_text, scores):
            i['score'] = float(s)
    else:
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

    results = []
    for target_data, prefix in analogy_types:

        for target_model in language_models.keys():

            # if not os.path.exists(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv"):
            #     _df = analogy_solver(target_model, target_data, data_prefix=prefix, instruction_type="instruction-a")
            #     _df.to_csv(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv", index=False)
            # else:
            #     _df = pd.read_csv(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv")
            # results.append(
            #     {
            #         'accuracy': _df['accuracy'].mean(),
            #         'model': target_model,
            #         'approach': 'instruction',
            #         'prefix': prefix,
            #         'data': target_data
            #     }
            # )

            score_file = f"results/scores/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.json"
            breakdown_file = f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.csv"
            if not os.path.exists(breakdown_file):

                if os.path.dirname(breakdown_file) != '':
                    os.makedirs(os.path.dirname(breakdown_file), exist_ok=True)

                if os.path.dirname(score_file) != '':
                    os.makedirs(os.path.dirname(score_file), exist_ok=True)

                _scores_texts = None
                if os.path.exists(score_file):
                    with open(score_file) as f:
                        _scores_texts = json.load(f)

                _df, _scores_texts = analogy_solver(target_model, target_data, data_prefix=prefix, scores_texts=_scores_texts)
                _df.to_csv(breakdown_file, index=False)

                with open(score_file, 'w') as f:
                    json.dump(_scores_texts, f)

            else:
                _df = pd.read_csv(breakdown_file)

            print(target_data, prefix, target_model, _df['accuracy'].mean())
            results.append(
                {
                    'accuracy': _df['accuracy'].mean(),
                    'model': target_model,
                    'approach': 'prompt',
                    'prefix': prefix,
                    'data': target_data
                }
            )

    pd.DataFrame(results).to_csv('results/full_result.csv', index=False)
