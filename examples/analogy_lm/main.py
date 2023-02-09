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

import logging
import os
from typing import List

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
              instruction_type: str = None):
    template_header = templates[template_type][0].replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    if instruction_type is None:
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
    # "roberta-base": lmppl.MaskedLM,  # 110M
    # "roberta-large": lmppl.MaskedLM,  # 355M
    # "microsoft/deberta-v3-xsmall": lmppl.MaskedLM,  # 70M
    # "microsoft/deberta-v3-small": lmppl.MaskedLM,  # 142M
    # "microsoft/deberta-v3-base": lmppl.MaskedLM,  # 184M
    # "microsoft/deberta-v3-large": lmppl.MaskedLM,  # 434M
    # "microsoft/deberta-v2-xlarge": lmppl.MaskedLM,  # 900M
    # "microsoft/deberta-v2-xxlarge": lmppl.MaskedLM,  # 1.5B
    "gpt2": lmppl.LM,  # 124M
    "gpt2-medium": lmppl.LM,  # 355M
    "gpt2-large": lmppl.LM,  # 774M
    "gpt2-xl": lmppl.LM,  # 1.5B
    "facebook/opt-125m": lmppl.LM,  # 125M
    "facebook/opt-350m": lmppl.LM,  # 350M
    "facebook/opt-1.3b": lmppl.LM,  # 1.3B
    "facebook/opt-iml-1.3b": lmppl.LM,  # 1.3B
    "facebook/opt-iml-max-1.3b": lmppl.LM,  # 1.3B
    "t5-small": lmppl.EncoderDecoderLM,  # 60M
    "t5-base": lmppl.EncoderDecoderLM,  # 220M
    "t5-large": lmppl.EncoderDecoderLM,  # 770M
    "t5-3b": lmppl.EncoderDecoderLM,  # 3B
    "google/flan-t5-small": lmppl.EncoderDecoderLM,  # 60M
    "google/flan-t5-base": lmppl.EncoderDecoderLM,  # 220M
    "google/flan-t5-large": lmppl.EncoderDecoderLM,  # 770M
    "google/flan-t5-xl": lmppl.EncoderDecoderLM,  # 3B
}


def analogy_solver(
        model,
        data_name,
        data_prefix: str = None,
        batch_size: int = 64,
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
    lm_class = language_models[model]
    if lm_class is lmppl.MaskedLM:
        scorer = lm_class(model, max_length=256)
    else:
        scorer = lm_class(model)

    # get scores
    if lm_class is lmppl.EncoderDecoderLM:
        scores = scorer.get_perplexity(
            input_texts=[x[0] for x in dataset_flat],
            output_texts=[x[1] for x in dataset_flat],
            batch=batch_size
        )
    else:
        scores = scorer.get_perplexity(
            input_texts=[f"{x[0]} {x[1]}" for x in dataset_flat],
            batch=batch_size
        )
    index_score = list(zip(dataset_index, scores))
    scores_aligned = [(i, [b for a, b in index_score if a == i]) for i in sorted(list(set(dataset_index)))]
    prediction = [i[1].index(min(i[1])) for i in scores_aligned]

    # compute accuracy
    df = dataset.to_pandas()
    df['choice'] = [[_i.tolist() for _i in i] for i in df['choice']]
    df['prediction'] = prediction
    df['accuracy'] = df['prediction'] == df['answer']
    return df


if __name__ == '__main__':
    os.makedirs('results/breakdown', exist_ok=True)

    results = []
    for target_data, prefix in analogy_types:

        for target_model in language_models.keys():

            if not os.path.exists(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv"):
                _df = analogy_solver(target_model, target_data, batch_size=16, instruction_type="instruction-a")
                _df.to_csv(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv", index=False)
            else:
                _df = pd.read_csv(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.csv")
            results.append(
                {'accuracy': _df['accuracy'].mean(), 'model': target_model, 'approach': 'instruction', 'prefix': prefix,
                 'data': target_data}
            )

            if not os.path.exists(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.csv"):
                _df = analogy_solver(target_model, target_data, batch_size=16)
                _df.to_csv(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.csv", index=False)
            else:
                _df = pd.read_csv(f"results/breakdown/{os.path.basename(target_model)}_{target_data}_{prefix}.prompt.csv")

            results.append(
                {'accuracy': _df['accuracy'].mean(), 'model': target_model, 'approach': 'prompt', 'prefix': prefix, 'data': target_data}
            )

    pd.DataFrame(results).to_csv('results/full_result.csv', index=False)
