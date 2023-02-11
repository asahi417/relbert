import os
import json
import gc
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

instruction = ["Which one of the following is an analogy?", "The correct answer is"]
cot = 'Answer the following question by reasoning step-by-step.'
template = "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>"
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
    # "gpt2": 64,  # 124M
    # "gpt2-medium": 64,  # 355M
    # "gpt2-large": 32,  # 774M
    # "gpt2-xl": 16,  # 1.5B
    # "EleutherAI/gpt-j-6B": 1,  # 6B
    # "facebook/opt-125m": 64,  # 125M
    # "facebook/opt-350m": 64,  # 350M
    # "facebook/opt-1.3b": 16,  # 1.3B
    # "facebook/opt-iml-1.3b": 16,  # 1.3B
    # "facebook/opt-iml-max-1.3b": 16,  # 1.3B
    # "t5-small": 64,  # 60M
    # "t5-base": 64,  # 220M
    # "t5-large": 32,  # 770M
    # "t5-3b": 16,  # 3B
    # "t5-11b": 1,  # 11B
    "google/flan-t5-small": 64,  # 60M
    "google/flan-t5-base": 64,  # 220M
    "google/flan-t5-large": 32,  # 770M
    "google/flan-t5-xl": 32,  # 3B
    "google/flan-t5-xxl": 8  # 11B
}
# language_models.update({
#     "EleutherAI/gpt-neox-20b": 1,  # 20B
#     "facebook/opt-30b": 1,  # 30B
#     "facebook/opt-iml-30b": 1,  # 30B
#     "facebook/opt-iml-max-30b": 1,  # 30B
#     "google/switch-large-128": [lmppl.EncoderDecoderLM, 2],  # 770M
# })


def get_input(query_pair: List, candidate_pairs: List, use_cot: bool = False):
    tmp = template.replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    tmp = "\n".join([f"{n+1}) {tmp.replace('<subj-b>', a).replace('<obj-b>', b)}" for n, (a, b) in enumerate(candidate_pairs)])
    tmp = f"{instruction[0]}\n{tmp}\n{instruction[1]}"
    if use_cot:
        return f"{cot}\n{tmp}"
    return tmp


def get_generation(pipeline_generator, data_name, batch_size, data_prefix: str = None, use_cot: bool = False):

    # dataset setup
    dataset = load_dataset('relbert/analogy_questions', data_name, split='test')
    if data_prefix is not None:
        dataset = dataset.filter(lambda x: x['prefix'] == data_prefix)
        assert len(dataset) > 0

    # prompt data
    model_input = [get_input(x['stem'], x['choice'], use_cot=use_cot) for x in dataset]
    answers = [x['answer'] for x in dataset]

    # generate output
    output = pipeline_generator(model_input, batch_size=batch_size)
    model_output = [{'input': i, 'output': o['generated_text'], 'answer': a} for i, o, a in zip(model_input, output, answers)]
    return model_output


if __name__ == '__main__':
    os.makedirs('results/breakdown', exist_ok=True)

    results = []
    for target_model in language_models.keys():
        generator = None

        for target_data, prefix in analogy_types:
            print(target_model, target_data)

            # instruction
            score_file = f"results/instruction_output/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.json"
            if os.path.exists(score_file):
                with open(score_file) as f:
                    _model_output = [json.loads(x) for x in f.read().split('\n') if len(x) > 0]
            else:
                # generator = pipeline(model=target_model, device=device) if generator is None else generator
                generator = pipeline(model=target_model, device_map='auto') if generator is None else generator
                if os.path.dirname(score_file) != '':
                    os.makedirs(os.path.dirname(score_file), exist_ok=True)
                _model_output = get_generation(generator, target_data, language_models[target_model], data_prefix=prefix)
                with open(score_file, 'w') as f:
                    f.write('\n'.join([json.dumps(x) for x in _model_output]))

            # instruction with CoT
            score_file = f"results/instruction_output/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.cot.json"
            if os.path.exists(score_file):
                with open(score_file) as f:
                    _model_output = [json.loads(x) for x in f.read().split('\n') if len(x) > 0]
            else:
                # generator = pipeline(model=target_model, device=device) if generator is None else generator
                generator = pipeline(model=target_model, device_map='auto') if generator is None else generator
                if os.path.dirname(score_file) != '':
                    os.makedirs(os.path.dirname(score_file), exist_ok=True)
                _model_output = get_generation(generator, target_data, language_models[target_model], data_prefix=prefix, use_cot=True)
                with open(score_file, 'w') as f:
                    f.write('\n'.join([json.dumps(x) for x in _model_output]))

        del generator
        gc.collect()
        torch.cuda.empty_cache()
