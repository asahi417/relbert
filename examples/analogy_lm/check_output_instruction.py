import os
import json
import gc
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

instruction = "Which one of the following is an analogy?"
template = "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>"
analogy_types = [['sat_metaphor', '0'], ['sat_metaphor', '1'], ['sat_metaphor', '2']]
language_models = {
    "google/flan-t5-xxl": 8,  # 11B
    "google/flan-t5-xl": 16,  # 3B
    "google/flan-t5-large": 256,  # 770M
    "google/flan-t5-base": 512,  # 220M
    "google/flan-t5-small": 1024  # 60M
}


def get_input(query_pair: List, candidate_pairs: List):
    tmp = template.replace('<subj-a>', query_pair[0]).replace('<obj-a>', query_pair[1])
    tmp = "\n".join([f"{n+1}) {tmp.replace('<subj-b>', a).replace('<obj-b>', b)}" for n, (a, b) in enumerate(candidate_pairs)])
    return f"{instruction}\n{tmp}"


def get_generation(pipeline_generator, data_name, batch_size, data_prefix: str = None):
    # dataset setup
    dataset = load_dataset('relbert/analogy_questions', data_name, split='test')
    if data_prefix is not None:
        dataset = dataset.filter(lambda x: x['prefix'] == data_prefix)
        assert len(dataset) > 0
    # prompt data
    model_input = [get_input(x['stem'], x['choice']) for x in dataset]
    # generate output
    output = pipeline_generator(model_input, batch_size=batch_size)
    return [{'input': i, 'output': o['generated_text'], 'answer': a['answer']} for i, o, a in zip(model_input, output, dataset)]


if __name__ == '__main__':
    os.makedirs('results/instruction_output', exist_ok=True)
    results = []
    for target_model in language_models.keys():
        generator = None

        for target_data, prefix in analogy_types:
            score_file = f"results/instruction_output/{os.path.basename(target_model)}_{target_data}_{prefix}.instruction.json"
            if not os.path.exists(score_file):
                generator = pipeline(model=target_model, device_map='auto') if generator is None else generator
                _output = get_generation(generator, target_data, language_models[target_model], data_prefix=prefix)
                with open(score_file, 'w') as f:
                    f.write('\n'.join([json.dumps(x) for x in _output]))

        del generator
        gc.collect()
        torch.cuda.empty_cache()
