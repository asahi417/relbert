""" Fine-tune T5 on analogy generation.

- Small Models
```
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch3'
python finetune_t5_analogy.py -e 6 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch6'
python finetune_t5_analogy.py -m 'google/flan-t5-small' --skip-train --skip-validation -o 'analogy_models/flan-t5-small-analogy-epoch3' --repo-id 'relbert/flan-t5-small-analogy'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch1-p' --add-permutation
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch3-p' --add-permutation
python finetune_t5_analogy.py -m 'google/flan-t5-small' --skip-train --skip-validation -o 'analogy_models/flan-t5-small-analogy-epoch3-p' --repo-id 'relbert/flan-t5-small-analogy-permutation'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch1-pd' --add-permutation-domain
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch3-pd' --add-permutation-domain
python finetune_t5_analogy.py -m 'google/flan-t5-small' --skip-train --skip-validation -o 'analogy_models/flan-t5-small-analogy-epoch3-pd' --repo-id 'relbert/flan-t5-small-analogy-permutation-domain'
```

- Base Models
```
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-base' -o 'analogy_models/flan-t5-base-analogy-epoch3'
python finetune_t5_analogy.py -e 6 -m 'google/flan-t5-base' -o 'analogy_models/flan-t5-base-analogy-epoch6'
python finetune_t5_analogy.py -m 'google/flan-t5-base' --skip-train --skip-validation -o 'analogy_models/flan-t5-base-analogy-epoch3' --repo-id 'relbert/flan-t5-base-analogy'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-base' -o 'analogy_models/flan-t5-base-analogy-epoch1-p' --add-permutation
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-base' -o 'analogy_models/flan-t5-base-analogy-epoch3-p' --add-permutation
python finetune_t5_analogy.py -m 'google/flan-t5-base' --skip-train --skip-validation -o 'analogy_models/flan-t5-base-analogy-epoch3-p' --repo-id 'relbert/flan-t5-base-analogy-permutation'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-base' -o 'analogy_models/flan-t5-base-analogy-epoch1-pd' --add-permutation-domain
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-base' -o 'analogy_models/flan-t5-base-analogy-epoch3-pd' --add-permutation-domain
python finetune_t5_analogy.py -m 'google/flan-t5-base' --skip-train --skip-validation -o 'analogy_models/flan-t5-base-analogy-epoch3-pd' --repo-id 'relbert/flan-t5-base-analogy-permutation-domain'
```

- Large Models
```
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-large' -o 'analogy_models/flan-t5-large-analogy-epoch3'
python finetune_t5_analogy.py -e 6 -m 'google/flan-t5-large' -o 'analogy_models/flan-t5-large-analogy-epoch6'
python finetune_t5_analogy.py -m 'google/flan-t5-large' --skip-train --skip-validation -o 'analogy_models/flan-t5-large-analogy-epoch3' --repo-id 'relbert/flan-t5-large-analogy'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-large' -o 'analogy_models/flan-t5-large-analogy-epoch1-p' --add-permutation
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-large' -o 'analogy_models/flan-t5-large-analogy-epoch3-p' --add-permutation
python finetune_t5_analogy.py -m 'google/flan-t5-large' --skip-train --skip-validation -o 'analogy_models/flan-t5-large-analogy-epoch3-p' --repo-id 'relbert/flan-t5-large-analogy-permutation'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-large' -o 'analogy_models/flan-t5-large-analogy-epoch1-pd' --add-permutation-domain
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-large' -o 'analogy_models/flan-t5-large-analogy-epoch3-pd' --add-permutation-domain
python finetune_t5_analogy.py -m 'google/flan-t5-large' --skip-train --skip-validation -o 'analogy_models/flan-t5-large-analogy-epoch3-pd' --repo-id 'relbert/flan-t5-large-analogy-permutation-domain'
```

- XL Models
```
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-xl' -o 'analogy_models/flan-t5-xl-analogy-epoch3' --gradient-checkpointing --batch-size-eval 8
python finetune_t5_analogy.py -e 6 -m 'google/flan-t5-xl' -o 'analogy_models/flan-t5-xl-analogy-epoch6' --gradient-checkpointing --batch-size-eval 8
python finetune_t5_analogy.py -m 'google/flan-t5-xl' --skip-train --skip-validation -o 'analogy_models/flan-t5-xl-analogy-epoch6' --repo-id 'relbert/flan-t5-xl-analogy'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-xl' -o 'analogy_models/flan-t5-xl-analogy-epoch1-p' --gradient-checkpointing --batch-size-eval 8 --add-permutation
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-xl' -o 'analogy_models/flan-t5-xl-analogy-epoch3-p' --gradient-checkpointing --batch-size-eval 8 --add-permutation
python finetune_t5_analogy.py -m 'google/flan-t5-xl' --skip-train --skip-validation -o 'analogy_models/flan-t5-xl-analogy-epoch1-p' --repo-id 'relbert/flan-t5-xl-analogy-permutation'

python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-xl' -o 'analogy_models/flan-t5-xl-analogy-epoch1-pd' --add-permutation-domain
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-xl' -o 'analogy_models/flan-t5-xl-analogy-epoch3-pd' --add-permutation-domain
python finetune_t5_analogy.py -m 'google/flan-t5-xl' --skip-train --skip-validation -o 'analogy_models/flan-t5-xl-analogy-epoch1-p' --repo-id 'relbert/flan-t5-xl-analogy-permutation-domain'
```
"""
import argparse
import os
import json
import logging
import shutil
from os.path import join as pj
from itertools import permutations, chain
from statistics import mean
from distutils.dir_util import copy_tree

import pandas as pd
import torch
import transformers
from datasets import load_dataset
from lmppl import EncoderDecoderLM
from huggingface_hub import create_repo


#############
# Arguments #
#############
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description='Fine-tuning T5 on analogy generation.')
parser.add_argument('-m', '--model', default='google/flan-t5-small', type=str)
parser.add_argument('-o', '--output-dir', default='runs', type=str)
parser.add_argument('-s', '--random-seed', default=42, type=int)
parser.add_argument('-e', '--epoch', default=5, type=int)
parser.add_argument('-l', '--lr', default=1e-4, type=float)
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('--batch-size-eval', default=32, type=int)
parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
parser.add_argument('-d', '--data', default='relbert/semeval2012_relational_similarity', type=str)
parser.add_argument('--split-train', default='train', type=str)
parser.add_argument('--split-validation', default='validation', type=str)
parser.add_argument('--skip-train', help='', action='store_true')
parser.add_argument('--skip-validation', help='', action='store_true')
parser.add_argument('--gradient-checkpointing', help='', action='store_true')
parser.add_argument('--push-to-hub', help='', action='store_true')
parser.add_argument('--display-prediction', help='', action='store_true')
parser.add_argument('--fp16', help='', action='store_true')
parser.add_argument('--add-permutation', help='', action='store_true')
parser.add_argument('--add-permutation-domain', help='', action='store_true')
parser.add_argument('--repo-id', default=None, type=str)
opt = parser.parse_args()

##########
# Config #
##########
task_prefix = 'generate analogy:'
template_header = "<subj-a> is to <obj-a>"
template_footer = "<subj-b> is to <obj-b>"
is_parallel = False
if not opt.skip_train:
    ##############
    # Load Model #
    ##############
    model_config = transformers.AutoConfig.from_pretrained(opt.model)
    model = transformers.T5ForConditionalGeneration.from_pretrained(opt.model, config=model_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(opt.model)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        is_parallel = True
    if torch.cuda.device_count() == 1:
        model.to('cuda')

    #######################
    # Dataset Preparation #
    #######################

    def permute_same_domain(a, b, c, d):
        # return [[a, b, c, d], [b, a, d, c], [c, d, a, b], [d, c, b, a]]
        return [[a, b, c, d], [b, a, d, c]]

    def permute(a, b, c, d):
        return [[a, b, c, d], [b, a, d, c], [a, c, b, d], [d, b, c, a]]
        # return [[a, b, c, d], [a, c, b, d], [b, a, d, c], [b, d, a, c],
        #         [c, d, a, b], [c, a, d, b], [d, c, b, a], [d, b, c, a]]

    def encode(x, y):
        model_inputs = tokenizer(f"{task_prefix} {x}", truncation=True)
        model_inputs['labels'] = tokenizer(text_target=y, truncation=True)['input_ids']
        return model_inputs

    # prompting input
    df = load_dataset(opt.data, split=opt.split_train).to_pandas()
    tokenized_dataset = []
    for _, g in df.groupby("relation_type"):
        positives = [[a, b, c, d] for (a, b), (c, d) in permutations([i.tolist() for i in g['positives'].values[0].tolist()], 2)]
        if opt.add_permutation:
            positives = list(chain(*[permute(*i) for i in positives]))
        if opt.add_permutation_domain:
            positives = list(chain(*[permute_same_domain(*i) for i in positives]))
        tokenized_dataset += [encode(
            template_header.replace('<subj-a>', a).replace('<obj-a>', b),
            template_footer.replace('<subj-b>', c).replace('<obj-b>', d)) for a, b, c, d in positives]

    ##################
    # Model Training #
    ##################
    training_args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=opt.batch_size,
        gradient_checkpointing=opt.gradient_checkpointing,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        warmup_steps=0,
        weight_decay=0.01,
        learning_rate=opt.lr,
        num_train_epochs=opt.epoch,
        output_dir=pj(opt.output_dir, 'runs/'),
        logging_dir=pj(opt.output_dir, 'logging/'),
        logging_steps=100,
        evaluation_strategy="no",
        save_strategy='no',
        seed=opt.random_seed,
        fp16=opt.fp16)
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    trainer.train()
    finetuing_config = {'finetuing_config': {
        "task_prefix": task_prefix,
        "template_header": template_header,
        "template_footer": template_footer,
        "epoch": opt.epoch,
        'learning_rate': opt.lr,
        'batch_size': opt.batch_size,
        'random_seed': opt.random_seed,
        'data': opt.data,
        'model': opt.model,
        'gradient_accumulation_steps': opt.gradient_accumulation_steps}}
    if is_parallel:
        trainer.model.module.update(finetuing_config)
    else:
        trainer.model.config.update(finetuing_config)
    trainer.save_model(pj(opt.output_dir, "model"))
    tokenizer.save_pretrained(pj(opt.output_dir, "model"))
assert os.path.exists(pj(opt.output_dir, "model"))

if not opt.skip_validation:
    data_valid = load_dataset(opt.data, split=opt.split_validation)
    if opt.display_prediction:
        #######################
        # Qualitative Example #
        #######################
        pipe = transformers.pipeline('text2text-generation', model=f"{opt.output_dir}/model")
        logging.info("Generate examples...")
        for i in data_valid['positives']:
            model_input = f"{task_prefix} {template_header.replace('<subj-a>', i[0][0]).replace('<obj-a>', i[0][1])}"
            output = pipe(model_input)[0]['generated_text']
            logging.info(f"[input] {model_input} \n\t>>> {output}")

    ####################
    # Model Validation #
    ####################
    query = [f"{task_prefix} {template_header.replace('<subj-a>', i[0][0]).replace('<obj-a>', i[0][1])}" for i in data_valid['positives']]
    gold = [template_footer.replace('<subj-b>', i[1][0]).replace('<obj-b>', i[1][1]) for i in data_valid['positives']]
    choice = [[template_footer.replace('<subj-b>', i[0]).replace('<obj-b>', i[1]) for i in l] for l in data_valid['negatives']]
    scorer = EncoderDecoderLM(f"{opt.output_dir}/model")
    # get score for gold answer
    gold_score = scorer.get_perplexity(input_texts=query, output_texts=gold, batch=opt.batch_size_eval)
    # get score for the other choices
    query_flat = list(chain(*[[q] * len(c) for c, q in zip(choice, query)]))
    choice_flat = list(chain(*choice))
    choice_score = scorer.get_perplexity(
        input_texts=query_flat,
        output_texts=choice_flat,
        batch=opt.batch_size_eval)
    # compute accuracy
    index = list(chain(*[[n] * len(c) for n, c in enumerate(choice)]))
    df = pd.DataFrame([{"index": i, "score": s} for i, s in zip(index, choice_score)])
    score_dict = {i: g['score'].values.tolist() for i, g in df.groupby("index")}
    accuracy = mean([all(_v > gold_score[k] for _v in v) for k, v in score_dict.items()])
    with open(pj(opt.output_dir, "model", "validation_accuracy.json"), "w") as f:
        json.dump({"accuracy": accuracy, 'dataset': opt.data, 'split': opt.split_validation}, f)

if opt.repo_id is not None:
    #####################
    # Push to Model hub #
    #####################
    create_repo(repo_id=opt.repo_id, exist_ok=True, repo_type="model")
    transformers.T5ForConditionalGeneration.from_pretrained(f"{opt.output_dir}/model").push_to_hub(opt.repo_id)
    transformers.AutoTokenizer.from_pretrained(f"{opt.output_dir}/model").push_to_hub(opt.repo_id)

    model_dir = os.path.basename(opt.repo_id)
    if not os.path.exists(model_dir):
        os.system(f"git clone https://huggingface.co/{opt.repo_id}")
    # upload remaining files
    copy_tree(f"{opt.output_dir}/model", model_dir)
    readme = f"""
---
widget:
- text: "{task_prefix} mammal is to whale"
  example_title: "Analogy Example 1 (semantic relation)"
- text: "{task_prefix} wedding is to marriage"
  example_title: "Analogy Example 1 (semantic relation, metaphor)"
- text: "{task_prefix} London is to U.K."
  example_title: "Analogy Example 2 (entity)"
- text: "{task_prefix} actual is to actually"
  example_title: "Analogy Example 3 (morphological)"
---
# {opt.repo_id}

This is [{opt.model}](https://huggingface.co/{opt.model}) fine-tuned on [{opt.data}](https://huggingface.co/datasets/{opt.data}) 
for analogy generation, which is to generate a word pair (eg. `bird is to crow`) given a query (eg. `mammal is to whale`) 
so that the query and the generated word pair form an analogy statement.  

### Usage

```python
from transformers import pipeline

pipe = pipeline('text2text-generation', model="{opt.repo_id}")
output = pipe("{task_prefix} {template_header.replace('<subj-a>', 'mammal').replace('<obj-a>', 'whale')}")
print(output)
>>> [{{'generated_text': 'bird is to crow'}}]
```
"""
    with open(f"{model_dir}/README.md", 'w') as f:
        f.write(readme)
    os.system(f"cd {model_dir} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(model_dir)
