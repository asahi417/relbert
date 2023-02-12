"""
python finetune_t5_analogy.py -e 1 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch1'
python finetune_t5_analogy.py -e 3 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch3'
python finetune_t5_analogy.py -e 5 -m 'google/flan-t5-small' -o 'analogy_models/flan-t5-small-analogy-epoch5'
"""
import argparse
import os
import json
import logging
from itertools import permutations, chain
from statistics import mean

import pandas as pd
import torch
import transformers
from datasets import load_dataset
from lmppl import EncoderDecoderLM
from huggingface_hub import create_repo

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description='Fine-tuning T5 on analogy generation.')
parser.add_argument('-m', '--model', default='google/flan-t5-small', type=str)
parser.add_argument('-o', '--output-dir', default='runs', type=str)
parser.add_argument('-s', '--random-seed', default=42, type=int)
parser.add_argument('-e', '--epoch', default=5, type=int)
parser.add_argument('-l', '--lr', default=1e-4, type=float)
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('--batch-size-eval', default=32, type=int)
parser.add_argument('-d', '--data', default='relbert/semeval2012_relational_similarity', type=str)
parser.add_argument('--split-train', default='train', type=str)
parser.add_argument('--split-validation', default='validation', type=str)
parser.add_argument('--skip-train', help='', action='store_true')
parser.add_argument('--skip-validation', help='', action='store_true')
parser.add_argument('--push-to-hub', help='', action='store_true')
parser.add_argument('-a', '--model-alias', default=None, type=str)
opt = parser.parse_args()

##########
# Config #
##########
task_prefix = 'generate analogy:'
template_header = "<subj-a> is to <obj-a>"
template_footer = "<subj-b> is to <obj-b>"

##############
# Load Model #
##############
model_config = transformers.AutoConfig.from_pretrained(opt.model)
model = transformers.T5ForConditionalGeneration.from_pretrained(opt.model, config=model_config)
tokenizer = transformers.AutoTokenizer.from_pretrained(opt.model)
if torch.cuda.device_count() > 0:
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to('cuda')
logging.info(f'language model ({opt.model}) running on {model.device}')


#######################
# Dataset Preparation #
#######################
def encode(x, y):
    model_inputs = tokenizer(f"{task_prefix} {x}", truncation=True)
    model_inputs['labels'] = tokenizer(text_target=y, truncation=True)['input_ids']
    return model_inputs


# prompting input
dataset = load_dataset(opt.data)
df = dataset[opt.split_train].to_pandas()
tokenized_dataset = []
for _, g in df.groupby("relation_type"):
    positives = [i.tolist() for i in g['positives'].values[0].tolist()]
    tokenized_dataset += [encode(
        template_header.replace('<subj-a>', h_a).replace('<obj-a>', t_a),
        template_footer.replace('<subj-b>', h_b).replace('<obj-b>', t_b)) for
        (h_a, t_a), (h_b, t_b) in permutations(positives, 2)]

if not opt.skip_train:
    ##################
    # Model Training #
    ##################
    training_args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=opt.batch_size,
        warmup_steps=0,
        weight_decay=0.01,
        learning_rate=opt.lr,
        num_train_epochs=opt.epoch,
        output_dir=f'{opt.output_dir}/runs/',
        logging_dir=f'{opt.output_dir}/logging/',
        logging_steps=100,
        evaluation_strategy="no",
        save_strategy='no',
        seed=opt.random_seed
    )
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    trainer.train()
    trainer.model.config.update({'finetuing_config': {
        "task_prefix": task_prefix,
        "template_header": template_header,
        "template_footer": template_footer,
        "epoch": opt.epoch,
        'learning_rate': opt.lr,
        'batch_size': opt.batch_size,
        'random_seed': opt.random_seed,
        'data': opt.data,
        'model': opt.model}})
    trainer.save_model(f"{opt.output_dir}/model")
    tokenizer.save_pretrained(f"{opt.output_dir}/model")
assert os.path.exists(f"{opt.output_dir}/model")

if not opt.skip_validation:
    #######################
    # Qualitative Example #
    #######################
    data_valid = dataset[opt.split_validation]
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
    with open(f"{opt.output_dir}/model/validation_accuracy.json", "w") as f:
        json.dump({"accuracy": accuracy, 'datasaet': opt.data, 'split': opt.split_validation}, f)

if opt.push_to_hub:
    assert opt.hf_organization is not None, f'specify hf organization `--hf-organization`'
    assert opt.model_alias is not None, f'specify hf organization `--model-alias`'
    url = create_repo(opt.model_alias, organization=opt.hf_organization, exist_ok=True)
    # if not opt.skip_train:
    args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.hf_organization}
    trainer.model.push_to_hub(opt.model_alias, **args)
    tokenizer.push_to_hub(opt.model_alias, **args)
    if os.path.exists(summary_file):
        shutil.copy2(summary_file, opt.model_alias)
