import os
import json
import argparse
import logging
from random import shuffle, seed, randint
from itertools import permutations
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer


template_header = "<subj-a> is to <obj-a>"
template_join = "what"
template_footer = "<subj-b> is to <obj-b>"
instruction_header = "Which one of the following is an analogy?"
instruction_footer = "The correct answer is"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description='Fine-tuning T5 on analogy generation.')
parser.add_argument('-m', '--model', required=True, type=str)
parser.add_argument('-d', '--data', default='relbert/semeval2012_relational_similarity', type=str)
parser.add_argument('--instruction', help='', action='store_true')
opt = parser.parse_args()


logging.info('loading dataset')
data = load_dataset(opt.data, split='train')
df = data.to_pandas()
if opt.instruction:
    seed(42)

    def prompting(positieve_pair, negative_pair):
        for a, p in permutations(positives, 2):
            header = f"{template_header.replace('<subj-a>', a[0]).replace('<obj-a>', a[1])}"
            false_choices = [f"{header} {template_join} {template_footer.replace('<subj-b>', h).replace('<obj-b>', t)}" for h, t in negatives]
            true_choice = f"{header} {template_join} {template_footer.replace('<subj-b>', p[0]).replace('<obj-b>', p[1])}"
            answer_ind = randint(0, len(false_choices))
            choice = false_choices[:answer_ind] + [true_choice] + false_choices[answer_ind:]
            assert choice[answer_ind] == true_choice
            choice_string = "\n".join([f"{n+1}) {i}" for n, i in enumerate(choice)])
            prompt = f"{instruction_header}\n{choice_string}\n{instruction_footer}"

        return [[template_header.replace('<subj-a>', h_a).replace('<obj-a>', t_a),
                 template_footer.replace('<subj-b>', h_b).replace('<obj-b>', t_b)] for
                (h_a, t_a), (h_b, t_b) in permutations(list_paris, 2)]

    for _, g in df.groupby("relation_type"):
        positives = [i.tolist() for i in g['positives'].values[0].tolist()]
        negatives = [i.tolist() for i in g['negatives'].values[0].tolist()]


else:

    def prompting(list_paris):
        return [[template_header.replace('<subj-a>', h_a).replace('<obj-a>', t_a),
                 template_footer.replace('<subj-b>', h_b).replace('<obj-b>', t_b)] for
                (h_a, t_a), (h_b, t_b) in permutations(list_paris, 2)]


    model_input_output = []
    for _, g in df.groupby("relation_type"):
        model_input_output += prompting([i.tolist() for i in g['positives'].values[0].tolist()])



corpus = load_dataset("eth_py150_open", split='train')

model = transformers.T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained('Salesforce/codet5-small')

def preprocess(examples):
    model_inputs = tokenizer(examples['filepath'], truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['license'], truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = corpus.map(preprocess_function, batched=True)

training_args = transformers.Seq2SeqTrainingArguments( #general training arguments
    per_device_train_batch_size = 8,
    warmup_steps = 0,
    weight_decay = 0.01,
    learning_rate = 1e-4,
    num_train_epochs = 12,
    output_dir = './runs/run2/output/',
    logging_dir = './runs/run2/logging/',
    logging_steps = 50,
    save_steps= 10000,
    remove_unused_columns=False,
)

data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = transformers.Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset,
    data_collator=data_collator
)