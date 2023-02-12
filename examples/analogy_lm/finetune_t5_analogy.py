import argparse
import logging
from random import seed, randint
from itertools import permutations
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelWithLMHead


template_header = "<subj-a> is to <obj-a>"
template_join = "what"
template_footer = "<subj-b> is to <obj-b>"
instruction_header = "Which one of the following is an analogy?"
instruction_footer = "The correct answer is"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser(description='Fine-tuning T5 on analogy generation.')
parser.add_argument('-m', '--model', default='google/flan-t5-small', type=str)
parser.add_argument('-d', '--data', default='relbert/semeval2012_relational_similarity', type=str)
parser.add_argument('--instruction', help='', action='store_true')
opt = parser.parse_args()


logging.info('loading dataset')
data = load_dataset(opt.data, split='train')
df = data.to_pandas()
model_config = AutoConfig.from_pretrained(opt.model)


def prompting(positives, negatives):
    prompt_in_out = []
    if opt.instruction:
        seed(42)
        for a, p in permutations(positives, 2):
            header = f"{template_header.replace('<subj-a>', a[0]).replace('<obj-a>', a[1])}"
            false_choices = [f"{header} {template_join} {template_footer.replace('<subj-b>', h).replace('<obj-b>', t)}" for h, t in negatives]
            true_choice = f"{header} {template_join} {template_footer.replace('<subj-b>', p[0]).replace('<obj-b>', p[1])}"
            answer_ind = randint(0, len(false_choices))
            choice = false_choices[:answer_ind] + [true_choice] + false_choices[answer_ind:]
            assert choice[answer_ind] == true_choice
            choice_string = "\n".join([f"{n+1}) {i}" for n, i in enumerate(choice)])
            prompt = f"{instruction_header}\n{choice_string}"
            if model_config.model_type != 't5':
                prompt = f"{prompt}\n{instruction_footer}"
            prompt_in_out.append([prompt, true_choice])
        return prompt_in_out
    else:
        if model_config.model_type != 't5':
            return [[template_header.replace('<subj-a>', h_a).replace('<obj-a>', t_a) + f" {template_join}",
                     template_footer.replace('<subj-b>', h_b).replace('<obj-b>', t_b)] for
                    (h_a, t_a), (h_b, t_b) in permutations(positives, 2)]
        else:
            return [[template_header.replace('<subj-a>', h_a).replace('<obj-a>', t_a),
                     template_footer.replace('<subj-b>', h_b).replace('<obj-b>', t_b)] for
                    (h_a, t_a), (h_b, t_b) in permutations(positives, 2)]


# prompting input
model_input_output = []
for _, g in df.groupby("relation_type"):
    model_input_output += prompting(
        positives=[i.tolist() for i in g['positives'].values[0].tolist()],
        negatives=[i.tolist() for i in g['negatives'].values[0].tolist()])

# load model
is_t5 = model_config.model_type == 't5'
if is_t5:
    model = transformers.T5ForConditionalGeneration.from_pretrained(opt.model, config=model_config)
else:
    model = AutoModelWithLMHead.from_pretrained(opt.model, config=model_config)
if torch.cuda.device_count() > 0:
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to('cuda')
logging.info(f'language model running on {model.device}')

# load tokenizer & tokenization
tokenizer = AutoTokenizer.from_pretrained(opt.model)


def preprocess(examples):
    model_inputs = tokenizer(examples[0], truncation=True)
    if is_t5:
        model_inputs['labels'] = tokenizer(text_target=examples[1], truncation=True)['input_ids']
    else:
        model_inputs['labels'] = tokenizer(examples[1], truncation=True)['input_ids']
    return model_inputs

tokenized_dataset = [preprocess(i) for i in model_input_output]
tokenized_dataset =
corpus = load_dataset("eth_py150_open", split='train')


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