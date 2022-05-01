""" Implementation of Analogical Proportion Score https://aclanthology.org/2021.acl-long.280/

>>> from relbert import AnalogyScore
>>> query = ['word', 'language']
>>> options = [['paint', 'portrait'], ['poetry', 'rhythm'], ['note', 'music'], ['tale', 'story'], ['week', 'year']]
>>> scorer = AnalogyScore('roberta-large')
>>> score = scorer.analogy_score(_q, _options)
"""
import os
import re
import logging
import math
from math import log
from typing import List
from itertools import chain
from tqdm import tqdm

import transformers
import torch


os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
TEMPLATES = {
    'is-to-what': "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>",  # to-what
    'is-to-as': "<subj-a> is to <obj-a> as <subj-b> is to <obj-b>",  # to-as
    'rel-same': 'The relation between <subj-a> and <obj-a> is the same as the relation between <subj-b> and <obj-b>',  # rel-same
    'what-is-to': 'what <subj-a> is to <obj-a>, <subj-b> is to <obj-b>',  # what-to
    'she-to-as': 'She explained to him that <subj-a> is to <obj-a> as <subj-b> is to <obj-b>.',  # she-as
    'as-what-same': 'As I explained earlier, what <subj-a> is to <obj-a> is essentially the same as what <subj-b> is to <obj-b>.'  # as-what
}


def get_permutation(word_pairs, positive=True):
    a, b, c, d = word_pairs
    if positive:
        perm = [(a, b, c, d), (a, c, b, d), (b, a, d, c), (b, d, a, c), (c, d, a, b), (c, a, d, b), (d, c, b, a), (d, b, c, a)]
    else:
        perm = [(a, b, d, c), (a, c, d, b), (a, d, b, c), (a, d, c, b), (b, a, c, d), (b, c, a, d), (b, c, d, a), (b, d, c, a),
                (c, a, b, d), (c, b, a, d), (c, b, d, a), (c, d, b, a), (d, a, b, c), (d, a, c, b), (d, b, a, c), (d, c, a, b)]
    return perm


def check_position(text, positions, tokens):
    for p, t in zip(positions, tokens):
        assert text[p[0]: p[1]] == t, '{} != {}'.format(text[p[0]: p[1]], t)


def prompting_relation(relation_words, template_type: str = 'is-to-what'):
    """ to convert a SAT style analogy set into a natural sentence with a template """
    assert template_type in TEMPLATES.keys(), 'choose one from {}'.format(TEMPLATES.keys())
    template = TEMPLATES[template_type]
    subject_a, object_a, subject_b, object_b = relation_words
    position = []
    for i, m in zip(['<subj-a>', '<obj-a>', '<subj-b>', '<obj-b>'], [subject_a, object_a, subject_b, object_b]):
        position += [[len(template.split(i)[0]), len(template.split(i)[0]) + len(m)]]
        template = template.replace(i, m)
    check_position(template, position, [subject_a, object_a, subject_b, object_b])
    return template, position


def get_partition(_list):
    length = list(map(lambda x: len(x), _list))
    return list(map(lambda x: [sum(length[:x]), sum(length[:x + 1])], range(len(length))))


def find_position(tokenizer, mask_position, text, token: List = None):
    if token is None:
        token = tokenizer.tokenize(text)
    start, end = mask_position
    token_to_mask = text[start:end]
    start = len(re.sub(r'\s*\Z', '', text[:start]))
    token_before = tokenizer.tokenize(text[:start])
    assert token[:len(token_before)] == token_before, 'wrong token\n `{}` vs `{}`'.format(
        token[:len(token_before)], token_before)
    i = len(token_before)
    while i < len(token):
        i += 1
        decode = tokenizer.convert_tokens_to_string(token[:i])
        tmp_decode = decode.replace(' ', '')
        if token_to_mask in tmp_decode:
            break
    return [len(token_before), i]


def input_ids_to_labels(input_ids, label_position: List = None, label_id: List = None, is_causal: bool = False, pad_token_id=None):
    assert len(label_position) == len(label_id)
    label = [PAD_TOKEN_LABEL_ID] * len(input_ids)
    if is_causal:  # shift the label sequence for causal inference
        label = list(map(lambda x: PAD_TOKEN_LABEL_ID if x == pad_token_id else x, input_ids))
        label = label[1:] + [PAD_TOKEN_LABEL_ID]
    else:
        for p, i in zip(label_position, label_id):
            label[p] = i
    return label


class Dataset(torch.utils.data.Dataset):
    """ `torch.utils.data.Dataset` """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data  # a list of dictionaries

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class AnalogyScore:

    def __init__(self, model: str, max_length: int = 32):
        self.max_length = max_length

        # model setup
        self.is_causal = 'gpt' in model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.config = transformers.AutoConfig.from_pretrained(model)
        if self.is_causal:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model, config=self.config)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = transformers.AutoModelForMaskedLM.from_pretrained(model, config=self.config)
        self.model.eval()

        # gpu setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        logging.info('running on {} GPU'.format(torch.cuda.device_count))

        # sentence prefix tokens
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
        self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]

    def __get_nll(self, data_loader):
        """ Negative log likelihood (NLL) """
        assert self.model
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        nll = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                labels = encode.pop('labels')
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                loss = loss.view(len(prediction_scores), -1)
                loss = torch.sum(loss, -1)
                nll += list(map(
                    lambda x: x[0] / sum(map(lambda y: y != PAD_TOKEN_LABEL_ID, x[1])),
                    zip(loss.cpu().tolist(), labels.cpu().tolist())
                ))
        return nll

    def encode_plus_perplexity(self, word: List, template_type: str):
        """ An output from `encode_plus` for perplexity computation """
        text, position = prompting_relation(word, template_type=template_type)
        if self.is_causal:
            encode = self.tokenizer.encode_plus(text, max_length=self.max_length, truncation=True, padding='max_length')
            encode['labels'] = input_ids_to_labels(encode['input_ids'], is_causal=True, pad_token_id=self.tokenizer.pad_token_id)
            return [encode]
        token_list = self.tokenizer.tokenize(text)

        def encode_with_single_mask_id(mask_position: int):
            _token_list = token_list.copy()  # can not be encode outputs because of prefix
            masked_token_id = self.tokenizer.convert_tokens_to_ids(_token_list[mask_position])
            _token_list[mask_position] = self.tokenizer.mask_token
            tmp_string = self.tokenizer.convert_tokens_to_string(_token_list)
            _encode = self.tokenizer.encode_plus(tmp_string, max_length=self.max_length, truncation=True, padding='max_length')
            _encode['labels'] = input_ids_to_labels(
                _encode['input_ids'],
                label_position=[mask_position + len(self.sp_token_prefix)],
                label_id=[masked_token_id],
                is_causal=self.is_causal
            )
            return _encode

        length = min(self.max_length - len(self.sp_token_prefix), len(token_list))
        return [encode_with_single_mask_id(i) for i in range(length)]

    def batch_encode_plus_perplexity(self, batch_word: List, template_type: str, batch_size: int = None):
        """ Batch version of `self.encode_plus_perplexity` """
        batch_size = len(batch_word) if batch_size is None else batch_size
        data = []
        for x in batch_word:
            data.append(self.encode_plus_perplexity(template_type=template_type, word=x))
        partition = get_partition(data)
        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))), batch_size=batch_size, shuffle=False, drop_last=False
        )
        return data_loader, partition

    def get_perplexity(self, word_pairs: List, template_type: str = 'is-to-what', batch_size: int = 32):
        """ (pseudo) Perplexity """
        data_loader, partition = self.batch_encode_plus_perplexity(
            template_type=template_type,
            batch_word=word_pairs,
            batch_size=batch_size)
        nll = self.__get_nll(data_loader)
        return list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))

    def analogy_score(self,
                      query_word_pair: List,
                      option_word_pairs: List,
                      weight_head: float = 0.5,
                      weight_tail: float = 0.5,
                      score_type: str = 'pmi',
                      template: str = 'is-to-what',
                      batch_size: int = 32,
                      positive_permutation: int = 0,
                      negative_permutation: int = None,
                      weight_negative: float = 1.0):

        length = len(option_word_pairs)

        def compute_negative_pmi(ppl):

            # conditional negative log likelihood (fixed head and tail tokens)
            ppl_in_option = list(map(lambda x: ppl[length * x + x], range(length)))
            negative_log_likelihood_cond = list(map(lambda x: log(x / sum(ppl_in_option)), ppl_in_option))

            # marginal negative log likelihood (tail token)
            ppl_out_option = list(map(lambda x: sum(map(lambda y: ppl[x + length * y], range(length))), range(length)))
            negative_log_likelihood_mar_t = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

            # marginal negative log likelihood (head token)
            ppl_out_option = list(map(lambda x: sum(ppl[x * length: (x + 1) * length]), range(length)))
            negative_log_likelihood_mar_h = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

            # negative pmi approx by perplexity difference: higher is better
            neg_pmi = list(map(
                lambda x: x[0] - weight_head * x[1] - weight_tail * x[2],
                zip(negative_log_likelihood_cond, negative_log_likelihood_mar_h, negative_log_likelihood_mar_t)))
            return neg_pmi

        if score_type == 'pmi':
            # setup language model/data
            option_word_pairs = list(chain(*[[[i[0], m[1]] for m in option_word_pairs] for i in option_word_pairs]))

        model_input = [query_word_pair + list(i) for i in option_word_pairs]

        logging.info('get prediction from language model')
        model_input_positive = [get_permutation(i)[positive_permutation] for i in model_input]
        ppl_positive = self.get_perplexity(model_input_positive, batch_size=batch_size, template_type=template)
        if score_type == 'pmi':
            score = compute_negative_pmi(ppl_positive)
        else:
            score = ppl_positive

        if negative_permutation is not None:
            logging.info('get prediction from language model (negative permutation)')
            model_input_negative = [get_permutation(i, positive=False)[negative_permutation] for i in model_input]
            ppl_negative = self.get_perplexity(model_input_negative, batch_size=batch_size, template_type=template)
            if score_type == 'pmi':
                score_negative = compute_negative_pmi(ppl_negative)
            else:
                score_negative = ppl_negative
        else:
            score_negative = [0] * len(score)
        score = [neg * weight_negative - pos for neg, pos in zip(score_negative, score)]
        return score


if __name__ == '__main__':
    config_roberta = {"model": "roberta-large", "weight_head": 0.2, "weight_tail": 0.2, "template": 'as-what-same',
                      "positive_permutation": 4, "negative_permutation": 10, "weight_negative": 0.2}
    config_gpt2 = {"model": "gpt2-xl", "weight_head": -0.4, "weight_tail": 0.2, "template": 'rel-same',
                   "positive_permutation": 2, "negative_permutation": 0, "weight_negative": 0.8}
    config_bert = {"model": "bert-large-cased", "weight_head": -0.2, "weight_tail": -0.4, "template": 'what-is-to',
                   "positive_permutation": 4, "negative_permutation": 4, "weight_negative": 0.2}

    _q = ['word', 'language']
    _options = [['paint', 'portrait'], ['poetry', 'rhythm'], ['note', 'music'], ['tale', 'story'], ['week', 'year']]

    _scorer = AnalogyScore(config_roberta.pop('model'))
    _s = _scorer.analogy_score(_q, _options, score_type='ppl', **config_roberta)
    print(_s)
    print(_options[_s.index(max(_s))])
