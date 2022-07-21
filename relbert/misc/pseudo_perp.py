""" Pseudo Perplexity
>>> scorer = PPL('bert-base-cased')
>>> sentence = ['Red is the color of courage, of a warrior and a martyr.',
                'His father was a tailor and his mother was a midwife.',]
>>> print(scorer.get_perplexity(sentence))
[3.896801383195375, 6.4809147517537955]
"""
import os
import logging
import math
from typing import List
from itertools import chain
from tqdm import tqdm

import transformers
import torch

os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index


def get_partition(_list):
    length = list(map(lambda x: len(x), _list))
    return list(map(lambda x: [sum(length[:x]), sum(length[:x + 1])], range(len(length))))


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


class PPL:

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
        self.mask_token = self.tokenizer.mask_token

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

    def encode_plus_perplexity(self, text: str):
        """ An output from `encode_plus` for perplexity computation """
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

    def batch_encode_plus_perplexity(self, batch_text: List, batch_size: int = None):
        """ Batch version of `self.encode_plus_perplexity` """
        batch_size = len(batch_text) if batch_size is None else batch_size
        data = []
        for x in batch_text:
            data.append(self.encode_plus_perplexity(x))
        partition = get_partition(data)
        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))), batch_size=batch_size, shuffle=False, drop_last=False
        )
        return data_loader, partition

    def get_perplexity(self, batch_text: List, batch_size: int = 32):
        """ (pseudo) Perplexity """
        data_loader, partition = self.batch_encode_plus_perplexity(batch_text=batch_text, batch_size=batch_size)
        nll = self.__get_nll(data_loader)
        return list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))


if __name__ == '__main__':
    scorer = PPL('bert-base-cased')
    sentence = ['Red is the color of courage, of a warrior and a martyr.',
                'His father was a tailor and his mother was a midwife.', ]
    s = scorer.get_perplexity(sentence)
    print(s)
