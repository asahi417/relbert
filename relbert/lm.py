""" RelBERT: Get relational embedding from transformers language model. """
import os
import logging
from typing import List
from multiprocessing import Pool

import transformers
import torch

from .util import internet_connection

__all__ = 'RelBERT'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message


def custom_prompter(word_pair, template: str, mask_token: str = None):
    """ Transform word pair into string prompt. """
    token_mask = '<mask>'
    token_subject = '<subj>'
    token_object = '<obj>'
    assert len(word_pair) == 2, word_pair
    subj, obj = word_pair
    assert token_subject not in subj and token_object not in subj and token_mask not in subj
    assert token_subject not in obj and token_object not in obj and token_mask not in obj
    prompt = template.replace(token_subject, subj).replace(token_object, obj)
    if mask_token is not None:
        prompt = prompt.replace(token_mask, mask_token)
    return prompt


class Dataset(torch.utils.data.Dataset):

    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self,
                 tokenizer,
                 max_length: int,
                 template: str = None,
                 aggregation_mode: str = 'average_no_mask',
                 truncate_exceed_tokens: bool = True):
        self.tokenizer = tokenizer
        self.template = template
        self.aggregation_mode = aggregation_mode
        self.truncate_exceed_tokens = truncate_exceed_tokens
        self.m_len = self.tokenizer.model_max_length if max_length is None else max_length
        assert self.tokenizer.model_max_length >= self.m_len, f'{self.tokenizer.model_max_length} < {self.m_len}'

    def __call__(self, word_pair: List):
        """ Encoding a word pair or sentence. If the word pair is given use custom template. """
        sentence = custom_prompter(word_pair, self.template, self.tokenizer.mask_token)
        encode = self.tokenizer.encode_plus(sentence, max_length=self.m_len, truncation=True, padding='max_length')
        if not self.truncate_exceed_tokens:
            assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, f"exceeded length {encode['input_ids']}"
        encode['labels'] = self.input_ids_to_labels(encode['input_ids'])
        return encode

    def input_ids_to_labels(self, input_ids: List):
        if self.aggregation_mode == 'mask':
            assert self.tokenizer.mask_token_id in input_ids
            return list(map(lambda x: 1 if x == self.tokenizer.mask_token_id else 0, input_ids))
        elif self.aggregation_mode == 'average':
            return list(map(lambda x: 0 if x == self.tokenizer.pad_token_id else 1, input_ids))
        elif self.aggregation_mode == 'average_no_mask':
            return list(map(lambda x: 0 if x in [self.tokenizer.pad_token_id, self.tokenizer.mask_token_id] else 1, input_ids))
        else:
            raise ValueError(f'unknown aggregation mode: {self.aggregation_mode}')


class RelBERT:
    """ RelBERT: Get relational embedding from transformers language model. """

    def __init__(self,
                 model: str = 'relbert/relbert-roberta-large',
                 max_length: int = 64,
                 aggregation_mode: str = 'average_no_mask',
                 template: str = None,
                 truncate_exceed_tokens: bool = True):
        """ Get embedding from transformers language model.

        Parameters
        ----------
        model : str
            Transformers model alias.
        max_length : int
            Model length.
        aggregation_mode : str
            - `mask` to get the embedding for a word pair by [MASK] token, eg) (A, B) -> A [MASK] B
            - `average` to average embeddings over the context.
            - `cls` to get the embedding on the [CLS] token
        """
        self.truncate_exceed_tokens = truncate_exceed_tokens
        self.max_length = max_length
        self.model_name = model
        self.internet = internet_connection()
        self.model_config = transformers.AutoConfig.from_pretrained(model, local_files_only=not self.internet)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=not self.internet)

        self.template = None
        self.custom_template = None
        if 'relbert_config' in self.model_config.to_dict().keys():
            logging.info(f'loading finetuned RelBERT model {model}')
            self.aggregation_mode = self.model_config.relbert_config['aggregation_mode']
            self.template = self.model_config.relbert_config['template']
            self.is_trained = True
        else:
            assert template is not None, 'template is required for untrained RelBERT'
            self.aggregation_mode = aggregation_mode
            self.template = template
            self.is_trained = False
            self.model_config.update({'relbert_config': {'aggregation_mode': aggregation_mode, 'template': self.template}})
        self.model = transformers.AutoModel.from_pretrained(model, config=self.model_config, local_files_only=not self.internet)
        self.hidden_size = self.model.config.hidden_size

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.info(f'language model running on {torch.cuda.device_count()} GPU')

    def train(self): self.model.train()

    def eval(self): self.model.eval()

    def save(self, cache_dir):
        if self.parallel:
            self.model.module.save_pretrained(cache_dir)
        else:
            self.model.save_pretrained(cache_dir)
        self.tokenizer.save_pretrained(cache_dir)

    def encode_word_pairs(self, word_pairs, parallel: bool = True):
        """ Return a dictionary of encoded word_pair."""
        logging.info('encode word pairs')
        assert all(type(i) is tuple or list for i in word_pairs), word_pairs
        logging.info(f'\t original: {len(word_pairs)}')
        word_pairs_dict = {'__'.join(p): p for p in word_pairs}
        word_pairs_dict_key = sorted(list(set(word_pairs_dict.keys())))
        word_pairs = [word_pairs_dict[k] for k in word_pairs_dict_key]
        logging.info(f'\t deduplicate: {len(word_pairs_dict_key)}')
        if parallel:
            pool = Pool()
            encode = pool.map(EncodePlus(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                template=self.template,
                aggregation_mode=self.aggregation_mode,
                truncate_exceed_tokens=self.truncate_exceed_tokens), word_pairs)
            pool.close()
        else:
            process = EncodePlus(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                template=self.template,
                aggregation_mode=self.aggregation_mode,
                truncate_exceed_tokens=self.truncate_exceed_tokens)
            encode = []
            for p in word_pairs:
                encode.append(process(p))
        return {k: e for k, e in zip(word_pairs_dict_key, encode)}

    def to_embedding(self, encode, batch_size: int = None):
        """ Compute embedding from batch of encode. """
        labels = encode.pop('labels')
        size = len(labels)
        if batch_size is None or batch_size > size:
            output = self.model(**{k: v.to(self.device) for k, v in encode.items()}, return_dict=True)
            return (output['last_hidden_state'] * labels.to(self.device).reshape(len(labels), -1, 1)).sum(1)
        else:
            chunks = list(range(0, size, batch_size)) + [size]
            segment = [(a, b) for a, b in zip(chunks[:-1], chunks[1:])]
            last_hidden_state = []
            for s, e in segment:
                output = self.model(**{k: v[s:e].to(self.device) for k, v in encode.items()}, return_dict=True)
                last_hidden_state.append(output['last_hidden_state'])
            last_hidden_state = torch.concat(last_hidden_state)
            labels = labels[:len(last_hidden_state)].to(self.device)
            embeddings = (last_hidden_state * labels.reshape(len(labels), -1, 1)).sum(1)
            return embeddings

    def get_embedding(self, x: List, batch_size: int = None):
        """ Get embedding from RelBERT (no gradient).

        Parameters
        ----------
        x : list
            List of word pairs or sentence
        batch_size : int
            Batch size.

        Returns
        -------
        Embedding (len(x), n_hidden)
        """
        is_single_list = False
        if len(x) == 2 and type(x[0]) is str and type(x[1]) is str:
            x = [x]
            is_single_list = True

        if not all(type(i) is tuple for i in x):
            x = [tuple(i) for i in x]

        batch_size = len(x) if batch_size is None else batch_size
        encoded_pair_dict = self.encode_word_pairs(x)
        pair_key = list(encoded_pair_dict.keys())
        data_loader = torch.utils.data.DataLoader(
            Dataset([encoded_pair_dict[k] for k in pair_key]), batch_size=batch_size, shuffle=False, drop_last=False)
        logging.debug('\t * run LM inference')
        h_list = []
        with torch.no_grad():
            for encode in data_loader:
                h_list += self.to_embedding(encode).cpu().tolist()
        h_dict = {p: h for h, p in zip(h_list, pair_key)}
        h_return = [h_dict['__'.join(p)] for p in x]
        if is_single_list:
            assert len(h_return) == 1
            return h_return[0]
        return h_return
