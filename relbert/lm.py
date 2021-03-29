""" RelBERT: Get relational embedding from transformers language model. """
import os
import logging
from typing import Dict, List
from itertools import combinations, chain
from multiprocessing import Pool
from random import randint

import transformers
import torch

from .prompt import word_pair_prompter
from .list_keeper import ListKeeper

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = 'BERT'


class Dataset(torch.utils.data.Dataset):
    """ Dataset loader for triplet loss. """
    float_tensors = ['attention_mask']

    def __init__(self, positive_samples: Dict, negative_samples: Dict = None, pairwise_input: bool = True):
        if negative_samples is not None:
            assert positive_samples.keys() == negative_samples.keys()
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.pairwise_input = pairwise_input
        if self.pairwise_input:
            self.keys = sorted(list(positive_samples.keys()))
            self.positive_pattern_id = {k: list(combinations(range(len(self.positive_samples[k])), 2)) for k in self.keys}
        else:
            NotImplementedError()
            # assert self.negative_samples is None
            # self.keys =
            # self.positive_pattern_id = {k: list(range(len(self.positive_samples[k]))) for k in self.keys}
            # self.key_mapper = list(chain(*[[k] * len(self.positive_pattern_id[k]) for k in keys]))
            # self.positive_pattern_flat = list(chain(*[self.positive_pattern_id[k] for k in keys]))

    def __len__(self):
        return len(self.keys)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        relation_type = self.keys[idx]
        patterns = self.positive_pattern_id[relation_type]
        if self.pairwise_input:
            i = randint(0, len(patterns) - 1)
            a, b = patterns[i]
            # a, b = self.positive_pattern_flat[idx]
            positive_a = self.positive_samples[relation_type][a]
            tensor_positive_a = {k: self.to_tensor(k, v) for k, v in positive_a.items()}
            positive_b = self.positive_samples[relation_type][b]
            tensor_positive_b = {k: self.to_tensor(k, v) for k, v in positive_b.items()}

            negative_list = self.negative_samples[relation_type]
            i = randint(0, len(negative_list) - 1)
            tensor_negative = {k: self.to_tensor(k, v) for k, v in negative_list[i].items()}
            return {'positive_a': tensor_positive_a, 'positive_b': tensor_positive_b, 'negative': tensor_negative}
        else:
            NotImplementedError()
            # a = self.positive_pattern_flat[idx]
            # positive_a = self.positive_samples[relation_type][a]
            # tensor_positive_a = {k: self.to_tensor(k, v) for k, v in positive_a.items()}
            # return {'positive_a': tensor_positive_a}


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self, tokenizer, max_length: int, template_type: str = 'a', mode: str = 'mask'):
        self.template_type = template_type
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.mode = mode
        if max_length is not None:
            assert self.max_length >= max_length, '{} < {}'.format(self.max_length, max_length)
            self.max_length = max_length

    def __call__(self, word_pair: List):
        """ Encoding a word pair or sentence. """
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        sentence = word_pair_prompter(word_pair, template_type=self.template_type, mask_token=self.tokenizer.mask_token)
        encode = self.tokenizer.encode_plus(sentence, **param)
        assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded length {}'.format(encode['input_ids'])
        encode['labels'] = self.input_ids_to_labels(encode['input_ids'])
        return encode

    def input_ids_to_labels(self, input_ids: List):
        if self.mode == 'mask':
            assert self.tokenizer.mask_token_id in input_ids
            return list(map(lambda x: 1 if x == self.tokenizer.mask_token_id else 0, input_ids))
        elif self.mode == 'average':
            return list(map(lambda x: 0 if x == self.tokenizer.pad_token_id else 1, input_ids))
        elif self.mode == 'average_no_mask':
            return list(map(lambda x: 0 if x in [self.tokenizer.pad_token_id, self.tokenizer.mask_token_id] else 1, input_ids))
        else:
            raise ValueError('unknown mode {}'.format(self.mode))


class RelBERT:
    """ RelBERT: Get relational embedding from transformers language model. """

    def __init__(self,
                 model: str,
                 max_length: int = 32,
                 cache_dir: str = None,
                 mode: str = 'mask',
                 template_type: str = 'a'):
        """ Get embedding from transformers language model.

        Parameters
        ----------
        model : str
            Transformers model alias.
        max_length : int
            Model length.
        cache_dir : str
        mode : str
            - `mask` to get the embedding for a word pair by [MASK] token, eg) (A, B) -> A [MASK] B
            - `average` to average embeddings over the context.
            - `cls` to get the embedding on the [CLS] token
        """
        assert 'bert' in model, '{} is not BERT'.format(model)
        self.model_name = model
        self.cache_dir = cache_dir
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        except ValueError:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir,
                                                                        local_files_only=True)
        try:
            self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir)
        except ValueError:
            self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir,
                                                                  local_files_only=True)
        # check if the language model is RelBERT trained or not.
        if 'relbert_config' in self.config.to_dict().keys():
            self.mode = self.config.relbert_config['mode']
            self.template_type = self.config.relbert_config['template_type']
            self.is_trained = True
        else:
            self.config.update({'relbert_config': {'mode': mode, 'template_type': template_type}})
            self.mode = mode
            self.template_type = template_type
            self.is_trained = False
        try:
            self.model = transformers.AutoModel.from_pretrained(
                self.model_name, config=self.config, cache_dir=self.cache_dir)
        except ValueError:
            self.model = transformers.AutoModel.from_pretrained(
                self.model_name, config=self.config, cache_dir=self.cache_dir, local_files_only=True)

        # classifier weight
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.max_length = max_length

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        logging.info('BERT running on {} GPU'.format(torch.cuda.device_count()))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, cache_dir):
        self.model.save_pretrained(cache_dir)
        self.tokenizer.save_pretrained(cache_dir)

    def preprocess(self,
                   positive_samples,
                   negative_sample: List = None,
                   parallel: bool = True,
                   pairwise_input: bool = True):
        """ Preprocess textual data.

        Parameters
        ----------
        positive_samples : List
        negative_sample : List
        parallel : bool
            Parallelize data processing part over CPUs.

        Returns
        -------
        torch.utils.data.Dataset
        """
        if type(positive_samples) is not dict:
            assert len(positive_samples) > 0, len(positive_samples)
            positive_samples = [positive_samples] if type(positive_samples[0]) is str else positive_samples
            positive_samples = {k: v for k, v in enumerate(positive_samples)}

        key = list(positive_samples.keys())
        positive_samples_list = ListKeeper([positive_samples[k] for k in key])

        logging.debug('{} positive data to encode'.format(len(positive_samples)))
        negative_sample_list = None
        if negative_sample is not None:
            logging.debug('preparing negative data')
            assert len(negative_sample) > 0, len(negative_sample)
            if type(negative_sample) is not dict:
                negative_sample = [negative_sample] if type(negative_sample[0]) is str else negative_sample
                negative_sample = {k: v for k, v in enumerate(negative_sample)}
            assert negative_sample.keys() == positive_samples.keys()
            negative_sample_list = ListKeeper([negative_sample[k] for k in key])

        shared = {'tokenizer': self.tokenizer, 'max_length': self.max_length, 'template_type': self.template_type, 'mode': self.mode}

        def pool_map(_list):
            if parallel:
                pool = Pool()
                out = pool.map(EncodePlus(**shared), _list)
                pool.close()
            else:
                out = list(map(EncodePlus(**shared), _list))
            return out

        positive_embedding = pool_map(positive_samples_list.flatten_list)
        positive_embedding = positive_samples_list.restore_structure(positive_embedding)
        positive_embedding = {key[n]: v for n, v in enumerate(positive_embedding)}

        negative_embedding = None
        if negative_sample_list is not None:
            negative_embedding = pool_map(negative_sample_list.flatten_list)
            negative_embedding = negative_sample_list.restore_structure(negative_embedding)
            negative_embedding = {key[n]: v for n, v in enumerate(negative_embedding)}

        return Dataset(positive_samples=positive_embedding,
                       negative_samples=negative_embedding,
                       pairwise_input=pairwise_input)

    def to_embedding(self, encode):
        """ Compute embedding from batch of encode. """
        encode = {k: v.to(self.device) for k, v in encode.items()}
        labels = encode.pop('labels')
        output = self.model(**encode, return_dict=True)
        batch_embedding_tensor = (output['last_hidden_state'] * labels.reshape(len(labels), -1, 1)).sum(1)
        return batch_embedding_tensor

    def get_embedding(self, x: List, batch_size: int = None, num_worker: int = 0, parallel: bool = True):
        """ Get embedding from BERT.

        Parameters
        ----------
        x : list
            List of word pairs.
        batch_size : int
            Batch size.
        num_worker : int
            Dataset worker number.
        parallel : boo;
            Parallelize data processing part over CPUs.

        Returns
        -------
        Embedding (len(x), n_hidden).
        """
        data = self.preprocess(x, parallel=parallel)
        batch_size = len(x) if batch_size is None else batch_size
        data_loader = torch.utils.data.DataLoader(
            data, num_workers=num_worker, batch_size=batch_size, shuffle=False, drop_last=False)

        logging.debug('\t* run LM inference')
        h_list = []
        for encode in data_loader:
            h_list += self.to_embedding(encode[0]).cpu().to_list()
        return h_list
