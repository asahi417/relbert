""" RelBERT: Get relational embedding from transformers language model. """
import os
import logging
import json
from typing import Dict, List
from multiprocessing import Pool

import transformers
import torch

from .list_keeper import ListKeeper
from .util import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = 'RelBERT'
preset_templates = {
        "a": "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>",
        "b": "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is A's <mask>",
        "c": "Today, I finally discovered the relation between <subj> and <obj> : <mask>",
        "d": "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>",
        "e": "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is A’s <mask>",
        "f": "The teacher explained how <subj> is related to <obj> : it is <obj> 's <mask>",
        "g": "The teacher explained how <subj> is related to <obj> : it is <mask>",
        "h": "The teacher explained how <subj> is related to <obj> : <mask>",
        "i": "The teacher explained how <subj> is related to <obj> : it is the <mask>"
    }


def custom_prompter(word_pair, template_type: str = 'a', mask_token: str = None):
    """ Transform word pair into string prompt. """

    token_mask = '<mask>'
    token_subject = '<subj>'
    token_object = '<obj>'

    # if custom_template is not None:
    #     assert token_mask in custom_template, 'mask token not found: {}'.format(custom_template)
    #     assert token_subject in custom_template, 'subject token not found: {}'.format(custom_template)
    #     assert token_object in custom_template, 'object token not found: {}'.format(custom_template)
    #     template = custom_template
    # else:
    template = preset_templates[template_type]

    assert len(word_pair) == 2, word_pair
    subj, obj = word_pair
    assert token_subject not in subj and token_object not in subj and token_mask not in subj
    assert token_subject not in obj and token_object not in obj and token_mask not in obj
    prompt = template.replace(token_subject, subj).replace(token_object, obj)
    if mask_token is not None:
        prompt = prompt.replace(token_mask, mask_token)
    return prompt


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self,
                 tokenizer,
                 max_length: int,
                 custom_template_type: str = 'a',
                 template: Dict = None,
                 mode: str = 'average_no_mask'):
        assert custom_template_type or template
        self.custom_template_type = custom_template_type
        self.template = template
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.mode = mode
        if max_length is not None:
            assert self.max_length >= max_length, '{} < {}'.format(self.max_length, max_length)
            self.max_length = max_length

    def __call__(self, word_pair: List):
        """ Encoding a word pair or sentence. If the word pair is given use custom template."""
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        if all(type(i) is str for i in word_pair):
            logging.warning('receive sentence instead of word: {}'.format(word_pair))
            sentence = word_pair
        else:
            if self.template:
                top = self.template['top']
                mid = self.template['mid']
                bottom = self.template['bottom']
                h, t = word_pair
                mask = self.tokenizer.mask_token
                assert h != mask and t != mask
                token_ids = self.tokenizer.encode(
                    ' '.join([mask] * len(top) + [h] + [mask] * len(mid) + [t] + [mask] * len(bottom)))
                token_ids = [-100 if i == self.tokenizer.mask_token_id else i for i in token_ids]
                for i in top + mid + bottom:
                    token_ids[token_ids.index(-100)] = i
                sentence = self.tokenizer.decode(token_ids)
            else:
                sentence = custom_prompter(word_pair, self.custom_template_type, self.tokenizer.mask_token)
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
                 max_length: int = 128,
                 cache_dir: str = None,
                 mode: str = 'average_no_mask',
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
        template_type : str
            Custom template type or path to prompt json file that contains 'top'/'mid'/'bottom'.
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
        self.custom_template_type = None
        self.template = None
        if 'relbert_config' in self.config.to_dict().keys():
            logging.info('loading finetuned RelBERT model')
            self.mode = self.config.relbert_config['mode']
            self.custom_template_type = self.config.relbert_config['custom_template_type']
            self.template = self.config.relbert_config['template']
            self.is_trained = True
        else:
            self.mode = mode
            self.is_trained = False
            if template_type in preset_templates:
                self.custom_template_type = template_type
            else:
                with open(template_type, 'r') as f:
                    self.template = json.load(f)
            self.config.update({'relbert_config': {'mode': mode, 'custom_template_type': self.custom_template_type,
                                                   'template': self.template}})
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
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.info('language model running on {} GPU'.format(torch.cuda.device_count()))
        logging.info('\t * template       : {}'.format(self.template))
        logging.info('\t * custom template: {}'.format(self.custom_template_type))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, cache_dir):
        if self.parallel:
            self.model.module.save_pretrained(cache_dir)
        else:
            self.model.save_pretrained(cache_dir)
        self.tokenizer.save_pretrained(cache_dir)

    def preprocess(self,
                   positive_samples,
                   negative_sample: Dict = None,
                   relation_structure: Dict = None,
                   parallel: bool = True,
                   pairwise_input: bool = True):
        """ Preprocess textual data.

        Parameters
        ----------
        positive_samples : List or Dict
            1D array with string (for prediction) or dictionary with 2D array (for training)
        negative_sample : Dict
        parallel : bool
            Parallelize data processing part over CPUs.

        Returns
        -------
        torch.utils.data.Dataset
        """
        if type(positive_samples) is not dict:
            assert relation_structure is None
            assert negative_sample is None
            assert len(positive_samples) > 0, len(positive_samples)
            assert type(positive_samples) is list and all(type(i) is tuple for i in positive_samples)
            positive_samples = {k: [v] for k, v in enumerate(positive_samples)}

        key = list(positive_samples.keys())
        positive_samples_list = ListKeeper([positive_samples[k] for k in key])

        logging.debug('{} positive data to encode'.format(len(positive_samples)))
        negative_sample_list = None
        if negative_sample is not None:
            logging.debug('preparing negative data')
            assert len(negative_sample) > 0, len(negative_sample)
            assert negative_sample.keys() == positive_samples.keys()
            negative_sample_list = ListKeeper([negative_sample[k] for k in key])

        shared = {'tokenizer': self.tokenizer, 'max_length': self.max_length, 'mode': self.mode,
                  'template': self.template, 'custom_template_type': self.custom_template_type}

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
                       relation_structure=relation_structure,
                       pairwise_input=pairwise_input)

    def to_embedding(self, encode):
        """ Compute embedding from batch of encode. """
        encode = {k: v.to(self.device) for k, v in encode.items()}
        labels = encode.pop('labels')
        output = self.model(**encode, return_dict=True)
        batch_embedding_tensor = (output['last_hidden_state'] * labels.reshape(len(labels), -1, 1)).sum(1)
        return batch_embedding_tensor

    def get_embedding(self, x: List, batch_size: int = None, num_worker: int = 1, parallel: bool = True):
        """ Get embedding from RelBERT (no gradient).

        Parameters
        ----------
        x : list
            List of word pairs or sentence
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

        data = self.preprocess(x, parallel=parallel, pairwise_input=False)
        batch_size = len(x) if batch_size is None else batch_size
        data_loader = torch.utils.data.DataLoader(
            data, num_workers=num_worker, batch_size=batch_size, shuffle=False, drop_last=False)

        logging.debug('\t * run LM inference')
        h_list = []
        with torch.no_grad():
            for encode in data_loader:
                h_list += self.to_embedding(encode).cpu().tolist()
        return h_list
