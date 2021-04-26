import os
import logging
import json
from typing import List, Dict

import torch
from .discrete_tuning import preprocess
from ..config import Config
from ..util import load_language_model
from ..trainer import BaseTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = 'ContinuousTriggerEmbedding'


class PromptEmbedding(torch.nn.Module):
    """ LSTM trigger embedding for ptuning. """

    def __init__(self,
                 pseudo_token: str,
                 hidden_size: int,
                 n_trigger_b: int = 1,
                 n_trigger_i: int = 1,
                 n_trigger_e: int = 1,
                 dropout: float = 0.0,
                 device: str = 'cpu'):
        super().__init__()
        self.pseudo_token = pseudo_token
        self.embedding = torch.nn.Embedding(n_trigger_i + n_trigger_b + n_trigger_e, hidden_size).to(device)
        self.n_trigger_i, self.n_trigger_b, self.n_trigger_e = n_trigger_i, n_trigger_b, n_trigger_e
        self.lstm_head = torch.nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=2, dropout=dropout,
            bidirectional=True, batch_first=True).to(device)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size)
        ).to(device)
        self.seq_indices = torch.arange(self.embedding.num_embeddings).to(device)  # input

    def forward(self):
        out = self.embedding(self.seq_indices).unsqueeze(0)
        return self.mlp_head(self.lstm_head(out)[0]).squeeze()

    def save(self, export_file: str):
        assert export_file.endswith('.json')
        with open(export_file, 'w') as f:
            json.dump({'n_trigger_b': self.n_trigger_b,
                       'n_trigger_i': self.n_trigger_i,
                       'n_trigger_e': self.n_trigger_e,
                       'pseudo_token': self.pseudo_token,
                       'embedding': self.embedding.weight.cpu().tolist()}, f)
        logging.debug('exported to {}'.format(export_file))


class ContinuousTriggerEmbedding(BaseTrainer):

    def __init__(self,
                 export: str,
                 epoch: int = 5,
                 momentum: float = 0.9,
                 lr: float = 0.001,
                 lr_warmup: int = 100,
                 lr_decay: bool = False,
                 optimizer: str = 'adam',
                 weight_decay: float = 0.0,
                 pseudo_token: str = '<prompt>',
                 n_trigger_i: int = 1,
                 n_trigger_b: int = 1,
                 n_trigger_e: int = 1,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 data: str = 'semeval2012',
                 n_sample: int = 5,
                 softmax_loss: bool = True,
                 in_batch_negative: bool = True,
                 parent_contrast: bool = True,
                 mse_margin: float = 1,
                 batch: int = 16,
                 random_seed: int = 0,
                 cache_dir: str = None,
                 fp16: bool = False):
        super(ContinuousTriggerEmbedding, self).__init__(cache_dir=cache_dir)
        self.config = Config(
            export=export,
            optimizer=optimizer,
            weight_decay=weight_decay,
            lr=lr,
            lr_warmup=lr_warmup,
            lr_decay=lr_decay,
            momentum=momentum,
            epoch=epoch,
            batch=batch,
            pseudo_token=pseudo_token,
            n_trigger_i=n_trigger_i,
            n_trigger_b=n_trigger_b,
            n_trigger_e=n_trigger_e,
            model=model,
            max_length=max_length,
            data=data,
            n_sample=n_sample,
            softmax_loss=softmax_loss,
            in_batch_negative=in_batch_negative,
            parent_contrast=parent_contrast,
            mse_margin=mse_margin,
            random_seed=random_seed,
            fp16=fp16)
        # model setup
        self.tokenizer, self.model, _ = load_language_model(self.config.model, cache_dir)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.config.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.config.pseudo_token)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        try:
            embedding_size = self.model.config.embedding_size
        except AttributeError:
            embedding_size = self.model.config.hidden_size
        self.input_embeddings = self.model.get_input_embeddings()

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.prompter = PromptEmbedding(
            pseudo_token=self.config.pseudo_token,
            hidden_size=embedding_size,
            n_trigger_b=self.config.n_trigger_b,
            n_trigger_i=self.config.n_trigger_i,
            n_trigger_e=self.config.n_trigger_e,
            device=self.device)
        self.model_parameters = list(self.prompter.named_parameters())

        logging.info('language model running on {} GPU'.format(torch.cuda.device_count()))
        self.setup()

    def save(self, current_epoch=None):
        if current_epoch is None:
            self.prompter.save('{}/prompt.json'.format(self.config.cache_dir))
        else:
            self.prompter.save('{}/prompt.{}.json'.format(self.config.cache_dir, current_epoch))

    def model_output(self, encode):
        encode = {k: v.to(self.device) for k, v in encode.items()}
        labels = encode.pop('labels')
        input_ids = encode.pop('input_ids')
        mask = input_ids == self.pseudo_token_id
        input_ids[mask] = self.tokenizer.unk_token_id
        embedding = self.input_embeddings(input_ids)
        trigger_embedding = self.prompter()
        for i in range(len(mask)):
            embedding[i][mask[i], :] = trigger_embedding
        encode['inputs_embeds'] = embedding
        output = self.model(**encode, return_dict=True)
        batch_embedding_tensor = (output['last_hidden_state'] * labels.reshape(len(labels), -1, 1)).sum(1)
        return batch_embedding_tensor

    def convert_pairs_to_prompt(self, pairs):
        def convert_pair_to_prompt(h, t):
            prompt = [self.config.pseudo_token] * self.config.n_trigger_b + [h]
            prompt += [self.config.pseudo_token] * self.config.n_trigger_i + [t]
            prompt += [self.config.pseudo_token] * self.config.n_trigger_e
            return ' '.join(prompt)
        return [convert_pair_to_prompt(_h, _t) for _h, _t in pairs]

    def get_prompt(self, num_workers: int = 1):
        self.train(num_workers)
        self.save()

    def preprocess(self, positive_samples, negative_samples: Dict = None, relation_structure: Dict = None):
        """ Encoding data and returns torch.utils.data.Dataset. """
        return preprocess(self.tokenizer, self.config.max_length, self.convert_pairs_to_prompt, False,
                          positive_samples, negative_samples, relation_structure)
