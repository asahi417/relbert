import os
import logging
import pickle
import json
import random
from itertools import chain
from typing import List
from multiprocessing import Pool
from itertools import combinations, product
from tqdm import tqdm
import torch

from ..list_keeper import ListKeeper
from ..config import Config
from ..util import fix_seed, load_language_model, triplet_loss, Dataset, get_linear_schedule_with_warmup
from ..data import get_training_data
from ..trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = ''


class PromptEmbedding(torch.nn.Module):
    """ LSTM trigger embedding for ptuning. """

    def __init__(self,
                 hidden_size: int = None,
                 path_to_weight: str = None,
                 n_trigger_b: int = 1,
                 n_trigger_i: int = 1,
                 n_trigger_e: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        if path_to_weight is not None:
            with open(path_to_weight, 'rb') as f:
                weight = pickle.load(f)
            self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(weight))
            self.mlp_head = self.lstm_head = None
        else:
            assert hidden_size
            length = n_trigger_i + n_trigger_b + n_trigger_e
            self.embedding = torch.nn.Embedding(length, hidden_size)
            self.lstm_head = torch.nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=2, dropout=dropout,
                bidirectional=True, batch_first=True)
            self.mlp_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
        self.seq_indices = torch.arange(self.embedding.num_embeddings)  # input

    def forward(self):
        out = self.embedding(self.seq_indices).unsqueeze(0)
        if self.mlp_head is not None and self.lstm_head is not None:
            out = self.mlp_head(self.lstm_head(out)[0]).squeeze()
        return out

    def save(self, export_path: str):
        with open(export_path, 'wb') as f:
            pickle.dump(self.embedding.weight.cpu().tolist(), f)


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing for Ptuning. """

    def __init__(self, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or self.tokenizer.model_max_length
        assert self.max_length <= self.tokenizer.model_max_length, '{} < {}'.format(
            self.max_length, self.tokenizer.model_max_length)

    def __call__(self, sentence: str):
        encode = self.tokenizer.encode_plus(sentence, max_length=self.max_length, truncation=True, padding='max_length')
        assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded length {}'.format(encode['input_ids'])
        # label for token to be aggregated as an embedding
        encode['labels'] = list(map(lambda x: 0 if x == self.tokenizer.pad_token_id else 1, encode['input_ids']))
        return encode


class ContinuousTriggerEmbedding(Trainer):

    def __init__(self,
                 export: str,
                 epoch: int = 5,
                 momentum: float = 0.9,
                 lr: float = 0.001,
                 lr_warmpu: int = 100,
                 lr_decay: bool = False,
                 optimizer: str = 'adam',
                 pseudo_token: str = '[PROMPT]',
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
        super(ContinuousTriggerEmbedding, self).__init__()
        fix_seed(random_seed)
        self.config = Config(
            export=export,
            config_name='prompter_config',
            optimizer=optimizer,
            lr=lr,
            lr_warmpu=lr_warmpu,
            lr_decay=lr_decay,
            momentum=momentum,
            epoch=epoch,
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
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_string(self.config.pseudo_token)
        self.model.eval()
        self.prompter = PromptEmbedding(
            hidden_size=self.model.config.hidden_size, n_trigger_b=self.config.n_trigger_b,
            n_trigger_i=self.config.n_trigger_i, n_trigger_e=self.config.n_trigger_e)
        self.input_embeddings = self.model.get_input_embeddings()

        # cache config
        self.batch = batch
        self.checkpoint_dir = self.config.cache_dir

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        self.prompter.to(self.device)
        logging.info('language model running on {} GPU'.format(torch.cuda.device_count()))
        # get dataset
        self.all_positive, self.all_negative, self.relation_structure = get_training_data(
            data_name=self.config.data, n_sample=self.config.n_sample, cache_dir=cache_dir)

        # calculate the number of trial to cover all combination in batch
        n_pos = min(len(i) for i in self.all_positive.values())
        n_neg = min(len(i) for i in self.all_negative.values())
        self.n_trial = len(list(product(combinations(range(n_pos), 2), range(n_neg))))

        # setup optimizer
        model_parameters = list(self.prompter.named_parameters())
        self.linear = None
        self.discriminative_loss = None
        if softmax_loss:
            logging.info('add linear layer for softmax_loss')
            self.linear = torch.nn.Linear(self.model.config.hidden_size * 3, 1)  # three way feature
            self.linear.weight.data.normal_(std=0.02)
            self.discriminative_loss = torch.nn.BCELoss()
            self.linear.to(self.device)
            model_parameters += list(self.linear.named_parameters())

        if self.config.optimizer == 'adamax':
            self.optimizer = torch.optim.Adamax(model_parameters, lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model_parameters, lr=self.config.lr, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.AdamW(model_parameters, lr=self.config.lr)
        else:
            raise ValueError('unknown optimizer: {}'.format(self.config.optimizer))

        # scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.lr_warmup,
            num_training_steps=self.config.epoch if self.config.lr_decay else None)

        assert self.all_negative.keys() == self.all_positive.keys()
        logging.info('{} positive data/{} negative data'.format(len(self.all_positive), len(self.all_negative)))
        logging.info('{} trial'.format(self.n_trial))

    def convert_pairs_to_prompt(self, pairs):
        def convert_pair_to_prompt(h, t):
            prompt = [self.config.pseudo_token] * self.config.n_trigger_b + [h]
            prompt += [self.config.pseudo_token] * self.config.n_trigger_i + [t]
            prompt += [self.config.pseudo_token] * self.config.n_trigger_i
            return ' '.join(prompt)
        return [convert_pair_to_prompt(_h, _t) for _h, _t in pairs]

    def preprocess(self):
        """ Encoding data and returns torch.utils.data.Dataset. """

        shared = {'tokenizer': self.tokenizer, 'max_length': self.config.max_length}

        def pool_map(_list):
            pool = Pool()
            out = pool.map(EncodePlus(**shared), _list)
            pool.close()
            return out

        key = list(self.all_positive.keys())
        positive_samples_list = ListKeeper([self.all_positive[k] for k in key])
        p_prompt_out = self.convert_pairs_to_prompt(positive_samples_list.flatten_list)

        negative_samples_list = ListKeeper([self.all_negative[k] for k in key])
        n_prompt_out = self.convert_pairs_to_prompt(negative_samples_list.flatten_list)
        positive_embedding = pool_map(p_prompt_out)
        negative_embedding = pool_map(n_prompt_out)
        if any(i is None for i in positive_embedding) or any(i is None for i in negative_embedding):
            return None
        positive_embedding = positive_samples_list.restore_structure(positive_embedding)
        positive_embedding = {key[n]: v for n, v in enumerate(positive_embedding)}
        negative_embedding = negative_samples_list.restore_structure(negative_embedding)
        negative_embedding = {key[n]: v for n, v in enumerate(negative_embedding)}
        if self.config.parent_contrast:
            return dict(positive_samples=positive_embedding, negative_samples=negative_embedding,
                        relation_structure=self.relation_structure)
        else:
            return dict(positive_samples=positive_embedding, negative_samples=negative_embedding)

    def get_prompt(self, num_workers: int = 1):
        """ Train prompter.

        Parameter
        ----------
        num_workers : int
            Workers for DataLoader.
        """
        # logging.info('start prompt generation')
        # loss = None
        # for i in range(self.config.last_iter, self.config.n_iteration):
        #     filter_matrix, loss = self.__single_iteration(num_workers, filter_matrix)
        #     if loss is None:
        #         logging.info('early exit: no more updates')
        #         break
        #     logging.info('iteration {}/{}: {}\t loss {}'.format(
        #         i + 1, self.config.n_iteration, self.tokenizer.convert_ids_to_tokens(self.prompter.triggers), loss))
        #     self.prompter.save('{}/prompt.{}.json'.format(self.config.cache_dir, i), loss)
        #     mode = 'a' if os.path.exists('{}/loss.txt'.format(self.config.cache_dir)) else 'w'
        #     with open('{}/loss.txt'.format(self.config.cache_dir), mode) as f:
        #         f.write('{}\n'.format(loss))
        # self.prompter.save('{}/prompt.json'.format(self.config.cache_dir), loss)

        logging.info('start model training')
        param = self.preprocess()
        batch_index = list(range(self.n_trial))
        global_step = 0

        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.config.epoch):  # loop over the epoch
                random.shuffle(batch_index)
                for n, bi in enumerate(batch_index):
                    dataset = Dataset(deterministic_index=bi, **param)
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=self.config.batch, shuffle=True, num_workers=num_workers, drop_last=True)
                    mean_loss, global_step = self.train_single_epoch(loader, global_step=global_step)
                    inst_lr = self.optimizer.param_groups[0]['lr']
                    logging.info('[epoch {}/{}, batch_id {}/{}] average loss: {}, lr: {}'.format(
                        e, self.config.epoch, n, self.n_trial, round(mean_loss, 3), inst_lr))

        cache_dir = '{}/epoch_{}'.format(self.checkpoint_dir, e + 1)
        os.makedirs(cache_dir, exist_ok=True)
        self.save(cache_dir)
        logging.info('complete training: model ckpt was saved at {}'.format(self.checkpoint_dir))

    def get_logit(self, encode):
        encode.pop()


    def train_single_epoch(self, data_loader, global_step: int):
        total_loss = 0
        bce = torch.nn.BCELoss()
        step_in_epoch = len(data_loader)
        for x in data_loader:
            global_step += 1
            self.optimizer.zero_grad()
            positive_a = {k: v.to(self.device) for k, v in x['positive_a'].items()}
            positive_b = {k: v.to(self.device) for k, v in x['positive_b'].items()}
            negative = {k: v.to(self.device) for k, v in x['negative'].items()}
            if self.config.parent_contrast:
                positive_hc = {k: v.to(self.device) for k, v in x['positive_parent'].items()}
                negative_hc = {k: v.to(self.device) for k, v in x['negative_parent'].items()}
                encode = {k: torch.cat([positive_a[k], positive_b[k], negative[k], positive_hc[k], negative_hc[k]])
                          for k in positive_a.keys()}

                embedding = self.get_logit(encode)
                v_anchor, v_positive, v_negative, v_positive_hc, v_negative_hc = embedding.chunk(5)

                # contrastive loss
                loss = triplet_loss(v_anchor, v_positive, v_negative, v_positive_hc, v_negative_hc,
                                    margin=self.config.mse_margin, in_batch_negative=self.config.in_batch_negative)
            else:
                encode = {k: torch.cat([positive_a[k], positive_b[k], negative[k]]) for k in positive_a.keys()}
                embedding = self.lm.to_embedding(encode)
                v_anchor, v_positive, v_negative = embedding.chunk(3)

                # contrastive loss
                loss = triplet_loss(v_anchor, v_positive, v_negative,
                                    margin=self.config.mse_margin, in_batch_negative=self.config.in_batch_negative)

            if self.linear is not None:
                # the 3-way discriminative loss used in SBERT
                feature_positive = torch.cat([v_anchor, v_positive, torch.abs(v_anchor - v_positive)], dim=1)
                feature_negative = torch.cat([v_anchor, v_negative, torch.abs(v_anchor - v_negative)], dim=1)
                feature = torch.cat([feature_positive, feature_negative])
                pred = torch.sigmoid(self.linear(feature))
                label = torch.tensor([1] * len(feature_positive) + [0] * len(feature_negative),
                                     dtype=torch.float32, device=self.lm.device)
                loss += bce(pred, label.unsqueeze(-1))

            # backward: calculate gradient
            self.scaler.scale(loss).backward()
            inst_loss = loss.cpu().item()

            # aggregate average loss over epoch
            total_loss += inst_loss

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        return total_loss / step_in_epoch, global_step
