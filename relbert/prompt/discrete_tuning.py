""" Gradient based trigger search inspired by AutoPrompt https://github.com/ucinlp/autoprompt """
import os
import logging
import json
import random
from itertools import chain
from typing import List, Dict
from multiprocessing import Pool
from itertools import combinations, product
from tqdm import tqdm
import torch

from ..list_keeper import ListKeeper
from ..config import Config
from ..util import fix_seed, load_language_model, triplet_loss, Dataset
from ..data import get_training_data
from ..lm import EncodePlus

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = 'GradientTriggerSearch'
MAX_GRADIENT_VALUE = 2000


def preprocess(tokenizer, max_length, prompting_module, trigger_mode,
               positive_samples, negative_samples: Dict = None, relation_structure: Dict = None):
    """ Encoding data and returns torch.utils.data.Dataset. """
    # use average aggregation while optimizing prompt
    shared = {'tokenizer': tokenizer, 'max_length': max_length, 'trigger_mode': trigger_mode, 'mode': 'average'}

    def pool_map(_list):
        pool = Pool()
        out = pool.map(EncodePlus(**shared), _list)
        pool.close()
        return out

    key = list(positive_samples.keys())
    positive_samples_list = ListKeeper([positive_samples[k] for k in key])
    p_prompt_out = prompting_module(positive_samples_list.flatten_list)
    negative_samples_list = ListKeeper([negative_samples[k] for k in key])
    n_prompt_out = prompting_module(negative_samples_list.flatten_list)
    positive_embedding = pool_map(p_prompt_out)
    negative_embedding = pool_map(n_prompt_out)
    if any(i is None for i in positive_embedding) or any(i is None for i in negative_embedding):
        return None
    positive_embedding = positive_samples_list.restore_structure(positive_embedding)
    positive_embedding = {key[n]: v for n, v in enumerate(positive_embedding)}
    negative_embedding = negative_samples_list.restore_structure(negative_embedding)
    negative_embedding = {key[n]: v for n, v in enumerate(negative_embedding)}
    return dict(positive_samples=positive_embedding, negative_samples=negative_embedding,
                relation_structure=relation_structure)


class PromptGenerator:
    """ Prompt generator with triggers. """

    def __init__(self, n_trigger_i: int, n_trigger_b: int, n_trigger_e: int, tokenizer=None):
        """ Prompt generator with triggers.

        Parameters
        ----------
        n_trigger_i : int
            The number of mask in between word_pair.
        n_trigger_b : int
            The number of mask at the beginning of the template.
        n_trigger_e : int
            The number of mask at the end of the template.
        tokenizer : transformers tokenizer.
        """
        self.tokenizer = tokenizer
        # initialize triggers with mask
        self.triggers = [self.tokenizer.mask_token_id] * n_trigger_b + [self.tokenizer.mask_token_id] * n_trigger_i + \
                        [self.tokenizer.mask_token_id] * n_trigger_e
        self.b = n_trigger_b
        self.i = n_trigger_i
        self.e = n_trigger_e
        self.n_trigger = len(self.triggers)

    def __call__(self, word_pair: List):
        """ Prompting batch of word with triggers.

        Parameters
        ----------
        word_pair : list
            A list of two words.

        Returns
        -------
        List of token_ids (list of ids) and trigger token position (list of [0, 1])
        """
        return list(map(self.__single_word_pair_prompt, word_pair))

    def save(self, export_file: str):
        assert export_file.endswith('.json')
        with open(export_file, 'w') as f:
            json.dump({'top': self.triggers[:self.b], 'mid': self.triggers[self.b:self.b + self.i],
                       'bottom': self.triggers[self.b + self.i:]}, f)
        logging.debug('exported to {}'.format(export_file))

    @staticmethod
    def sub_seq(_list, sub, mask):
        s, e = [(i, i + len(sub)) for i in range(len(_list)) if _list[i:i + len(sub)] == sub][0]
        mask[s:e] = [1] * (e - s)
        _list[s:e] = [-100] * (e - s)
        return mask, _list

    def update_trigger(self, trigger_index: int, trigger_id: str):
        self.triggers[trigger_index] = trigger_id

    def get_trigger(self, trigger_index: int):
        return self.triggers[trigger_index]

    def __single_word_pair_prompt(self, word_pair: List):
        assert len(word_pair) == 2 and type(word_pair) in (list, tuple)
        h, t = word_pair
        mask = self.tokenizer.mask_token
        assert h != mask and t != mask
        token_ids = self.tokenizer.encode(' '.join([mask] * self.b + [h] + [mask] * self.i + [t] + [mask] * self.e))
        trigger = [int(i == self.tokenizer.mask_token_id) for i in token_ids]
        token_ids = [-100 if i == self.tokenizer.mask_token_id else i for i in token_ids]
        for i in self.triggers:
            token_ids[token_ids.index(-100)] = i
        assert self.n_trigger == sum(trigger), (self.triggers, sum(trigger), trigger)
        return token_ids, trigger


class GradientStorage:
    """ This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained. """

    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class GradientTriggerSearch:

    def __init__(self,
                 export: str=None,
                 topk: int = -1,
                 n_trigger_i: int = 1,
                 n_trigger_b: int = 1,
                 n_trigger_e: int = 1,
                 n_iteration: int = 10,
                 filter_label: bool = False,
                 filter_pn: bool = False,
                 trigger_selection: str = 'random',
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 data: str = 'semeval2012',
                 n_sample: int = 5,
                 in_batch_negative: bool = False,
                 parent_contrast: bool = False,
                 mse_margin: float = 1,
                 batch: int = 16,
                 batch_no_grad: int = 64,
                 random_seed: int = 0,
                 cache_dir: str = None,
                 checkpoint_path: str = None):
        fix_seed(random_seed)
        self.config = Config(
            config_name='prompter_config',
            export=export,
            checkpoint_path=checkpoint_path,
            topk=topk,
            n_trigger_i=n_trigger_i,
            n_trigger_b=n_trigger_b,
            n_trigger_e=n_trigger_e,
            n_iteration=n_iteration,
            filter_label=filter_label,
            filter_pn=filter_pn,
            trigger_selection=trigger_selection,
            batch=batch,
            batch_no_grad=batch_no_grad,
            model=model,
            max_length=max_length,
            data=data,
            n_sample=n_sample,
            in_batch_negative=in_batch_negative,
            parent_contrast=parent_contrast,
            mse_margin=mse_margin,
            random_seed=random_seed)
        # model setup
        self.tokenizer, self.model, _ = load_language_model(self.config.model, cache_dir)
        self.model.eval()
        self.input_embeddings = self.model.get_input_embeddings()
        self.gradient_store = GradientStorage(self.input_embeddings)
        # cache config
        self.checkpoint_dir = self.config.cache_dir
        if self.config.last_iter != 0:
            ckpt = '{}/prompt.{}.json'.format(self.config.cache_dir, self.config.last_iter - 1)
            logging.info('loading last prompt checkpoint from {}'.format(ckpt))
            # restore last prompt
            with open(ckpt, 'r') as f:
                tmp = json.load(f)
            # self.prompter = PromptGenerator(len(tmp['top']), len(tmp['mid']), len(tmp['bottom']), self.tokenizer)
            self.prompter = PromptGenerator(len(tmp['mid']), len(tmp['top']), len(tmp['bottom']), self.tokenizer)
            for n, i in enumerate(tmp['top'] + tmp['mid'] + tmp['bottom']):
                self.prompter.update_trigger(n, i)
        else:
            self.prompter = PromptGenerator(n_trigger_i, n_trigger_b, n_trigger_e, self.tokenizer)

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.info('language model running on {} GPU'.format(torch.cuda.device_count()))
        # get dataset
        self.all_positive, self.all_negative, self.relation_structure = get_training_data(
            data_name=self.config.data, n_sample=self.config.n_sample, cache_dir=cache_dir)

        # calculate the number of trial to cover all combination in batch
        n_pos = min(len(i) for i in self.all_positive.values())
        n_neg = min(len(i) for i in self.all_negative.values())
        self.n_trial = len(list(product(combinations(range(n_pos), 2), range(n_neg))))

        assert self.all_negative.keys() == self.all_positive.keys()
        logging.info('{} positive data/{} negative data'.format(len(self.all_positive), len(self.all_negative)))
        logging.info('{} trial'.format(self.n_trial))

    def get_filtering_matrix(self):
        key = list(self.all_positive.keys())
        positive_samples_list = ListKeeper([self.all_positive[k] for k in key])
        p_prompt_out = self.prompter(positive_samples_list.flatten_list)
        negative_samples_list = ListKeeper([self.all_negative[k] for k in key])
        n_prompt_out = self.prompter(negative_samples_list.flatten_list)
        filter_vocab = list(self.tokenizer.all_special_ids)
        if self.config.filter_label:
            v = set(list(chain(*[i for i, _ in n_prompt_out])) + list(chain(*[i for i, _ in p_prompt_out])))
            filter_vocab += list(filter(lambda x: x not in self.tokenizer.all_special_ids, v))
        return filter_vocab

    def get_prompt(self, num_workers: int = 1):
        """ Train prompter.

        Parameter
        ----------
        num_workers : int
            Workers for DataLoader.
        """
        logging.info('start prompt generation')
        filter_matrix = None
        loss = None
        for i in range(self.config.last_iter, self.config.n_iteration):
            filter_matrix, loss = self.__single_iteration(num_workers, filter_matrix)
            if loss is None:
                logging.info('early exit: no more updates')
                break
            logging.info('iteration {}/{}: {}\t loss {}'.format(
                i + 1, self.config.n_iteration, self.tokenizer.convert_ids_to_tokens(self.prompter.triggers), loss))
            self.prompter.save('{}/prompt.{}.json'.format(self.config.cache_dir, i))
            mode = 'a' if os.path.exists('{}/loss.txt'.format(self.config.cache_dir)) else 'w'
            with open('{}/loss.txt'.format(self.config.cache_dir), mode) as f:
                f.write('{}\n'.format(loss))
        self.prompter.save('{}/prompt.json'.format(self.config.cache_dir))

    def __single_iteration(self, num_workers: int = 1, filter_matrix=None):

        def aggregate_loss_single_trial(loader, sum_grad, n_grad, total_loss, no_grad: bool = False):
            for i, x in enumerate(loader):
                positive_a = {k: v.to(self.device) for k, v in x['positive_a'].items()}
                positive_b = {k: v.to(self.device) for k, v in x['positive_b'].items()}
                negative = {k: v.to(self.device) for k, v in x['negative'].items()}
                if self.config.parent_contrast:
                    positive_hc = {k: v.to(self.device) for k, v in x['positive_parent'].items()}
                    negative_hc = {k: v.to(self.device) for k, v in x['negative_parent'].items()}
                    encode = {k: torch.cat([positive_a[k], positive_b[k], negative[k], positive_hc[k], negative_hc[k]])
                              for k in positive_a.keys()}
                else:
                    encode = {k: torch.cat([positive_a[k], positive_b[k], negative[k]]) for k in positive_a.keys()}

                # get model prediction
                encode = {k: _v.to(self.device) for k, _v in encode.items()}
                labels = encode.pop('labels')
                trigger = encode.pop('trigger')
                output = self.model(**encode, return_dict=True)
                batch_embedding_tensor = (output['last_hidden_state'] * labels.reshape(len(labels), -1, 1)).sum(1)
                if self.config.parent_contrast:
                    v_anchor, v_positive, v_negative, v_positive_hc, v_negative_hc = batch_embedding_tensor.chunk(5)
                else:
                    v_anchor, v_positive, v_negative = batch_embedding_tensor.chunk(3)
                    v_positive_hc = v_negative_hc = None

                # contrastive loss
                loss = triplet_loss(
                    v_anchor, v_positive, v_negative, v_positive_hc, v_negative_hc, margin=self.config.mse_margin,
                    in_batch_negative=self.config.in_batch_negative)

                total_loss += loss.sum().cpu().item()
                n_grad += len(labels)
                if no_grad:
                    continue
                # backward: calculate gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRADIENT_VALUE)
                loss.backward()
                grad = self.gradient_store.get()
                # remove nan
                grad[grad != grad] = 0
                # assert (grad != grad).sum().cpu().item() == 0, (grad != grad).sum()
                batch_size, _, emb_dim = grad.size()
                trigger_position = trigger.unsqueeze(-1) == 1
                grad = torch.masked_select(grad, trigger_position)
                grad = grad.view(batch_size, self.prompter.n_trigger, emb_dim)
                sum_grad += grad.sum(dim=0)

            return sum_grad, n_grad, total_loss

        def aggregate_loss(no_grad: bool = False):
            if self.config.parent_contrast:
                shared_param = preprocess(self.tokenizer, self.config.max_length, self.prompter, True,
                                          self.all_positive, self.all_negative, self.relation_structure)
            else:
                shared_param = preprocess(self.tokenizer, self.config.max_length, self.prompter, True,
                                          self.all_positive, self.all_negative)
            if shared_param is None:
                return None, None
            total_loss = 0
            sum_grad = 0
            n_grad = 0
            # As the data feeder is stochastic (randomly sample combination of positive and negative), we ensure the
            # gradients are aggregated from large enough sets to behave as a global gradient by conducting several
            # individual runs.
            batch = self.config.batch_no_grad if no_grad else self.config.batch
            for d in tqdm(list(range(self.n_trial))):
                data = Dataset(deterministic_index=d, **shared_param)
                loader = torch.utils.data.DataLoader(data, batch_size=batch, num_workers=num_workers)
                sum_grad, n_grad, total_loss = aggregate_loss_single_trial(loader, sum_grad, n_grad, total_loss, no_grad)
            return sum_grad/n_grad, total_loss/n_grad

        logging.info('compute candidate trigger')
        if filter_matrix is None:
            vocab = self.get_filtering_matrix()
            logging.debug('construct filtering vocab matrix')
            filter_matrix = torch.zeros(self.tokenizer.vocab_size, dtype=torch.float32, device=self.device)
            for __v in vocab:
                filter_matrix[__v] = -1e32
            if self.config.filter_pn:
                for word, idx in self.tokenizer.get_vocab().items():
                    # https://github.com/ucinlp/autoprompt/blob/master/autoprompt/create_trigger.py#L274
                    if len(word) == 1:
                        continue
                    if idx in vocab or self.tokenizer.decode([idx])[0].isupper():
                        logging.debug('\t filtered: {}'.format(word))
                        filter_matrix[idx] = -1e32
        logging.debug('compute gradient')
        average_grad, average_loss = aggregate_loss()
        logging.info('\t * tmp loss: {}'.format(average_loss))

        # create trigger que: evaluation will be done in the order
        if self.config.trigger_selection == 'random':
            # random
            trigger_index = list(range(self.prompter.n_trigger))
            random.shuffle(trigger_index)
        elif self.config.trigger_selection == 'best':
            # choose trigger with the largest gradient norm (tend to be fixed to specific position)
            grad_norm = ((average_grad ** 2).sum(-1) ** 0.5).cpu().tolist()
            trigger_index = [grad_norm.index(i) for i in sorted(grad_norm, reverse=True)]
        else:
            raise NotImplementedError()

        original_trigger, trigger_to_flip, best_trigger, best_loss = None, None, None, None
        logging.info('evaluate to get the best trigger: {}'.format(trigger_index))

        # whenever it finds loss improvement, exit and update with it
        for trigger_to_flip in trigger_index:
            logging.info('evaluate trigger: {}'.format(trigger_to_flip))
            candidate = self.top_candidate(average_grad[trigger_to_flip], filter_matrix)
            original_trigger = self.prompter.get_trigger(trigger_to_flip)
            for m, c in enumerate(candidate):
                c_str = self.tokenizer.convert_ids_to_tokens(c)
                self.prompter.update_trigger(trigger_to_flip, c)
                with torch.no_grad():
                    _, cand_loss = aggregate_loss(no_grad=True)
                if cand_loss is None:
                    logging.info('\t - candidate: {} \tSKIP (invalid token)'.format(c_str))
                    continue
                loss_gain = average_loss - cand_loss
                if loss_gain > 0:
                    logging.info('\t - candidate: {} \tUPDATE (loss decrease: {})'.format(c_str, loss_gain))
                    best_trigger, best_loss = c, cand_loss
                    break
                logging.info('\t - candidate: {} \tSKIP (no loss decrease, loss:{})'.format(c_str, cand_loss))
                if self.config.topk != -1 and m > self.config.topk:
                    logging.info('reached max candidate (no trigger found at {})'.format(trigger_to_flip))
                    break
            self.prompter.update_trigger(trigger_to_flip, original_trigger)
            if best_loss is not None:
                break

        if best_loss is None:
            logging.info('no triggers found')
            return filter_matrix, None
        new = self.tokenizer.convert_ids_to_tokens(best_trigger)
        old = self.tokenizer.convert_ids_to_tokens(original_trigger)
        logging.info('update `{}`: {} -> {}'.format(trigger_to_flip, old, new))
        self.prompter.update_trigger(trigger_to_flip, best_trigger)
        return filter_matrix, best_loss

    def top_candidate(self, averaged_grad, filter_matrix):
        """ Returns the top candidate replacements."""
        with torch.no_grad():
            gradient_dot_embedding_matrix = filter_matrix - torch.matmul(self.input_embeddings.weight, averaged_grad)
            logging.debug('\t - max gradient score:{}'.format(gradient_dot_embedding_matrix.max()))
            _, top_k_ids = gradient_dot_embedding_matrix.topk(len(gradient_dot_embedding_matrix))
        return top_k_ids.cpu().tolist()
