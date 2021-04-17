""" Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """
import os
import logging
from itertools import product, combinations
import random
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .lm import RelBERT
from .data import get_training_data
from .config import Config
from .util import get_linear_schedule_with_warmup, triplet_loss, fix_seed, module_output_dir, Dataset


class Trainer:
    """ Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """

    def __init__(self,
                 export: str = None,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 mode: str = 'average',
                 data: str = 'semeval2012',
                 n_sample: int = 5,
                 template_type: str = 'a',
                 softmax_loss: bool = True,
                 in_batch_negative: bool = True,
                 parent_contrast: bool = True,
                 mse_margin: float = 1,
                 epoch: int = 5,
                 batch: int = 64,
                 lr: float = 0.001,
                 lr_decay: bool = False,
                 lr_warmup: int = 100,
                 weight_decay: float = 0,
                 optimizer: str = 'adam',
                 momentum: float = 0.9,
                 fp16: bool = False,
                 random_seed: int = 0,
                 cache_dir: str = None):
        """ Initialize training instance to finetune relation BERT model.

        Parameters
        ----------
        model
        max_length
        mode
        data
        n_sample
        template_type
        softmax_loss
        in_batch_negative
        parent_contrast
        mse_margin
        epoch
        batch
        lr
        lr_decay
        lr_warmup
        weight_decay
        optimizer
        momentum
        fp16
        random_seed
        export_dir
        export_name
        cache_dir
        """

        fix_seed(random_seed)
        self.cache_dir = cache_dir

        # load language model
        self.lm = RelBERT(
            model=model, max_length=max_length, cache_dir=self.cache_dir, mode=mode, template_type=template_type)

        assert not self.lm.is_trained, '{} is already trained'.format(model)
        self.lm.train()

        # config
        self.config = Config(
            model=model,
            max_length=max_length,
            mode=mode,
            data=data,
            n_sample=n_sample,
            custom_template_type=self.lm.custom_template_type,
            template=self.lm.template,
            softmax_loss=softmax_loss,
            in_batch_negative=in_batch_negative,
            parent_contrast=parent_contrast,
            mse_margin=mse_margin,
            epoch=epoch,
            lr_warmup=lr_warmup,
            batch=batch,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            optimizer=optimizer,
            momentum=momentum,
            fp16=fp16,
            random_seed=random_seed,
            export=export,
        )

        # calculate the number of trial to cover all combination in batch
        self.n_trial = len(list(product(combinations(range(self.config.n_sample), 2), range(self.config.n_sample))))

        # model size
        self.checkpoint_dir = self.config.cache_dir

        # get dataset
        self.all_positive, self.all_negative, self.relation_structure = get_training_data(
            data_name=self.config.data, n_sample=self.config.n_sample, cache_dir=self.cache_dir)

        model_parameters = list(self.lm.model.named_parameters())
        self.linear = None
        self.discriminative_loss = None
        if softmax_loss:
            logging.info('add linear layer for softmax_loss')
            self.linear = nn.Linear(self.lm.hidden_size * 3, 1)  # three way feature
            self.linear.weight.data.normal_(std=0.02)
            self.discriminative_loss = nn.BCELoss()
            self.linear.to(self.lm.device)
            model_parameters += list(self.linear.named_parameters())
            if self.lm.parallel:
                self.linear = torch.nn.DataParallel(self.linear)

        # setup optimizer
        if self.config.weight_decay is not None or self.config.weight_decay != 0:
            no_decay = ["bias", "LayerNorm.weight"]
            model_parameters = [
                {"params": [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.config.weight_decay},
                {"params": [p for n, p in model_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]

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

        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)

    def train(self, num_workers: int = 1, epoch_save: int = 10):
        """ Train model.

        Parameters
        ----------
        num_workers : int
            Workers for DataLoader.
        epoch_save : int
            Epoch to run validation eg) Every 100000 epoch, it will save model weight as default.
        """
        writer = SummaryWriter(log_dir=self.config.cache_dir)

        if self.config.parent_contrast:
            dataset = self.lm.preprocess(self.all_positive, self.all_negative, self.relation_structure)
        else:
            dataset = self.lm.preprocess(self.all_positive, self.all_negative)

        logging.info('start model training')
        # logging.info('\t * train data: {}, batch number: {}'.format(len(dataset), len(loader)))
        batch_index = list(range(self.n_trial))
        global_step = 0

        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.config.epoch):  # loop over the epoch
                random.shuffle(batch_index)
                for n, bi in enumerate(batch_index):
                    dataset = Dataset(deterministic_index=bi, **dataset)
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=self.config.batch, shuffle=True, num_workers=num_workers, drop_last=True)
                    mean_loss, global_step = self.train_single_epoch(loader, global_step=global_step, writer=writer)
                    inst_lr = self.optimizer.param_groups[0]['lr']
                    logging.info('[epoch {}/{}, batch_id {}/{}] average loss: {}, lr: {}'.format(
                        e, self.config.epoch, n, self.n_trial, round(mean_loss, 3), inst_lr))
                if (e + 1) % epoch_save == 0 and (e + 1) != 0:
                    cache_dir = '{}/epoch_{}'.format(self.checkpoint_dir, e + 1)
                    os.makedirs(cache_dir, exist_ok=True)
                    self.lm.save(cache_dir)

        writer.close()
        cache_dir = '{}/epoch_{}'.format(self.checkpoint_dir, e + 1)
        os.makedirs(cache_dir, exist_ok=True)
        self.lm.save(cache_dir)
        logging.info('complete training: model ckpt was saved at {}'.format(self.checkpoint_dir))

    def train_single_epoch(self, data_loader, global_step: int, writer):
        self.lm.train()
        total_loss = 0
        bce = nn.BCELoss()
        step_in_epoch = len(data_loader)
        for x in data_loader:
            global_step += 1

            self.optimizer.zero_grad()

            positive_a = {k: v.to(self.lm.device) for k, v in x['positive_a'].items()}
            positive_b = {k: v.to(self.lm.device) for k, v in x['positive_b'].items()}
            negative = {k: v.to(self.lm.device) for k, v in x['negative'].items()}
            if self.config.parent_contrast:
                positive_hc = {k: v.to(self.lm.device) for k, v in x['positive_parent'].items()}
                negative_hc = {k: v.to(self.lm.device) for k, v in x['negative_parent'].items()}
                encode = {k: torch.cat([positive_a[k], positive_b[k], negative[k], positive_hc[k], negative_hc[k]])
                          for k in positive_a.keys()}
                embedding = self.lm.to_embedding(encode)
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
            writer.add_scalar('train/loss', inst_loss, global_step)

            # update optimizer
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', inst_lr, global_step)

            # aggregate average loss over epoch
            total_loss += inst_loss

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        return total_loss / step_in_epoch, global_step
