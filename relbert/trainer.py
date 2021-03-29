""" Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """
import os
import logging

from .data import get_semeval_data
from .config import Config
from .util import fix_seed
from .lm import RelBERT
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .util import get_linear_schedule_with_warmup, triplet_loss


class Trainer:
    """ Train relation BERT with prompted relation pairs from SemEval 2012 task 2. """

    def __init__(self,
                 model: str = 'roberta-large',
                 max_length: int = 64,
                 mode: str = 'mask',
                 data: str = 'semeval2012',
                 n_sample: int = 10,
                 template_type: str = 'a',
                 softmax_loss: bool = True,
                 in_batch_negative: bool = True,
                 mse_margin: float = 1,
                 epoch: int = 10,
                 epoch_warmup: int = 10,
                 batch: int = 64,
                 lr: float = 0.001,
                 lr_decay: bool = False,
                 weight_decay: float = 0,
                 optimizer: str = 'adam',
                 momentum: float = 0.9,
                 fp16: bool = False,
                 random_seed: int = 0,
                 export_dir: str = './ckpt',
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
        mse_margin
        epoch
        epoch_warmup
        batch
        lr
        lr_decay
        weight_decay
        optimizer
        momentum
        fp16
        random_seed
        export_dir
        cache_dir
        """

        fix_seed(random_seed)
        self.cache_dir = cache_dir

        # load language model
        self.lm = RelBERT(
            model=model, max_length=max_length, cache_dir=self.cache_dir, mode=mode, template_type=template_type)
        assert not self.lm.is_trained, '{} is already trained'.format(model)

        # config
        self.config = Config(
            model=model,
            max_length=max_length,
            mode=mode,
            data=data,
            n_sample=n_sample,
            template_type=template_type,
            softmax_loss=softmax_loss,
            in_batch_negative=in_batch_negative,
            mse_margin=mse_margin,
            epoch=epoch,
            epoch_warmup=epoch_warmup,
            batch=batch,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            optimizer=optimizer,
            momentum=momentum,
            fp16=fp16,
            random_seed=random_seed,
            export_dir=export_dir)

        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler('{}/training.log'.format(self.config.cache_dir))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)

        # model size
        self.checkpoint_dir = self.config.cache_dir

        # get dataset
        if self.config.data == 'semeval2012':
            all_positive, all_negative, _ = get_semeval_data(
                n_sample=self.config.n_sample, cache_dir=self.cache_dir)
            self.dataset = self.lm.preprocess(positive_samples=all_positive, negative_sample=all_negative)
        else:
            raise ValueError('unknown data: {}'.format(self.config.data))

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
            num_warmup_steps=self.config.epoch_warmup,
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

        logging.info('start model training')
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch, shuffle=True, num_workers=num_workers, drop_last=True)
        logging.info('\t * train data: {}, batch number: {}'.format(len(self.dataset), len(loader)))

        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.config.epoch):  # loop over the epoch
                mean_loss = self.train_single_epoch(loader, epoch_n=e, writer=writer)
                inst_lr = self.optimizer.param_groups[0]['lr']
                logging.info('[epoch {}/{}] average loss: {}, lr: {}'.format(
                    e, self.config.epoch, round(mean_loss, 3), inst_lr))

                if e % epoch_save == 0 and e != 0:
                    cache_dir = '{}/epoch_{}'.format(self.checkpoint_dir, e)
                    os.makedirs(cache_dir)
                    self.lm.save(cache_dir)
                self.scheduler.step()

        writer.close()
        self.lm.save(self.checkpoint_dir)
        logging.info('complete training: model ckpt was saved at {}'.format(self.checkpoint_dir))

    def train_single_epoch(self, data_loader, epoch_n: int, writer):
        self.lm.train()
        total_loss = 0
        bce = nn.BCELoss()
        step_in_epoch = len(data_loader)
        for i, x in enumerate(data_loader):
            positive_a = {k: v.to(self.lm.device) for k, v in x['positive_a'].items()}
            positive_b = {k: v.to(self.lm.device) for k, v in x['positive_b'].items()}
            negative = {k: v.to(self.lm.device) for k, v in x['negative'].items()}

            # zero the parameter gradients
            self.optimizer.zero_grad()

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
                label = torch.tensor([1] * len(feature_positive) + [0] * len(feature_negative), dtype=torch.float32)
                loss += bce(pred, label.unsqueeze(-1))

            # backward: calculate gradient
            self.scaler.scale(loss).backward()

            inst_loss = loss.cpu().item()
            writer.add_scalar('train/loss', inst_loss, i + epoch_n * step_in_epoch)

            # update optimizer
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', inst_lr, i + epoch_n * step_in_epoch)

            # aggregate average loss over epoch
            total_loss += inst_loss

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss / step_in_epoch


