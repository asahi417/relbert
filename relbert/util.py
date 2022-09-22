import random
import logging
from itertools import product

import numpy as np
import torch


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def stack_sum(_list):
    if len(_list) == 0:
        return 0
    return torch.mean(torch.stack(_list))


cos_2d = torch.nn.CosineSimilarity(dim=1)
cos_1d = torch.nn.CosineSimilarity(dim=0)
bce = torch.nn.BCELoss()


class NCELoss:
    
    def __init__(self,
                 loss_function,
                 temperature_nce_constant,
                 temperature_nce_rank=None,
                 classification_loss: bool = False,
                 hidden_size: int = None,
                 margin: int = 1,
                 boundary: int = 0,
                 device=None,
                 parallel=False):
        self.loss_function = loss_function
        self.temperature_nce_constant = temperature_nce_constant
        self.temperature_nce_rank = temperature_nce_rank
        self.classification_loss = classification_loss
        self.margin = margin
        self.boundary = boundary
        self.hidden_size = hidden_size
        self.device = device
        if classification_loss:
            logging.info('activate classification loss')
            assert self.hidden_size, '`hidden_size` should be specified'
            assert self.device, '`device` should be specified'
            self.linear = torch.nn.Linear(self.hidden_size * 3, 1)  # three way feature
            self.linear.weight.data.normal_(std=0.02)
            self.linear.to(self.device)
            if parallel:
                self.linear = torch.nn.DataParallel(self.linear)
        else:
            self.linear = self.discriminative_loss = None

    def get_rank_temperature(self, i, n):
        assert i <= n, f"{i}, {n}"
        if self.temperature_nce_rank['type'] == 'linear':
            _min = self.temperature_nce_rank['min']
            _max = self.temperature_nce_rank['max']
            return (_min - _max) / (1 - n) * (i - 1) + _min
        raise ValueError(f"unknown type: {self.temperature_nce_rank['type']}")

    def __call__(self,
                 embedding_p,
                 embedding_n,
                 rank=None):
        loss = []
        batch_size_positive = len(embedding_p)
        if self.loss_function == 'nce_rank':
            assert rank is not None
            rank_map = {r: 1 + n for n, r in enumerate(sorted(rank))}
            rank = [rank_map[r] for r in rank]
            for i in range(batch_size_positive):
                assert type(rank[i]) == int, rank[i]
                tau = self.get_rank_temperature(rank[i], batch_size_positive)
                deno_n = torch.sum(torch.exp(cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / tau))
                dist = torch.exp(cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / tau)
                nume_p = stack_sum([d for n, d in enumerate(dist) if rank[n] >= rank[i]])
                deno_p = stack_sum([d for n, d in enumerate(dist) if rank[n] < rank[i]])
                loss.append(- torch.log(nume_p / (deno_p + deno_n)))
        elif self.loss_function in ['nce_logout', 'info_loob']:
            for i in range(batch_size_positive):
                deno_n = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / self.temperature_nce_constant))
                for p in range(batch_size_positive):
                    logit_p = torch.exp(
                        cos_1d(embedding_p[i], embedding_p[p]) / self.temperature_nce_constant
                    )
                    if self.loss_function == 'info_loob':
                        loss.append(- torch.log(logit_p / deno_n))
                    else:
                        loss.append(- torch.log(logit_p / (logit_p + deno_n)))
        elif self.loss_function == 'nce_login':
            for i in range(batch_size_positive):
                deno_n = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / self.temperature_nce_constant))
                logit_p = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / self.temperature_nce_constant))
                loss.append(- torch.log(logit_p / (logit_p + deno_n)))
        elif self.loss_function == 'triplet':
            for i in range(batch_size_positive):
                distance_positive = []
                distance_negative = []
                for p in range(batch_size_positive):
                    if i != p:
                        distance_positive.append(torch.sum((embedding_p[i] - embedding_p[p]) ** 2, -1) ** 0.5)
                for n in range(len(embedding_n)):
                    distance_negative.append(torch.sum((embedding_p[i] - embedding_n[n]) ** 2, -1) ** 0.5)
                for d_p, d_n in product(distance_positive, distance_negative):
                    loss.append(torch.sum(torch.clip(d_p - d_n - self.margin, min=self.boundary)))
        else:
            raise ValueError(f"unknown loss function {self.loss_function}")
        loss = stack_sum(loss)
        if self.linear is not None:
            for i in range(batch_size_positive):
                features = []
                labels = []
                for j in range(batch_size_positive):
                    feature = torch.cat(
                        [embedding_p[i], embedding_p[j], torch.abs(embedding_p[i] - embedding_p[j])],
                        dim=0)
                    features.append(feature)
                    labels.append([1])
                for j in range(len(embedding_n)):
                    feature = torch.cat(
                        [embedding_p[i], embedding_n[j], torch.abs(embedding_p[i] - embedding_n[j])],
                        dim=0)
                    features.append(feature)
                    labels.append([0])
                pred = torch.sigmoid(self.linear(torch.stack(features)))
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                loss += bce(pred, labels)
        return loss
