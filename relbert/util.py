import random
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


class NCELoss:
    
    def __init__(self,
                 loss_function,
                 temperature_nce_constant,
                 temperature_nce_rank=None):
        self.loss_function = loss_function
        self.temperature_nce_constant = temperature_nce_constant
        self.temperature_nce_rank = temperature_nce_rank

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
        elif self.loss_function == 'nce_logout':
            for i in range(batch_size_positive):
                deno_n = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / self.temperature_nce_constant))
                for p in range(batch_size_positive):
                    logit_p = torch.exp(
                        cos_1d(embedding_p[i], embedding_p[p]) / self.temperature_nce_constant)
                    loss.append(- torch.log(logit_p / (logit_p + deno_n)))
        elif self.loss_function == 'nce_login':
            for i in range(batch_size_positive):
                deno_n = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_n) / self.temperature_nce_constant))
                logit_p = torch.sum(torch.exp(
                    cos_2d(embedding_p[i].unsqueeze(0), embedding_p) / self.temperature_nce_constant))
                loss.append(- torch.log(logit_p / (logit_p + deno_n)))
        else:
            raise ValueError(f"unknown loss function {self.loss_function}")
        loss = stack_sum(loss)
        return loss
