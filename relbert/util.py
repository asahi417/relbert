import random
import urllib.request

import numpy as np
import torch
from torch import nn


def internet_connection():
    try:
        urllib.request.urlopen('http://google.com')
        return True
    except Exception:
        return False


def fix_seed(seed: int = 12, cuda: bool = True):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loss_triplet(
        tensor_anchor,
        tensor_positive,
        tensor_negative,
        tensor_positive_parent=None,
        tensor_negative_parent=None,
        margin: float = 1.0,
        linear=None,
        device: str = 'gpu'):
    """ Triplet Loss """
    bce = nn.BCELoss()

    def classification_loss(v_anchor, v_positive, v_negative):
        # the 3-way discriminative loss used in SBERT
        if linear is not None:
            feature_positive = torch.cat([v_anchor, v_positive, torch.abs(v_anchor - v_positive)], dim=1)
            feature_negative = torch.cat([v_anchor, v_negative, torch.abs(v_anchor - v_negative)], dim=1)
            feature = torch.cat([feature_positive, feature_negative])
            pred = torch.sigmoid(linear(feature))
            label = torch.tensor([1] * len(feature_positive) + [0] * len(feature_negative), dtype=torch.float32, device=device)
            return bce(pred, label.unsqueeze(-1))
        return 0

    def main_loss(v_anchor, v_positive, v_negative):
        distance_positive = torch.sum((v_anchor - v_positive) ** 2, -1) ** 0.5
        distance_negative = torch.sum((v_anchor - v_negative) ** 2, -1) ** 0.5
        return torch.sum(torch.clip(distance_positive - distance_negative - margin, min=0))

    loss = main_loss(tensor_anchor, tensor_positive, tensor_negative)
    loss += main_loss(tensor_positive, tensor_anchor, tensor_negative)
    loss += classification_loss(tensor_anchor, tensor_positive, tensor_negative)

    def sample_augmentation(v_anchor, v_positive):
        v_anchor_aug = v_anchor.unsqueeze(-1).permute(2, 0, 1).repeat(len(v_anchor), 1, 1).reshape(len(v_anchor), -1)
        v_positive_aug = v_positive.unsqueeze(-1).permute(2, 0, 1).repeat(len(v_positive), 1, 1).reshape(
            len(v_positive), -1)
        v_negative_aug = v_positive.unsqueeze(-1).permute(0, 2, 1).repeat(1, len(v_positive), 1).reshape(
            len(v_positive), -1)
        return v_anchor_aug, v_positive_aug, v_negative_aug

    # In-batch Negative Sampling
    # No elements in single batch share same relation type, so here we construct negative sample within batch
    # by regarding positive sample from other entries as its negative. The original negative is the hard
    # negatives from same relation type and the in batch negative is easy negative from other relation types.
    a, p, n = sample_augmentation(tensor_anchor, tensor_positive)
    loss += main_loss(a, p, n)
    a, p, n = sample_augmentation(tensor_positive, tensor_anchor)
    loss += main_loss(a, p, n)

    # contrastive loss of the parent class
    if tensor_positive_parent is not None and tensor_negative_parent is not None:
        loss += main_loss(tensor_anchor, tensor_positive_parent, tensor_negative_parent)
        loss += main_loss(tensor_positive_parent, tensor_anchor, tensor_negative_parent)
        loss += classification_loss(tensor_anchor, tensor_positive_parent, tensor_negative_parent)
    return loss


def loss_nce(tensor_positive,
             tensor_negative,
             temperature: float = 1.0,
             info_loob: bool = False,
             linear=None,
             device: str = 'gpu'):
    """ NCE loss"""
    bce = nn.BCELoss()
    cos_3d = torch.nn.CosineSimilarity(dim=2)
    eps = 1e-5
    logit_n = torch.exp(
        cos_3d(tensor_positive.unsqueeze(1), tensor_negative.unsqueeze(0)) / temperature
    )
    deno_n = torch.sum(logit_n, dim=-1)  # sum over negative
    logit_p = torch.exp(
        cos_3d(tensor_positive.unsqueeze(1), tensor_positive.unsqueeze(0)) / temperature
    )
    if info_loob:
        loss = torch.sum(- torch.log(logit_p / (deno_n.unsqueeze(-1) + eps)))
    else:
        loss = torch.sum(- torch.log(logit_p / (deno_n.unsqueeze(-1) + logit_p + eps)))
    if linear is not None:
        batch_size_positive = len(tensor_positive)
        for i in range(batch_size_positive):
            features = []
            labels = []
            for j in range(batch_size_positive):
                feature = torch.cat(
                    [tensor_positive[i], tensor_positive[j], torch.abs(tensor_positive[i] - tensor_positive[j])],
                    dim=0)
                features.append(feature)
                labels.append([1])
            for j in range(len(tensor_negative)):
                feature = torch.cat(
                    [tensor_positive[i], tensor_negative[j], torch.abs(tensor_positive[i] - tensor_negative[j])],
                    dim=0)
                features.append(feature)
                labels.append([0])
            pred = torch.sigmoid(linear(torch.stack(features)))
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
            loss += bce(pred, labels)
    return loss
