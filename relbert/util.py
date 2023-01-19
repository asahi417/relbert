import random
import os
import tarfile
import zipfile
import gzip
import requests
import urllib.request

import gdown
import numpy as np
import torch
from torch import nn

home_dir = f"{os.path.expanduser('~')}/.cache/relbert"


def internet_connection():
    try:
        urllib.request.urlopen('http://google.com')
        return True
    except Exception:
        return False


def cosine_similarity(a, b):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/norm_a/norm_b


def euclidean_distance(a, b):
    return sum(map(lambda x: (x[0] - x[1])**2, zip(a, b))) ** 0.5


def wget(url, cache_dir: str = './cache', gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    path = _wget(url, cache_dir, gdrive_filename=gdrive_filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)


def _wget(url: str, cache_dir, gdrive_filename: str = None):
    """ get data from web """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        return gdown.download(url, f'{cache_dir}/{gdrive_filename}', quiet=False)
    filename = os.path.basename(url)
    with open(f'{cache_dir}/{filename}', "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return f'{cache_dir}/{filename}'


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def triplet_loss(tensor_anchor,
                 tensor_positive,
                 tensor_negative,
                 tensor_positive_parent=None,
                 tensor_negative_parent=None,
                 margin: int = 1,
                 linear=None,
                 device: str = 'gpu'):
    """ Compute contrastive triplet loss with in batch augmentation which enables to propagate error on quadratic
    of batch size. """
    bce = nn.BCELoss()

    def classification_loss(v_anchor, v_positive, v_negative):
        # the 3-way discriminative loss used in SBERT
        if linear is not None:
            feature_positive = torch.cat([v_anchor, v_positive, torch.abs(v_anchor - v_positive)], dim=1)
            feature_negative = torch.cat([v_anchor, v_negative, torch.abs(v_anchor - v_negative)], dim=1)
            feature = torch.cat([feature_positive, feature_negative])
            pred = torch.sigmoid(linear(feature))
            label = torch.tensor([1] * len(feature_positive) + [0] * len(feature_negative),
                                 dtype=torch.float32, device=device)
            return bce(pred, label.unsqueeze(-1))
        return 0

    def main_loss(v_anchor, v_positive, v_negative):
        distance_positive = torch.sum((v_anchor - v_positive) ** 2, -1) ** 0.5
        distance_negative = torch.sum((v_anchor - v_negative) ** 2, -1) ** 0.5
        return torch.sum(torch.clip(distance_positive - distance_negative - margin, min=0))

    def sample_augmentation(v_anchor, v_positive):
        v_anchor_aug = v_anchor.unsqueeze(-1).permute(2, 0, 1).repeat(len(v_anchor), 1, 1).reshape(len(v_anchor), -1)
        v_positive_aug = v_positive.unsqueeze(-1).permute(2, 0, 1).repeat(len(v_positive), 1, 1).reshape(len(v_positive), -1)
        v_negative_aug = v_positive.unsqueeze(-1).permute(0, 2, 1).repeat(1, len(v_positive), 1).reshape(len(v_positive), -1)
        return v_anchor_aug, v_positive_aug, v_negative_aug

    loss = main_loss(tensor_anchor, tensor_positive, tensor_negative)
    loss += main_loss(tensor_positive, tensor_anchor, tensor_negative)
    loss += classification_loss(tensor_anchor, tensor_positive, tensor_negative)

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
