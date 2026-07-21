import torch
import torch.nn.functional as F


def cosine_similarity(feat1, feat2):
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)

    return torch.sum(feat1 * feat2, dim=-1)


def cosine_distance(feat1, feat2):
    return 1 - cosine_similarity(feat1, feat2)


def l2_distance(feat1, feat2):
    return torch.norm(feat1 - feat2, p=2, dim=-1)