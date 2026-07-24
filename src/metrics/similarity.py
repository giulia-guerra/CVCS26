# Questo file contiene le funzioni utilizzate per confrontare le feature estratte
# dalle immagini reference e distorted nella pipeline IQA. Implementa la
# similarità coseno, che misura quanto due vettori di feature siano vicini nello
# spazio delle rappresentazioni, la relativa distanza coseno e la distanza L2
# (distanza euclidea) per quantificare la differenza tra le feature dei due input

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