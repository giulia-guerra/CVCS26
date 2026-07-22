import torch
import torch.nn.functional as F


def cosine_distance(x1, x2):
    """
    Cosine distance = 1 - cosine similarity
    """

    similarity = F.cosine_similarity(x1, x2)

    return 1 - similarity



def l2_distance(x1, x2):
    """
    Euclidean distance
    """

    return torch.norm(x1 - x2, dim=1)