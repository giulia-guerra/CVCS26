# Questo file raccoglie le metriche principali utilizzate nella pipeline di
# valutazione IQA. Contiene le implementazioni di SRCC e PLCC compatibili con
# input PyTorch o NumPy e include funzioni aggiuntive per il confronto delle
# feature estratte dai modelli vision tramite distanza coseno e distanza L2.


import torch
from scipy.stats import spearmanr, pearsonr


# ==================================================
# Metriche IQA
# ==================================================

def srcc(pred, target):
    """
    Spearman Rank Correlation Coefficient
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    score, _ = spearmanr(pred, target)

    return float(score)


def plcc(pred, target):
    """
    Pearson Linear Correlation Coefficient
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    score, _ = pearsonr(pred, target)

    return float(score)


# ==================================================
# Alias compatibili con i test
# ==================================================

def compute_srcc(pred, target):
    return srcc(pred, target)


def compute_plcc(pred, target):
    return plcc(pred, target)


# ==================================================
# Distanze tra feature
# ==================================================

def cosine_distance(x, y):
    """
    1 - cosine similarity
    """

    if x.ndim == 1:
        similarity = torch.nn.functional.cosine_similarity(
            x.unsqueeze(0),
            y.unsqueeze(0),
            dim=1
        )
        return 1 - similarity.squeeze()

    similarity = torch.nn.functional.cosine_similarity(
        x,
        y,
        dim=-1
    )

    return 1 - similarity


def l2_distance(x, y):
    """
    Distanza euclidea
    """

    return torch.norm(
        x - y,
        p=2,
        dim=-1 if x.ndim > 1 else 0
    )