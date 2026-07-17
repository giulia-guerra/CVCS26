import torch
from scipy.stats import spearmanr, pearsonr


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


def cosine_distance(x, y):

    similarity = torch.nn.functional.cosine_similarity(
        x,
        y,
        dim=0
    )

    return 1 - similarity


def l2_distance(x, y):

    return torch.norm(
        x - y,
        p=2
    )