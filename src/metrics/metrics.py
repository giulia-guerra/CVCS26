import torch
from scipy.stats import pearsonr, spearmanr


def srcc(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    score, _ = spearmanr(pred, target)
    return float(score)


def plcc(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    score, _ = pearsonr(pred, target)
    return float(score)