import numpy as np
from scipy.stats import spearmanr, pearsonr


def srcc(predictions, targets):
    """
    Spearman Rank Correlation Coefficient.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if len(predictions) < 2:
        return float("nan")

    if np.std(predictions) == 0 or np.std(targets) == 0:
        return float("nan")

    value = spearmanr(predictions, targets).statistic

    return float(value)


def plcc(predictions, targets):
    """
    Pearson Linear Correlation Coefficient.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if len(predictions) < 2:
        return float("nan")

    if np.std(predictions) == 0 or np.std(targets) == 0:
        return float("nan")

    value = pearsonr(predictions, targets).statistic

    return float(value)