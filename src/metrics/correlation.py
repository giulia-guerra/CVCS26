import numpy as np
from scipy.stats import pearsonr, spearmanr


def PLCC(predictions, targets):
    """
    Pearson Linear Correlation Coefficient

    predictions: valori predetti dal modello
    targets: MOS/DMOS reali
    """

    predictions = np.array(predictions)
    targets = np.array(targets)

    score, _ = pearsonr(
        predictions,
        targets
    )

    return score



def SRCC(predictions, targets):
    """
    Spearman Rank Correlation Coefficient
    """

    predictions = np.array(predictions)
    targets = np.array(targets)

    score, _ = spearmanr(
        predictions,
        targets
    )

    return score