# Implementazione delle metriche di correlazione utilizzate per valutare i modelli IQA.
# Calcola PLCC (Pearson Linear Correlation Coefficient) e SRCC (Spearman Rank
# Correlation Coefficient) confrontando i valori predetti dal modello con i valori
# MOS/DMOS reali del dataset.
# Le metriche vengono utilizzate per misurare rispettivamente la correlazione lineare
# e la correlazione basata sui ranghi tra qualità stimata e qualità percepita.

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