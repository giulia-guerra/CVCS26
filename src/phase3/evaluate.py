# Il codice valuta un modello già addestrato su un dataset: 
# per ogni batch genera le predizioni del MOS a partire dalle feature, 
# le confronta con i valori reali e calcola l’MSE. 
# Al termine calcola anche SRCC e PLCC per misurare la correlazione tra predizioni e MOS reali, 
# restituendo inoltre tutte le predizioni e i target.

import torch
from torch.utils.data import DataLoader

from src.phase3.metrics import srcc, plcc


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on a dataloader.

    Returns:
        mse
        srcc
        plcc
        predictions
        targets
    """

    model.eval()

    predictions = []
    targets = []

    total_squared_error = 0.0
    total_samples = 0

    criterion = torch.nn.MSELoss(reduction="sum")

    for batch in dataloader:

        features = batch["features"].to(device)
        mos = batch["mos"].to(device)

        output = model(features)

        loss = criterion(output, mos)

        total_squared_error += loss.item()
        total_samples += mos.size(0)

        predictions.extend(output.cpu().numpy())
        targets.extend(mos.cpu().numpy())

    mse = total_squared_error / total_samples

    correlation_srcc = srcc(predictions, targets)
    correlation_plcc = plcc(predictions, targets)

    return (
        mse,
        correlation_srcc,
        correlation_plcc,
        predictions,
        targets,
    )