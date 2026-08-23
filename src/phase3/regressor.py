# Il codice definisce un MLP (Multi-Layer Perceptron) per la regressione del MOS.
# Riceve in input le feature estratte dall'encoder e le elabora attraverso
# due layer fully connected con ReLU e dropout, producendo infine un singolo
# valore che rappresenta il MOS predetto per ogni immagine.


import torch.nn as nn


class IQARegressor(nn.Module):
    """
    Simple MLP regression model for Phase 3.

    Input:
        feature_dim

    Output:
        predicted MOS
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)