# FeatureDataset implementa il dataset utilizzato nella Fase 3 per il training supervisionato. 
# Il codice carica le feature estratte dagli encoder salvate nei file .pt, seleziona il layer richiesto, 
# costruisce il vettore di input come differenza assoluta tra le feature dell'immagine di riferimento e di quella distorta, 
# e associa a ogni campione il relativo valore MOS da utilizzare come target durante l'addestramento.

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    Dataset for Phase 3 supervised IQA training.

    Expected .pt format:

        ref_features:  [num_layers, num_samples, feature_dim]
        dist_features: [num_layers, num_samples, feature_dim]
        mos:           [num_samples]

    For a selected layer, the input is:

        abs(ref_features - dist_features)

    and the target is:

        MOS
    """

    def __init__(self, pt_path, layer=0):
        self.pt_path = pt_path
        self.layer = layer

        data = torch.load(pt_path, map_location="cpu")

        required_keys = [
            "ref_features",
            "dist_features",
            "mos",
        ]

        for key in required_keys:
            if key not in data:
                raise KeyError(
                    f"Missing key '{key}' in {pt_path}. "
                    f"Available keys: {list(data.keys())}"
                )

        ref_features = data["ref_features"]
        dist_features = data["dist_features"]
        mos = data["mos"]

        if ref_features.ndim != 3:
            raise ValueError(
                f"Expected ref_features to have 3 dimensions, "
                f"got {ref_features.shape}"
            )

        if dist_features.ndim != 3:
            raise ValueError(
                f"Expected dist_features to have 3 dimensions, "
                f"got {dist_features.shape}"
            )

        num_layers = ref_features.shape[0]

        if not 0 <= layer < num_layers:
            raise ValueError(
                f"Invalid layer {layer}. "
                f"Available layers: 0-{num_layers - 1}"
            )

        if ref_features.shape != dist_features.shape:
            raise ValueError(
                "ref_features and dist_features have different shapes: "
                f"{ref_features.shape} vs {dist_features.shape}"
            )

        ref = ref_features[layer].float()
        dist = dist_features[layer].float()
        mos = mos.float().flatten()

        if ref.shape[0] != mos.shape[0]:
            raise ValueError(
                f"Number of samples does not match MOS: "
                f"{ref.shape[0]} vs {mos.shape[0]}"
            )

        # Phase 3 baseline:
        # use absolute difference between reference and distorted features.
        features = torch.abs(ref - dist)

        self.features = features
        self.mos = mos

        self.image_names = data.get(
            "image_names",
            [str(i) for i in range(len(mos))]
        )

        self.model_config = data.get("model_config", "unknown")

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "mos": self.mos[index],
            "name": self.image_names[index],
        }