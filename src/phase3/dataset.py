# FeatureDataset implementa il dataset utilizzato nella Fase 3 per il training supervisionato. 
# Il codice carica le feature estratte dagli encoder salvate nei file .pt, seleziona il layer richiesto, 
# costruisce il vettore di input come differenza assoluta tra le feature dell'immagine di riferimento e di quella distorta, 
# e associa a ogni campione il relativo valore MOS da utilizzare come target durante l'addestramento.


import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    Dataset for the Advanced Attention Aggregator.

    Loads SigLIP2 Base and Large features.

    Stored feature format:

        ref_features:
            [num_layers, num_samples, feature_dim]

        dist_features:
            [num_layers, num_samples, feature_dim]

        mos:
            [num_samples]

    Returned sample:

        ref_base:
            [num_layers_base, feature_dim_base]

        dist_base:
            [num_layers_base, feature_dim_base]

        ref_large:
            [num_layers_large, feature_dim_large]

        dist_large:
            [num_layers_large, feature_dim_large]

        mos:
            scalar

    The DataLoader therefore produces:

        ref_base:
            [batch_size, num_layers_base, feature_dim_base]

        ref_large:
            [batch_size, num_layers_large, feature_dim_large]

    IMPORTANT:
        No flattening is performed.
    """

    def __init__(
        self,
        features_base_path,
        features_large_path,
    ):

        self.data_base = torch.load(
            features_base_path,
            map_location="cpu",
            weights_only=False,
        )

        self.data_large = torch.load(
            features_large_path,
            map_location="cpu",
            weights_only=False,
        )

        # --------------------------------------------------
        # Load features
        # --------------------------------------------------

        self.ref_base = self.data_base["ref_features"].float()
        self.dist_base = self.data_base["dist_features"].float()

        self.ref_large = self.data_large["ref_features"].float()
        self.dist_large = self.data_large["dist_features"].float()

        self.mos_base = self.data_base["mos"].float().flatten()
        self.mos_large = self.data_large["mos"].float().flatten()

        # --------------------------------------------------
        # Check dimensions
        # --------------------------------------------------

        for name, tensor in {
            "ref_base": self.ref_base,
            "dist_base": self.dist_base,
            "ref_large": self.ref_large,
            "dist_large": self.dist_large,
        }.items():

            if tensor.ndim != 3:
                raise ValueError(
                    f"{name} must be 3D "
                    f"[num_layers, num_samples, feature_dim]. "
                    f"Found {tensor.shape}"
                )

        # --------------------------------------------------
        # Check ref/dist shapes
        # --------------------------------------------------

        if self.ref_base.shape != self.dist_base.shape:
            raise ValueError(
                "Base reference/distorted shapes do not match: "
                f"{self.ref_base.shape} vs "
                f"{self.dist_base.shape}"
            )

        if self.ref_large.shape != self.dist_large.shape:
            raise ValueError(
                "Large reference/distorted shapes do not match: "
                f"{self.ref_large.shape} vs "
                f"{self.dist_large.shape}"
            )

        # --------------------------------------------------
        # Check number of samples
        # --------------------------------------------------

        n_base = self.ref_base.shape[1]
        n_large = self.ref_large.shape[1]

        if n_base != n_large:
            raise ValueError(
                f"Base and Large have different number of samples: "
                f"{n_base} vs {n_large}"
            )

        if len(self.mos_base) != n_base:
            raise ValueError(
                f"Base MOS length {len(self.mos_base)} "
                f"does not match samples {n_base}"
            )

        if len(self.mos_large) != n_large:
            raise ValueError(
                f"Large MOS length {len(self.mos_large)} "
                f"does not match samples {n_large}"
            )

        # --------------------------------------------------
        # Check MOS alignment
        # --------------------------------------------------

        if not torch.allclose(
            self.mos_base,
            self.mos_large,
            atol=1e-6,
        ):
            raise ValueError(
                "Base and Large MOS vectors are different."
            )

        self.mos = self.mos_base

        # --------------------------------------------------
        # Image names
        # --------------------------------------------------

        self.image_names = self.data_base.get(
            "image_names",
            [str(i) for i in range(len(self.mos))]
        )

        if len(self.image_names) != len(self.mos):
            raise ValueError(
                "Number of image names does not match "
                "number of samples."
            )

        # --------------------------------------------------
        # Dimensions
        # --------------------------------------------------

        self.num_layers_base = self.ref_base.shape[0]
        self.feature_dim_base = self.ref_base.shape[2]

        self.num_layers_large = self.ref_large.shape[0]
        self.feature_dim_large = self.ref_large.shape[2]

        print("\nFeatureDataset initialized")
        print(f"  Base:  {self.ref_base.shape}")
        print(f"  Large: {self.ref_large.shape}")
        print(f"  MOS:   {self.mos.shape}")
        print(f"  Base feature dim:  {self.feature_dim_base}")
        print(f"  Large feature dim: {self.feature_dim_large}")

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, index):

        return {
            "ref_base": self.ref_base[:, index, :],
            "dist_base": self.dist_base[:, index, :],

            "ref_large": self.ref_large[:, index, :],
            "dist_large": self.dist_large[:, index, :],

            "mos": self.mos[index],

            "name": self.image_names[index],
        }