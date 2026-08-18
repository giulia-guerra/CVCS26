# FeatureDataset implementa il dataset utilizzato nella Fase 3 per il training supervisionato. 
# Il codice carica le feature estratte dagli encoder salvate nei file .pt, seleziona il layer richiesto, 
# costruisce il vettore di input come differenza assoluta tra le feature dell'immagine di riferimento e di quella distorta, 
# e associa a ogni campione il relativo valore MOS da utilizzare come target durante l'addestramento.


import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    Dataset for Phase 3 DualEncoderFusion.

    Loads features from two encoders:

        - SigLIP2 Base
        - SigLIP2 Large

    Expected .pt format:

        ref_features:
            [num_layers, num_samples, feature_dim]

        dist_features:
            [num_layers, num_samples, feature_dim]

        mos:
            [num_samples]

    The dataset returns all layers for both encoders.

    Example for PIPAL:

        Base:
            [13, 23200, 768]

        Large:
            [25, 23200, 1024]

    Therefore:

        Base flattened dimension:
            13 * 768 = 9984

        Large flattened dimension:
            25 * 1024 = 25600

    The absolute difference between reference and distorted
    features is NOT computed here.

    It is computed inside DualEncoderFusion / IQAFeatureAggregator.
    """

    def __init__(
        self,
        features_base_path,
        features_large_path,
    ):
        """
        Args:
            features_base_path:
                Path to SigLIP2 Base .pt file.

            features_large_path:
                Path to SigLIP2 Large .pt file.
        """

        self.features_base_path = features_base_path
        self.features_large_path = features_large_path

        # ====================================================
        # LOAD FEATURE FILES
        # ====================================================

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

        # ====================================================
        # CHECK REQUIRED KEYS
        # ====================================================

        required_keys = [
            "ref_features",
            "dist_features",
            "mos",
        ]

        for key in required_keys:

            if key not in self.data_base:
                raise KeyError(
                    f"Missing key '{key}' in "
                    f"Base feature file: "
                    f"{features_base_path}"
                )

            if key not in self.data_large:
                raise KeyError(
                    f"Missing key '{key}' in "
                    f"Large feature file: "
                    f"{features_large_path}"
                )

        # ====================================================
        # LOAD FEATURES
        # ====================================================

        self.ref_base = self.data_base[
            "ref_features"
        ].float()

        self.dist_base = self.data_base[
            "dist_features"
        ].float()

        self.ref_large = self.data_large[
            "ref_features"
        ].float()

        self.dist_large = self.data_large[
            "dist_features"
        ].float()

        # ====================================================
        # CHECK FEATURE DIMENSIONS
        # ====================================================

        if self.ref_base.ndim != 3:
            raise ValueError(
                "Base ref_features must have shape "
                "[num_layers, num_samples, feature_dim]. "
                f"Found shape: {self.ref_base.shape}"
            )

        if self.dist_base.ndim != 3:
            raise ValueError(
                "Base dist_features must have shape "
                "[num_layers, num_samples, feature_dim]. "
                f"Found shape: {self.dist_base.shape}"
            )

        if self.ref_large.ndim != 3:
            raise ValueError(
                "Large ref_features must have shape "
                "[num_layers, num_samples, feature_dim]. "
                f"Found shape: {self.ref_large.shape}"
            )

        if self.dist_large.ndim != 3:
            raise ValueError(
                "Large dist_features must have shape "
                "[num_layers, num_samples, feature_dim]. "
                f"Found shape: {self.dist_large.shape}"
            )

        # ====================================================
        # CHECK REF / DIST SHAPES WITHIN EACH MODEL
        # ====================================================

        if self.ref_base.shape != self.dist_base.shape:

            raise ValueError(
                "Base reference and distorted features "
                "have different shapes:\n"
                f"  ref_base:  {self.ref_base.shape}\n"
                f"  dist_base: {self.dist_base.shape}"
            )

        if self.ref_large.shape != self.dist_large.shape:

            raise ValueError(
                "Large reference and distorted features "
                "have different shapes:\n"
                f"  ref_large:  {self.ref_large.shape}\n"
                f"  dist_large: {self.dist_large.shape}"
            )

        # ====================================================
        # LOAD MOS
        # ====================================================

        self.mos_base = self.data_base[
            "mos"
        ].float().flatten()

        self.mos_large = self.data_large[
            "mos"
        ].float().flatten()

        # ====================================================
        # CHECK NUMBER OF SAMPLES
        # ====================================================

        num_samples_base = self.ref_base.shape[1]

        num_samples_large = self.ref_large.shape[1]

        if num_samples_base != num_samples_large:

            raise ValueError(
                "Base and Large contain different "
                "numbers of samples:\n"
                f"  Base:  {num_samples_base}\n"
                f"  Large: {num_samples_large}"
            )

        if len(self.mos_base) != num_samples_base:

            raise ValueError(
                "Number of Base MOS values does not "
                "match number of Base samples:\n"
                f"  MOS:     {len(self.mos_base)}\n"
                f"  Samples: {num_samples_base}"
            )

        if len(self.mos_large) != num_samples_large:

            raise ValueError(
                "Number of Large MOS values does not "
                "match number of Large samples:\n"
                f"  MOS:     {len(self.mos_large)}\n"
                f"  Samples: {num_samples_large}"
            )

        # ====================================================
        # CHECK MOS ALIGNMENT
        # ====================================================

        if not torch.allclose(
            self.mos_base,
            self.mos_large,
            atol=1e-6,
        ):
            raise ValueError(
                "MOS vectors from Base and Large "
                "are different. The two feature files "
                "may not refer to the same sample ordering."
            )

        # Use one common MOS vector
        self.mos = self.mos_base

        # ====================================================
        # IMAGE NAMES
        # ====================================================

        self.image_names = self.data_base.get(
            "image_names",
            [str(i) for i in range(len(self.mos))]
        )

        if len(self.image_names) != len(self.mos):

            raise ValueError(
                "Number of image names does not match "
                "number of samples:\n"
                f"  Names:   {len(self.image_names)}\n"
                f"  Samples: {len(self.mos)}"
            )

        # ====================================================
        # MODEL CONFIGURATION
        # ====================================================

        self.model_config_base = self.data_base.get(
            "model_config",
            "unknown",
        )

        self.model_config_large = self.data_large.get(
            "model_config",
            "unknown",
        )

        # ====================================================
        # FEATURE SHAPES / DIMENSIONS
        # ====================================================

        self.num_layers_base = self.ref_base.shape[0]
        self.feature_dim_base = self.ref_base.shape[2]

        self.num_layers_large = self.ref_large.shape[0]
        self.feature_dim_large = self.ref_large.shape[2]

        # Flattened dimensions used by DualEncoderFusion
        self.dim_base = (
            self.num_layers_base
            * self.feature_dim_base
        )

        self.dim_large = (
            self.num_layers_large
            * self.feature_dim_large
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print("\nFeatureDataset initialized:")
        print(
            f"  Base features:  {self.ref_base.shape}"
        )
        print(
            f"  Large features: {self.ref_large.shape}"
        )
        print(
            f"  Samples:        {len(self.mos)}"
        )
        print(
            f"  Base dim:       {self.dim_base}"
        )
        print(
            f"  Large dim:      {self.dim_large}"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):
        return len(self.mos)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        return {
            # ------------------------------------------------
            # SigLIP2 Base
            # Shape:
            # [num_layers_base, feature_dim_base]
            # ------------------------------------------------

            "ref_base": self.ref_base[
                :, index, :
            ],

            "dist_base": self.dist_base[
                :, index, :
            ],

            # ------------------------------------------------
            # SigLIP2 Large
            # Shape:
            # [num_layers_large, feature_dim_large]
            # ------------------------------------------------

            "ref_large": self.ref_large[
                :, index, :
            ],

            "dist_large": self.dist_large[
                :, index, :
            ],

            # ------------------------------------------------
            # MOS
            # ------------------------------------------------

            "mos": self.mos[index],

            # ------------------------------------------------
            # Image name
            # ------------------------------------------------

            "name": self.image_names[index],
        }