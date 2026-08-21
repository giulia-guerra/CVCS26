import torch
from torch.utils.data import Dataset


class AdvancedFeatureDataset(Dataset):
    """
    Dataset for the Advanced Phase 3 experiment.

    The dataset is loaded from a single .pt file containing:

        ref_base
        dist_base
        ref_large
        dist_large
        mos

    Expected shapes:

        ref_base:
            [N, L_base, 768]

        dist_base:
            [N, L_base, 768]

        ref_large:
            [N, L_large, 1024]

        dist_large:
            [N, L_large, 1024]

        mos:
            [N]

    Each sample returned by __getitem__ is a dictionary.
    """

    def __init__(self, feature_file):

        super().__init__()

        print("\nLoading Advanced dataset...")
        print(f"Feature file: {feature_file}")

        data = torch.load(
            feature_file,
            map_location="cpu",
        )

        # ==================================================
        # REQUIRED KEYS
        # ==================================================

        required_keys = [
            "ref_base",
            "dist_base",
            "ref_large",
            "dist_large",
            "mos",
        ]

        for key in required_keys:

            if key not in data:

                raise KeyError(
                    f"Missing key '{key}' in feature file. "
                    f"Expected keys: {required_keys}"
                )

        # ==================================================
        # LOAD DATA
        # ==================================================

        self.ref_base = data["ref_base"].float()
        self.dist_base = data["dist_base"].float()

        self.ref_large = data["ref_large"].float()
        self.dist_large = data["dist_large"].float()

        self.mos = data["mos"].float()

        # ==================================================
        # CHECK NUMBER OF SAMPLES
        # ==================================================

        num_samples = len(self.mos)

        if len(self.ref_base) != num_samples:
            raise ValueError(
                "ref_base and mos have different "
                "number of samples."
            )

        if len(self.dist_base) != num_samples:
            raise ValueError(
                "dist_base and mos have different "
                "number of samples."
            )

        if len(self.ref_large) != num_samples:
            raise ValueError(
                "ref_large and mos have different "
                "number of samples."
            )

        if len(self.dist_large) != num_samples:
            raise ValueError(
                "dist_large and mos have different "
                "number of samples."
            )

        # ==================================================
        # CHECK DIMENSIONS
        # ==================================================

        if self.ref_base.ndim != 3:
            raise ValueError(
                f"ref_base must be 3D, got "
                f"{self.ref_base.shape}"
            )

        if self.dist_base.ndim != 3:
            raise ValueError(
                f"dist_base must be 3D, got "
                f"{self.dist_base.shape}"
            )

        if self.ref_large.ndim != 3:
            raise ValueError(
                f"ref_large must be 3D, got "
                f"{self.ref_large.shape}"
            )

        if self.dist_large.ndim != 3:
            raise ValueError(
                f"dist_large must be 3D, got "
                f"{self.dist_large.shape}"
            )

        # ==================================================
        # REFERENCE / DISTORTED SHAPE CHECK
        # ==================================================

        if self.ref_base.shape != self.dist_base.shape:

            raise ValueError(
                "ref_base and dist_base must have "
                "the same shape."
            )

        if self.ref_large.shape != self.dist_large.shape:

            raise ValueError(
                "ref_large and dist_large must have "
                "the same shape."
            )

        # ==================================================
        # FEATURE DIMENSION CHECK
        # ==================================================

        base_dim = self.ref_base.shape[-1]
        large_dim = self.ref_large.shape[-1]

        if base_dim != 768:

            raise ValueError(
                f"Expected Base feature dimension 768, "
                f"got {base_dim}"
            )

        if large_dim != 1024:

            raise ValueError(
                f"Expected Large feature dimension 1024, "
                f"got {large_dim}"
            )

        # ==================================================
        # PRINT INFORMATION
        # ==================================================

        print("\n" + "=" * 60)
        print("ADVANCED FEATURE DATASET")
        print("=" * 60)

        print(f"Samples:        {num_samples}")

        print(
            f"Base features:  {self.ref_base.shape}"
        )

        print(
            f"Large features: {self.ref_large.shape}"
        )

        print(
            f"MOS:            {self.mos.shape}"
        )

        print(
            f"Base layers:    {self.ref_base.shape[1]}"
        )

        print(
            f"Large layers:   {self.ref_large.shape[1]}"
        )

        print(
            f"Base dim:       {self.ref_base.shape[2]}"
        )

        print(
            f"Large dim:      {self.ref_large.shape[2]}"
        )

        print("=" * 60)

    # ======================================================
    # LENGTH
    # ======================================================

    def __len__(self):

        return len(self.mos)

    # ======================================================
    # GET ITEM
    # ======================================================

    def __getitem__(self, idx):

        return {
            "ref_base": self.ref_base[idx],
            "dist_base": self.dist_base[idx],
            "ref_large": self.ref_large[idx],
            "dist_large": self.dist_large[idx],
            "mos": self.mos[idx],
        }