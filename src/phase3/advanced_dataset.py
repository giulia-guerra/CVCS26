# Il codice definisce un Dataset PyTorch per l'esperimento Advanced Phase 3.
# Carica le feature di tutti i layer degli encoder SigLIP2 Base e Large,
# convertendole dal formato [layer, sample, feature] a [sample, layer, feature].
# Il dataset mantiene separate le feature reference e distorted e associa a ogni
# immagine il relativo MOS, verificando inoltre la coerenza dei dati tra i due encoder.


import torch
from torch.utils.data import Dataset


class AdvancedFeatureDataset(Dataset):
    """
    Dataset for the Advanced Phase 3 experiment.

    Loads directly:

        siglip2_base_all_layers.pt
        siglip2_large_all_layers.pt

    Original feature format:

        [num_layers, num_samples, feature_dim]

    Converted internally to:

        [num_samples, num_layers, feature_dim]
    """

    def __init__(
        self,
        features_base_path,
        features_large_path,
    ):

        super().__init__()

        print("\nLoading Base features...")
        print(features_base_path)

        base_data = torch.load(
            features_base_path,
            map_location="cpu",
            weights_only=False,
        )

        print("\nLoading Large features...")
        print(features_large_path)

        large_data = torch.load(
            features_large_path,
            map_location="cpu",
            weights_only=False,
        )

        # ==================================================
        # BASE
        # ==================================================

        self.ref_base = (
            base_data["ref_features"]
            .permute(1, 0, 2)
            .float()
        )

        self.dist_base = (
            base_data["dist_features"]
            .permute(1, 0, 2)
            .float()
        )

        # ==================================================
        # LARGE
        # ==================================================

        self.ref_large = (
            large_data["ref_features"]
            .permute(1, 0, 2)
            .float()
        )

        self.dist_large = (
            large_data["dist_features"]
            .permute(1, 0, 2)
            .float()
        )

        # ==================================================
        # MOS
        # ==================================================

        self.mos = base_data["mos"].float()

        # ==================================================
        # CHECKS
        # ==================================================

        if not torch.equal(
            base_data["mos"],
            large_data["mos"],
        ):
            raise ValueError(
                "MOS vectors differ between "
                "Base and Large files."
            )

        num_samples = len(self.mos)

        assert len(self.ref_base) == num_samples
        assert len(self.dist_base) == num_samples
        assert len(self.ref_large) == num_samples
        assert len(self.dist_large) == num_samples

        # ==================================================
        # INFO
        # ==================================================

        print("\n" + "=" * 60)
        print("ADVANCED FEATURE DATASET")
        print("=" * 60)

        print(f"Samples: {num_samples}")

        print(
            f"Base:  {self.ref_base.shape}"
        )

        print(
            f"Large: {self.ref_large.shape}"
        )

        print(
            f"MOS:   {self.mos.shape}"
        )

        print("=" * 60)

    def __len__(self):

        return len(self.mos)

    def __getitem__(self, idx):

        return {
            "ref_base": self.ref_base[idx],
            "dist_base": self.dist_base[idx],
            "ref_large": self.ref_large[idx],
            "dist_large": self.dist_large[idx],
            "mos": self.mos[idx],
        }