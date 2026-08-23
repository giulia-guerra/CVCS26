# Il codice definisce lo script di valutazione della Phase 3, che combina
# i dataset LIVE e TID2013 utilizzando le feature di tutti i layer di SigLIP2
# Base e Large. Carica il checkpoint del modello DualEncoderFusion, applica
# la normalizzazione delle feature e del MOS e valuta le predizioni sia sul
# dataset combinato sia separatamente su LIVE e TID2013 tramite MSE, SRCC e PLCC.
# Infine, salva le predizioni e i relativi valori MOS in un file CSV.


import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.phase3.aggregation import DualEncoderFusion
from src.phase3.metrics import srcc, plcc


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_feature_file(path):

    data = torch.load(
        path,
        map_location="cpu",
    )

    required_keys = [
        "ref_features",
        "dist_features",
        "mos",
        "image_names",
    ]

    for key in required_keys:

        if key not in data:

            raise KeyError(
                f"Missing key '{key}' in {path}"
            )

    return data


# ============================================================
# MIXTURE DATASET
# ============================================================

class MixtureFeatureDataset(Dataset):

    def __init__(
        self,
        live_base,
        live_large,
        tid_base,
        tid_large,
    ):

        live_base_data = load_feature_file(
            live_base
        )

        live_large_data = load_feature_file(
            live_large
        )

        tid_base_data = load_feature_file(
            tid_base
        )

        tid_large_data = load_feature_file(
            tid_large
        )

        # ----------------------------------------------------
        # Compatibility checks
        # ----------------------------------------------------

        self._check_pair(
            live_base_data,
            live_large_data,
            "LIVE",
        )

        self._check_pair(
            tid_base_data,
            tid_large_data,
            "TID2013",
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        self.ref_base = torch.cat(
            [
                live_base_data["ref_features"],
                tid_base_data["ref_features"],
            ],
            dim=1,
        )

        self.dist_base = torch.cat(
            [
                live_base_data["dist_features"],
                tid_base_data["dist_features"],
            ],
            dim=1,
        )

        self.ref_large = torch.cat(
            [
                live_large_data["ref_features"],
                tid_large_data["ref_features"],
            ],
            dim=1,
        )

        self.dist_large = torch.cat(
            [
                live_large_data["dist_features"],
                tid_large_data["dist_features"],
            ],
            dim=1,
        )

        self.mos = torch.cat(
            [
                live_base_data["mos"].float(),
                tid_base_data["mos"].float(),
            ],
            dim=0,
        )

        self.image_names = (
            [
                f"LIVE/{x}"
                for x in live_base_data["image_names"]
            ]
            +
            [
                f"TID2013/{x}"
                for x in tid_base_data["image_names"]
            ]
        )

        self.dataset_names = (
            ["LIVE"] * len(
                live_base_data["mos"]
            )
            +
            ["TID2013"] * len(
                tid_base_data["mos"]
            )
        )

        self.dim_base = (
            self.ref_base.shape[0]
            *
            self.ref_base.shape[2]
        )

        self.dim_large = (
            self.ref_large.shape[0]
            *
            self.ref_large.shape[2]
        )

    @staticmethod
    def _check_pair(
        base_data,
        large_data,
        name,
    ):

        if (
            base_data["ref_features"].shape[1]
            !=
            large_data["ref_features"].shape[1]
        ):
            raise ValueError(
                f"{name}: Base/Large sample count mismatch."
            )

        if not torch.allclose(
            base_data["mos"].float(),
            large_data["mos"].float(),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                f"{name}: Base/Large MOS mismatch."
            )

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):

        return {
            "ref_base":
                self.ref_base[:, idx, :],

            "dist_base":
                self.dist_base[:, idx, :],

            "ref_large":
                self.ref_large[:, idx, :],

            "dist_large":
                self.dist_large[:, idx, :],

            "mos":
                self.mos[idx],

            "name":
                self.image_names[idx],

            "dataset":
                self.dataset_names[idx],
        }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sample(
    tensor,
    stats,
    name,
):

    return (
        tensor - stats[name]["mean"]
    ) / stats[name]["std"]


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_checkpoint(
    model,
    dataset,
    indices,
    feature_stats,
    mos_stats,
    device,
):

    model.eval()

    predictions = []
    targets = []
    dataset_labels = []
    names = []

    for idx in indices:

        dataset_name = (
            dataset.dataset_names[idx]
        )

        ref_base = normalize_sample(
            dataset.ref_base[:, idx, :],
            feature_stats,
            "ref_base",
        ).unsqueeze(0).to(device)

        dist_base = normalize_sample(
            dataset.dist_base[:, idx, :],
            feature_stats,
            "dist_base",
        ).unsqueeze(0).to(device)

        ref_large = normalize_sample(
            dataset.ref_large[:, idx, :],
            feature_stats,
            "ref_large",
        ).unsqueeze(0).to(device)

        dist_large = normalize_sample(
            dataset.dist_large[:, idx, :],
            feature_stats,
            "dist_large",
        ).unsqueeze(0).to(device)

        mos_mean = mos_stats[
            dataset_name
        ]["mean"].to(device)

        mos_std = mos_stats[
            dataset_name
        ]["std"].to(device)

        target_norm = (
            (
                dataset.mos[idx]
                -
                mos_stats[
                    dataset_name
                ]["mean"]
            )
            /
            mos_stats[
                dataset_name
            ]["std"]
        ).to(device)

        pred_norm = model(
            ref_base,
            dist_base,
            ref_large,
            dist_large,
        )

        pred = (
            pred_norm.squeeze()
            *
            mos_std
            +
            mos_mean
        )

        target = (
            target_norm
            *
            mos_std
            +
            mos_mean
        )

        predictions.append(
            pred.item()
        )

        targets.append(
            target.item()
        )

        dataset_labels.append(
            dataset_name
        )

        names.append(
            dataset.image_names[idx]
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    dataset_labels = np.asarray(
        dataset_labels
    )

    results = {}

    # --------------------------------------------------------
    # MIXED
    # --------------------------------------------------------

    results["all_mse"] = np.mean(
        (predictions - targets) ** 2
    )

    results["all_srcc"] = srcc(
        predictions,
        targets,
    )

    results["all_plcc"] = plcc(
        predictions,
        targets,
    )

    # --------------------------------------------------------
    # PER DATASET
    # --------------------------------------------------------

    for dataset_name in [
        "LIVE",
        "TID2013",
    ]:

        mask = (
            dataset_labels
            ==
            dataset_name
        )

        p = predictions[mask]
        t = targets[mask]

        results[
            f"{dataset_name.lower()}_mse"
        ] = np.mean(
            (p - t) ** 2
        )

        results[
            f"{dataset_name.lower()}_srcc"
        ] = srcc(
            p,
            t,
        )

        results[
            f"{dataset_name.lower()}_plcc"
        ] = plcc(
            p,
            t,
        )

    results["predictions"] = predictions
    results["targets"] = targets
    results["dataset_labels"] = dataset_labels
    results["names"] = names

    return results


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    results,
    output_path,
):

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "name",
                "dataset",
                "target_mos",
                "predicted_mos",
            ]
        )

        for row in zip(
            results["names"],
            results["dataset_labels"],
            results["targets"],
            results["predictions"],
        ):

            writer.writerow(
                [
                    row[0],
                    row[1],
                    float(row[2]),
                    float(row[3]),
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--live-base",
        required=True,
    )

    parser.add_argument(
        "--live-large",
        required=True,
    )

    parser.add_argument(
        "--tid-base",
        required=True,
    )

    parser.add_argument(
        "--tid-large",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "PHASE 3.5 - MIXTURE EVALUATION"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False,
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Saved best SRCC: "
        f"{checkpoint['best_srcc']:.6f}"
    )

    print(
        f"Saved best PLCC: "
        f"{checkpoint['best_plcc']:.6f}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = MixtureFeatureDataset(
        live_base=args.live_base,
        live_large=args.live_large,
        tid_base=args.tid_base,
        tid_large=args.tid_large,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = DualEncoderFusion(
        dim_base=checkpoint["dim_base"],
        dim_large=checkpoint["dim_large"],
        variant=checkpoint["variant"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Variant: {checkpoint['variant']}"
    )

    # ========================================================
    # VALIDATION INDICES
    # ========================================================

    val_indices = checkpoint[
        "val_indices"
    ]

    feature_stats = checkpoint[
        "feature_normalization_stats"
    ]

    mos_stats = checkpoint[
        "mos_normalization_stats"
    ]

    print(
        f"Validation samples: "
        f"{len(val_indices)}"
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate_checkpoint(
        model=model,
        dataset=dataset,
        indices=val_indices,
        feature_stats=feature_stats,
        mos_stats=mos_stats,
        device=device,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\nRESULTS")
    print("-" * 70)

    print(
        f"MIXED   | "
        f"MSE: {results['all_mse']:.4f} | "
        f"SRCC: {results['all_srcc']:.6f} | "
        f"PLCC: {results['all_plcc']:.6f}"
    )

    print(
        f"LIVE    | "
        f"MSE: {results['live_mse']:.4f} | "
        f"SRCC: {results['live_srcc']:.6f} | "
        f"PLCC: {results['live_plcc']:.6f}"
    )

    print(
        f"TID2013 | "
        f"MSE: {results['tid2013_mse']:.4f} | "
        f"SRCC: {results['tid2013_srcc']:.6f} | "
        f"PLCC: {results['tid2013_plcc']:.6f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    if args.output is None:

        checkpoint_path = Path(
            args.checkpoint
        )

        output_path = (
            checkpoint_path.parent
            /
            (
                checkpoint_path.stem
                +
                "_predictions.csv"
            )
        )

    else:

        output_path = Path(
            args.output
        )

    save_predictions(
        results,
        output_path,
    )

    print(
        f"\nPredictions saved to:"
        f"\n{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()