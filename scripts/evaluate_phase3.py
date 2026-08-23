# Valutazione del modello DualEncoderFusion addestrato nella Fase 3.
# Il codice:
#   1. carica il checkpoint migliore;
#   2. ricostruisce il dataset;
#   3. utilizza ESATTAMENTE il validation split salvato nel checkpoint;
#   4. carica le statistiche di normalizzazione salvate durante il training;
#   5. normalizza le feature nello stesso modo usato durante il training;
#   6. ricostruisce la stessa architettura DualEncoderFusion;
#   7. carica i pesi del checkpoint;
#   8. calcola MSE, SRCC e PLCC sul validation set.


import argparse
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


from src.phase3.dataset import FeatureDataset
from src.phase3.aggregation import DualEncoderFusion
from src.phase3.metrics import srcc, plcc


# ============================================================
# NORMALIZED DATASET
# ============================================================

class NormalizedEvaluationDataset(Dataset):
    """
    Dataset used during evaluation.

    It reproduces the same feature normalization used during
    training.

    IMPORTANT:
    The normalization statistics come exclusively from the
    training set and are loaded from the checkpoint.
    """

    def __init__(
        self,
        dataset,
        indices,
        feature_stats,
        mos_mean,
        mos_std,
    ):

        self.dataset = dataset

        self.indices = torch.as_tensor(
            indices,
            dtype=torch.long,
        )

        self.feature_stats = feature_stats

        self.mos_mean = mos_mean
        self.mos_std = mos_std

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]

        # ----------------------------------------------------
        # Raw features
        # ----------------------------------------------------

        ref_base = self.dataset.ref_base[
            :, idx, :
        ]

        dist_base = self.dataset.dist_base[
            :, idx, :
        ]

        ref_large = self.dataset.ref_large[
            :, idx, :
        ]

        dist_large = self.dataset.dist_large[
            :, idx, :
        ]

        # ----------------------------------------------------
        # Normalize Base
        # ----------------------------------------------------

        ref_base = (
            ref_base
            - self.feature_stats["ref_base"]["mean"]
        ) / self.feature_stats["ref_base"]["std"]

        dist_base = (
            dist_base
            - self.feature_stats["dist_base"]["mean"]
        ) / self.feature_stats["dist_base"]["std"]

        # ----------------------------------------------------
        # Normalize Large
        # ----------------------------------------------------

        ref_large = (
            ref_large
            - self.feature_stats["ref_large"]["mean"]
        ) / self.feature_stats["ref_large"]["std"]

        dist_large = (
            dist_large
            - self.feature_stats["dist_large"]["mean"]
        ) / self.feature_stats["dist_large"]["std"]

        # ----------------------------------------------------
        # Normalize MOS
        # ----------------------------------------------------

        mos = self.dataset.mos[idx]

        mos = (
            mos - self.mos_mean
        ) / self.mos_std

        return {
            "ref_base": ref_base,
            "dist_base": dist_base,
            "ref_large": ref_large,
            "dist_large": dist_large,
            "mos": mos,
            "name": self.dataset.image_names[idx],
        }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    mos_mean,
    mos_std,
):
    """
    Evaluate the DualEncoderFusion model.

    Predictions and targets are converted back to the
    original MOS scale before computing MSE, SRCC and PLCC.
    """

    model.eval()

    predictions = []
    targets = []

    mos_mean = mos_mean.to(device)
    mos_std = mos_std.to(device)

    for batch in loader:

        # ----------------------------------------------------
        # Move features to device
        # ----------------------------------------------------

        ref_base = batch[
            "ref_base"
        ].to(
            device,
            non_blocking=True,
        )

        dist_base = batch[
            "dist_base"
        ].to(
            device,
            non_blocking=True,
        )

        ref_large = batch[
            "ref_large"
        ].to(
            device,
            non_blocking=True,
        )

        dist_large = batch[
            "dist_large"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Normalized target
        # ----------------------------------------------------

        y_norm = batch[
            "mos"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        pred_norm = model(
            ref_base,
            dist_base,
            ref_large,
            dist_large,
        )

        # ----------------------------------------------------
        # Convert prediction to original MOS scale
        # ----------------------------------------------------

        pred = (
            pred_norm
            * mos_std
            + mos_mean
        )

        # ----------------------------------------------------
        # Convert target to original MOS scale
        # ----------------------------------------------------

        y = (
            y_norm
            * mos_std
            + mos_mean
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        targets.extend(
            y.cpu().numpy()
        )

    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )

    if len(predictions) == 0:

        raise RuntimeError(
            "Validation loader is empty."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mse = np.mean(
        (
            predictions
            - targets
        ) ** 2
    )

    srcc_value = srcc(
        predictions,
        targets,
    )

    plcc_value = plcc(
        predictions,
        targets,
    )

    return (
        mse,
        srcc_value,
        plcc_value,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Phase 3 DualEncoderFusion "
            "checkpoint."
        )
    )

    # --------------------------------------------------------
    # Feature files
    # --------------------------------------------------------

    parser.add_argument(
        "--features-base",
        required=True,
        help=(
            "Path to the SigLIP2 Base "
            "all-layers .pt file."
        ),
    )

    parser.add_argument(
        "--features-large",
        required=True,
        help=(
            "Path to the SigLIP2 Large "
            "all-layers .pt file."
        ),
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Phase 3 checkpoint.",
    )

    # --------------------------------------------------------
    # Variant
    # --------------------------------------------------------

    parser.add_argument(
        "--variant",
        choices=[
            "small",
            "medium",
            "large",
        ],
        default=None,
        help=(
            "Model variant. If omitted, the value stored "
            "in the checkpoint is used."
        ),
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Validation batch size.",
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
    print("PHASE 3 - DUAL ENCODER EVALUATION")
    print("=" * 70)

    print(
        f"Base features:   {args.features_base}"
    )

    print(
        f"Large features:  {args.features_large}"
    )

    print(
        f"Checkpoint:      {args.checkpoint}"
    )

    print(
        f"Device:          {device}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    print("\nCheckpoint information:")

    if "epoch" in checkpoint:

        print(
            f"  Epoch:          "
            f"{checkpoint['epoch']}"
        )

    if "best_srcc" in checkpoint:

        print(
            f"  Best SRCC:      "
            f"{checkpoint['best_srcc']:.6f}"
        )

    if "best_plcc" in checkpoint:

        print(
            f"  Best PLCC:      "
            f"{checkpoint['best_plcc']:.6f}"
        )

    # ========================================================
    # CHECK NORMALIZATION STATISTICS
    # ========================================================

    if "normalization_stats" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'normalization_stats'. "
            "This checkpoint is not compatible with "
            "the current train_phase3.py."
        )

    if "mos_mean" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain 'mos_mean'."
        )

    if "mos_std" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain 'mos_std'."
        )

    feature_stats = checkpoint[
        "normalization_stats"
    ]

    mos_mean = checkpoint[
        "mos_mean"
    ]

    mos_std = checkpoint[
        "mos_std"
    ]

    required_feature_stats = [
        "ref_base",
        "dist_base",
        "ref_large",
        "dist_large",
    ]

    for key in required_feature_stats:

        if key not in feature_stats:

            raise KeyError(
                f"Normalization statistics missing "
                f"'{key}'."
            )

        if "mean" not in feature_stats[key]:

            raise KeyError(
                f"Normalization statistics for "
                f"'{key}' are missing 'mean'."
            )

        if "std" not in feature_stats[key]:

            raise KeyError(
                f"Normalization statistics for "
                f"'{key}' are missing 'std'."
            )

    print(
        "\nNormalization statistics: "
        "loaded from checkpoint."
    )

    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    dim_base = checkpoint.get(
        "dim_base",
        None,
    )

    dim_large = checkpoint.get(
        "dim_large",
        None,
    )

    variant = (
        args.variant
        if args.variant is not None
        else checkpoint.get(
            "variant",
            "medium",
        )
    )

    print(
        f"\nModel variant: {variant}"
    )

    if dim_base is not None:

        print(
            f"Checkpoint Base dimension: "
            f"{dim_base}"
        )

    if dim_large is not None:

        print(
            f"Checkpoint Large dimension: "
            f"{dim_large}"
        )

    # ========================================================
    # DATASET
    # ========================================================

    print("\nLoading dataset...")

    dataset = FeatureDataset(
        features_base_path=args.features_base,
        features_large_path=args.features_large,
    )

    print("\nDataset:")

    print(
        f"  Samples:          {len(dataset)}"
    )

    print(
        f"  Base shape:       "
        f"{dataset.ref_base.shape}"
    )

    print(
        f"  Large shape:      "
        f"{dataset.ref_large.shape}"
    )

    print(
        f"  Base total dim:   "
        f"{dataset.dim_base}"
    )

    print(
        f"  Large total dim:  "
        f"{dataset.dim_large}"
    )

    # ========================================================
    # CHECK DIMENSIONS
    # ========================================================

    if dim_base is not None:

        if dataset.dim_base != dim_base:

            raise ValueError(
                "Base feature dimension mismatch:\n"
                f"  Checkpoint: {dim_base}\n"
                f"  Dataset:    {dataset.dim_base}"
            )

    if dim_large is not None:

        if dataset.dim_large != dim_large:

            raise ValueError(
                "Large feature dimension mismatch:\n"
                f"  Checkpoint: {dim_large}\n"
                f"  Dataset:    {dataset.dim_large}"
            )

    # ========================================================
    # VALIDATION SPLIT
    # ========================================================

    if "val_indices" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain 'val_indices'. "
            "The current training script saves the exact "
            "validation split, so evaluation requires it."
        )

    val_indices = checkpoint[
        "val_indices"
    ]

    print(
        "\nValidation split: "
        "loaded directly from checkpoint."
    )

    # ========================================================
    # NORMALIZED VALIDATION DATASET
    # ========================================================

    val_dataset = NormalizedEvaluationDataset(
        dataset=dataset,
        indices=val_indices,
        feature_stats=feature_stats,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    use_cuda = torch.cuda.is_available()

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    print(
        f"  Total samples:    {len(dataset)}"
    )

    print(
        f"  Validation:       {len(val_dataset)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = DualEncoderFusion(
        dim_base=dataset.dim_base,
        dim_large=dataset.dim_large,
        variant=variant,
    ).to(device)

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print("\nModel loaded successfully.")

    # ========================================================
    # EVALUATION
    # ========================================================

    print("\nRunning evaluation...")

    (
        mse,
        srcc_value,
        plcc_value,
    ) = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"MSE :  {mse:.6f}"
    )

    print(
        f"SRCC:  {srcc_value:.6f}"
    )

    print(
        f"PLCC:  {plcc_value:.6f}"
    )

    print("-" * 70)

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "best_srcc" in checkpoint:

        print(
            f"Saved best SRCC:  "
            f"{checkpoint['best_srcc']:.6f}"
        )

    if "best_plcc" in checkpoint:

        print(
            f"Saved best PLCC:  "
            f"{checkpoint['best_plcc']:.6f}"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()