# Il codice definisce lo script di valutazione dell'Advanced Phase 3.
# Carica il dataset con le feature di tutti i layer di SigLIP2 Base e Large
# e il checkpoint del modello Transformer già addestrato. Il modello viene
# valutato sul validation set, riportando le predizioni alla scala originale
# del MOS e calcolando MSE, SRCC e PLCC per misurare le prestazioni della rete.


import argparse

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, random_split

from src.phase3.advanced_dataset import AdvancedFeatureDataset
from src.phase3.aggregation import AdvancedAttentionAggregator


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
    mos_mean,
    mos_std,
):
    """
    Evaluate the Advanced model.

    The model predicts normalized MOS.

    Predictions are converted back to the original MOS
    scale before computing MSE, SRCC and PLCC.
    """

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            ref_base = batch["ref_base"].to(device)
            dist_base = batch["dist_base"].to(device)

            ref_large = batch["ref_large"].to(device)
            dist_large = batch["dist_large"].to(device)

            mos = batch["mos"].to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            prediction_normalized = model(
                ref_base,
                dist_base,
                ref_large,
                dist_large,
            )

            # ------------------------------------------------
            # Convert prediction back to original MOS scale
            # ------------------------------------------------

            prediction = (
                prediction_normalized * mos_std
                + mos_mean
            )

            predictions.append(
                prediction.cpu()
            )

            targets.append(
                mos.cpu()
            )

    # ========================================================
    # CONCATENATE
    # ========================================================

    predictions = torch.cat(
        predictions
    ).numpy()

    targets = torch.cat(
        targets
    ).numpy()

    # Make sure they are 1D
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    # ========================================================
    # MSE
    # ========================================================

    mse = np.mean(
        (predictions - targets) ** 2
    )

    # ========================================================
    # SRCC
    # ========================================================

    srcc = spearmanr(
        predictions,
        targets,
    ).statistic

    # ========================================================
    # PLCC
    # ========================================================

    plcc = pearsonr(
        predictions,
        targets,
    ).statistic

    return mse, srcc, plcc


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Phase 3 Advanced "
            "Transformer Attention Aggregator"
        )
    )

    # ========================================================
    # FEATURE FILES
    # ========================================================

    parser.add_argument(
        "--features-base",
        required=True,
        help=(
            "Path to SigLIP2 Base "
            "all-layers feature file."
        ),
    )

    parser.add_argument(
        "--features-large",
        required=True,
        help=(
            "Path to SigLIP2 Large "
            "all-layers feature file."
        ),
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained Advanced checkpoint.",
    )

    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    parser.add_argument(
        "--dim-base",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--dim-large",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--proj-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    print("\n" + "=" * 70)
    print("PHASE 3 - ADVANCED EVALUATION")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = AdvancedFeatureDataset(
        features_base_path=args.features_base,
        features_large_path=args.features_large,
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    val_size = int(
        len(dataset) * args.val_ratio
    )

    train_size = (
        len(dataset)
        - val_size
    )

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    print("\n" + "=" * 70)
    print("VALIDATION SPLIT")
    print("=" * 70)

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Validation ratio: {args.val_ratio}"
    )

    print(
        f"Random seed: {args.seed}"
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = AdvancedAttentionAggregator(
        dim_base=args.dim_base,
        dim_large=args.dim_large,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Check model state
    # --------------------------------------------------------

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # LOAD MOS NORMALIZATION
    # ========================================================

    if (
        "mos_mean" not in checkpoint
        or "mos_std" not in checkpoint
    ):

        raise KeyError(
            "Checkpoint does not contain "
            "'mos_mean' and/or 'mos_std'. "
            "This checkpoint was probably created "
            "with the old training script without "
            "MOS normalization."
        )

    mos_mean = checkpoint["mos_mean"].to(device)
    mos_std = checkpoint["mos_std"].to(device)

    if mos_std.item() <= 0:

        raise ValueError(
            "Invalid MOS standard deviation "
            "stored in checkpoint."
        )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    print(
        f"MOS mean: {mos_mean.item():.6f}"
    )

    print(
        f"MOS std:  {mos_std.item():.6f}"
    )

    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "best_srcc" in checkpoint:

        print(
            f"Saved best SRCC: "
            f"{checkpoint['best_srcc']:.6f}"
        )

    if "best_plcc" in checkpoint:

        print(
            f"Saved best PLCC: "
            f"{checkpoint['best_plcc']:.6f}"
        )

    if "best_mse" in checkpoint:

        print(
            f"Saved best MSE: "
            f"{checkpoint['best_mse']:.6f}"
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    mse, srcc, plcc = evaluate(
        model,
        val_loader,
        device,
        mos_mean,
        mos_std,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"MSE:  {mse:.6f}"
    )

    print(
        f"SRCC: {srcc:.6f}"
    )

    print(
        f"PLCC: {plcc:.6f}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()