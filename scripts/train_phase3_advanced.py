# Il codice implementa il training dell'Advanced Phase 3 con un aggregatore
# basato su Transformer Attention. Utilizza le feature di tutti i layer di
# SigLIP2 Base e Large, normalizza il MOS usando esclusivamente i dati di
# training e ottimizza il modello tramite MSELoss e AdamW. Ad ogni epoca
# valuta il modello sul validation set tramite MSE, SRCC e PLCC e salva
# automaticamente il checkpoint con il miglior SRCC, utilizzando anche
# l'early stopping per evitare overfitting.


import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, random_split

from src.phase3.advanced_dataset import AdvancedFeatureDataset
from src.phase3.aggregation import AdvancedAttentionAggregator


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    Evaluate the model.

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
            # Normalize target MOS
            # ------------------------------------------------

            mos_normalized = (
                (mos - mos_mean)
                / mos_std
            )

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
# TRAIN
# ============================================================

def train(args):

    # ========================================================
    # SEED
    # ========================================================

    set_seed(args.seed)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("PHASE 3 - ADVANCED TRAINING")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # FEATURE FILES
    # ========================================================

    print("\nLoading existing feature files...")

    print(
        f"Base features : {args.features_base}"
    )

    print(
        f"Large features: {args.features_large}"
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

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples:   {len(val_dataset)}"
    )

    print(
        f"Validation ratio: {args.val_ratio}"
    )

    print(
        f"Random seed:      {args.seed}"
    )

    # ========================================================
    # MOS NORMALIZATION
    # ========================================================
    #
    # IMPORTANT:
    # Mean and std are calculated ONLY on the training set.
    #
    # This avoids leaking validation information into training.
    # ========================================================

    train_indices = train_dataset.indices

    train_mos = dataset.mos[
        train_indices
    ].float()

    mos_mean = train_mos.mean().to(device)

    mos_std = train_mos.std().to(device)

    # Safety check
    if mos_std.item() <= 0:

        raise ValueError(
            "MOS standard deviation is zero. "
            "Cannot normalize MOS."
        )

    print("\n" + "=" * 70)
    print("MOS NORMALIZATION")
    print("=" * 70)

    print(
        f"Training MOS mean: "
        f"{mos_mean.item():.6f}"
    )

    print(
        f"Training MOS std:  "
        f"{mos_std.item():.6f}"
    )

    print(
        "\nTraining target:"
    )

    print(
        "MOS_normalized = "
        "(MOS - mean) / std"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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

    print("\nModel:")
    print(model)

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.MSELoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        args.output_dir,
        "best_advanced.pt",
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_srcc = -float("inf")
    best_plcc = -float("inf")
    best_mse = float("inf")

    patience_counter = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for batch in train_loader:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            ref_base = batch["ref_base"].to(device)
            dist_base = batch["dist_base"].to(device)

            ref_large = batch["ref_large"].to(device)
            dist_large = batch["dist_large"].to(device)

            mos = batch["mos"].to(device)

            # ------------------------------------------------
            # Normalize MOS
            # ------------------------------------------------

            mos_normalized = (
                (mos - mos_mean)
                / mos_std
            )

            # ------------------------------------------------
            # Reset gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            prediction_normalized = model(
                ref_base,
                dist_base,
                ref_large,
                dist_large,
            )

            # ------------------------------------------------
            # MSE on normalized MOS
            # ------------------------------------------------

            loss = criterion(
                prediction_normalized,
                mos_normalized,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Accumulate loss
            # ------------------------------------------------

            total_loss += (
                loss.item()
                * mos.size(0)
            )

        # ====================================================
        # TRAIN LOSS
        # ====================================================

        train_loss = (
            total_loss
            / len(train_dataset)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_mse, srcc, plcc = evaluate(
            model,
            val_loader,
            device,
            mos_mean,
            mos_std,
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"SRCC: {srcc:.6f} | "
            f"PLCC: {plcc:.6f}"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if srcc > best_srcc:

            best_srcc = srcc
            best_plcc = plcc
            best_mse = val_mse

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_srcc":
                        srcc,

                    "best_plcc":
                        plcc,

                    "best_mse":
                        val_mse,

                    # ----------------------------------------
                    # MOS normalization statistics
                    # ----------------------------------------

                    "mos_mean":
                        mos_mean.cpu(),

                    "mos_std":
                        mos_std.cpu(),

                    # ----------------------------------------
                    # Training configuration
                    # ----------------------------------------

                    "args":
                        vars(args),
                },
                checkpoint_path,
            )

            print(
                f"  -> Best checkpoint saved "
                f"(SRCC={srcc:.6f})"
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/"
                f"{args.patience})"
            )

            if (
                patience_counter
                >= args.patience
            ):

                print(
                    "\nEarly stopping."
                )

                break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best SRCC: {best_srcc:.6f}"
    )

    print(
        f"Best PLCC: {best_plcc:.6f}"
    )

    print(
        f"Best MSE:  {best_mse:.6f}"
    )

    print(
        f"MOS mean:  {mos_mean.item():.6f}"
    )

    print(
        f"MOS std:   {mos_std.item():.6f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 Advanced "
            "Transformer Attention Training"
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
    # TRAINING
    # ========================================================

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    # ========================================================
    # ADVANCED MODEL
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
    # OUTPUT
    # ========================================================

    parser.add_argument(
        "--output-dir",
        default="results/phase3/advanced",
    )

    args = parser.parse_args()

    train(args)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()