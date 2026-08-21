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
):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            ref_base = batch["ref_base"].to(device)
            dist_base = batch["dist_base"].to(device)

            ref_large = batch["ref_large"].to(device)
            dist_large = batch["dist_large"].to(device)

            mos = batch["mos"].to(device)

            prediction = model(
                ref_base,
                dist_base,
                ref_large,
                dist_large,
            )

            predictions.append(
                prediction.cpu()
            )

            targets.append(
                mos.cpu()
            )

    predictions = torch.cat(
        predictions
    ).numpy()

    targets = torch.cat(
        targets
    ).numpy()

    mse = np.mean(
        (predictions - targets) ** 2
    )

    srcc = spearmanr(
        predictions,
        targets,
    ).statistic

    plcc = pearsonr(
        predictions,
        targets,
    ).statistic

    return mse, srcc, plcc


# ============================================================
# TRAIN
# ============================================================

def train(args):

    set_seed(args.seed)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 60)
    print("PHASE 3 - ADVANCED TRAINING")
    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = AdvancedFeatureDataset(
        feature_file=args.features,
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    val_size = int(
        len(dataset) * args.val_ratio
    )

    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            args.seed
        ),
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

    print("\nDataset split:")
    print(
        f"Train samples: {len(train_dataset)}"
    )
    print(
        f"Val samples:   {len(val_dataset)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

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

            ref_base = batch["ref_base"].to(device)
            dist_base = batch["dist_base"].to(device)

            ref_large = batch["ref_large"].to(device)
            dist_large = batch["dist_large"].to(device)

            mos = batch["mos"].to(device)

            optimizer.zero_grad()

            prediction = model(
                ref_base,
                dist_base,
                ref_large,
                dist_large,
            )

            loss = criterion(
                prediction,
                mos,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * mos.size(0)
            )

        train_loss = (
            total_loss
            / len(train_dataset)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_mse, srcc, plcc = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"SRCC: {srcc:.6f} | "
            f"PLCC: {plcc:.6f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

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
                    "best_srcc": srcc,
                    "best_plcc": plcc,
                    "best_mse": val_mse,
                    "args": vars(args),
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

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best SRCC: {best_srcc:.6f}"
    )

    print(
        f"Best PLCC: {best_plcc:.6f}"
    )

    print(
        f"Best MSE:  {best_mse:.4f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print("=" * 60)


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
    # FEATURES
    # ========================================================

    parser.add_argument(
        "--features",
        required=True,
        help=(
            "Path to the single .pt file "
            "containing ref_base, dist_base, "
            "ref_large, dist_large and mos."
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


if __name__ == "__main__":
    main()