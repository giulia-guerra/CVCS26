# Implementa il training supervisionato dell'MLP baseline per l'Image Quality Assessment. 
# Il codice carica le feature estratte dagli encoder frozen, 
# divide il dataset in training e validation set, normalizza feature e MOS, 
# addestra un regressore MLP usando la loss MSE, monitora le metriche SRCC e PLCC sul validation set, 
# salva automaticamente il miglior checkpoint e interrompe il training tramite early stopping 
# quando le prestazioni non migliorano più.


import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.phase3.dataset import FeatureDataset
from src.phase3.regressor import IQARegressor
from src.phase3.metrics import srcc, plcc


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
# SPLIT
# ============================================================

def train_val_split(n, val_ratio, seed):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()

    val_size = int(n * val_ratio)

    return indices[val_size:], indices[:val_size]


# ============================================================
# NORMALIZED DATASET
# ============================================================

class NormalizedDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        feature_mean,
        feature_std,
        mos_mean,
        mos_std,
    ):
        self.dataset = dataset
        self.indices = indices

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.mos_mean = mos_mean
        self.mos_std = mos_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = self.dataset.features[idx]
        y = self.dataset.mos[idx]

        x = (x - self.feature_mean) / self.feature_std
        y = (y - self.mos_mean) / self.mos_std

        return {
            "features": x,
            "mos": y,
        }


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

def get_stats(dataset, train_indices):

    x = dataset.features[train_indices]
    y = dataset.mos[train_indices]

    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0)

    # Avoid division by zero
    feature_std[feature_std < 1e-8] = 1.0

    mos_mean = y.mean()
    mos_std = y.std()

    if mos_std < 1e-8:
        raise ValueError("MOS standard deviation is too small.")

    return (
        feature_mean,
        feature_std,
        mos_mean,
        mos_std,
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in loader:

        x = batch["features"].to(device)
        y = batch["mos"].to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        n = y.size(0)

        total_loss += loss.item() * n
        total_n += n

    return total_loss / total_n


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    mos_mean,
    mos_std,
):

    model.eval()

    predictions = []
    targets = []

    for batch in loader:

        x = batch["features"].to(device)
        y_norm = batch["mos"].to(device)

        pred_norm = model(x)

        # Back to original MOS scale
        pred = pred_norm * mos_std.to(device) + mos_mean.to(device)
        y = y_norm * mos_std.to(device) + mos_mean.to(device)

        predictions.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    mse = np.mean((predictions - targets) ** 2)
    correlation_srcc = srcc(predictions, targets)
    correlation_plcc = plcc(predictions, targets)

    return mse, correlation_srcc, correlation_plcc


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_mse",
                "val_srcc",
                "val_plcc",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Phase 3 supervised IQA training"
    )

    parser.add_argument(
        "--features",
        required=True,
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience",
    )

    parser.add_argument(
        "--output-dir",
        default="results/phase3",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("PHASE 3 - SUPERVISED IQA TRAINING")
    print("=" * 65)

    print(f"Features:       {args.features}")
    print(f"Layer:          {args.layer}")
    print(f"Device:         {device}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Hidden dim:     {args.hidden_dim}")
    print(f"Dropout:        {args.dropout}")
    print(f"Val ratio:      {args.val_ratio}")
    print(f"Patience:       {args.patience}")
    print(f"Seed:           {args.seed}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = FeatureDataset(
        pt_path=args.features,
        layer=args.layer,
    )

    print("\nDataset:")
    print(f"  Model:       {dataset.model_config}")
    print(f"  Samples:     {len(dataset)}")
    print(f"  Feature dim: {dataset.features.shape[1]}")

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_idx, val_idx = train_val_split(
        len(dataset),
        args.val_ratio,
        args.seed,
    )

    print("\nSplit:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    (
        feature_mean,
        feature_std,
        mos_mean,
        mos_std,
    ) = get_stats(
        dataset,
        train_idx,
    )

    print("\nNormalization:")
    print(f"  MOS mean: {mos_mean.item():.4f}")
    print(f"  MOS std:  {mos_std.item():.4f}")

    train_dataset = NormalizedDataset(
        dataset,
        train_idx,
        feature_mean,
        feature_std,
        mos_mean,
        mos_std,
    )

    val_dataset = NormalizedDataset(
        dataset,
        val_idx,
        feature_mean,
        feature_std,
        mos_mean,
        mos_std,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_dim = dataset.features.shape[1]

    model = IQARegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss / Optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        output_dir
        / f"best_{dataset.model_config}_layer{args.layer}.pt"
    )

    history_file = (
        output_dir
        / f"history_{dataset.model_config}_layer{args.layer}.csv"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_srcc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_mse, val_srcc, val_plcc = evaluate(
            model,
            val_loader,
            device,
            mos_mean,
            mos_std,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_srcc": val_srcc,
            "val_plcc": val_plcc,
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"SRCC: {val_srcc:.6f} | "
            f"PLCC: {val_plcc:.6f}"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if not np.isnan(val_srcc) and val_srcc > best_srcc:

            best_srcc = val_srcc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),

                    "model_config": dataset.model_config,
                    "layer": args.layer,

                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,

                    "best_srcc": val_srcc,
                    "best_plcc": val_plcc,
                    "val_mse": val_mse,

                    "seed": args.seed,

                    # Normalization
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "mos_mean": mos_mean,
                    "mos_std": mos_std,
                },
                checkpoint,
            )

            print(
                f"  -> Best checkpoint saved "
                f"(SRCC={val_srcc:.6f})"
            )

        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= args.patience:

            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for "
                f"{args.patience} epochs)."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(
        history,
        history_file,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)

    print(f"Best epoch:    {best_epoch}")
    print(f"Best Val SRCC: {best_srcc:.6f}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"History:       {history_file}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()