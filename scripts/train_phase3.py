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
# TRAIN / VALIDATION SPLIT
# ============================================================

def train_val_split(n, val_ratio, seed):

    if not 0 < val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        n,
        generator=generator
    ).tolist()

    val_size = int(n * val_ratio)

    if val_size == 0 or val_size == n:
        raise ValueError(
            f"Invalid validation size: {val_size} "
            f"for dataset of size {n}."
        )

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    return train_indices, val_indices


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

        # Normalize using TRAINING statistics
        x = (
            x - self.feature_mean
        ) / self.feature_std

        y = (
            y - self.mos_mean
        ) / self.mos_std

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

    # Compute statistics ONLY on training samples
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0)

    # Avoid division by zero for constant features
    feature_std[feature_std < 1e-8] = 1.0

    mos_mean = y.mean()
    mos_std = y.std()

    if mos_std < 1e-8:
        raise ValueError(
            "MOS standard deviation is too small."
        )

    return (
        feature_mean,
        feature_std,
        mos_mean,
        mos_std,
    )


# ============================================================
# TRAIN ONE EPOCH
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
    total_samples = 0

    for batch in loader:

        x = batch["features"].to(device)
        y = batch["mos"].to(device)

        optimizer.zero_grad()

        predictions = model(x)

        loss = criterion(
            predictions,
            y
        )

        loss.backward()

        optimizer.step()

        batch_size = y.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Training loader is empty."
        )

    return total_loss / total_samples


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

        # Model predicts normalized MOS
        pred_norm = model(x)

        # Convert predictions back to original MOS scale
        pred = (
            pred_norm * mos_std.to(device)
            + mos_mean.to(device)
        )

        # Convert ground truth back to original MOS scale
        y = (
            y_norm * mos_std.to(device)
            + mos_mean.to(device)
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        targets.extend(
            y.cpu().numpy()
        )

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if len(predictions) == 0:
        raise RuntimeError(
            "Validation loader is empty."
        )

    # MSE on original MOS scale
    mse = np.mean(
        (predictions - targets) ** 2
    )

    correlation_srcc = srcc(
        predictions,
        targets
    )

    correlation_plcc = plcc(
        predictions,
        targets
    )

    return (
        mse,
        correlation_srcc,
        correlation_plcc,
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

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
        description=(
            "Phase 3 supervised IQA training "
            "with an MLP regressor."
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--features",
        required=True,
        help="Path to the .pt feature file."
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Feature layer used for training."
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience."
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default="results/phase3",
    )

    args = parser.parse_args()

    # ========================================================
    # SETUP
    # ========================================================

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
    print(f"Seed:            {args.seed}")

    # ========================================================
    # DATASET
    # ========================================================

    dataset = FeatureDataset(
        pt_path=args.features,
        layer=args.layer,
    )

    print("\nDataset:")
    print(
        f"  Model:       "
        f"{dataset.model_config}"
    )
    print(
        f"  Samples:     "
        f"{len(dataset)}"
    )
    print(
        f"  Feature dim: "
        f"{dataset.features.shape[1]}"
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    train_idx, val_idx = train_val_split(
        len(dataset),
        args.val_ratio,
        args.seed,
    )

    print("\nSplit:")
    print(
        f"  Train: {len(train_idx)}"
    )
    print(
        f"  Val:   {len(val_idx)}"
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

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
    print(
        f"  MOS mean: "
        f"{mos_mean.item():.4f}"
    )
    print(
        f"  MOS std:  "
        f"{mos_std.item():.4f}"
    )

    # ========================================================
    # NORMALIZED DATASETS
    # ========================================================

    train_dataset = NormalizedDataset(
        dataset=dataset,
        indices=train_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )

    val_dataset = NormalizedDataset(
        dataset=dataset,
        indices=val_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    # ========================================================
    # MODEL
    # ========================================================

    input_dim = dataset.features.shape[1]

    model = IQARegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print("\nModel:")
    print(model)

    # ========================================================
    # LOSS / OPTIMIZER
    # ========================================================

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        output_dir
        / (
            f"best_"
            f"{dataset.model_config}"
            f"_layer{args.layer}.pt"
        )
    )

    history_path = (
        output_dir
        / (
            f"history_"
            f"{dataset.model_config}"
            f"_layer{args.layer}.csv"
        )
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    best_srcc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_mse,
            val_srcc,
            val_plcc,
        ) = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mos_mean=mos_mean,
            mos_std=mos_std,
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_srcc": val_srcc,
            "val_plcc": val_plcc,
        })

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"SRCC: {val_srcc:.6f} | "
            f"PLCC: {val_plcc:.6f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            not np.isnan(val_srcc)
            and val_srcc > best_srcc
        ):

            best_srcc = val_srcc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    # Training state
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    # Dataset / model
                    "model_config":
                        dataset.model_config,
                    "layer":
                        args.layer,
                    "input_dim":
                        input_dim,
                    "hidden_dim":
                        args.hidden_dim,
                    "dropout":
                        args.dropout,

                    # Performance
                    "best_srcc":
                        val_srcc,
                    "best_plcc":
                        val_plcc,
                    "val_mse":
                        val_mse,

                    # Reproducibility
                    "seed":
                        args.seed,
                    "val_ratio":
                        args.val_ratio,

                    # EXACT split used
                    "train_indices":
                        train_idx,
                    "val_indices":
                        val_idx,

                    # Normalization statistics
                    "feature_mean":
                        feature_mean,
                    "feature_std":
                        feature_std,
                    "mos_mean":
                        mos_mean,
                    "mos_std":
                        mos_std,
                },
                checkpoint_path,
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

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch} "
                f"(no improvement for "
                f"{args.patience} epochs)."
            )

            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    save_history(
        history,
        history_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)

    print(
        f"Best epoch:    "
        f"{best_epoch}"
    )

    print(
        f"Best Val SRCC: "
        f"{best_srcc:.6f}"
    )

    print(
        f"Checkpoint:    "
        f"{checkpoint_path}"
    )

    print(
        f"History:       "
        f"{history_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()