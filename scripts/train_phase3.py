# Implementa il training supervisionato dell'MLP baseline per l'Image Quality Assessment. 
# Il codice carica le feature estratte dagli encoder frozen, 
# divide il dataset in training e validation set, normalizza feature e MOS, 
# addestra un regressore MLP usando la loss MSE, monitora le metriche SRCC e PLCC sul validation set, 
# salva automaticamente il miglior checkpoint e interrompe il training tramite early stopping 
# quando le prestazioni non migliorano più.

# ============================================================
# PHASE 3 - DUAL ENCODER SUPERVISED TRAINING
# ============================================================

# Implements supervised training of the DualEncoderFusion
# MLP baseline for Image Quality Assessment.

# The code:
#   1. loads frozen encoder features;
#   2. creates a reproducible train/validation split;
#   3. computes normalization statistics ONLY on the training set;
#   4. normalizes features and MOS;
#   5. trains DualEncoderFusion using MSE loss;
#   6. evaluates SRCC and PLCC on the validation set;
#   7. saves the best checkpoint according to SRCC;
#   8. saves the training history after EVERY epoch;
#   9. uses early stopping based on validation SRCC.

# The CSV history is written incrementally so that it is not
# lost if the training process is interrupted or killed.
# ============================================================

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
from src.phase3.aggregation import DualEncoderFusion
from src.phase3.metrics import srcc, plcc


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
# TRAIN / VALIDATION SPLIT
# ============================================================

def train_val_split(n, val_ratio, seed):
    """
    Create a reproducible train/validation split.
    """

    if not 0 < val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        n,
        generator=generator,
    ).tolist()

    val_size = int(n * val_ratio)

    if val_size == 0 or val_size == n:
        raise ValueError(
            f"Invalid validation size: {val_size} "
            f"for dataset of size {n}."
        )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

def compute_feature_stats(
    dataset,
    train_indices,
):
    """
    Compute feature normalization statistics ONLY
    on the training samples.

    Four independent statistics are computed:

        ref_base
        dist_base
        ref_large
        dist_large
    """

    train_indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Extract training samples
    # --------------------------------------------------------

    ref_base = dataset.ref_base[
        :, train_indices, :
    ]

    dist_base = dataset.dist_base[
        :, train_indices, :
    ]

    ref_large = dataset.ref_large[
        :, train_indices, :
    ]

    dist_large = dataset.dist_large[
        :, train_indices, :
    ]

    # --------------------------------------------------------
    # Compute mean/std per layer and feature dimension
    # --------------------------------------------------------

    ref_base_mean = ref_base.mean(dim=1)
    ref_base_std = ref_base.std(dim=1)

    dist_base_mean = dist_base.mean(dim=1)
    dist_base_std = dist_base.std(dim=1)

    ref_large_mean = ref_large.mean(dim=1)
    ref_large_std = ref_large.std(dim=1)

    dist_large_mean = dist_large.mean(dim=1)
    dist_large_std = dist_large.std(dim=1)

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    ref_base_std[
        ref_base_std < 1e-8
    ] = 1.0

    dist_base_std[
        dist_base_std < 1e-8
    ] = 1.0

    ref_large_std[
        ref_large_std < 1e-8
    ] = 1.0

    dist_large_std[
        dist_large_std < 1e-8
    ] = 1.0

    return {
        "ref_base": {
            "mean": ref_base_mean,
            "std": ref_base_std,
        },
        "dist_base": {
            "mean": dist_base_mean,
            "std": dist_base_std,
        },
        "ref_large": {
            "mean": ref_large_mean,
            "std": ref_large_std,
        },
        "dist_large": {
            "mean": dist_large_mean,
            "std": dist_large_std,
        },
    }


# ============================================================
# MOS NORMALIZATION
# ============================================================

def compute_mos_stats(
    dataset,
    train_indices,
):
    """
    Compute MOS normalization statistics ONLY
    on training samples.
    """

    train_indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    mos = dataset.mos[
        train_indices
    ]

    mos_mean = mos.mean()
    mos_std = mos.std()

    if mos_std < 1e-8:
        raise ValueError(
            "MOS standard deviation is too small."
        )

    return mos_mean, mos_std


# ============================================================
# NORMALIZED DATASET
# ============================================================

class NormalizedDataset(Dataset):
    """
    Dataset wrapper that:

        1. selects train/validation indices;
        2. normalizes the four feature tensors;
        3. normalizes MOS.

    Statistics always come from the training set.
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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        ref_base = batch["ref_base"].to(
            device,
            non_blocking=True,
        )

        dist_base = batch["dist_base"].to(
            device,
            non_blocking=True,
        )

        ref_large = batch["ref_large"].to(
            device,
            non_blocking=True,
        )

        dist_large = batch["dist_large"].to(
            device,
            non_blocking=True,
        )

        y = batch["mos"].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions = model(
            ref_base,
            dist_base,
            ref_large,
            dist_large,
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            predictions,
            y,
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        batch_size = y.size(0)

        total_loss += (
            loss.item()
            * batch_size
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
    """
    Evaluate the model on the validation set.

    Predictions and targets are converted back to
    the original MOS scale before computing
    MSE, SRCC and PLCC.
    """

    model.eval()

    predictions = []
    targets = []

    mos_mean = mos_mean.to(device)
    mos_std = mos_std.to(device)

    for batch in loader:

        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        ref_base = batch["ref_base"].to(
            device,
            non_blocking=True,
        )

        dist_base = batch["dist_base"].to(
            device,
            non_blocking=True,
        )

        ref_large = batch["ref_large"].to(
            device,
            non_blocking=True,
        )

        dist_large = batch["dist_large"].to(
            device,
            non_blocking=True,
        )

        y_norm = batch["mos"].to(
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
            predictions - targets
        ) ** 2
    )

    correlation_srcc = srcc(
        predictions,
        targets,
    )

    correlation_plcc = plcc(
        predictions,
        targets,
    )

    return (
        mse,
        correlation_srcc,
        correlation_plcc,
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(
    history,
    path,
):
    """
    Save the complete training history to CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
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
# APPEND ONE EPOCH TO CSV
# ============================================================

def append_history_row(
    row,
    path,
):
    """
    Append one epoch to the CSV immediately.

    This guarantees that the history is preserved even
    if the process is interrupted before training finishes.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = path.exists()

    with open(
        path,
        "a",
        newline="",
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

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

        # Force the data to disk immediately.
        f.flush()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 supervised IQA training "
            "with DualEncoderFusion."
        )
    )

    # ========================================================
    # DATASET
    # ========================================================

    parser.add_argument(
        "--features-base",
        required=True,
        help=(
            "Path to SigLIP2 Base "
            "all-layers .pt file."
        ),
    )

    parser.add_argument(
        "--features-large",
        required=True,
        help=(
            "Path to SigLIP2 Large "
            "all-layers .pt file."
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    parser.add_argument(
        "--variant",
        type=str,
        default="medium",
        choices=[
            "small",
            "medium",
            "large",
        ],
        help=(
            "MLP variant used by DualEncoderFusion."
        ),
    )

    # ========================================================
    # TRAINING
    # ========================================================

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

    # ========================================================
    # VALIDATION
    # ========================================================

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help=(
            "Early stopping patience based on "
            "validation SRCC."
        ),
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

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

    print("=" * 70)
    print("PHASE 3 - DUAL ENCODER SUPERVISED TRAINING")
    print("=" * 70)

    print(
        f"Features Base:  {args.features_base}"
    )

    print(
        f"Features Large: {args.features_large}"
    )

    print(
        f"Variant:        {args.variant}"
    )

    print(
        f"Device:         {device}"
    )

    print(
        f"Epochs:         {args.epochs}"
    )

    print(
        f"Batch size:     {args.batch_size}"
    )

    print(
        f"Learning rate:  {args.lr}"
    )

    print(
        f"Weight decay:   {args.weight_decay}"
    )

    print(
        f"Validation:     {args.val_ratio}"
    )

    print(
        f"Patience:       {args.patience}"
    )

    print(
        f"Seed:            {args.seed}"
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / (
            f"best_dual_siglip2_"
            f"{args.variant}.pt"
        )
    )

    history_path = (
        output_dir
        / (
            f"history_dual_siglip2_"
            f"{args.variant}.csv"
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Create the CSV immediately.
    #
    # If an old CSV exists, remove it because this is a new
    # training run.
    # --------------------------------------------------------

    if history_path.exists():
        history_path.unlink()

    # Create CSV header immediately.
    with open(
        history_path,
        "w",
        newline="",
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

    print(
        f"\nTraining history will be saved to:"
        f"\n  {history_path}"
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
        f"  Samples:       {len(dataset)}"
    )

    print(
        f"  Base shape:    {dataset.ref_base.shape}"
    )

    print(
        f"  Large shape:   {dataset.ref_large.shape}"
    )

    print(
        f"  Base dim:      {dataset.dim_base}"
    )

    print(
        f"  Large dim:     {dataset.dim_large}"
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

    print(
        "\nComputing normalization statistics "
        "from TRAINING SET ONLY..."
    )

    feature_stats = compute_feature_stats(
        dataset,
        train_idx,
    )

    mos_mean, mos_std = compute_mos_stats(
        dataset,
        train_idx,
    )

    print("\nMOS normalization:")

    print(
        f"  Mean: {mos_mean.item():.6f}"
    )

    print(
        f"  Std:  {mos_std.item():.6f}"
    )

    print("\nFeature normalization:")

    print(
        "  ref_base  mean/std:",
        feature_stats["ref_base"]["mean"].mean().item(),
        "/",
        feature_stats["ref_base"]["std"].mean().item(),
    )

    print(
        "  dist_base mean/std:",
        feature_stats["dist_base"]["mean"].mean().item(),
        "/",
        feature_stats["dist_base"]["std"].mean().item(),
    )

    print(
        "  ref_large mean/std:",
        feature_stats["ref_large"]["mean"].mean().item(),
        "/",
        feature_stats["ref_large"]["std"].mean().item(),
    )

    print(
        "  dist_large mean/std:",
        feature_stats["dist_large"]["mean"].mean().item(),
        "/",
        feature_stats["dist_large"]["std"].mean().item(),
    )

    # ========================================================
    # NORMALIZED DATASETS
    # ========================================================

    train_dataset = NormalizedDataset(
        dataset=dataset,
        indices=train_idx,
        feature_stats=feature_stats,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )

    val_dataset = NormalizedDataset(
        dataset=dataset,
        indices=val_idx,
        feature_stats=feature_stats,
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

    dim_base = dataset.dim_base
    dim_large = dataset.dim_large

    model = DualEncoderFusion(
        dim_base=dim_base,
        dim_large=dim_large,
        variant=args.variant,
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
    # TRAINING LOOP
    # ========================================================

    best_srcc = -float("inf")
    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # VALIDATION
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
        # HISTORY ROW
        # ----------------------------------------------------

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_srcc": val_srcc,
            "val_plcc": val_plcc,
        }

        history.append(
            history_row
        )

        # ----------------------------------------------------
        # SAVE CSV IMMEDIATELY
        # ----------------------------------------------------

        append_history_row(
            history_row,
            history_path,
        )

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"SRCC: {val_srcc:.6f} | "
            f"PLCC: {val_plcc:.6f}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        #
        # SRCC is the primary IQA metric.
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
                    # ------------------------------------------------
                    # Model identification
                    # ------------------------------------------------

                    "model_type":
                        "DualEncoderFusion",

                    "variant":
                        args.variant,

                    "dim_base":
                        dim_base,

                    "dim_large":
                        dim_large,

                    "base_shape":
                        tuple(
                            dataset.ref_base.shape
                        ),

                    "large_shape":
                        tuple(
                            dataset.ref_large.shape
                        ),

                    # ------------------------------------------------
                    # Training state
                    # ------------------------------------------------

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    # ------------------------------------------------
                    # Performance
                    # ------------------------------------------------

                    "best_srcc":
                        val_srcc,

                    "best_plcc":
                        val_plcc,

                    "val_mse":
                        val_mse,

                    # ------------------------------------------------
                    # Reproducibility
                    # ------------------------------------------------

                    "seed":
                        args.seed,

                    "val_ratio":
                        args.val_ratio,

                    "train_indices":
                        train_idx,

                    "val_indices":
                        val_idx,

                    # ------------------------------------------------
                    # Normalization
                    # ------------------------------------------------

                    "normalization_stats":
                        feature_stats,

                    "mos_mean":
                        mos_mean,

                    "mos_std":
                        mos_std,

                    # ------------------------------------------------
                    # Dataset information
                    # ------------------------------------------------

                    "features_base":
                        str(
                            args.features_base
                        ),

                    "features_large":
                        str(
                            args.features_large
                        ),

                    "model_config_base":
                        dataset.model_config_base,

                    "model_config_large":
                        dataset.model_config_large,

                    # ------------------------------------------------
                    # Hyperparameters
                    # ------------------------------------------------

                    "learning_rate":
                        args.lr,

                    "weight_decay":
                        args.weight_decay,

                    "batch_size":
                        args.batch_size,

                    "patience":
                        args.patience,
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
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch} "
                f"(no SRCC improvement for "
                f"{args.patience} epochs)."
            )

            break

    # ========================================================
    # FINAL HISTORY SAVE
    #
    # This is kept as an extra safety measure.
    # ========================================================

    save_history(
        history,
        history_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best epoch:    {best_epoch}"
    )

    print(
        f"Best Val SRCC: {best_srcc:.6f}"
    )

    print(
        f"Checkpoint:    {checkpoint_path}"
    )

    print(
        f"History:       {history_path}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()