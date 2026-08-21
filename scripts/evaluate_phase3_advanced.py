import argparse

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, random_split

from src.phase3.advanced_dataset import AdvancedFeatureDataset
from src.phase3.aggregation import AdvancedAttentionAggregator


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader, device):

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

    # ========================================================
    # CONCATENATE RESULTS
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
    # FEATURES
    # ========================================================

    parser.add_argument(
        "--features",
        required=True,
        help=(
            "Path to the single .pt file containing "
            "ref_base, dist_base, ref_large, "
            "dist_large and mos."
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

    print("\n" + "=" * 60)
    print("PHASE 3 - ADVANCED ATTENTION EVALUATION")
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

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    print("\nValidation split:")
    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # ========================================================
    # VALIDATION DATALOADER
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

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint: {args.checkpoint}"
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

    print("\nRunning evaluation...")

    mse, srcc, plcc = evaluate(
        model,
        val_loader,
        device,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"MSE:  {mse:.6f}"
    )

    print(
        f"SRCC: {srcc:.6f}"
    )

    print(
        f"PLCC: {plcc:.6f}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()