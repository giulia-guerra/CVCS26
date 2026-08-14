# carica un modello MLP già addestrato (checkpoint), 
# ricostruisce automaticamente il dataset e il layer utilizzati durante il training, 
# rigenera lo stesso validation split tramite il seed salvato nel checkpoint 
# e valuta le prestazioni del modello calcolando MSE, SRCC e PLCC.


import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from src.phase3.dataset import FeatureDataset
from src.phase3.regressor import IQARegressor
from src.phase3.metrics import srcc, plcc


def make_split(n, val_ratio, seed):

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        n,
        generator=generator
    ).tolist()

    val_size = int(n * val_ratio)

    return (
        indices[val_size:],
        indices[:val_size]
    )


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a Phase 3 IQA MLP checkpoint"
    )

    parser.add_argument(
        "--features",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("PHASE 3 - EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False
    )

    layer = checkpoint["layer"]
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    dropout = checkpoint["dropout"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Layer:      {layer}")
    print(f"Input dim:  {input_dim}")
    print(f"Device:     {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = FeatureDataset(
        pt_path=args.features,
        layer=layer
    )

    if dataset.features.shape[1] != input_dim:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"checkpoint={input_dim}, "
            f"dataset={dataset.features.shape[1]}"
        )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    if "val_indices" in checkpoint:

        val_indices = checkpoint["val_indices"]

        print("Validation split: loaded from checkpoint")

    else:

        seed = checkpoint["seed"]
        val_ratio = checkpoint.get(
            "val_ratio",
            0.2
        )

        _, val_indices = make_split(
            len(dataset),
            val_ratio,
            seed
        )

        print(
            "Validation split: "
            "reconstructed from seed"
        )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"Samples:    {len(dataset)}")
    print(f"Validation: {len(val_dataset)}")

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    required_stats = [
        "feature_mean",
        "feature_std",
        "mos_mean",
        "mos_std",
    ]

    for key in required_stats:

        if key not in checkpoint:
            raise KeyError(
                f"Missing '{key}' in checkpoint."
            )

    feature_mean = checkpoint["feature_mean"].to(device)
    feature_std = checkpoint["feature_std"].to(device)
    mos_mean = checkpoint["mos_mean"].to(device)
    mos_std = checkpoint["mos_std"].to(device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = IQARegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in val_loader:

            x = batch["features"].to(device)
            y = batch["mos"].to(device)

            # Same normalization used during training
            x = (
                x - feature_mean
            ) / feature_std

            # Model predicts normalized MOS
            pred_norm = model(x)

            # Convert prediction back to original MOS scale
            pred = (
                pred_norm * mos_std
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mse = np.mean(
        (predictions - targets) ** 2
    )

    srcc_value = srcc(
        predictions,
        targets
    )

    plcc_value = plcc(
        predictions,
        targets
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("RESULTS")
    print("-" * 60)
    print(f"MSE :  {mse:.6f}")
    print(f"SRCC:  {srcc_value:.6f}")
    print(f"PLCC:  {plcc_value:.6f}")
    print("-" * 60)

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


if __name__ == "__main__":
    main()