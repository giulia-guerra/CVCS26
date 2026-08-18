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

from src.phase3.aggregation import DualEncoderFusion
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

        print("\nLoading LIVE Base...")
        live_base_data = load_feature_file(
            live_base
        )

        print("Loading LIVE Large...")
        live_large_data = load_feature_file(
            live_large
        )

        print("Loading TID2013 Base...")
        tid_base_data = load_feature_file(
            tid_base
        )

        print("Loading TID2013 Large...")
        tid_large_data = load_feature_file(
            tid_large
        )

        # ----------------------------------------------------
        # Check dataset compatibility
        # ----------------------------------------------------

        self._check_compatibility(
            live_base_data,
            live_large_data,
            "LIVE",
        )

        self._check_compatibility(
            tid_base_data,
            tid_large_data,
            "TID2013",
        )

        # ----------------------------------------------------
        # Check feature layer dimensions
        # ----------------------------------------------------

        if (
            live_base_data["ref_features"].shape[0]
            != tid_base_data["ref_features"].shape[0]
        ):
            raise ValueError(
                "LIVE and TID2013 Base have "
                "different numbers of layers."
            )

        if (
            live_large_data["ref_features"].shape[0]
            != tid_large_data["ref_features"].shape[0]
        ):
            raise ValueError(
                "LIVE and TID2013 Large have "
                "different numbers of layers."
            )

        if (
            live_base_data["ref_features"].shape[2]
            != tid_base_data["ref_features"].shape[2]
        ):
            raise ValueError(
                "LIVE and TID2013 Base have "
                "different feature dimensions."
            )

        if (
            live_large_data["ref_features"].shape[2]
            != tid_large_data["ref_features"].shape[2]
        ):
            raise ValueError(
                "LIVE and TID2013 Large have "
                "different feature dimensions."
            )

        # ----------------------------------------------------
        # Concatenate features
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

        # ----------------------------------------------------
        # Names
        # ----------------------------------------------------

        self.image_names = (
            [
                f"LIVE/{name}"
                for name in live_base_data["image_names"]
            ]
            +
            [
                f"TID2013/{name}"
                for name in tid_base_data["image_names"]
            ]
        )

        # ----------------------------------------------------
        # Dataset labels
        # ----------------------------------------------------

        self.dataset_names = (
            ["LIVE"] *
            len(live_base_data["mos"])
            +
            ["TID2013"] *
            len(tid_base_data["mos"])
        )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

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

        self.n_live = len(
            live_base_data["mos"]
        )

        self.n_tid = len(
            tid_base_data["mos"]
        )

        print("\nMixtureFeatureDataset initialized:")
        print(
            f"  LIVE samples:     {self.n_live}"
        )
        print(
            f"  TID2013 samples:  {self.n_tid}"
        )
        print(
            f"  Total samples:    {len(self)}"
        )
        print(
            f"  Base shape:       {self.ref_base.shape}"
        )
        print(
            f"  Large shape:      {self.ref_large.shape}"
        )
        print(
            f"  Base dim:         {self.dim_base}"
        )
        print(
            f"  Large dim:        {self.dim_large}"
        )

    @staticmethod
    def _check_compatibility(
        base_data,
        large_data,
        dataset_name,
    ):

        if (
            base_data["ref_features"].shape[1]
            != large_data["ref_features"].shape[1]
        ):
            raise ValueError(
                f"{dataset_name}: "
                "Base and Large sample counts differ."
            )

        if (
            base_data["mos"].shape
            != large_data["mos"].shape
        ):
            raise ValueError(
                f"{dataset_name}: "
                "MOS shapes differ."
            )

        if not torch.allclose(
            base_data["mos"].float(),
            large_data["mos"].float(),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                f"{dataset_name}: "
                "MOS values differ between Base and Large."
            )

        if (
            len(base_data["image_names"])
            != len(large_data["image_names"])
        ):
            raise ValueError(
                f"{dataset_name}: "
                "image_names lengths differ."
            )

    def __len__(self):
        return self.mos.shape[0]

    def __getitem__(self, idx):

        return {
            "ref_base": self.ref_base[:, idx, :],
            "dist_base": self.dist_base[:, idx, :],
            "ref_large": self.ref_large[:, idx, :],
            "dist_large": self.dist_large[:, idx, :],
            "mos": self.mos[idx],
            "name": self.image_names[idx],
            "dataset": self.dataset_names[idx],
        }


# ============================================================
# STRATIFIED DATASET SPLIT
# ============================================================

def mixture_train_val_split(
    dataset,
    val_ratio,
    seed,
):

    if not 0 < val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    live_indices = [
        i
        for i, name in enumerate(
            dataset.dataset_names
        )
        if name == "LIVE"
    ]

    tid_indices = [
        i
        for i, name in enumerate(
            dataset.dataset_names
        )
        if name == "TID2013"
    ]

    def split_indices(indices):

        indices = torch.tensor(
            indices,
            dtype=torch.long,
        )

        permutation = torch.randperm(
            len(indices),
            generator=generator,
        )

        indices = indices[permutation]

        val_size = max(
            1,
            int(len(indices) * val_ratio)
        )

        val_idx = indices[
            :val_size
        ].tolist()

        train_idx = indices[
            val_size:
        ].tolist()

        return train_idx, val_idx

    live_train, live_val = split_indices(
        live_indices
    )

    tid_train, tid_val = split_indices(
        tid_indices
    )

    train_indices = (
        live_train +
        tid_train
    )

    val_indices = (
        live_val +
        tid_val
    )

    return (
        train_indices,
        val_indices,
        live_train,
        live_val,
        tid_train,
        tid_val,
    )


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def compute_feature_stats(
    dataset,
    train_indices,
):

    indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    stats = {}

    tensors = {
        "ref_base": dataset.ref_base,
        "dist_base": dataset.dist_base,
        "ref_large": dataset.ref_large,
        "dist_large": dataset.dist_large,
    }

    for name, tensor in tensors.items():

        train_tensor = tensor[
            :,
            indices,
            :,
        ]

        mean = train_tensor.mean(
            dim=1
        )

        std = train_tensor.std(
            dim=1,
            unbiased=False,
        )

        std = torch.where(
            std < 1e-8,
            torch.ones_like(std),
            std,
        )

        stats[name] = {
            "mean": mean,
            "std": std,
        }

    return stats


# ============================================================
# DATASET-SPECIFIC MOS NORMALIZATION
# ============================================================

def compute_dataset_mos_stats(
    dataset,
    train_indices,
):

    stats = {}

    train_indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    for dataset_name in [
        "LIVE",
        "TID2013",
    ]:

        selected = [
            idx
            for idx in train_indices.tolist()
            if dataset.dataset_names[idx]
            == dataset_name
        ]

        selected = torch.as_tensor(
            selected,
            dtype=torch.long,
        )

        mos = dataset.mos[selected]

        mean = mos.mean()
        std = mos.std(
            unbiased=False
        )

        if std < 1e-8:
            raise ValueError(
                f"{dataset_name}: "
                "MOS standard deviation is too small."
            )

        stats[dataset_name] = {
            "mean": mean,
            "std": std,
        }

    return stats


# ============================================================
# NORMALIZED DATASET
# ============================================================

class NormalizedMixtureDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        feature_stats,
        mos_stats,
    ):

        self.dataset = dataset

        self.indices = torch.as_tensor(
            indices,
            dtype=torch.long,
        )

        self.feature_stats = feature_stats
        self.mos_stats = mos_stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]

        dataset_name = (
            self.dataset.dataset_names[idx]
        )

        ref_base = (
            self.dataset.ref_base[:, idx, :]
            -
            self.feature_stats[
                "ref_base"
            ]["mean"]
        ) / self.feature_stats[
            "ref_base"
        ]["std"]

        dist_base = (
            self.dataset.dist_base[:, idx, :]
            -
            self.feature_stats[
                "dist_base"
            ]["mean"]
        ) / self.feature_stats[
            "dist_base"
        ]["std"]

        ref_large = (
            self.dataset.ref_large[:, idx, :]
            -
            self.feature_stats[
                "ref_large"
            ]["mean"]
        ) / self.feature_stats[
            "ref_large"
        ]["std"]

        dist_large = (
            self.dataset.dist_large[:, idx, :]
            -
            self.feature_stats[
                "dist_large"
            ]["mean"]
        ) / self.feature_stats[
            "dist_large"
        ]["std"]

        mos_mean = self.mos_stats[
            dataset_name
        ]["mean"]

        mos_std = self.mos_stats[
            dataset_name
        ]["std"]

        mos = (
            self.dataset.mos[idx]
            -
            mos_mean
        ) / mos_std

        return {
            "ref_base": ref_base,
            "dist_base": dist_base,
            "ref_large": ref_large,
            "dist_large": dist_large,
            "mos": mos,
            "name": self.dataset.image_names[idx],
            "dataset": dataset_name,
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

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

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

        y = batch[
            "mos"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            ref_base,
            dist_base,
            ref_large,
            dist_large,
        )

        loss = criterion(
            pred,
            y,
        )

        loss.backward()

        optimizer.step()

        batch_size = y.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        /
        total_samples
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    mos_stats,
):

    model.eval()

    predictions = []
    targets = []
    dataset_labels = []
    names = []

    for batch in loader:

        ref_base = batch[
            "ref_base"
        ].to(device, non_blocking=True)

        dist_base = batch[
            "dist_base"
        ].to(device, non_blocking=True)

        ref_large = batch[
            "ref_large"
        ].to(device, non_blocking=True)

        dist_large = batch[
            "dist_large"
        ].to(device, non_blocking=True)

        y_norm = batch[
            "mos"
        ].to(device, non_blocking=True)

        pred_norm = model(
            ref_base,
            dist_base,
            ref_large,
            dist_large,
        )

        dataset_batch = batch[
            "dataset"
        ]

        pred_original = []
        target_original = []

        for i, dataset_name in enumerate(
            dataset_batch
        ):

            mean = mos_stats[
                dataset_name
            ]["mean"].to(device)

            std = mos_stats[
                dataset_name
            ]["std"].to(device)

            pred_original.append(
                pred_norm[i]
                *
                std
                +
                mean
            )

            target_original.append(
                y_norm[i]
                *
                std
                +
                mean
            )

        pred_original = torch.stack(
            pred_original
        )

        target_original = torch.stack(
            target_original
        )

        predictions.extend(
            pred_original
            .squeeze(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        targets.extend(
            target_original
            .squeeze(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        dataset_labels.extend(
            dataset_batch
        )

        names.extend(
            batch["name"]
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
    # DATASET-SPECIFIC
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

        if mask.sum() == 0:
            continue

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
# SAVE HISTORY
# ============================================================

def save_history(
    history,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",

        "all_mse",
        "all_srcc",
        "all_plcc",

        "live_mse",
        "live_srcc",
        "live_plcc",

        "tid2013_mse",
        "tid2013_srcc",
        "tid2013_plcc",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(history)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    results,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
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

        for name, dataset_name, target, pred in zip(
            results["names"],
            results["dataset_labels"],
            results["targets"],
            results["predictions"],
        ):

            writer.writerow(
                [
                    name,
                    dataset_name,
                    float(target),
                    float(pred),
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3.5 LIVE + TID2013 "
            "mixture training."
        )
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--variant",
        choices=[
            "small",
            "medium",
            "large",
        ],
        default="medium",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
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

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default="results/phase3_mixture",
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
    print(
        "PHASE 3.5 - LIVE + TID2013 MIXTURE"
    )
    print("=" * 70)

    print(f"Variant:       {args.variant}")
    print(f"Device:        {device}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay:  {args.weight_decay}")
    print(f"Validation:    {args.val_ratio}")
    print(f"Patience:      {args.patience}")
    print(f"Seed:           {args.seed}")

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
    # SPLIT
    # ========================================================

    (
        train_idx,
        val_idx,
        live_train,
        live_val,
        tid_train,
        tid_val,
    ) = mixture_train_val_split(
        dataset,
        args.val_ratio,
        args.seed,
    )

    print("\nSplit:")

    print(
        f"  LIVE train:      {len(live_train)}"
    )

    print(
        f"  LIVE validation: {len(live_val)}"
    )

    print(
        f"  TID2013 train:   {len(tid_train)}"
    )

    print(
        f"  TID2013 val:     {len(tid_val)}"
    )

    print(
        f"  Total train:     {len(train_idx)}"
    )

    print(
        f"  Total val:       {len(val_idx)}"
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print(
        "\nComputing feature normalization "
        "from TRAINING SET ONLY..."
    )

    feature_stats = compute_feature_stats(
        dataset,
        train_idx,
    )

    print(
        "\nComputing dataset-specific MOS "
        "normalization from TRAINING SET ONLY..."
    )

    mos_stats = compute_dataset_mos_stats(
        dataset,
        train_idx,
    )

    print("\nMOS normalization:")

    for dataset_name in [
        "LIVE",
        "TID2013",
    ]:

        print(
            f"  {dataset_name}: "
            f"mean={mos_stats[dataset_name]['mean'].item():.6f}, "
            f"std={mos_stats[dataset_name]['std'].item():.6f}"
        )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = NormalizedMixtureDataset(
        dataset,
        train_idx,
        feature_stats,
        mos_stats,
    )

    val_dataset = NormalizedMixtureDataset(
        dataset,
        val_idx,
        feature_stats,
        mos_stats,
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

    print("\nInitializing DualEncoderFusion:")

    model = DualEncoderFusion(
        dim_base=dataset.dim_base,
        dim_large=dataset.dim_large,
        variant=args.variant,
    ).to(device)

    print(f"  Variant:       {args.variant}")
    print(f"  Base dim:      {dataset.dim_base}")
    print(f"  Large dim:     {dataset.dim_large}")
    print(
        f"  Total input:   "
        f"{dataset.dim_base + dataset.dim_large}"
    )

    print("\nModel:")
    print(model)

    # ========================================================
    # OPTIMIZER
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
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        /
        f"best_mixture_siglip2_{args.variant}.pt"
    )

    history_path = (
        output_dir
        /
        f"history_mixture_siglip2_{args.variant}.csv"
    )

    predictions_path = (
        output_dir
        /
        f"predictions_mixture_siglip2_{args.variant}.csv"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_srcc = -float("inf")
    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        results = evaluate(
            model,
            val_loader,
            device,
            mos_stats,
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        row = {
            "epoch": epoch,
            "train_loss": train_loss,

            "all_mse": results["all_mse"],
            "all_srcc": results["all_srcc"],
            "all_plcc": results["all_plcc"],

            "live_mse": results["live_mse"],
            "live_srcc": results["live_srcc"],
            "live_plcc": results["live_plcc"],

            "tid2013_mse": results["tid2013_mse"],
            "tid2013_srcc": results["tid2013_srcc"],
            "tid2013_plcc": results["tid2013_plcc"],
        }

        history.append(row)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:03d}/{args.epochs}"
        )

        print(
            f"  Train Loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"  MIXED   | "
            f"MSE: {results['all_mse']:.3f} | "
            f"SRCC: {results['all_srcc']:.6f} | "
            f"PLCC: {results['all_plcc']:.6f}"
        )

        print(
            f"  LIVE    | "
            f"MSE: {results['live_mse']:.3f} | "
            f"SRCC: {results['live_srcc']:.6f} | "
            f"PLCC: {results['live_plcc']:.6f}"
        )

        print(
            f"  TID2013 | "
            f"MSE: {results['tid2013_mse']:.3f} | "
            f"SRCC: {results['tid2013_srcc']:.6f} | "
            f"PLCC: {results['tid2013_plcc']:.6f}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        current_srcc = results["all_srcc"]

        if (
            not np.isnan(current_srcc)
            and current_srcc > best_srcc
        ):

            best_srcc = current_srcc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_type":
                        "DualEncoderFusion",

                    "variant":
                        args.variant,

                    "dim_base":
                        dataset.dim_base,

                    "dim_large":
                        dataset.dim_large,

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_srcc":
                        results["all_srcc"],

                    "best_plcc":
                        results["all_plcc"],

                    "val_mse":
                        results["all_mse"],

                    "live_mse":
                        results["live_mse"],

                    "live_srcc":
                        results["live_srcc"],

                    "live_plcc":
                        results["live_plcc"],

                    "tid2013_mse":
                        results["tid2013_mse"],

                    "tid2013_srcc":
                        results["tid2013_srcc"],

                    "tid2013_plcc":
                        results["tid2013_plcc"],

                    "seed":
                        args.seed,

                    "val_ratio":
                        args.val_ratio,

                    "train_indices":
                        train_idx,

                    "val_indices":
                        val_idx,

                    "live_train_indices":
                        live_train,

                    "live_val_indices":
                        live_val,

                    "tid_train_indices":
                        tid_train,

                    "tid_val_indices":
                        tid_val,

                    "feature_normalization_stats":
                        feature_stats,

                    "mos_normalization_stats":
                        mos_stats,

                    "features_live_base":
                        str(args.live_base),

                    "features_live_large":
                        str(args.live_large),

                    "features_tid_base":
                        str(args.tid_base),

                    "features_tid_large":
                        str(args.tid_large),

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

            # Save predictions of best model
            save_predictions(
                results,
                predictions_path,
            )

            print(
                f"  -> BEST CHECKPOINT SAVED "
                f"(SRCC={best_srcc:.6f})"
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
                "\nEarly stopping."
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
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "MIXTURE TRAINING FINISHED"
    )
    print("=" * 70)

    print(
        f"Best epoch:     {best_epoch}"
    )

    print(
        f"Best SRCC:      {best_srcc:.6f}"
    )

    print(
        f"Checkpoint:     {checkpoint_path}"
    )

    print(
        f"History CSV:    {history_path}"
    )

    print(
        f"Predictions:    {predictions_path}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()