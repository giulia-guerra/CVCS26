import argparse

import torch
from torch.utils.data import DataLoader, Subset

from src.phase3.dataset import FeatureDataset
from src.phase3.regressor import IQARegressor
from src.phase3.evaluate import evaluate


def make_split(
    dataset_size,
    val_ratio,
    seed
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    val_size = int(dataset_size * val_ratio)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    return train_indices, val_indices


def main():

    parser = argparse.ArgumentParser()

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
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PHASE 3 - EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    layer = checkpoint["layer"]
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    dropout = checkpoint["dropout"]
    seed = checkpoint["seed"]

    print(f"Layer: {layer}")
    print(f"Input dim: {input_dim}")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = FeatureDataset(
        pt_path=args.features,
        layer=layer
    )

    _, val_indices = make_split(
        len(dataset),
        val_ratio=0.2,
        seed=seed
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = IQARegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    (
        mse,
        srcc_value,
        plcc_value,
        _,
        _
    ) = evaluate(
        model=model,
        dataloader=val_loader,
        device=device
    )

    print()
    print("RESULTS")
    print("-" * 60)
    print(f"MSE :  {mse:.6f}")
    print(f"SRCC:  {srcc_value:.6f}")
    print(f"PLCC:  {plcc_value:.6f}")
    print("-" * 60)


if __name__ == "__main__":
    main()