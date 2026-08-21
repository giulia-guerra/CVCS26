import os
import argparse

import torch


def load_features(path, name):

    print("\n" + "=" * 60)
    print(f"Loading {name}")
    print("=" * 60)

    print(f"File: {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "ref_features",
        "dist_features",
        "mos",
    ]

    for key in required_keys:

        if key not in data:
            raise KeyError(
                f"Missing key '{key}' in {path}"
            )

    ref = data["ref_features"].float()
    dist = data["dist_features"].float()
    mos = data["mos"].float()

    print(f"Reference features: {ref.shape}")
    print(f"Distorted features: {dist.shape}")
    print(f"MOS:                {mos.shape}")

    if ref.ndim != 3:
        raise ValueError(
            f"{name} ref_features must be 3D. "
            f"Got {ref.shape}"
        )

    if dist.ndim != 3:
        raise ValueError(
            f"{name} dist_features must be 3D. "
            f"Got {dist.shape}"
        )

    if ref.shape != dist.shape:
        raise ValueError(
            f"{name} reference/distorted shapes differ: "
            f"{ref.shape} vs {dist.shape}"
        )

    if ref.shape[1] != len(mos):
        raise ValueError(
            f"{name}: number of samples does not match MOS. "
            f"Features: {ref.shape[1]}, MOS: {len(mos)}"
        )

    # --------------------------------------------------------
    # Convert:
    #
    # [layers, samples, dim]
    #
    # to:
    #
    # [samples, layers, dim]
    # --------------------------------------------------------

    ref = ref.permute(
        1, 0, 2
    ).contiguous()

    dist = dist.permute(
        1, 0, 2
    ).contiguous()

    print("\nAfter permutation:")
    print(f"Reference: {ref.shape}")
    print(f"Distorted: {dist.shape}")

    return ref, dist, mos


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create the combined Advanced Phase 3 "
            "feature dataset from SigLIP2 Base "
            "and SigLIP2 Large features."
        )
    )

    parser.add_argument(
        "--base",
        required=True,
        help="Path to SigLIP2 Base .pt file.",
    )

    parser.add_argument(
        "--large",
        required=True,
        help="Path to SigLIP2 Large .pt file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the Advanced .pt file.",
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD BASE
    # ========================================================

    ref_base, dist_base, mos_base = load_features(
        args.base,
        "SigLIP2 Base",
    )

    # ========================================================
    # LOAD LARGE
    # ========================================================

    ref_large, dist_large, mos_large = load_features(
        args.large,
        "SigLIP2 Large",
    )

    # ========================================================
    # CHECK COMPATIBILITY
    # ========================================================

    print("\n" + "=" * 60)
    print("CHECKING COMPATIBILITY")
    print("=" * 60)

    if len(mos_base) != len(mos_large):

        raise ValueError(
            "Base and Large contain different "
            "numbers of samples."
        )

    if not torch.equal(
        mos_base,
        mos_large,
    ):

        raise ValueError(
            "Base and Large MOS vectors are not identical."
        )

    if ref_base.shape[0] != ref_large.shape[0]:

        raise ValueError(
            "Base and Large sample dimensions differ."
        )

    print("Number of samples:", len(mos_base))
    print("MOS vectors: identical")
    print("Base and Large: compatible")

    # ========================================================
    # CHECK FEATURE DIMENSIONS
    # ========================================================

    if ref_base.shape[-1] != 768:

        raise ValueError(
            "Expected Base feature dimension 768, "
            f"got {ref_base.shape[-1]}"
        )

    if ref_large.shape[-1] != 1024:

        raise ValueError(
            "Expected Large feature dimension 1024, "
            f"got {ref_large.shape[-1]}"
        )

    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    advanced_data = {
        "ref_base": ref_base,
        "dist_base": dist_base,
        "ref_large": ref_large,
        "dist_large": dist_large,
        "mos": mos_base,
    }

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # ========================================================
    # SAVE
    # ========================================================

    print("\n" + "=" * 60)
    print("SAVING ADVANCED DATASET")
    print("=" * 60)

    print(f"Output: {args.output}")

    torch.save(
        advanced_data,
        args.output,
    )

    # ========================================================
    # VERIFY SAVED FILE
    # ========================================================

    print("\nVerifying saved dataset...")

    check = torch.load(
        args.output,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Keys:",
        check.keys(),
    )

    for key, value in check.items():

        if hasattr(value, "shape"):

            print(
                f"{key}: {value.shape}"
            )

    print("\n" + "=" * 60)
    print("ADVANCED DATASET CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()