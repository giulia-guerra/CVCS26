import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(
        description="Plot Phase 3 model comparison"
    )

    parser.add_argument(
        "--results",
        required=True,
        help="CSV file containing Phase 3 results",
    )

    parser.add_argument(
        "--output-dir",
        default="results/phase3/plots",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load results
    # --------------------------------------------------

    df = pd.read_csv(args.results)

    required_columns = [
        "model",
        "dataset",
        "mse",
        "srcc",
        "plcc",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' in CSV"
            )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Plot SRCC
    # --------------------------------------------------

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            subset["model"],
            subset["srcc"],
        )

        plt.ylabel("SRCC")
        plt.xlabel("Model")
        plt.title(
            f"Phase 3 - SRCC Comparison ({dataset})"
        )

        plt.ylim(0, 1)

        plt.xticks(
            rotation=20,
            ha="right",
        )

        plt.tight_layout()

        output_path = os.path.join(
            args.output_dir,
            f"phase3_srcc_{dataset.lower()}.png",
        )

        plt.savefig(
            output_path,
            dpi=300,
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------
    # Plot PLCC
    # --------------------------------------------------

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            subset["model"],
            subset["plcc"],
        )

        plt.ylabel("PLCC")
        plt.xlabel("Model")
        plt.title(
            f"Phase 3 - PLCC Comparison ({dataset})"
        )

        plt.ylim(0, 1)

        plt.xticks(
            rotation=20,
            ha="right",
        )

        plt.tight_layout()

        output_path = os.path.join(
            args.output_dir,
            f"phase3_plcc_{dataset.lower()}.png",
        )

        plt.savefig(
            output_path,
            dpi=300,
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()