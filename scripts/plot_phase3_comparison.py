# Il codice genera i grafici di confronto tra i modelli della Phase 3
# a partire dai risultati salvati in un file CSV. Per ciascun dataset,
# confronta le prestazioni dei diversi modelli utilizzando MSE, SRCC e PLCC,
# producendo grafici a barre separati. I risultati vengono salvati
# automaticamente nella directory di output specificata [results/].


import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot Phase 3 Baseline vs Advanced "
            "model comparison"
        )
    )

    # --------------------------------------------------------
    # RESULTS CSV
    # --------------------------------------------------------

    parser.add_argument(
        "--results",
        required=True,
        help=(
            "CSV file containing Phase 3 comparison "
            "results"
        ),
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default="results/phase3/plots",
        help=(
            "Directory where plots will be saved"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("PHASE 3 - MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\nLoading results from:\n"
        f"{args.results}"
    )

    df = pd.read_csv(
        args.results
    )

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

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
                f"Missing required column "
                f"'{column}' in CSV"
            )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print(
        f"\nOutput directory:\n"
        f"{args.output_dir}"
    )

    # ========================================================
    # PRINT DATA
    # ========================================================

    print("\nResults:")
    print(df.to_string(index=False))

    # ========================================================
    # DATASETS
    # ========================================================

    datasets = df["dataset"].unique()

    # ========================================================
    # SRCC
    # ========================================================

    for dataset in datasets:

        subset = df[
            df["dataset"] == dataset
        ].copy()

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            subset["model"],
            subset["srcc"],
        )

        plt.ylabel(
            "SRCC"
        )

        plt.xlabel(
            "Model"
        )

        plt.title(
            f"Phase 3 - SRCC Comparison ({dataset})"
        )

        plt.ylim(
            0,
            1,
        )

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
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nSaved SRCC plot: "
            f"{output_path}"
        )

    # ========================================================
    # PLCC
    # ========================================================

    for dataset in datasets:

        subset = df[
            df["dataset"] == dataset
        ].copy()

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            subset["model"],
            subset["plcc"],
        )

        plt.ylabel(
            "PLCC"
        )

        plt.xlabel(
            "Model"
        )

        plt.title(
            f"Phase 3 - PLCC Comparison ({dataset})"
        )

        plt.ylim(
            0,
            1,
        )

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
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nSaved PLCC plot: "
            f"{output_path}"
        )

    # ========================================================
    # MSE
    # ========================================================

    for dataset in datasets:

        subset = df[
            df["dataset"] == dataset
        ].copy()

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            subset["model"],
            subset["mse"],
        )

        plt.ylabel(
            "MSE"
        )

        plt.xlabel(
            "Model"
        )

        plt.title(
            f"Phase 3 - MSE Comparison ({dataset})"
        )

        plt.xticks(
            rotation=20,
            ha="right",
        )

        plt.tight_layout()

        output_path = os.path.join(
            args.output_dir,
            f"phase3_mse_{dataset.lower()}.png",
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nSaved MSE plot: "
            f"{output_path}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PLOTTING COMPLETED")
    print("=" * 60)

    print(
        f"Plots saved in:\n"
        f"{args.output_dir}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()