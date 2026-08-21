import argparse

import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description="Summarize Phase 3 results"
    )

    parser.add_argument(
        "--results",
        required=True,
        help="CSV file containing Phase 3 results",
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

    # --------------------------------------------------
    # Print complete table
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS")
    print("=" * 70)

    print(
        df[
            [
                "model",
                "dataset",
                "mse",
                "srcc",
                "plcc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------
    # Best model by SRCC
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST MODEL BY SRCC")
    print("=" * 70)

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        best = subset.loc[
            subset["srcc"].idxmax()
        ]

        print(
            f"{dataset}: "
            f"{best['model']} "
            f"(SRCC={best['srcc']:.6f})"
        )

    # --------------------------------------------------
    # Best model by PLCC
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST MODEL BY PLCC")
    print("=" * 70)

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        best = subset.loc[
            subset["plcc"].idxmax()
        ]

        print(
            f"{dataset}: "
            f"{best['model']} "
            f"(PLCC={best['plcc']:.6f})"
        )

    print()


if __name__ == "__main__":
    main()