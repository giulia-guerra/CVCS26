import argparse
import os

import pandas as pd


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Summarize Phase 3 Baseline "
            "and Advanced model results"
        )
    )

    # --------------------------------------------------------
    # INPUT CSV
    # --------------------------------------------------------

    parser.add_argument(
        "--results",
        required=True,
        help=(
            "CSV file containing Phase 3 "
            "comparison results"
        ),
    )

    # --------------------------------------------------------
    # OUTPUT TXT
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        default="results/phase3/tables/phase3_summary.txt",
        help=(
            "Output text file where the complete "
            "summary will be saved"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    df = pd.read_csv(
        args.results
    )

    # ========================================================
    # CHECK COLUMNS
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
    # BUILD SUMMARY
    # ========================================================

    lines = []

    def add(text=""):
        lines.append(text)

    # ========================================================
    # HEADER
    # ========================================================

    add("=" * 70)
    add("PHASE 3 RESULTS")
    add("=" * 70)

    add("")
    add(f"Loading results from:")
    add(f"{args.results}")

    # ========================================================
    # COMPLETE RESULTS
    # ========================================================

    add("")
    add("=" * 70)
    add("COMPLETE RESULTS")
    add("=" * 70)

    complete_table = df[
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

    add(complete_table)

    # ========================================================
    # BEST MODEL BY SRCC
    # ========================================================

    add("")
    add("=" * 70)
    add("BEST MODEL BY SRCC")
    add("=" * 70)

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        best = subset.loc[
            subset["srcc"].idxmax()
        ]

        add(
            f"{dataset}: "
            f"{best['model']} "
            f"(SRCC={best['srcc']:.6f})"
        )

    # ========================================================
    # BEST MODEL BY PLCC
    # ========================================================

    add("")
    add("=" * 70)
    add("BEST MODEL BY PLCC")
    add("=" * 70)

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        best = subset.loc[
            subset["plcc"].idxmax()
        ]

        add(
            f"{dataset}: "
            f"{best['model']} "
            f"(PLCC={best['plcc']:.6f})"
        )

    # ========================================================
    # BEST MODEL BY MSE
    # ========================================================

    add("")
    add("=" * 70)
    add("BEST MODEL BY MSE")
    add("=" * 70)

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        best = subset.loc[
            subset["mse"].idxmin()
        ]

        add(
            f"{dataset}: "
            f"{best['model']} "
            f"(MSE={best['mse']:.6f})"
        )

    # ========================================================
    # ADVANCED VS BASELINE
    # ========================================================

    add("")
    add("=" * 70)
    add("ADVANCED VS BASELINE")
    add("=" * 70)

    for dataset in df["dataset"].unique():

        subset = df[
            df["dataset"] == dataset
        ]

        # ----------------------------------------------------
        # Find baseline
        # ----------------------------------------------------

        baseline_rows = subset[
            subset["model"]
            .str.lower()
            .str.contains("baseline")
        ]

        # ----------------------------------------------------
        # Find advanced
        # ----------------------------------------------------

        advanced_rows = subset[
            subset["model"]
            .str.lower()
            .str.contains("advanced")
        ]

        # ----------------------------------------------------
        # Check
        # ----------------------------------------------------

        if (
            baseline_rows.empty
            or advanced_rows.empty
        ):

            add(
                f"\n{dataset}: "
                "Baseline or Advanced model "
                "not found."
            )

            continue

        baseline = baseline_rows.iloc[0]
        advanced = advanced_rows.iloc[0]

        # ----------------------------------------------------
        # Absolute differences
        # ----------------------------------------------------

        delta_srcc = (
            advanced["srcc"]
            - baseline["srcc"]
        )

        delta_plcc = (
            advanced["plcc"]
            - baseline["plcc"]
        )

        delta_mse = (
            advanced["mse"]
            - baseline["mse"]
        )

        # ----------------------------------------------------
        # Percentage improvements
        # ----------------------------------------------------

        srcc_improvement = (
            delta_srcc
            / abs(baseline["srcc"])
            * 100
        )

        plcc_improvement = (
            delta_plcc
            / abs(baseline["plcc"])
            * 100
        )

        # For MSE, LOWER is better.
        # Therefore improvement is:
        # (baseline - advanced) / baseline

        mse_improvement = (
            (
                baseline["mse"]
                - advanced["mse"]
            )
            / baseline["mse"]
            * 100
        )

        # ----------------------------------------------------
        # Print / save comparison
        # ----------------------------------------------------

        add("")
        add(f"Dataset: {dataset}")

        add("")
        add("Baseline:")

        add(
            f"  SRCC: {baseline['srcc']:.6f}"
        )

        add(
            f"  PLCC: {baseline['plcc']:.6f}"
        )

        add(
            f"  MSE:  {baseline['mse']:.6f}"
        )

        add("")
        add("Advanced Attention:")

        add(
            f"  SRCC: {advanced['srcc']:.6f}"
        )

        add(
            f"  PLCC: {advanced['plcc']:.6f}"
        )

        add(
            f"  MSE:  {advanced['mse']:.6f}"
        )

        # ----------------------------------------------------
        # Absolute differences
        # ----------------------------------------------------

        add("")
        add("Absolute differences:")

        add(
            f"  Δ SRCC: {delta_srcc:+.6f}"
        )

        add(
            f"  Δ PLCC: {delta_plcc:+.6f}"
        )

        add(
            f"  Δ MSE:  {delta_mse:+.6f}"
        )

        # ----------------------------------------------------
        # Percentage improvements
        # ----------------------------------------------------

        add("")
        add("Percentage improvement:")

        add(
            f"  SRCC: {srcc_improvement:+.2f}%"
        )

        add(
            f"  PLCC: {plcc_improvement:+.2f}%"
        )

        add(
            f"  MSE:  {mse_improvement:+.2f}%"
        )

    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    add("")
    add("=" * 70)
    add("SUMMARY COMPLETED")
    add("=" * 70)

    add("")
    add("Higher SRCC and PLCC are better.")
    add("Lower MSE is better.")

    add("=" * 70)

    # ========================================================
    # FINAL TEXT
    # ========================================================

    summary = "\n".join(lines)

    # --------------------------------------------------------
    # PRINT TO TERMINAL
    # --------------------------------------------------------

    print("\n")
    print(summary)

    # --------------------------------------------------------
    # SAVE TO FILE
    # --------------------------------------------------------

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(summary)
        f.write("\n")

    print(
        f"\nSummary saved to:\n"
        f"{args.output}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()