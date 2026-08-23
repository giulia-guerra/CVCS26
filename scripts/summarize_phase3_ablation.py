# Il codice raccoglie e sintetizza i risultati dell'ablation study della Phase 3.
# Cerca i file di history delle varianti small, medium e large, seleziona per ciascuna
# la configurazione con il miglior SRCC sul dataset Mixed e raccoglie le metriche
# MSE, SRCC e PLCC per Mixed, LIVE e TID2013. Infine, ordina i risultati in base
# al Mixed SRCC, li salva in un file CSV e stampa una tabella riepilogativa.


import argparse
import csv
from pathlib import Path


# ============================================================
# FIND BEST ROW
# ============================================================

def find_best_row(
    csv_path,
):

    with open(
        csv_path,
        "r",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"Empty CSV: {csv_path}"
        )

    best = max(
        rows,
        key=lambda row:
            float(row["all_srcc"])
    )

    return best


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create Phase 3.5 mixture "
            "ablation summary."
        )
    )

    parser.add_argument(
        "--results-dir",
        default="results",
    )

    parser.add_argument(
        "--output",
        default="results/phase3_mixture_ablation.csv",
    )

    args = parser.parse_args()

    results_dir = Path(
        args.results_dir
    )

    output_path = Path(
        args.output
    )

    # ========================================================
    # SEARCH
    # ========================================================

    variants = [
        "small",
        "medium",
        "large",
    ]

    summary = []

    for variant in variants:

        # ----------------------------------------------------
        # Search recursively
        # ----------------------------------------------------

        candidates = list(
            results_dir.glob(
                f"**/history_mixture_siglip2_{variant}.csv"
            )
        )

        if not candidates:

            print(
                f"[WARNING] No CSV found "
                f"for variant '{variant}'"
            )

            continue

        # Use newest file if multiple exist
        candidates.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        csv_path = candidates[0]

        print(
            f"Reading {variant}: "
            f"{csv_path}"
        )

        best = find_best_row(
            csv_path
        )

        summary.append(
            {
                "variant": variant,

                "best_epoch":
                    int(best["epoch"]),

                "mixed_mse":
                    float(best["all_mse"]),

                "mixed_srcc":
                    float(best["all_srcc"]),

                "mixed_plcc":
                    float(best["all_plcc"]),

                "live_mse":
                    float(best["live_mse"]),

                "live_srcc":
                    float(best["live_srcc"]),

                "live_plcc":
                    float(best["live_plcc"]),

                "tid2013_mse":
                    float(best["tid2013_mse"]),

                "tid2013_srcc":
                    float(best["tid2013_srcc"]),

                "tid2013_plcc":
                    float(best["tid2013_plcc"]),
            }
        )

    if not summary:

        raise RuntimeError(
            "No mixture history CSV files found."
        )

    # ========================================================
    # SORT BY MIXED SRCC
    # ========================================================

    summary.sort(
        key=lambda row:
            row["mixed_srcc"],
        reverse=True,
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "variant",
        "best_epoch",

        "mixed_mse",
        "mixed_srcc",
        "mixed_plcc",

        "live_mse",
        "live_srcc",
        "live_plcc",

        "tid2013_mse",
        "tid2013_srcc",
        "tid2013_plcc",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    # ========================================================
    # PRINT TABLE
    # ========================================================

    print("\n")
    print("=" * 110)
    print(
        "PHASE 3.5 - MIXTURE ABLATION"
    )
    print("=" * 110)

    print(
        f"{'Variant':<10}"
        f"{'Epoch':>8}"
        f"{'Mixed SRCC':>14}"
        f"{'Mixed PLCC':>14}"
        f"{'LIVE SRCC':>14}"
        f"{'LIVE PLCC':>14}"
        f"{'TID SRCC':>14}"
        f"{'TID PLCC':>14}"
    )

    print("-" * 110)

    for row in summary:

        print(
            f"{row['variant']:<10}"
            f"{row['best_epoch']:>8}"
            f"{row['mixed_srcc']:>14.4f}"
            f"{row['mixed_plcc']:>14.4f}"
            f"{row['live_srcc']:>14.4f}"
            f"{row['live_plcc']:>14.4f}"
            f"{row['tid2013_srcc']:>14.4f}"
            f"{row['tid2013_plcc']:>14.4f}"
        )

    print("=" * 110)

    print(
        f"\nSaved summary to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()