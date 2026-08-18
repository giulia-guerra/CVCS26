import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot Phase 3 mixture ablation results."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="results/phase3_mixture_ablation.csv",
        help="Input ablation CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3_mixture_ablation_plots",
        help="Directory where plots will be saved.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    df = pd.read_csv(input_path)

    required_columns = [
        "variant",
        "mixed_srcc",
        "mixed_plcc",
        "live_srcc",
        "live_plcc",
        "tid2013_srcc",
        "tid2013_plcc",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}"
        )

    # Keep a consistent order
    variant_order = ["small", "medium", "large"]

    df["variant"] = pd.Categorical(
        df["variant"],
        categories=variant_order,
        ordered=True,
    )

    df = df.sort_values("variant")

    variants = df["variant"].astype(str).tolist()

    # ---------------------------------------------------------
    # PLOT 1: SRCC
    # ---------------------------------------------------------

    x = range(len(variants))

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        df["mixed_srcc"],
        marker="o",
        linewidth=2,
        label="Mixed",
    )

    plt.plot(
        x,
        df["live_srcc"],
        marker="o",
        linewidth=2,
        label="LIVE",
    )

    plt.plot(
        x,
        df["tid2013_srcc"],
        marker="o",
        linewidth=2,
        label="TID2013",
    )

    plt.xticks(x, [v.capitalize() for v in variants])
    plt.ylabel("SRCC")
    plt.xlabel("Model Variant")
    plt.title("Phase 3 Ablation - SRCC")
    plt.ylim(-1.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    srcc_path = output_dir / "phase3_ablation_srcc.png"
    plt.savefig(srcc_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: PLCC
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        df["mixed_plcc"],
        marker="o",
        linewidth=2,
        label="Mixed",
    )

    plt.plot(
        x,
        df["live_plcc"],
        marker="o",
        linewidth=2,
        label="LIVE",
    )

    plt.plot(
        x,
        df["tid2013_plcc"],
        marker="o",
        linewidth=2,
        label="TID2013",
    )

    plt.xticks(x, [v.capitalize() for v in variants])
    plt.ylabel("PLCC")
    plt.xlabel("Model Variant")
    plt.title("Phase 3 Ablation - PLCC")
    plt.ylim(-1.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plcc_path = output_dir / "phase3_ablation_plcc.png"
    plt.savefig(plcc_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: MIXED ONLY - SRCC vs PLCC
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        df["mixed_srcc"],
        marker="o",
        linewidth=2,
        label="SRCC",
    )

    plt.plot(
        x,
        df["mixed_plcc"],
        marker="o",
        linewidth=2,
        label="PLCC",
    )

    plt.xticks(x, [v.capitalize() for v in variants])
    plt.ylabel("Correlation")
    plt.xlabel("Model Variant")
    plt.title("Phase 3 Ablation - Mixed Dataset")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    mixed_path = output_dir / "phase3_ablation_mixed.png"
    plt.savefig(mixed_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 4: DATASET COMPARISON - SRCC
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    width = 0.25

    plt.bar(
        [i - width for i in x],
        df["mixed_srcc"],
        width=width,
        label="Mixed",
    )

    plt.bar(
        x,
        df["live_srcc"],
        width=width,
        label="LIVE",
    )

    plt.bar(
        [i + width for i in x],
        df["tid2013_srcc"],
        width=width,
        label="TID2013",
    )

    plt.xticks(x, [v.capitalize() for v in variants])
    plt.ylabel("SRCC")
    plt.xlabel("Model Variant")
    plt.title("Phase 3 Ablation - SRCC Comparison")
    plt.ylim(-1.0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    bar_path = output_dir / "phase3_ablation_srcc_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PRINT SUMMARY
    # ---------------------------------------------------------

    print("=" * 80)
    print("PHASE 3 - ABLATION PLOTS")
    print("=" * 80)

    print(f"Input CSV: {input_path}")
    print(f"Output directory: {output_dir}")
    print()

    print("Generated plots:")
    print(f"  - {srcc_path}")
    print(f"  - {plcc_path}")
    print(f"  - {mixed_path}")
    print(f"  - {bar_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()