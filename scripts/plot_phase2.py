# lettura del CSV prodotto da analyze_phase2.py
# creazione dei plot: layer vs SRCC
# salvataggio immagini in: results/phase2/plots/

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "phase2"
CSV_FILE = RESULTS_DIR / "csv" / "phase2_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric(df: pd.DataFrame, metric_col: str, title: str, out_name: str) -> None:
    plt.figure(figsize=(10, 6))

    for (dataset, model), group in df.groupby(["dataset", "model"]):
        group = group.sort_values("layer")
        plt.plot(group["layer"], group[metric_col], marker="o", label=f"{dataset} - {model}")

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(metric_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = PLOTS_DIR / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Salvato: {out_path}")


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV non trovato: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    plot_metric(
        df,
        metric_col="SRCC",
        title="Layer vs SRCC - Phase 2",
        out_name="phase2_layer_vs_srcc.png",
    )

    plot_metric(
        df,
        metric_col="PLCC",
        title="Layer vs PLCC - Phase 2",
        out_name="phase2_layer_vs_plcc.png",
    )


if __name__ == "__main__":
    main()
