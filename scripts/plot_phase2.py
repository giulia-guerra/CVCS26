# lettura del CSV prodotto da analyze_phase2.py
# creazione dei plot: layer vs SRCC
# salvataggio immagini in: results/phase2/plots/

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# CSV generato da analyze_phase2.py
#CSV_FILE = ROOT / "phase2_pipal_results.csv"
#CSV_FILE = ROOT / "phase2_live_results.csv"
CSV_FILE = ROOT / "phase2_tid2013_results.csv"

# Cartella output grafici
PLOTS_DIR = ROOT / "results" / "phase2" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric(df, metric_col, title, out_name):

    plt.figure(figsize=(10, 6))

    for model, group in df.groupby("model"):

        group = group.sort_values("layer")

        plt.plot(
            group["layer"],
            group[metric_col],
            marker="o",
            label=model
        )

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(metric_col.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = PLOTS_DIR / out_name

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Salvato: {out_path}")


def main():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV non trovato: {CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    print(df.head())

    plot_metric(
        df,
        metric_col="srcc",
        title="Layer vs SRCC (TID2013)",
        out_name="layer_vs_srcc.png"
    )

    plot_metric(
        df,
        metric_col="plcc",
        title="Layer vs PLCC (TID2013)",
        out_name="layer_vs_plcc.png"
    )


if __name__ == "__main__":
    main()