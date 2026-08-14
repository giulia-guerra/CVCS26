# genera due grafici distinti, pronti per essere inseriti nel report, in base ai dati ottenuti da 
# results/phase2/tables/final_phase2_comparison.csv:

# best_srcc_per_model.png --> ti permette di confrontare direttamente la performance massima ottenuta 
# dai diversi encoder sui tre dataset.

# best_layer_per_model.png --> mostra invece dove si trova la performance migliore. 
# È particolarmente utile per sostenere l'analisi della Fase 2: non stai semplicemente scegliendo 
# l'ultimo embedding, ma stai verificando quale livello intermedio contiene la rappresentazione 
# più utile per IQA.


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TABLE_FILE = (
    ROOT
    / "results"
    / "phase2"
    / "tables"
    / "final_phase2_comparison.csv"
)

PLOTS_DIR = (
    ROOT
    / "results"
    / "phase2"
    / "plots"
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load final Phase 2 table
# ============================================================

def load_data():

    if not TABLE_FILE.exists():
        raise FileNotFoundError(
            f"Final Phase 2 table not found:\n{TABLE_FILE}"
        )

    df = pd.read_csv(TABLE_FILE)

    required_columns = {
        "dataset",
        "model",
        "best_layer",
        "srcc",
        "plcc",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in final table: {missing}"
        )

    # Convert numeric columns
    df["best_layer"] = pd.to_numeric(
        df["best_layer"],
        errors="coerce"
    )

    df["srcc"] = pd.to_numeric(
        df["srcc"],
        errors="coerce"
    )

    df["plcc"] = pd.to_numeric(
        df["plcc"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["best_layer", "srcc"]
    )

    return df


# ============================================================
# Graph 1: Best SRCC per model
# ============================================================

def plot_best_srcc(df):

    pivot = df.pivot(
        index="model",
        columns="dataset",
        values="srcc"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8
    )

    ax.set_title(
        "Best SRCC by Encoder Model and Dataset",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Encoder Model",
        fontsize=12
    )

    ax.set_ylabel(
        "Best SRCC",
        fontsize=12
    )

    ax.set_ylim(0, 1)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        title="Dataset",
        fontsize=9,
        title_fontsize=10
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    output = PLOTS_DIR / "best_srcc_per_model.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Graph 2: Best Layer per model
# ============================================================

def plot_best_layer(df):

    pivot = df.pivot(
        index="model",
        columns="dataset",
        values="best_layer"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8
    )

    ax.set_title(
        "Best Layer by Encoder Model and Dataset",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Encoder Model",
        fontsize=12
    )

    ax.set_ylabel(
        "Best Layer",
        fontsize=12
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        title="Dataset",
        fontsize=9,
        title_fontsize=10
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    output = PLOTS_DIR / "best_layer_per_model.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    print("\nFinal Phase 2 table:")
    print(df.to_string(index=False))

    print("\nGenerating final Phase 2 plots...")

    plot_best_srcc(df)

    plot_best_layer(df)

    print("\nDone.")


if __name__ == "__main__":
    main()