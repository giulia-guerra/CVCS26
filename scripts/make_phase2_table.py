# lettura del CSV dei risultati
# selezione del layer migliore per ogni modello/dataset
# creazione della tabella finale della fase 2
# salvataggio in: results/phase2/tables/

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

#CSV_FILE = ROOT / "phase2_pipal_results.csv"
#CSV_FILE = ROOT / "phase2_live_results.csv"
CSV_FILE = ROOT / "phase2_tid2013_results.csv"

TABLE_DIR = ROOT / "results" / "phase2" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main():

    df = pd.read_csv(CSV_FILE)

    # ignora eventuali NaN (es. layer 0 di DINO)
    df = df.dropna()

    best_rows = []

    for model, group in df.groupby("model"):

        best = group.loc[group["srcc"].idxmax()]

        best_rows.append({
            "model": best["model"],
            "best_layer": int(best["layer"]),
            "srcc": best["srcc"],
            "plcc": best["plcc"]
        })

    result = pd.DataFrame(best_rows)

    print("\n=== BEST LAYERS ===")
    print(result)

    #csv_out = TABLE_DIR / "phase2_best_layers_pipal.csv"
    #csv_out = TABLE_DIR / "phase2_best_layers_live.csv"
    csv_out = TABLE_DIR / "phase2_best_layers_tid2013.csv"
    result.to_csv(csv_out, index=False)

    print(f"CSV salvato in: {csv_out}")


if __name__ == "__main__":
    main()