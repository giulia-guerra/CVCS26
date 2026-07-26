# lettura del CSV dei risultati
# selezione del layer migliore per ogni modello/dataset
# creazione della tabella finale della fase 2
# salvataggio in: results/phase2/tables/

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "phase2"
CSV_FILE = RESULTS_DIR / "csv" / "phase2_results.csv"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV non trovato: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    # Prende il layer migliore per ogni coppia dataset/model in base a SRCC
    best = (
        df.sort_values(["dataset", "model", "SRCC"], ascending=[True, True, False])
          .groupby(["dataset", "model"], as_index=False)
          .first()
    )

    # Tabella compatta finale
    table = best[["dataset", "model", "layer", "SRCC", "PLCC"]].copy()
    table = table.sort_values(["dataset", "model"])

    out_csv = TABLES_DIR / "phase2_table.csv"
    out_md = TABLES_DIR / "phase2_table.md"

    table.to_csv(out_csv, index=False)
    out_md.write_text(table.to_markdown(index=False), encoding="utf-8")

    print(f"Salvato: {out_csv}")
    print(f"Salvato: {out_md}")
    print(table)


if __name__ == "__main__":
    main()