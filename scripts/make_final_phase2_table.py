# legge i CSV generati dalla fase 2 per ogni dataset;
# trova il best layer (massimo SRCC);
# raccoglie:
    # dataset
    # modello
    # best layer
    # SRCC
    # PLCC
    # cosine
    # L2
# crea una tabella unica finale.

from pathlib import Path
import pandas as pd


# =====================================================
# PATH
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

TABLES_DIR = ROOT / "results" / "phase2" / "tables"

OUTPUT_FILE = TABLES_DIR / "final_phase2_comparison.csv"


# =====================================================
# INPUT TABLES
# =====================================================

FILES = {
    "LIVE": TABLES_DIR / "phase2_best_layers_live.csv",
    "PIPAL": TABLES_DIR / "phase2_best_layers_pipal.csv",
    "TID2013": TABLES_DIR / "phase2_best_layers_tid2013.csv",
}



# =====================================================
# MAIN
# =====================================================

def main():

    all_tables = []


    for dataset, file in FILES.items():

        print("\n======================")
        print(dataset)
        print("======================")

        if not file.exists():
            raise FileNotFoundError(
                f"File non trovato: {file}"
            )


        df = pd.read_csv(file)


        print("Colonne:")
        print(df.columns.tolist())


        # aggiunge nome dataset

        df.insert(
            0,
            "dataset",
            dataset
        )


        all_tables.append(df)



    # concatena LIVE + PIPAL + TID2013

    final_table = pd.concat(
        all_tables,
        ignore_index=True
    )


    # salva

    final_table.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\n================================")
    print("FINAL PHASE 2 COMPARISON")
    print("================================")

    print(final_table)


    print(
        "\nSalvata in:"
    )
    print(
        OUTPUT_FILE
    )



if __name__ == "__main__":
    main()