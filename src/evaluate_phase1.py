# Questo file contiene lo script principale per la valutazione iniziale della
# pipeline IQA (Phase 1). Carica il dataset LIVE tramite il relativo DataLoader,
# raccoglie i valori MOS reali e calcola le metriche SRCC e PLCC tra predizioni
# e ground truth. Attualmente utilizza una predizione placeholder uguale al MOS
# reale per verificare il corretto funzionamento della pipeline, del calcolo
# delle metriche e del sistema di logging dei risultati.


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd

from src.metrics.metrics import srcc, plcc
from src.metrics.similarity import cosine_similarity, l2_distance

FEATURES_DIR = Path("/work/cvcs2026/Cross_Entropy_Champions/features/PIPAL")


def evaluate_model(pt_file):
    print(f"\n=== {pt_file.name} ===")

    data = torch.load(pt_file, map_location="cpu")

    ref_features = data["ref_features"]
    dist_features = data["dist_features"]
    mos = data["mos"]

    # Ultimo layer
    ref = ref_features[-1]
    dist = dist_features[-1]

    print("Ref shape :", ref.shape)
    print("Dist shape:", dist.shape)

    # Similarità coseno
    cosine_scores = cosine_similarity(ref, dist)

    # Distanza L2
    l2_scores = l2_distance(ref, dist)

    results = []

    # Cosine
    srcc_cos = srcc(cosine_scores, mos)
    plcc_cos = plcc(cosine_scores, mos)

    # L2 (segno invertito)
    srcc_l2 = srcc(-l2_scores, mos)
    plcc_l2 = plcc(-l2_scores, mos)

    model_name = data["model_config"]

    results.append({
        "model": model_name,
        "metric": "cosine",
        "SRCC": srcc_cos,
        "PLCC": plcc_cos
    })

    results.append({
        "model": model_name,
        "metric": "l2",
        "SRCC": srcc_l2,
        "PLCC": plcc_l2
    })

    print(f"Cosine -> SRCC={srcc_cos:.4f} PLCC={plcc_cos:.4f}")
    print(f"L2     -> SRCC={srcc_l2:.4f} PLCC={plcc_l2:.4f}")

    return results


def main():
    all_results = []

    for pt_file in FEATURES_DIR.glob("*.pt"):
        all_results.extend(evaluate_model(pt_file))

    df = pd.DataFrame(all_results)

    output_csv = "phase1_pipal_results.csv"
    df.to_csv(output_csv, index=False)

    print("\n=== RISULTATI FINALI ===")
    print(df)

    print(f"\nCSV salvato in: {output_csv}")


if __name__ == "__main__":
    main()