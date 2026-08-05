# caricamento dei file .pt
# lettura delle feature per ogni dataset/model/layer:
# calcolo di SRCC/PLCC
# salvataggio dei risultati in un CSV

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd

from src.metrics.metrics import srcc, plcc
from src.metrics.similarity import cosine_similarity

FEATURES_DIR = Path("/work/cvcs2026/Cross_Entropy_Champions/features/PIPAL")


def analyze_model(pt_file):
    print(f"\nAnalizzo {pt_file.name}")

    data = torch.load(pt_file, map_location="cpu")

    ref_features = data["ref_features"]
    dist_features = data["dist_features"]
    mos = data["mos"]

    n_layers = ref_features.shape[0]

    results = []

    for layer_idx in range(n_layers):

        ref = ref_features[layer_idx]
        dist = dist_features[layer_idx]

        scores = cosine_similarity(ref, dist)

        layer_srcc = srcc(scores, mos)
        layer_plcc = plcc(scores, mos)

        print(
            f"Layer {layer_idx:02d} | "
            f"SRCC={layer_srcc:.4f} | "
            f"PLCC={layer_plcc:.4f}"
        )

        results.append({
            "model": data["model_config"],
            "layer": layer_idx,
            "srcc": layer_srcc,
            "plcc": layer_plcc
        })

    return results


def main():

    all_results = []

    for pt_file in FEATURES_DIR.glob("*.pt"):
        all_results.extend(analyze_model(pt_file))

    df = pd.DataFrame(all_results)

    output_file = "phase2_pipal_results.csv"
    df.to_csv(output_file, index=False)

    print("\nRisultati salvati in:")
    print(output_file)


if __name__ == "__main__":
    main()