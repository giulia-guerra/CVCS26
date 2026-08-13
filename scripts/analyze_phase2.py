# caricamento dei file .pt
# lettura delle feature per ogni dataset/model/layer:
# calcolo di SRCC/PLCC
# salvataggio dei risultati in un CSV

import sys
from pathlib import Path
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd

from src.metrics.metrics import srcc, plcc
from src.metrics.similarity import cosine_similarity

#FEATURES_DIR = Path("/work/cvcs2026/Cross_Entropy_Champions/features/PIPAL")
#FEATURES_DIR = Path("/work/cvcs2026/Cross_Entropy_Champions/features/LIVE")
FEATURES_DIR = Path("/work/cvcs2026/Cross_Entropy_Champions/features/TID2013")

def analyze_model(pt_file):
    print(f"\n=== Analizzo {pt_file.name} ===")

    data = torch.load(pt_file, map_location="cpu")

    print("Keys disponibili:", list(data.keys()))

    ref_features = data["ref_features"]
    dist_features = data["dist_features"]
    mos = data["mos"]

    model_name = data.get(
        "model_config",
        data.get(
            "model_name",
            pt_file.stem.replace("_all_layers", "")
        )
    )

    print(f"Modello: {model_name}")
    print(f"Layers: {ref_features.shape[0]}")
    print(f"Samples: {ref_features.shape[1]}")

    n_layers = ref_features.shape[0]

    results = []

    for layer_idx in range(n_layers):

        ref = ref_features[layer_idx]
        dist = dist_features[layer_idx]

        scores = cosine_similarity(ref, dist)

        layer_srcc = abs(srcc(scores, mos))
        layer_plcc = abs(plcc(scores, mos))

        print(
            f"Layer {layer_idx:02d} | "
            f"SRCC={layer_srcc:.4f} | "
            f"PLCC={layer_plcc:.4f}"
        )

        results.append({
            "model": model_name,
            "layer": layer_idx,
            "srcc": layer_srcc,
            "plcc": layer_plcc
        })

    # NOVITÀ: Valutazione Media di tutti i layer
    # ref_features e dist_features hanno shape [n_layers, n_samples, features_dim]
    mean_ref = ref_features.mean(dim=0)   # shape diventa [n_samples, features_dim]
    mean_dist = dist_features.mean(dim=0)
    
    # IMPORTANTE: Ri-normalizzare le feature dopo la media
    mean_ref = F.normalize(mean_ref, p=2, dim=-1)
    mean_dist = F.normalize(mean_dist, p=2, dim=-1)
    
    mean_scores = cosine_similarity(mean_ref, mean_dist)
    mean_srcc = abs(srcc(mean_scores, mos))
    mean_plcc = abs(plcc(mean_scores, mos))
    
    print(f"Media di tutti i layer | SRCC={mean_srcc:.4f} | PLCC={mean_plcc:.4f}")
    
    results.append({
        "model": model_name,
        "layer": "mean",  # Usiamo la stringa "mean" per distinguerlo nel CSV
        "srcc": mean_srcc,
        "plcc": mean_plcc
    })
    
    return results


def main():
    all_results = []

    pt_files = sorted(FEATURES_DIR.glob("*.pt"))

    print(f"Trovati {len(pt_files)} file .pt")

    for pt_file in pt_files:
        try:
            all_results.extend(analyze_model(pt_file))
        except Exception as e:
            print(f"\nERRORE con {pt_file.name}")
            print(e)

    df = pd.DataFrame(all_results)

    #output_file = "phase2_pipal_results.csv"
    #output_file = "phase2_live_results.csv"
    output_file = "phase2_tid2013_results.csv"

    df.to_csv(output_file, index=False)

    print("\n=== RISULTATI SALVATI ===")
    print(df.head())

    print(f"\nCSV salvato in: {output_file}")


if __name__ == "__main__":
    main()