# caricamento dei file .pt
# lettura delle feature per ogni dataset/model/layer:
# calcolo di SRCC/PLCC
# salvataggio dei risultati in un CSV

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.metrics import srcc, plcc
from src.metrics.similarity import cosine_similarity, l2_distance

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features" / "phase2"
RESULTS_DIR = ROOT / "results" / "phase2"
CSV_DIR = RESULTS_DIR / "csv"

CSV_DIR.mkdir(parents=True, exist_ok=True)


def load_pt_file(path: Path) -> Any:
    """
    Carica un file .pt.

    ATTENZIONE:
    Adatta questa funzione al formato reale dei file che ti passerà Anto.
    """
    import torch
    return torch.load(path, map_location="cpu")


def compute_layer_scores(sample_data: Any, metric: str = "cosine") -> list[float]:
    """
    Calcola uno score per ogni layer.

    ATTENZIONE:
    Qui devi adattare la logica al formato reale del .pt.
    L'idea è:
    - per ogni layer
    - prendi ref_feature e dist_feature
    - calcola una similarity/distance
    - restituisci una lista di score, uno per layer
    """
    layer_scores = []

    # ESEMPIO DI STRUTTURA ATTESA:
    # sample_data = {
    #     "layers": [
    #         {"ref": tensor(...), "dist": tensor(...)},
    #         {"ref": tensor(...), "dist": tensor(...)},
    #     ],
    #     "mos": 73.2,
    #     "name": "xxx"
    # }

    for layer in sample_data["layers"]:
        ref_feat = layer["ref"]
        dist_feat = layer["dist"]

        if metric == "cosine":
            score = cosine_similarity(ref_feat, dist_feat)
        elif metric == "l2":
            score = l2_distance(ref_feat, dist_feat)
        else:
            raise ValueError(f"Metric not supported: {metric}")

        layer_scores.append(float(score))

    return layer_scores


def analyze_file(pt_path: Path, dataset_name: str, model_name: str, metric: str = "cosine") -> pd.DataFrame:
    data = load_pt_file(pt_path)

    mos_values = []
    all_layer_scores = []

    # ESEMPIO ATTESO:
    # data = [
    #   {"layers": [...], "mos": 12.3, "name": "..."},
    #   {"layers": [...], "mos": 45.6, "name": "..."},
    # ]
    #
    # oppure:
    # data["samples"] = [...]

    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

    for sample in samples:
        mos = float(sample["mos"])
        layer_scores = compute_layer_scores(sample, metric=metric)

        mos_values.append(mos)
        all_layer_scores.append(layer_scores)

    all_layer_scores = list(map(list, zip(*all_layer_scores)))  # traspose: layer x samples

    rows = []
    for layer_idx, scores in enumerate(all_layer_scores):
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "layer": layer_idx,
                "metric": metric,
                "SRCC": srcc(mos_values, scores),
                "PLCC": plcc(mos_values, scores),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    results = []

    # Adatta questa lista ai nomi reali dei file .pt
    # Esempio:
    # features/phase2/live/dinov2_base.pt
    # features/phase2/live/siglip2_base.pt
    dataset_dirs = [p for p in FEATURES_DIR.iterdir() if p.is_dir()]

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        for pt_file in dataset_dir.glob("*.pt"):
            model_name = pt_file.stem
            df = analyze_file(pt_file, dataset_name, model_name, metric="cosine")
            results.append(df)

    if not results:
        raise RuntimeError("Nessun file .pt trovato in features/phase2/")

    final_df = pd.concat(results, ignore_index=True)
    out_csv = CSV_DIR / "phase2_results.csv"
    final_df.to_csv(out_csv, index=False)

    print(f"Salvato: {out_csv}")
    print(final_df.head())


if __name__ == "__main__":
    main()