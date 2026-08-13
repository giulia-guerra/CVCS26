import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import sys
from pathlib import Path

# Configura i path del tuo progetto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.metrics import srcc, plcc
from src.metrics.similarity import cosine_similarity

def load_and_concat_features(pt_file_path):
    """
    Carica un file .pt e concatena tutti i layer in un unico grande vettore per ogni immagine.
    """
    data = torch.load(pt_file_path, map_location="cpu")
    
    # Le feature hanno shape [n_layers, n_samples, embed_dim]
    # Spostiamo n_samples come prima dimensione: [n_samples, n_layers, embed_dim]
    ref = data["ref_features"].permute(1, 0, 2)
    dist = data["dist_features"].permute(1, 0, 2)
    mos = data["mos"]
    
    n_samples = ref.shape[0]
    
    # Flatten delle feature: uniamo i layer e l'embed_dim
    # La nuova shape sarà [n_samples, n_layers * embed_dim]
    ref_concat = ref.reshape(n_samples, -1)
    dist_concat = dist.reshape(n_samples, -1)
    
    return ref_concat, dist_concat, mos

def main():
    # Sostituisci il nome del file con il modello che stai analizzando (es. vincitore della Fase 1)
    
    model_file_name = "siglip2_base_all_layers.pt" 
    #model_file_name = "siglip2_large_all_layers.pt" 

    path_live = Path(f"/work/cvcs2026/Cross_Entropy_Champions/features/LIVE/{model_file_name}")
    path_tid = Path(f"/work/cvcs2026/Cross_Entropy_Champions/features/TID2013/{model_file_name}")
    path_pipal = Path(f"/work/cvcs2026/Cross_Entropy_Champions/features/PIPAL/{model_file_name}")

    print("1. Caricamento dati di TRAIN (LIVE + TID2013)...")
    ref_live, dist_live, _ = load_and_concat_features(path_live)
    ref_tid, dist_tid, _ = load_and_concat_features(path_tid)

    # Uniamo le feature di referenza di LIVE e TID2013 per allenare la PCA
    train_features = torch.cat([ref_live, ref_tid], dim=0)

    print(f"Shape dati di train per PCA: {train_features.shape}")
    
    # 2. Allenamento (Fit) della PCA
    print("2. Fittando la PCA a 256 dimensioni (puoi variare n_components)...")
    pca = PCA(n_components=256)
    pca.fit(train_features.numpy())

    print("3. Caricamento dati di TEST (PIPAL)...")
    ref_pipal, dist_pipal, mos_pipal = load_and_concat_features(path_pipal)

    # 3. Trasformazione delle feature di TEST
    print("4. Applicando la trasformazione PCA alle feature di PIPAL...")
    ref_pca = torch.tensor(pca.transform(ref_pipal.numpy()))
    dist_pca = torch.tensor(pca.transform(dist_pipal.numpy()))

    # 4. Ri-normalizzazione (Cruciale prima della Cosine Similarity)
    ref_pca = F.normalize(ref_pca, p=2, dim=-1)
    dist_pca = F.normalize(dist_pca, p=2, dim=-1)

    # 5. Calcolo metriche
    scores = cosine_similarity(ref_pca, dist_pca)
    final_srcc = srcc(scores, mos_pipal)
    final_plcc = plcc(scores, mos_pipal)

    print("\n=== RISULTATI PCA SU PIPAL ===")
    print(f"Modello: {model_file_name}")
    print(f"Dimensioni ridotte a: 256")
    print(f"SRCC: {final_srcc:.4f}")
    print(f"PLCC: {final_plcc:.4f}")

if __name__ == "__main__":
    main()