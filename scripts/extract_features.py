import os
import torch
from src.models import VisionEncoder, MODEL_REGISTRY
from tqdm import tqdm

# Importiamo la factory della tua collega
from dataloader.dataset import get_dataloader 

# ottimizzati per Gpu da 24GB in su (RTX6000, A5000, A40, L40S)
BATCH_CONFIG = {
    "dinov2_small": 128,
    "dinov2_base": 64,
    "dinov2_large": 16,

    "dinov3_small": 128,
    "dinov3_base": 64,
    "dinov3_large": 16,

    "siglip2_base": 64,
    "siglip2_large": 16,
    "siglip2_large_384": 8,
}

DATASET_PATHS = {
    "PIPAL": "/work/cvcs2026/Cross_Entropy_Champions/datasets/PIPAL",
    "LIVE": "/work/cvcs2026/Cross_Entropy_Champions/datasets/LIVEIQA_release2",
    "TID2013": "/work/cvcs2026/Cross_Entropy_Champions/datasets/tid2013"
}

def extract_and_save(model_key, dataset_name):
    # Controlliamo che il percorso del dataset esista
    root_dir = DATASET_PATHS.get(dataset_name)
    if not os.path.exists(root_dir):
        print(f"Skipping {dataset_name}: percorso {root_dir} non trovato.")
        return

    # Pre salvataggio
    save_dir = f"/work/cvcs2026/Cross_Entropy_Champions/features/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{model_key}_all_layers.pt")
    if os.path.exists(save_path):
        print(f"✅ {save_path} già presente, salto.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCaricamento modello {model_key} su {device}...")
    encoder = VisionEncoder(MODEL_REGISTRY[model_key], device=device)
    
    model_batch_size = BATCH_CONFIG.get(model_key, 32) 
    
    # Inizializziamo il dataloader usando la funzione della tua collega
    print(f"Inizializzazione dataloader per {dataset_name}...")
    loader = get_dataloader(
        name=dataset_name, 
        root_dir=root_dir, 
        batch_size=model_batch_size, 
        shuffle=False, # FONDAMENTALE: shuffle=False per l'estrazione! Vogliamo l'ordine esatto
        transform=None # Lasciamo None così ci arrivano le PIL Image pure
    ) 
    
    ref_features_list = []
    dist_features_list = []
    mos_list = []
    names_list = []

    print(f"Inizio estrazione {dataset_name}...")
    
    # Iteriamo sul dataloader (ora restituisce il dizionario della collate_fn)
    with torch.inference_mode():
        for step, batch in enumerate(tqdm(loader, desc=f"{dataset_name} - {model_key}")):   # progress bar con tqdm
            # Estraiamo le liste di PIL Image e il tensore dei MOS
            ref_imgs_pil = batch["ref_image"]
            dist_imgs_pil = batch["dist_image"]
            mos_scores = batch["mos"]
            img_names = batch["name"]
            
            # Passiamo le PIL Image al modello (il processor dentro VisionEncoder fa la magia)
            # features shape: [num_layers, batch, dim]
            try:
                ref_feats = encoder(ref_imgs_pil)
                dist_feats = encoder(dist_imgs_pil)

            except torch.cuda.OutOfMemoryError:
                print(f"\n❌ CUDA Out of Memory con {model_key}")
                print(f"Prova a ridurre la batch size da {model_batch_size}.")
                raise
            
            ref_features_list.append(ref_feats)
            dist_features_list.append(dist_feats)
            mos_list.append(mos_scores) 
            names_list.extend(img_names)

    # Concateniamo tutto
    # shape features: [num_layers, total_images, dim]
    final_ref = torch.cat(ref_features_list, dim=1)
    final_dist = torch.cat(dist_features_list, dim=1)
    
    # shape mos: [total_images]
    final_mos = torch.cat(mos_list, dim=0)
    
    # Salvataggio    
    torch.save({
        'ref_features': final_ref,
        'dist_features': final_dist,
        'mos': final_mos,
        'image_names': names_list,
        'model_key': model_key,
        'model_name': MODEL_REGISTRY[model_key],
    }, save_path)
    
    print(f"✅ Salvato con successo in {save_path}")

    del encoder                
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Il tutor suggerisce di partire da PIPAL.
    datasets_to_run = ["PIPAL", "LIVE", "TID2013"]
    
    # Modelli da testare (assicurati di averli mappati in MODEL_REGISTRY dentro src.models)
    models_to_run = ["dinov2_small", "dinov2_base", "dinov2_large",
                     "dinov3_small", "dinov3_base", "dinov3_large",
                     "siglip2_base", "siglip2_large"]
    
    for ds in datasets_to_run:
        for m in models_to_run: 
            extract_and_save(m, ds)