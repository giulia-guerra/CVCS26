# scripts/extract_features.py
import os
import torch
from src.models import VisionEncoder, MODEL_REGISTRY
# Assumi che la tua collega abbia creato un dataloader universale in src/datasets.py
from src.datasets import get_dataloader 

def extract_and_save(model_key, dataset_name):
    device = "cuda"
    encoder = VisionEncoder(MODEL_REGISTRY[model_key], device=device)
    loader = get_dataloader(dataset_name, batch_size=32) # Dataloader della collega
    
    features_list = []
    image_names = []

    print(f"Extracting {dataset_name} with {model_key}...")
    for batch_images, batch_names in loader:
        # features shape: [num_layers, batch, dim]
        features = encoder(batch_images)
        features_list.append(features)
        image_names.extend(batch_names)

    # Concateniamo tutto: [num_layers, total_images, dim]
    final_tensor = torch.cat(features_list, dim=1)
    
    save_dir = f"/work/cvcs2026/Cross_Entropy_Champions/features/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{model_key}_all_layers.pt")
    torch.save({
        'features': final_tensor,
        'image_names': image_names,
        'model_config': model_key
    }, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    for ds in ["LIVE", "TID2013", "PIPAL"]:
        for m in ["dinov2_small", "dinov2_base", "siglip2_base"]: # Inizia con questi
            extract_and_save(m, ds)