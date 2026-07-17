# test_layers.py (evoluzione di test_integration)
import torch
from src.models import VisionEncoder, MODEL_REGISTRY
from PIL import Image
import numpy as np

def check_layers():
    encoder = VisionEncoder(MODEL_REGISTRY["dinov2_small"])
    dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    
    # Ora ci aspettiamo un tensore 3D: [Layer, Batch, Dim]
    features = encoder([dummy_img])
    
    print(f"Modello: {encoder.model_name}")
    print(f"Numero di layer estratti: {features.shape[0]}")
    print(f"Dimensione delle feature: {features.shape[2]}")
    
    # Verifica che l'ultimo layer sia diverso dal primo (segno che stiamo estraendo layer diversi)
    diff = torch.norm(features[0] - features[-1])
    print(f"Differenza tra primo e ultimo layer: {diff.item():.4f}")

if __name__ == "__main__":
    check_layers()