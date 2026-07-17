import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class VisionEncoder(nn.Module):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        # 1. Carica il preprocessore specifico per il modello
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # 2. Carica il modello e spostalo su GPU
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 3. Mettiamo il modello in eval mode e congeliamo i pesi (Backbone frozen)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        images: Lista di PIL Images o singola PIL Image.
        Ritorna: Tensore di shape [batch_size, feature_dim]
        """
        # Il processor gestisce resize, normalizzazione (mean/std) in automatico
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Chiediamo esplicitamente al modello di restituire tutti i layer per ottenere gli stati intermedi
            outputs = self.model(**inputs, output_hidden_states=True)

            # outputs.hidden_states è una tupla di (batch_size, sequence_length, hidden_size)
            # Ne abbiamo uno per ogni layer + l'embedding iniziale

        all_layer_features = []
        for layer_features in outputs.hidden_states:
            # Estraiamo il CLS token (indice 0) per ogni layer
            # E normalizziamo (facilita la vita a Persona A per la similarità)
            cls_token = layer_features[:, 0, :] 
            cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=-1)
            all_layer_features.append(cls_token.cpu()) # Spostiamo in CPU per risparmiare VRAM

        # Ritorna un tensore di shape [num_layers, batch_size, feature_dim]
        return torch.stack(all_layer_features)
    
# Dizionario dei modelli da valutare (come suggerito dal tutor)
MODEL_REGISTRY = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base":  "facebook/dinov2-base",
    "dinov2_large": "facebook/dinov2-large",
    # Nota: DINOv3 potrebbe avere nomi diversi su HF o richiedere timm, controlla i repo Meta ufficiali
    "siglip2_base": "google/siglip-base-patch16-224", # Sostituisci con i path esatti di SigLIP2 se diversi
    "siglip2_large":"google/siglip-large-patch16-256",
}