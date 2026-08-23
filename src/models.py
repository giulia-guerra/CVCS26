# Il codice definisce un Vision Encoder generico basato sui modelli pre-addestrati
# disponibili tramite Hugging Face. Il preprocessore e il modello vengono caricati
# automaticamente, mentre il backbone viene mantenuto congelato durante l'estrazione
# delle feature. Per ogni immagine vengono estratte le rappresentazioni di tutti i layer,
# utilizzando il CLS token per DINO e il Global Average Pooling per SigLIP, seguite da
# normalizzazione L2. Il MODEL_REGISTRY raccoglie infine tutti gli encoder DINOv2,
# DINOv3 e SigLIP2 utilizzati negli esperimenti del progetto.


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
            # FIX: Se il modello ha un sottomodello visivo separato (come SigLIP/CLIP), usiamo solo quello
            if hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model(**inputs, output_hidden_states=True)

        all_layer_features = []
        for layer_features in outputs.hidden_states:
            if "dinov" in self.model_name:
                # DINO: prendiamo il CLS token (indice 0)
                feat = layer_features[:, 0, :]
            elif "siglip" in self.model_name:
                # SigLIP: facciamo la media di tutti i patch (Global Average Pooling)
                feat = layer_features.mean(dim=1)
            else:
                feat = layer_features[:, 0, :]
            
            # Normalizzazione L2 su ogni layer (fondamentale per metriche percettive)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            all_layer_features.append(feat.cpu()) # Spostiamo su CPU per non finire la RAM della GPU

        # shape: [num_layers, batch_size, feature_dim]
        return torch.stack(all_layer_features)
    
# Dizionario dei modelli da valutare (come suggerito dal tutor)
MODEL_REGISTRY = {
    # Baseline richieste (DINOv2)
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base":  "facebook/dinov2-base",
    "dinov2_large": "facebook/dinov2-large",
    
    # Vision Encoder Moderni (Contributo principale della Fase 1)
    "dinov3_small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_base":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m", # Raccomandato per 24GB VRAM
    
    "siglip2_base": "google/siglip2-base-patch16-224", 
    "siglip2_large":"google/siglip2-large-patch16-256",          # Ottimo per il benchmark
}