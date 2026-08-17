# Moduli della Fase 3 (MLP, Cross-Attention)
import torch
import torch.nn as nn

class IQAFeatureAggregator(nn.Module):
    """
    Modulo MLP per l'ablazione (Punto 4.5).
    Permette di testare architetture di dimensioni diverse passando il parametro 'variant'.
    """
    def __init__(self, input_dim, variant="medium"):
        super().__init__()
        
        self.variant = variant
        
        if variant == "small":
            # Baseline leggerissima per evitare overfitting
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(128, 1)
            )
            
        elif variant == "medium":
            # Baseline standard (Quella da usare come riferimento principale)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 1)
            )
            
        elif variant == "large":
            # Multi-Layer MLP profondo (Punto 4.5)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            raise ValueError("variant deve essere 'small', 'medium', o 'large'")

    def forward(self, ref_features, dist_features):
        """
        ref_features e dist_features sono tensori bidimensionali (già flattened):
        Shape: [batch_size, input_dim]
        """
        # TECNICA IQA STANDARD: Calcolo della differenza assoluta.
        # Spinge la rete a concentrarsi su "cosa è cambiato" rispetto alla reference.
        diff = torch.abs(ref_features - dist_features)
        
        # Facoltativo (ma consigliato in IQA): puoi anche calcolare il prodotto element-wise
        # mult = ref_features * dist_features
        # e poi concatenarli: diff = torch.cat([diff, mult], dim=1) -> *Richiede di raddoppiare input_dim
        
        score = self.mlp(diff)
        return score.squeeze(-1) # Da [batch_size, 1] a [batch_size]


class DualEncoderFusion(nn.Module):
    """
    Modulo finale da richiamare nel Training Loop.
    Gestisce la fusione dei due modelli vincitori della Fase 1.
    """
    def __init__(self, dim_base=9984, dim_large=25600, variant="medium"):
        # dim_base = 13 layer * 768 (shape di siglip2_base_all_layers)
        # dim_large = N layer * dim (devi verificare la shape esatta del tuo siglip2_large)
        super().__init__()
        
        # La dimensione di input totale sarà la somma delle feature appiattite dei due modelli
        total_input_dim = dim_base + dim_large
        
        print(f"Inizializzazione DualEncoderFusion (Variant: {variant})")
        print(f"Input features totali: {total_input_dim}")
        
        self.aggregator = IQAFeatureAggregator(input_dim=total_input_dim, variant=variant)

    def forward(self, ref_base, dist_base, ref_large, dist_large):
        """
        I tensori in input provengono dal Dataloader di Giuli, estratti dai due file .pt.
        Shape attesa per ciascuno: [batch_size, num_layers, feature_dim]
        """
        # 1. Appiattiamo le feature di entrambi i modelli (da 3D a 2D)
        r_base_flat = ref_base.flatten(start_dim=1)
        d_base_flat = dist_base.flatten(start_dim=1)
        
        r_large_flat = ref_large.flatten(start_dim=1)
        d_large_flat = dist_large.flatten(start_dim=1)
        
        # 2. Concateniamo le feature di base e large
        # Shape finale: [batch_size, dim_base + dim_large]
        ref_combined = torch.cat([r_base_flat, r_large_flat], dim=1)
        dist_combined = torch.cat([d_base_flat, d_large_flat], dim=1)
        
        # 3. Passiamo tutto al nostro MLP
        predicted_mos = self.aggregator(ref_combined, dist_combined)
        
        return predicted_mos