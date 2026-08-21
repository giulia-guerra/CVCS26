# Moduli della Fase 3 (MLP, Cross-Attention)
import torch
import torch.nn as nn


# ============================================================
# IQA FEATURE AGGREGATOR
# ============================================================

class IQAFeatureAggregator(nn.Module):
    """
    MLP regressor used for Phase 3.

    Input:
        Concatenated absolute difference between
        reference and distorted features from:

            - SigLIP2 Base
            - SigLIP2 Large

    Input shape:
        [batch_size, input_dim]

    Output:
        Predicted normalized MOS
        [batch_size]
    """

    def __init__(
        self,
        input_dim,
        variant="medium",
    ):
        super().__init__()

        self.variant = variant

        # ----------------------------------------------------
        # SMALL
        # ----------------------------------------------------

        if variant == "small":

            self.mlp = nn.Sequential(
                nn.Linear(
                    input_dim,
                    128,
                ),
                nn.ReLU(),
                nn.Dropout(
                    p=0.4
                ),
                nn.Linear(
                    128,
                    1,
                ),
            )

        # ----------------------------------------------------
        # MEDIUM
        # ----------------------------------------------------

        elif variant == "medium":

            self.mlp = nn.Sequential(
                nn.Linear(
                    input_dim,
                    256,
                ),
                nn.ReLU(),
                nn.Dropout(
                    p=0.4
                ),
                nn.Linear(
                    256,
                    64,
                ),
                nn.ReLU(),
                nn.Dropout(
                    p=0.2
                ),
                nn.Linear(
                    64,
                    1,
                ),
            )

        # ----------------------------------------------------
        # LARGE
        # ----------------------------------------------------

        elif variant == "large":

            self.mlp = nn.Sequential(
                nn.Linear(
                    input_dim,
                    512,
                ),
                nn.ReLU(),
                nn.Dropout(
                    p=0.5
                ),
                nn.Linear(
                    512,
                    128,
                ),
                nn.ReLU(),
                nn.Dropout(
                    p=0.3
                ),
                nn.Linear(
                    128,
                    32,
                ),
                nn.ReLU(),
                nn.Linear(
                    32,
                    1,
                ),
            )

        else:

            raise ValueError(
                "variant must be "
                "'small', 'medium', or 'large'"
            )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        ref_features,
        dist_features,
    ):
        """
        Compute absolute feature difference and
        predict MOS.

        Expected input:

            ref_features:
                [batch_size, input_dim]

            dist_features:
                [batch_size, input_dim]

        Returns:

            [batch_size]
        """

        # ----------------------------------------------------
        # Absolute reference-distorted difference
        # ----------------------------------------------------

        diff = torch.abs(
            ref_features - dist_features
        )

        # ----------------------------------------------------
        # MLP regression
        # ----------------------------------------------------

        score = self.mlp(
            diff
        )

        # [batch, 1] -> [batch]
        return score.squeeze(-1)


# ============================================================
# DUAL ENCODER FUSION
# ============================================================

class DualEncoderFusion(nn.Module):
    """
    Dual-encoder fusion model for Phase 3.

    The model receives all layers from:

        - SigLIP2 Base
        - SigLIP2 Large

    For each encoder:

        reference
            +
        distorted

    features are flattened and concatenated.

    Then the absolute difference between the
    reference and distorted representations is
    computed and passed to an MLP regressor.

    Example for PIPAL:

        Base:
            [batch, 13, 768]

        Large:
            [batch, 25, 1024]

        Base flattened:
            13 * 768 = 9984

        Large flattened:
            25 * 1024 = 25600

        Total:
            9984 + 25600 = 35584
    """

    def __init__(
        self,
        dim_base,
        dim_large,
        variant="medium",
    ):
        super().__init__()

        self.dim_base = dim_base
        self.dim_large = dim_large
        self.variant = variant

        # ----------------------------------------------------
        # Total input dimension
        # ----------------------------------------------------

        total_input_dim = (
            dim_base
            + dim_large
        )

        print(
            "Initializing DualEncoderFusion:"
        )

        print(
            f"  Variant:       {variant}"
        )

        print(
            f"  Base dim:      {dim_base}"
        )

        print(
            f"  Large dim:     {dim_large}"
        )

        print(
            f"  Total input:   {total_input_dim}"
        )

        # ----------------------------------------------------
        # MLP aggregator
        # ----------------------------------------------------

        self.aggregator = IQAFeatureAggregator(
            input_dim=total_input_dim,
            variant=variant,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        ref_base,
        dist_base,
        ref_large,
        dist_large,
    ):
        """
        Expected input shapes:

            ref_base:
                [batch, num_layers_base, feature_dim_base]

            dist_base:
                [batch, num_layers_base, feature_dim_base]

            ref_large:
                [batch, num_layers_large, feature_dim_large]

            dist_large:
                [batch, num_layers_large, feature_dim_large]
        """

        # ----------------------------------------------------
        # Flatten Base
        # ----------------------------------------------------

        ref_base_flat = (
            ref_base.flatten(
                start_dim=1
            )
        )

        dist_base_flat = (
            dist_base.flatten(
                start_dim=1
            )
        )

        # ----------------------------------------------------
        # Flatten Large
        # ----------------------------------------------------

        ref_large_flat = (
            ref_large.flatten(
                start_dim=1
            )
        )

        dist_large_flat = (
            dist_large.flatten(
                start_dim=1
            )
        )

        # ----------------------------------------------------
        # Concatenate encoders
        # ----------------------------------------------------

        ref_combined = torch.cat(
            [
                ref_base_flat,
                ref_large_flat,
            ],
            dim=1,
        )

        dist_combined = torch.cat(
            [
                dist_base_flat,
                dist_large_flat,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        expected_dim = (
            self.dim_base
            + self.dim_large
        )

        if ref_combined.shape[1] != expected_dim:

            raise ValueError(
                "Unexpected feature dimension: "
                f"expected {expected_dim}, "
                f"got {ref_combined.shape[1]}"
            )

        if dist_combined.shape[1] != expected_dim:

            raise ValueError(
                "Unexpected distorted feature "
                f"dimension: expected {expected_dim}, "
                f"got {dist_combined.shape[1]}"
            )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        predicted_mos = self.aggregator(
            ref_combined,
            dist_combined,
        )

        return predicted_mos

class AdvancedAttentionAggregator(nn.Module):
    """
    Modulo di Fusione Avanzato per lo Step 5 (Esperimenti Modello).
    Usa un'architettura in stile Transformer Encoder con un [CLS] token 
    imparabile per estrarre dinamicamente l'importanza dei layer.
    """
    def __init__(
        self, 
        dim_base=768, 
        dim_large=1024, # dimensione reale di Siglip2-Large
        proj_dim=256, 
        num_heads=4, 
        transformer_layers=1, 
        dropout=0.3
    ):
        super().__init__()
        
        print(f"Inizializzazione AdvancedAttentionAggregator (Heads: {num_heads}, Proj_Dim: {proj_dim})")
        
        # 1. Proiezioni lineari per portare i due modelli nello stesso spazio latente
        self.proj_base = nn.Linear(dim_base, proj_dim)
        self.proj_large = nn.Linear(dim_large, proj_dim)
        
        # 2. Il CLS Token imparabile (stile ViT/BERT)
        # Inizializzato in modo randomico, la rete imparerà come usarlo per "interrogare" i layer
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))
        
        # 3. Transformer Encoder (Self-Attention + Feed Forward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, 
            nhead=num_heads, 
            dim_feedforward=proj_dim * 2,
            batch_first=True, 
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 4. Head di regressione finale (Prende in input SOLO il CLS token)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )

    def forward(self, ref_base, dist_base, ref_large, dist_large):
        """
        Input attesi: Tensori 3D [batch_size, num_layers, feature_dim]
        (Giuli NON deve aver chiamato .flatten() sul dataloader per usare questo modello!)
        """
        # 1. Calcolo della differenza assoluta layer per layer
        diff_base = torch.abs(ref_base - dist_base)   
        diff_large = torch.abs(ref_large - dist_large) 
        
        # 2. Proiezione nello spazio comune
        p_base = self.proj_base(diff_base)     # [batch, num_layers_base, proj_dim]
        p_large = self.proj_large(diff_large)  # [batch, num_layers_large, proj_dim]
        
        # 3. Concatenazione dei token dei layer (Sequenza)
        # Se Base ha 13 layer e Large ne ha 26, tokens avrà shape [batch, 39, proj_dim]
        layer_tokens = torch.cat([p_base, p_large], dim=1)
        
        # 4. Aggiunta del CLS Token all'inizio della sequenza
        batch_size = layer_tokens.shape[0]
        # Espandiamo il CLS token per ogni elemento del batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [batch, 1, proj_dim]
        
        # Sequenza finale: [batch, 1 + 39, proj_dim]
        sequence = torch.cat([cls_tokens, layer_tokens], dim=1)
        
        # 5. Passaggio nel Transformer
        # L'Attenzione farà interagire il CLS token con tutti i layer e i layer tra di loro
        out_sequence = self.transformer(sequence)
        
        # 6. Estrazione del SOLO output del CLS token (indice 0)
        cls_out = out_sequence[:, 0, :] # [batch, proj_dim]
        
        # 7. Regressione a MOS
        score = self.head(cls_out)
        
        return score.squeeze(-1) # [batch_size]