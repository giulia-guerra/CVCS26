# Il codice definisce i moduli di aggregazione e regressione utilizzati nella Phase 3.
# IQAFeatureAggregator utilizza un MLP per trasformare la differenza assoluta
# tra feature reference e distorted nel MOS predetto, con diverse varianti di complessità.
# DualEncoderFusion estende questo approccio combinando le feature di tutti i layer
# di SigLIP2 Base e Large. Infine, AdvancedAttentionAggregator utilizza proiezioni
# lineari e un Transformer Encoder per modellare le relazioni tra i layer dei due encoder
# prima della regressione finale del MOS.



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
        reference and distorted features.

    Input shape:
        [batch_size, input_dim]

    Output:
        Predicted MOS
        [batch_size]
    """

    def __init__(
        self,
        input_dim,
        variant="medium",
    ):
        super().__init__()

        self.input_dim = input_dim
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
        Compute absolute reference-distorted
        feature difference and predict MOS.

        Inputs:
            ref_features:
                [B, input_dim]

            dist_features:
                [B, input_dim]

        Returns:
            [B]
        """

        if ref_features.shape != dist_features.shape:
            raise ValueError(
                "Reference and distorted features "
                "must have the same shape. "
                f"Got {ref_features.shape} and "
                f"{dist_features.shape}."
            )

        if ref_features.ndim != 2:
            raise ValueError(
                "IQAFeatureAggregator expects "
                "2D tensors [B, input_dim]. "
                f"Got shape {ref_features.shape}."
            )

        if ref_features.shape[1] != self.input_dim:
            raise ValueError(
                "Unexpected input dimension. "
                f"Expected {self.input_dim}, "
                f"got {ref_features.shape[1]}."
            )

        # ----------------------------------------------------
        # Absolute difference
        # ----------------------------------------------------

        diff = torch.abs(
            ref_features - dist_features
        )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        score = self.mlp(diff)

        return score.squeeze(-1)


# ============================================================
# DUAL ENCODER FUSION
# ============================================================

class DualEncoderFusion(nn.Module):
    """
    Dual-encoder MLP fusion model.

    This model receives all layers from:

        - SigLIP2 Base
        - SigLIP2 Large

    Expected inputs:

        ref_base:
            [B, L_base, D_base]

        dist_base:
            [B, L_base, D_base]

        ref_large:
            [B, L_large, D_large]

        dist_large:
            [B, L_large, D_large]

    The layer dimensions must be provided explicitly.

    Example:

        Base:
            L_base = 13
            D_base = 768

        Large:
            L_large = 25
            D_large = 1024

    Then:

        Base flattened:
            13 * 768 = 9984

        Large flattened:
            25 * 1024 = 25600

        Total:
            35584
    """

    def __init__(
        self,
        dim_base,
        dim_large,
        num_layers_base,
        num_layers_large,
        variant="medium",
    ):
        super().__init__()

        self.dim_base = dim_base
        self.dim_large = dim_large

        self.num_layers_base = num_layers_base
        self.num_layers_large = num_layers_large

        self.variant = variant

        # ----------------------------------------------------
        # Total input dimension
        # ----------------------------------------------------

        base_input_dim = (
            num_layers_base
            * dim_base
        )

        large_input_dim = (
            num_layers_large
            * dim_large
        )

        total_input_dim = (
            base_input_dim
            + large_input_dim
        )

        print(
            "Initializing DualEncoderFusion:"
        )

        print(
            f"  Variant:          {variant}"
        )

        print(
            f"  Base layers:      {num_layers_base}"
        )

        print(
            f"  Base dimension:   {dim_base}"
        )

        print(
            f"  Base flattened:   {base_input_dim}"
        )

        print(
            f"  Large layers:     {num_layers_large}"
        )

        print(
            f"  Large dimension:  {dim_large}"
        )

        print(
            f"  Large flattened:  {large_input_dim}"
        )

        print(
            f"  Total input:      {total_input_dim}"
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
        Forward pass.

        Inputs:

            ref_base:
                [B, L_base, D_base]

            dist_base:
                [B, L_base, D_base]

            ref_large:
                [B, L_large, D_large]

            dist_large:
                [B, L_large, D_large]

        Returns:

            predicted MOS:
                [B]
        """

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        if ref_base.shape != dist_base.shape:
            raise ValueError(
                "Base reference and distorted tensors "
                "must have the same shape."
            )

        if ref_large.shape != dist_large.shape:
            raise ValueError(
                "Large reference and distorted tensors "
                "must have the same shape."
            )

        if ref_base.ndim != 3:
            raise ValueError(
                "Expected Base tensors with shape "
                "[B, L_base, D_base]. "
                f"Got {ref_base.shape}."
            )

        if ref_large.ndim != 3:
            raise ValueError(
                "Expected Large tensors with shape "
                "[B, L_large, D_large]. "
                f"Got {ref_large.shape}."
            )

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if ref_base.shape[1] != self.num_layers_base:
            raise ValueError(
                "Unexpected number of Base layers. "
                f"Expected {self.num_layers_base}, "
                f"got {ref_base.shape[1]}."
            )

        if ref_base.shape[2] != self.dim_base:
            raise ValueError(
                "Unexpected Base feature dimension. "
                f"Expected {self.dim_base}, "
                f"got {ref_base.shape[2]}."
            )

        if ref_large.shape[1] != self.num_layers_large:
            raise ValueError(
                "Unexpected number of Large layers. "
                f"Expected {self.num_layers_large}, "
                f"got {ref_large.shape[1]}."
            )

        if ref_large.shape[2] != self.dim_large:
            raise ValueError(
                "Unexpected Large feature dimension. "
                f"Expected {self.dim_large}, "
                f"got {ref_large.shape[2]}."
            )

        # ----------------------------------------------------
        # Flatten Base
        # ----------------------------------------------------

        ref_base_flat = ref_base.flatten(
            start_dim=1
        )

        dist_base_flat = dist_base.flatten(
            start_dim=1
        )

        # ----------------------------------------------------
        # Flatten Large
        # ----------------------------------------------------

        ref_large_flat = ref_large.flatten(
            start_dim=1
        )

        dist_large_flat = dist_large.flatten(
            start_dim=1
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
        # Final dimension check
        # ----------------------------------------------------

        expected_dim = (
            self.num_layers_base
            * self.dim_base
            +
            self.num_layers_large
            * self.dim_large
        )

        if ref_combined.shape[1] != expected_dim:
            raise ValueError(
                "Unexpected combined reference "
                "feature dimension. "
                f"Expected {expected_dim}, "
                f"got {ref_combined.shape[1]}."
            )

        if dist_combined.shape[1] != expected_dim:
            raise ValueError(
                "Unexpected combined distorted "
                "feature dimension. "
                f"Expected {expected_dim}, "
                f"got {dist_combined.shape[1]}."
            )

        # ----------------------------------------------------
        # MLP regression
        # ----------------------------------------------------

        predicted_mos = self.aggregator(
            ref_combined,
            dist_combined,
        )

        return predicted_mos


# ============================================================
# ADVANCED ATTENTION AGGREGATOR
# ============================================================

class AdvancedAttentionAggregator(nn.Module):
    """
    Advanced Transformer-based IQA aggregator.

    This model keeps the encoder layers as individual
    tokens instead of flattening them.

    Inputs:

        ref_base:
            [B, L_base, 768]

        dist_base:
            [B, L_base, 768]

        ref_large:
            [B, L_large, 1024]

        dist_large:
            [B, L_large, 1024]

    Processing:

        1. Compute absolute reference-distorted difference.
        2. Project Base features to proj_dim.
        3. Project Large features to proj_dim.
        4. Concatenate Base and Large layer tokens.
        5. Add learnable CLS token.
        6. Apply Transformer Encoder.
        7. Use CLS representation for regression.

    Output:

        predicted MOS:
            [B]
    """

    def __init__(
        self,
        dim_base=768,
        dim_large=1024,
        proj_dim=256,
        num_heads=4,
        transformer_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Save configuration
        # ----------------------------------------------------

        self.dim_base = dim_base
        self.dim_large = dim_large
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.dropout = dropout

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if proj_dim % num_heads != 0:
            raise ValueError(
                "proj_dim must be divisible by "
                "num_heads. "
                f"Got proj_dim={proj_dim}, "
                f"num_heads={num_heads}."
            )

        if transformer_layers <= 0:
            raise ValueError(
                "transformer_layers must be > 0."
            )

        # ----------------------------------------------------
        # Base projection
        # ----------------------------------------------------

        self.proj_base = nn.Linear(
            dim_base,
            proj_dim,
        )

        # ----------------------------------------------------
        # Large projection
        # ----------------------------------------------------

        self.proj_large = nn.Linear(
            dim_large,
            proj_dim,
        )

        # ----------------------------------------------------
        # Learnable CLS token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                proj_dim,
            )
        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=num_heads,
                dim_feedforward=proj_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
        )

        # ----------------------------------------------------
        # Regression head
        # ----------------------------------------------------

        self.head = nn.Sequential(
            nn.Linear(
                proj_dim,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                128,
                1,
            ),
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
        Forward pass.

        Inputs:

            ref_base:
                [B, L_base, D_base]

            dist_base:
                [B, L_base, D_base]

            ref_large:
                [B, L_large, D_large]

            dist_large:
                [B, L_large, D_large]

        Returns:

            predicted MOS:
                [B]
        """

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        if ref_base.shape != dist_base.shape:
            raise ValueError(
                "Base reference and distorted "
                "features must have identical shapes. "
                f"Got {ref_base.shape} and "
                f"{dist_base.shape}."
            )

        if ref_large.shape != dist_large.shape:
            raise ValueError(
                "Large reference and distorted "
                "features must have identical shapes. "
                f"Got {ref_large.shape} and "
                f"{dist_large.shape}."
            )

        if ref_base.ndim != 3:
            raise ValueError(
                "Base features must have shape "
                "[B, L, D]. "
                f"Got {ref_base.shape}."
            )

        if ref_large.ndim != 3:
            raise ValueError(
                "Large features must have shape "
                "[B, L, D]. "
                f"Got {ref_large.shape}."
            )

        # ----------------------------------------------------
        # Feature dimension checks
        # ----------------------------------------------------

        if ref_base.shape[-1] != self.dim_base:
            raise ValueError(
                "Unexpected Base feature dimension. "
                f"Expected {self.dim_base}, "
                f"got {ref_base.shape[-1]}."
            )

        if ref_large.shape[-1] != self.dim_large:
            raise ValueError(
                "Unexpected Large feature dimension. "
                f"Expected {self.dim_large}, "
                f"got {ref_large.shape[-1]}."
            )

        # ----------------------------------------------------
        # 1. Reference / distorted difference
        # ----------------------------------------------------

        diff_base = torch.abs(
            ref_base - dist_base
        )

        diff_large = torch.abs(
            ref_large - dist_large
        )

        # ----------------------------------------------------
        # 2. Project Base and Large to common space
        # ----------------------------------------------------

        base_tokens = self.proj_base(
            diff_base
        )

        large_tokens = self.proj_large(
            diff_large
        )

        # ----------------------------------------------------
        # 3. Concatenate layer tokens
        # ----------------------------------------------------

        layer_tokens = torch.cat(
            [
                base_tokens,
                large_tokens,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # 4. Add CLS token
        # ----------------------------------------------------

        batch_size = layer_tokens.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        sequence = torch.cat(
            [
                cls_tokens,
                layer_tokens,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # 5. Transformer Encoder
        # ----------------------------------------------------

        output = self.transformer(
            sequence
        )

        # ----------------------------------------------------
        # 6. CLS representation
        # ----------------------------------------------------

        cls_output = output[
            :,
            0,
            :,
        ]

        # ----------------------------------------------------
        # 7. Regression head
        # ----------------------------------------------------

        score = self.head(
            cls_output
        )

        # ----------------------------------------------------
        # Output [B]
        # ----------------------------------------------------

        return score.squeeze(-1)