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