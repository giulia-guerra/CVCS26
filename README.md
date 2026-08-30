# Beyond the Pixel Limit

Computer Vision and Cognitive Systems Project 2026

## Team Contributions

Antonio D'agata, matricola 218794
Giulia Guerra,   matricola 214065 

### Data Pipeline and Evaluation

* Dataset loaders for LIVE, TID2013 and PIPAL.
* Full-reference IQA sample generation.
* Evaluation metrics (SRCC, PLCC, Cosine Similarity, L2 Distance).
* CSV logging utilities.
* Dataset validation and automated testing.
* Intermediate-layer analysis and result aggregation.

### Vision Encoder Integration

* DINOv2 encoder family integration.
* DINOv3 encoder family integration.
* SigLIP2 encoder family integration.
* Feature extraction pipeline.
* Generation of intermediate-layer feature representations used throughout the experiments.

### Experimental Results

The project successfully evaluated multiple vision encoder families across three IQA datasets (LIVE, TID2013 and PIPAL), identifying the most informative intermediate layers for perceptual quality prediction.


## Repository structure

* dataloader/
* datasets/
* results/
* scripts/
* scripts_slurm/
* src/
* tests/


---

## Dataset Pipeline

Supported datasets:

* LIVE
* TID2013
* PIPAL

Each sample returns:

```python
{
    "reference_image": ...,
    "distorted_image": ...,
    "mos": ...,
    "image_name": ...
}
```

Tests:

```bash
pytest tests/
```

---

## Phase 1 – Benchmark Pipeline

The objective of Phase 1 was to build and validate the complete IQA evaluation pipeline.

Implemented components:

* Dataset loaders for LIVE, TID2013 and PIPAL.
* Support for Full-Reference IQA samples (reference + distorted image).
* Similarity metrics:
   - Cosine Similarity
   - L2 Distance

* Correlation metrics:
  - SRCC (Spearman Rank Correlation Coefficient)
  - PLCC (Pearson Linear Correlation Coefficient)

* CSV logging utilities.
* Unit tests for datasets, metrics and feature processing.

Outcome:

* All dataset loaders were successfully validated.
* Feature extraction pipeline was integrated with the vision encoders.
* Evaluation metrics were verified through automated tests.

---

## Phase 2 – Intermediate Layer Analysis

The objective of Phase 2 was to investigate which intermediate layers of the vision encoders provide features that are most correlated with human perceptual quality.

Evaluated encoder families:

* DINOv2
  - Small
  - Base
  - Large

* DINOv3
  - Small
  - Base
  - Large

* SigLIP2
  - Base
  - Large

Datasets:
* LIVE
* TID2013
* PIPAL

For each model and each layer:

1. Features were extracted.
2. Similarity scores were computed.
3. SRCC and PLCC were measured against MOS.
4. The best-performing layer was identified.

### Main Results

#### PIPAL

Best performance:

| Model        | Best Layer | SRCC  | PLCC  |
| ------------ | ---------- | ----- | ----- |
| SigLIP2 Base | 7          | 0.579 | 0.595 |
| DINOv2 Small | 2          | 0.579 | 0.508 |
| DINOv3 Small | 2          | 0.570 | 0.518 |

Observations:

* Early and intermediate layers consistently performed better than deeper layers.
* SigLIP2 Base achieved the highest PLCC.

#### TID2013

Best performance:

| Model         | Best Layer | SRCC  | PLCC  |
| ------------- | ---------- | ----- | ----- |
| SigLIP2 Large | 20         | 0.804 | 0.676 |
| SigLIP2 Base  | 11         | 0.803 | 0.667 |
| DINOv3 Small  | 12         | 0.772 | 0.702 |

Observations:

* TID2013 produced the strongest correlations overall.
* SigLIP2 models consistently outperformed DINO variants.
* Intermediate layers were generally more informative than the final layers.

#### LIVE

Best performance:

| Model                    | Best Layer | SRCC  | PLCC  |
| ------------------------ | ---------- | ----- | ----- |
| DINOv3 Small             | 4          | 0.964 | 0.722 |
| DINOv2 Base              | 4          | 0.963 | 0.727 |
| SigLIP2 Base             | 8          | 0.962 | 0.626 |
| DINOv2 Large             | 9          | 0.954 | 0.760 |

Observations:

* All evaluated models achieved very high rank correlations (SRCC > 0.95).
* DINOv3 Small obtained the highest SRCC (0.964), closely followed by DINOv2 Base (0.963).
* The best PLCC was achieved by DINOv2 Large (0.760), indicating the strongest linear relationship with MOS scores.
* Most models reached their best performance in intermediate layers rather than in the final layer.
* DINO-based models consistently outperformed SigLIP2 models in terms of PLCC, while SRCC values were comparable across all architectures.

### Outputs

Final comparison table:

```text
results/phase2/tables/final_phase2_comparison.csv
```

#### Best Layer per Encoder and Dataset

| Dataset | Model         | Best Layer | SRCC  | PLCC  |
| ------- | ------------- | ---------- | ----- | ------|
| LIVE    | DINOv2 Base   | 4          | 0.963 | 0.727 |
| LIVE    | DINOv2 Large  | 9          | 0.954 | 0.760 |
| LIVE    | DINOv2 Small  | 4          | 0.959 | 0.720 |
| LIVE    | DINOv3 Base   | 4          | 0.960 | 0.759 |
| LIVE    | DINOv3 Large  | 6          | 0.956 | 0.665 |
| LIVE    | DINOv3 Small  | 4          | 0.964 | 0.722 |
| LIVE    | SigLIP2 Base  | 8          | 0.962 | 0.626 |
| LIVE    | SigLIP2 Large | 18         | 0.960 | 0.672 |
| ------- | ------------- | ---------- | ----- | ------|
| PIPAL   | DINOv2 Small  | 2          | 0.579 | 0.508 |
| PIPAL   | DINOv2 Base   | 1          | 0.578 | 0.486 |
| PIPAL   | DINOv2 Large  | 1          | 0.554 | 0.435 |
| PIPAL   | DINOv3 Base   | 1          | 0.539 | 0.406 |
| PIPAL   | DINOv3 Large  | 1          | 0.565 | 0.442 |
| PIPAL   | DINOv3 Small  | 2          | 0.570 | 0.518 |
| PIPAL   | SigLIP2 Large | 15         | 0.564 | 0.576 |
| PIPAL   | SigLIP2 Base  | 7          | 0.579 | 0.595 |
| ------- | ------------- | ---------- | ----- | ------|
| TID2013 | DINOv2 Base   | 12         | 0.709 | 0.656 |
| TID2013 | DINOv2 Large  | 5          | 0.691 | 0.667 |
| TID2013 | DINOv2 Small  | 12         | 0.759 | 0.692 |
| TID2013 | DINOv3 Base   | 12         | 0.730 | 0.679 |
| TID2013 | DINOv3 Large  | 24         | 0.708 | 0.639 |
| TID2013 | DINOv3 Small  | 12         | 0.772 | 0.702 |
| TID2013 | SigLIP2 Base  | 11         | 0.803 | 0.667 |
| TID2013 | SigLIP2 Large | 20         | 0.804 | 0.676 |

#### Main Findings

* The best-performing layer varies significantly across encoders and datasets, confirming that intermediate representations can be more informative than final-layer features for IQA.

* On LIVE, all evaluated encoders achieved very high SRCC values (> 0.95), with DINOv3 Small obtaining the highest SRCC (0.964). This indicates that the learned representations are highly correlated with perceived image quality on this dataset.

 * On PIPAL, the strongest performance was achieved by SigLIP2 Base (SRCC = 0.579, PLCC = 0.595), closely followed by DINOv2 Small. The lower overall correlations compared with LIVE suggest that PIPAL represents a more challenging scenario for direct feature-similarity-based IQA.

* On TID2013, SigLIP2 Large achieved the highest SRCC (0.804), while DINOv3 Small obtained the best PLCC among the DINO models (0.702), showing that encoder performance is strongly dependent on the dataset and distortion characteristics.

 * In several encoder-dataset combinations, the best intermediate layer outperformed the final-layer representation. This supports the hypothesis that the final encoder embedding is not necessarily the most informative representation for IQA.

* The optimal layer is highly dataset-dependent, suggesting that different distortion types and image-quality distributions benefit from representations at different levels of abstraction. Therefore, there is no single universally optimal layer across all encoders and datasets.

* The differences between SRCC and PLCC further indicate that feature similarity and perceived image quality do not always follow a purely linear relationship.

* The selected best layers will be used in Phase 3 to extract features for supervised MOS prediction. For each encoder-dataset pair, the layer identified during Phase 2 will be used to construct the feature representation provided to the supervised regression model.

* Pooling Strategy and Representation Structure (CLS vs. Patch Mean): SigLIP2 achieves particularly strong performance on TID2013 and PIPAL compared with the evaluated DINO variants. One possible explanation is the difference in representation aggregation strategies. DINO relies on a global [CLS] token representation, whereas SigLIP2 uses Global Average Pooling over patch tokens. This may help preserve information distributed across local image regions that is useful for perceptual quality assessment. However, the current experiments do not independently isolate pooling strategy, architecture, and pretraining objective, so the observed differences should be interpreted as an association rather than as evidence of a direct causal relationship.
 
* Layer Trends: The observed results suggest that different encoder families distribute perceptual information differently across network depth. In many cases, intermediate layers outperform the final representation, supporting the hypothesis that low-level distortion-sensitive information may become less prominent in highly semantic representations. The exact trend varies across datasets and encoder architectures.

Phase 2 successfully identified the most informative layer for each encoder and dataset. The results demonstrate that layer selection is an important component of the IQA pipeline and provide the basis for the supervised learning stage developed in Phase 3.


---

## Phase 3 – Supervised Learning & Regression

Phase 3 extends the IQA pipeline from feature-based correlation analysis to supervised learning.

The goal is to learn a regression function that maps encoder features to Mean Opinion Scores (MOS).

### Implemented components

* PyTorch supervised training pipeline.
* MLP regression head on top of frozen encoder features.
* Train/validation split with fixed random seed.
* Feature normalization computed from the training set only.
* Dataset-specific MOS normalization.
* MSE loss.
* SRCC and PLCC evaluation during training.
* Best-checkpoint selection based on validation SRCC.
* Early stopping.
* GPU support and SLURM-compatible training scripts.
* Checkpoint and prediction saving.
* Dedicated evaluation script for trained models.

### Dual SigLIP2 Baseline

The first Phase 3 experiment uses a dual-encoder architecture combining SigLIP2 Base and SigLIP2 Large features.

| Model            | Variant   | Best Epoch | MSE     | SRCC       | PLCC       |
| ---------------- | --------- | ---------- | ------- | ---------- | ---------- |
| Dual SigLIP2     | Small     | 15         | 4759.99 | 0.7952     | 0.8322     |
| Dual SigLIP2     | Medium    | 13         | 5411.94 | 0.8173     | 0.8410     |
| **Dual SigLIP2** | **Large** | **15**     | 6993.55 | **0.8222** | **0.8521** |

The **Dual SigLIP2 Large** configuration achieved the best Phase 3 baseline performance, with **SRCC = 0.8222** and **PLCC = 0.8521** on the PIPAL validation split.

---

## Phase 3 – LIVE + TID2013 Mixture Ablation

To evaluate the robustness of the regression architecture across datasets, a second experiment was performed by jointly training on **LIVE and TID2013**.

The experiment combines frozen SigLIP2 Base and SigLIP2 Large features and compares three regression-head capacities:

* **Small**
* **Medium**
* **Large**

All variants use the same feature extraction pipeline, train/validation split, normalization strategy, optimizer settings, and random seed. The only variable is the capacity of the regression head.

### Mixture Ablation Results

| Variant    | Mixed SRCC | Mixed PLCC | LIVE SRCC  | LIVE PLCC  | TID2013 SRCC | TID2013 PLCC |
| ---------- | ---------- | ---------- | ---------- | ---------- | ------------ | ------------ |
| **Medium** | **0.7500** | **0.8994** | 0.7843     | 0.8021     | **0.8542**   | **0.8949**   |
| Small      | 0.7301     | 0.8925     | **0.7921** | **0.8064** | 0.8070       | 0.8630       |
| Large      | 0.6250     | 0.5822     | -0.9598    | -0.8649    | 0.8117       | 0.8127       |

The **Medium** variant achieves the best overall performance on the mixed validation set, reaching **SRCC = 0.7500** and **PLCC = 0.8994**.

The Small variant performs slightly better on LIVE, while the Medium variant provides the best results on TID2013. The Large variant performs substantially worse and shows unstable training behavior: its best checkpoint was reached at **epoch 2**, after which validation performance deteriorated and early stopping was triggered.

### Main Findings

The Phase 3 experiments show that increasing the capacity of the regression head does not necessarily lead to better IQA performance.

The results suggest that the **Medium architecture provides the best overall trade-off between model capacity and generalization**, while the Large configuration can become unstable when trained on the mixed LIVE + TID2013 setting.

Overall, Phase 3 demonstrates that frozen visual encoder features can be successfully mapped to perceptual quality scores using a relatively lightweight supervised regression head.

## Phase 3 – Advanced Architectures and Ablation Studies

The advanced Phase 3 experiments extend the baseline regression approach by exploiting multi-layer representations and combining features from different SigLIP2 encoder sizes.

The objective is to investigate whether richer feature representations and attention-based fusion can improve perceptual quality prediction compared with the standard Phase 3 MLP baseline.

### Advanced Architecture

The main advanced model is the `AdvancedAttentionAggregator`.

It combines:

- SigLIP2 Base multi-layer features.
- SigLIP2 Large multi-layer features.
- Independent linear projections for Base and Large features.
- Multi-layer feature aggregation.
- A learnable CLS token.
- Transformer-based self-attention.
- Multi-head attention.
- A lightweight MLP regression head.

The architecture receives reference and distorted image features from both encoders and learns to aggregate information across encoder layers and encoder scales before predicting the MOS.

Unlike the baseline model, which operates on a selected feature representation, the advanced architecture exploits information from multiple layers and uses attention mechanisms to learn how these representations should be combined.

### Advanced PIPAL Experiment

The first advanced experiment was performed on PIPAL using:

- SigLIP2 Base all-layer features.
- SigLIP2 Large all-layer features.
- Joint attention-based feature aggregation.
- Frozen vision encoders.
- Supervised MOS regression.

The model uses a Transformer-based attention module to learn which feature representations and encoder layers are most informative for perceptual quality prediction.

Training configuration:

- Projection dimension: 256
- Transformer layers: 1
- Attention heads: 4
- Hidden dimension of regression head: 128
- Dropout: 0.3
- Optimizer: AdamW
- Loss: MSE
- Validation ratio: 0.2
- Random seed: 42
- Early stopping based on validation SRCC

### Advanced PIPAL Results

The advanced attention-based architecture was compared directly with the Phase 3 baseline on the PIPAL validation set.

| Model                       | Dataset   | MSE         | SRCC       | PLCC       |
| --------------------------- | --------- | ----------- | ---------- | ---------- |
| Phase 3 MLP Medium Baseline | PIPAL     | 6566.77     | 0.7214     | 0.7477     |
| **Advanced Attention**      | **PIPAL** | **4752.39** | **0.8131** | **0.8349** |

The **Advanced Attention** model achieves the best performance relative to the Phase 3 MLP Medium baseline for all three evaluation metrics.

Compared with the Phase 3 MLP Medium baseline:

| Metric | Baseline | Advanced    | Improvement      |
| ------ | -------- | ----------- | ---------------- |
| SRCC   | 0.7214   | **0.8131**  | **+12.71%**      |
| PLCC   | 0.7477   | **0.8349**  | **+11.65%**      |
| MSE    | 6566.77  | **4752.39** | **27.63% lower** |

The advanced model improves both rank and linear correlation with human perceptual quality while simultaneously reducing the prediction error.

The comparison is performed against the Phase 3 MLP Medium baseline. While the Advanced Attention architecture substantially improves over this baseline, the Dual SigLIP2 Large configuration achieves slightly higher SRCC and PLCC values (0.8222 and 0.8521 respectively). Therefore, the Advanced Attention model should not be interpreted as universally outperforming every Phase 3 configuration across all metrics.

In particular:

- **SRCC increases from 0.7214 to 0.8131**, indicating stronger agreement with the ranking of perceived image quality.
- **PLCC increases from 0.7477 to 0.8349**, indicating a stronger linear relationship between predicted and ground-truth MOS.
- **MSE decreases from 6566.77 to 4752.39**, corresponding to a **27.63% reduction in prediction error**.

These results demonstrate that multi-layer attention-based fusion of SigLIP2 Base and Large features provides a substantial improvement over the standard Phase 3 regression baseline on PIPAL.

### LIVE + TID2013 Mixture Experiment

A second experiment investigates whether the dual-encoder regression architecture generalizes across different IQA datasets.

The model is trained jointly on:

- LIVE
- TID2013

The experiment uses:

- SigLIP2 Base features.
- SigLIP2 Large features.
- Dataset-stratified train/validation splitting.
- Feature normalization computed only from the training set.
- Dataset-specific MOS normalization.
- Separate evaluation on LIVE and TID2013.
- Evaluation on the combined validation set.

This experiment evaluates whether the regression architecture can learn a common mapping from visual representations to perceptual quality across datasets with different distortion characteristics and MOS distributions.

### Mixture Ablation

A regression-head ablation was performed by comparing three `DualEncoderFusion` variants:

- `small`
- `medium`
- `large`

The three configurations use the same feature extraction pipeline, dataset split, normalization strategy, optimizer and random seed. The main variable is the capacity of the fusion/regression architecture.

The comparison evaluates:

- Mixed-dataset SRCC.
- Mixed-dataset PLCC.
- Mixed-dataset MSE.
- LIVE SRCC and PLCC.
- TID2013 SRCC and PLCC.

The **Medium** configuration provided the best overall trade-off between model capacity and generalization on the mixed LIVE + TID2013 validation set.

The Small configuration achieved slightly better performance on LIVE, while the Medium configuration provided the strongest overall performance on the mixed validation set and the best results on TID2013.

The Large configuration showed unstable validation behavior and substantially worse mixed-dataset performance. This indicates that increasing model capacity does not necessarily improve IQA performance and may lead to poorer generalization.

### Evaluation Metrics

The advanced experiments use:

- **SRCC** – Spearman Rank Correlation Coefficient
- **PLCC** – Pearson Linear Correlation Coefficient
- **MSE** – Mean Squared Error

Higher SRCC and PLCC indicate better agreement with human perceptual quality rankings, while lower MSE indicates more accurate MOS prediction.

### Outputs

The advanced experiments generate:

- Trained model checkpoints.
- Best-model checkpoints selected according to validation SRCC.
- Training history CSV files.
- Prediction CSV files.
- SRCC, PLCC and MSE results.
- Ablation tables.
- Comparison plots.
- Baseline vs. advanced model comparisons.

### Main Findings

The complete Phase 3 experiments show that:

- Frozen vision encoder features can be successfully used for supervised IQA regression.
- Combining SigLIP2 Base and Large representations can improve performance over simpler feature-based approaches.
- Multi-layer representations provide richer information than relying exclusively on a single encoder layer.
- Attention-based aggregation allows the model to learn which representations are most relevant for perceptual quality prediction.
- The **Advanced Attention** architecture substantially outperforms the Phase 3 MLP Medium baseline on PIPAL.
- Compared with the baseline, the Advanced Attention model improves **SRCC by 12.71%** and **PLCC by 11.65%**, while reducing **MSE by 27.63%**.
- The improvement across all three metrics indicates that the advanced architecture provides both stronger correlation with human perceptual judgments and more accurate MOS prediction.
- Increasing regression-head capacity does not necessarily improve generalization in the LIVE + TID2013 mixture experiment.
- The **Medium** `DualEncoderFusion` configuration provides the best overall trade-off between model capacity and generalization in the mixture setting.
- The Large configuration can become unstable when trained on the mixed LIVE + TID2013 setting.
- The optimal architecture depends on the dataset and experimental setting.
- Dynamic vs. Static Aggregation: While the standard MLP baseline applies static learned weights to the complete feature representation, the Advanced Attention Aggregator introduces a learnable [CLS] token and a structured information bottleneck. Through attention-based aggregation, the model can dynamically combine information from multiple encoder layers. This design may help reduce overfitting and contributes to the improved validation performance observed on PIPAL.


Overall, the advanced Phase 3 experiments demonstrate that exploiting **multi-layer visual representations and attention-based fusion** can significantly improve perceptual quality prediction compared with a simpler regression baseline.