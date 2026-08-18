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
  * Cosine Similarity
  * L2 Distance

* Correlation metrics:
  * SRCC (Spearman Rank Correlation Coefficient)
  * PLCC (Pearson Linear Correlation Coefficient)

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
  * Small
  * Base
  * Large

* DINOv3
  * Small
  * Base
  * Large

* SigLIP2
  * Base
  * Large

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

| Dataset | Model | Best Layer | SRCC | PLCC |
______________________________________________
| LIVE | DINOv2 Base | 4 | 0.963 | 0.727 |
| LIVE | DINOv2 Large | 9 | 0.954 | 0.760 |
| LIVE | DINOv2 Small | 4 | 0.959 | 0.720 |
| LIVE | DINOv3 Base | 4 | 0.960 | 0.759 |
| LIVE | DINOv3 Large | 6 | 0.956 | 0.665 |
| LIVE | DINOv3 Small | 4 | 0.964 | 0.722 |
| LIVE | SigLIP2 Base | 8 | 0.962 | 0.626 |
| LIVE | SigLIP2 Large | 18 | 0.960 | 0.672 |
______________________________________________
| PIPAL | DINOv2 Small | 2 | 0.579 | 0.508 |
| PIPAL | DINOv2 Base | 1 | 0.578 | 0.486 |
| PIPAL | DINOv2 Large | 1 | 0.554 | 0.435 |
| PIPAL | DINOv3 Base | 1 | 0.539 | 0.406 |
| PIPAL | DINOv3 Large | 1 | 0.565 | 0.442 |
| PIPAL | DINOv3 Small | 2 | 0.570 | 0.518 |
| PIPAL | SigLIP2 Large | 15 | 0.564 | 0.576 |
| PIPAL | SigLIP2 Base | 7 | 0.579 | 0.595 |
______________________________________________
| TID2013 | DINOv2 Base | 12 | 0.709 | 0.656 |
| TID2013 | DINOv2 Large | 5 | 0.691 | 0.667 |
| TID2013 | DINOv2 Small | 12 | 0.759 | 0.692 |
| TID2013 | DINOv3 Base | 12 | 0.730 | 0.679 |
| TID2013 | DINOv3 Large | 24 | 0.708 | 0.639 |
| TID2013 | DINOv3 Small | 12 | 0.772 | 0.702 |
| TID2013 | SigLIP2 Base | 11 | 0.803 | 0.667 |
| TID2013 | SigLIP2 Large | 20 | 0.804 | 0.676 |

#### Main Findings

* The best-performing layer varies significantly across encoders and datasets, confirming that intermediate representations can be more informative than final-layer features for IQA.

* On LIVE, all evaluated encoders achieved very high SRCC values (> 0.95), with DINOv3 Small obtaining the highest SRCC (0.964). This indicates that the learned representations are highly correlated with perceived image quality on this dataset.

 * On PIPAL, the strongest performance was achieved by SigLIP2 Base (SRCC = 0.579, PLCC = 0.595), closely followed by DINOv2 Small. The lower overall correlations compared with LIVE suggest that PIPAL represents a more challenging scenario for direct feature-similarity-based IQA.

* On TID2013, SigLIP2 Large achieved the highest SRCC (0.804), while DINOv3 Small obtained the best PLCC among the DINO models (0.702), showing that encoder performance is strongly dependent on the dataset and distortion characteristics.

 * In several encoder-dataset combinations, the best intermediate layer outperformed the final-layer representation. This supports the hypothesis that the final encoder embedding is not necessarily the most informative representation for IQA.

* The optimal layer is highly dataset-dependent, suggesting that different distortion types and image-quality distributions benefit from representations at different levels of abstraction. Therefore, there is no single universally optimal layer across all encoders and datasets.

* The differences between SRCC and PLCC further indicate that feature similarity and perceived image quality do not always follow a purely linear relationship.

* The selected best layers will be used in Phase 3 to extract features for supervised MOS prediction. For each encoder-dataset pair, the layer identified during Phase 2 will be used to construct the feature representation provided to the supervised regression model.

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

| Model | Variant | Best Epoch | MSE | SRCC | PLCC |
____________________________________________________
| Dual SigLIP2 | Small | 15 | 4759.99 | 0.7952 | 0.8322 |
| Dual SigLIP2 | Medium | 13 | 5411.94 | 0.8173 | 0.8410 |
| **Dual SigLIP2** | **Large** | **15** | 6993.55 | **0.8222** | **0.8521** |

The **Dual SigLIP2 Large** configuration achieved the best Phase 3 baseline performance, with **SRCC = 0.8222** and **PLCC = 0.8521** on the PIPAL validation split.

---

## Phase 3.5 – LIVE + TID2013 Mixture Ablation

To evaluate the robustness of the regression architecture across datasets, a second experiment was performed by jointly training on **LIVE and TID2013**.

The experiment combines frozen SigLIP2 Base and SigLIP2 Large features and compares three regression-head capacities:

* **Small**
* **Medium**
* **Large**

All variants use the same feature extraction pipeline, train/validation split, normalization strategy, optimizer settings, and random seed. The only variable is the capacity of the regression head.

### Mixture Ablation Results

| Variant | Mixed SRCC | Mixed PLCC | LIVE SRCC | LIVE PLCC | TID2013 SRCC | TID2013 PLCC |
___________________________________________________________________________________________
| **Medium** | **0.7500** | **0.8994** | 0.7843 | 0.8021 | **0.8542** | **0.8949** |
| Small | 0.7301 | 0.8925 | **0.7921** | **0.8064** | 0.8070 | 0.8630 |
| Large | 0.6250 | 0.5822 | -0.9598 | -0.8649 | 0.8117 | 0.8127 |

The **Medium** variant achieves the best overall performance on the mixed validation set, reaching **SRCC = 0.7500** and **PLCC = 0.8994**.

The Small variant performs slightly better on LIVE, while the Medium variant provides the best results on TID2013. The Large variant performs substantially worse and shows unstable training behavior: its best checkpoint was reached at **epoch 2**, after which validation performance deteriorated and early stopping was triggered.

### Main Findings

The Phase 3 experiments show that increasing the capacity of the regression head does not necessarily lead to better IQA performance.

The results suggest that the **Medium architecture provides the best overall trade-off between model capacity and generalization**, while the Large configuration can become unstable when trained on the mixed LIVE + TID2013 setting.

Overall, Phase 3 demonstrates that frozen visual encoder features can be successfully mapped to perceptual quality scores using a relatively lightweight supervised regression head.

### Ablation Plot

![Phase 3.5 Ablation](results/phase3_mixture_ablation_plots/phase3_ablation_mixed.png)