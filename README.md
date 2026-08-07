# Beyond the Pixel Limit

Computer Vision and Cognitive Systems Project 2026

## Team Contributions

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

* configs/
* dataloader/
* datasets/
* logs/
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

| Model         | Best Layer | SRCC  | PLCC  |
| ------------- | ---------- | ----- | ----- |
| DINOv2 Large  | 24         | 0.086 | 0.071 |
| SigLIP2 Large | 15         | 0.071 | 0.067 |

Observations:

* All models achieved very low correlations on LIVE.
* Further investigation may be required to understand the behavior on this dataset.

### Outputs

Final comparison table:

```text
results/phase2/tables/final_phase2_comparison.csv
```

The Phase 2 analysis successfully identified the best-performing layer for each encoder and dataset, providing the basis for the supervised learning stage.

---

## Next Step – Phase 3

The next phase introduces a supervised learning module.

Goals:

* Train a regression head on top of encoder features.
* Predict MOS scores directly.
* Implement a PyTorch training loop.
* Compute training and validation losses.
* Evaluate SRCC and PLCC during training.
* Prepare SLURM scripts for large-scale experiments on the cluster.
