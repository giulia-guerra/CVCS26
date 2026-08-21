#!/bin/bash

#SBATCH --job-name=cvcs26_phase3_adv
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

#SBATCH --output=results/phase3_slurm/advanced_%j.out
#SBATCH --error=results/phase3_slurm/advanced_%j.err


# ============================================================
# SAFETY
# ============================================================

set -e


# ============================================================
# HEADER
# ============================================================

echo "========================================"
echo "CVCS26 - PHASE 3 ADVANCED"
echo "========================================"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"

echo "========================================"


# ============================================================
# PROJECT
# ============================================================

cd /homes/gguerra/provaProgetto/CVCS26

source /homes/gguerra/envs/cvcs_env/bin/activate


# ============================================================
# PYTHON
# ============================================================

echo "Python:"
which python
python --version

echo "========================================"


# ============================================================
# PYTORCH / CUDA
# ============================================================

echo "PyTorch / CUDA"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
PY

echo "========================================"


# ============================================================
# NVIDIA
# ============================================================

echo "NVIDIA SMI"

nvidia-smi

echo "========================================"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

mkdir -p results/phase3_slurm/advanced


# ============================================================
# ADVANCED DATASET
# ============================================================

FEATURES="/work/cvcs2026/Cross_Entropy_Champions/features/PIPAL/advanced_features.pt"


echo "Advanced feature file:"
echo "$FEATURES"

if [ ! -f "$FEATURES" ]; then
    echo "ERROR: Advanced feature file not found:"
    echo "$FEATURES"
    exit 1
fi

echo "Advanced feature file found."

echo "========================================"


# ============================================================
# TRAINING
# ============================================================

echo "Starting Advanced training..."

srun python scripts/train_phase3_advanced.py \
    --features "$FEATURES" \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --val-ratio 0.2 \
    --patience 7 \
    --seed 42 \
    --num-workers 4 \
    --dim-base 768 \
    --dim-large 1024 \
    --proj-dim 256 \
    --num-heads 4 \
    --transformer-layers 1 \
    --dropout 0.3 \
    --output-dir results/phase3_slurm/advanced


# ============================================================
# FINISHED
# ============================================================

echo "========================================"
echo "PHASE 3 ADVANCED FINISHED"
echo "========================================"

date

# ============================================================
# This SLURM script runs the Advanced Phase 3 experiment
# on the GPU cluster.
#
# It:
#   1. Requests GPU, CPU, memory and execution time.
#   2. Activates the project Python environment.
#   3. Checks Python, PyTorch, CUDA and GPU availability.
#   4. Verifies that the Advanced .pt dataset exists.
#   5. Runs train_phase3_advanced.py.
#   6. Saves the checkpoint and logs in:
#
#      results/phase3_slurm/advanced/
#
# The Advanced model uses:
#
#   Base features  -> 768 dimensions
#   Large features -> 1024 dimensions
#   Projection     -> 256 dimensions
#   Transformer    -> 1 layer
#   Attention heads -> 4
#
# Unlike the Phase 3 baseline, there is no --layer argument
# because the Advanced model receives all layers from both
# encoders.
# ============================================================