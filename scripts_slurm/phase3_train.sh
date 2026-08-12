#!/bin/bash

#SBATCH --job-name=cvcs26_phase3
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

#SBATCH --output=results/phase3_slurm/slurm_%j.out
#SBATCH --error=results/phase3_slurm/slurm_%j.err

set -e

echo "========================================"
echo "CVCS26 - PHASE 3"
echo "========================================"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"

echo "========================================"

cd /homes/gguerra/provaProgetto/CVCS26

source /homes/gguerra/envs/cvcs_env/bin/activate

echo "Python:"
which python
python --version

echo "========================================"
echo "PyTorch / CUDA"
echo "========================================"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "========================================"
echo "NVIDIA SMI"
echo "========================================"

nvidia-smi

echo "========================================"

mkdir -p results/phase3_slurm

srun python scripts/train_phase3.py \
    --features /work/cvcs2026/Cross_Entropy_Champions/features/PIPAL/dinov2_small_all_layers.pt \
    --layer 2 \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --hidden-dim 256 \
    --dropout 0.2 \
    --val-ratio 0.2 \
    --patience 7 \
    --seed 42 \
    --output-dir results/phase3_slurm

echo "========================================"
echo "PHASE 3 FINISHED"
echo "========================================"

date