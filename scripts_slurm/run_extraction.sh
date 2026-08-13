#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=12:00:00 
#SBATCH --output=/work/cvcs2026/Cross_Entropy_Champions/logs/%x_%j.out
#SBATCH --error=/work/cvcs2026/Cross_Entropy_Champions/logs/%x_%j.err
#SBATCH --account=cvcs2026 

module purge
module load python/3.13.13

source /homes/$USER/envs/cvcs_env/bin/activate

# FONDAMENTALE: Salva i modelli enormi scaricati da HuggingFace su /work e non in /homes
export HF_HOME=/work/cvcs2026/Cross_Entropy_Champions/.cache/huggingface
export PYTHONPATH=.
cd /homes/$USER/prog/CVCS26

python -u scripts/extract_features.py