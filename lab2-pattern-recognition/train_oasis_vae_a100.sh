#!/bin/bash
# COMP3710 Lab 2 Part 4 Task 1 - OASIS VAE on a Rangpur A100.
# Usage: sbatch train_oasis_vae_a100.sh /actual/path/to/preprocessed_oasis

#SBATCH --job-name=oasis-vae
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=oasis_vae_%j.out
#SBATCH --error=oasis_vae_%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch train_oasis_vae_a100.sh /path/to/preprocessed_oasis"
  exit 2
fi

source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710/lab2-pattern-recognition"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

python pattern_recognition_demo2/part4_oasis_vae.py \
  --data-dir "$1" \
  --epochs 50 \
  --batch-size 64 \
  --latent-dim 16 \
  --beta 0.001 \
  --workers 4

echo "Finished $(date)"
