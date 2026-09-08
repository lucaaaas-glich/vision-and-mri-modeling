#!/bin/bash
# COMP3710 Lab 2 Part 4 Task 3 - OASIS WGAN-GP.
# Submit from the project directory with: sbatch train_oasis_wgan_a100.sh

#SBATCH --job-name=oasis-wgan
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=oasis_wgan_%j.out
#SBATCH --error=oasis_wgan_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710/lab2-pattern-recognition"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python pattern_recognition_demo2/part4_oasis_wgan_gp.py \
  --data-dir /home/groups/comp3710/OASIS/keras_png_slices_train \
  --epochs 50 \
  --image-size 128 \
  --batch-size 64 \
  --workers 4
echo "Finished $(date)"
