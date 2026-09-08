#!/bin/bash
# COMP3710 Lab 2 Part 4 Task 2 - four-class OASIS U-Net.
# Usage: sbatch train_oasis_unet_a100.sh

#SBATCH --job-name=oasis-unet
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=oasis_unet_%j.out
#SBATCH --error=oasis_unet_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710/lab2-pattern-recognition"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python pattern_recognition_demo2/part4_oasis_unet.py \
  --data-root /home/groups/comp3710/OASIS \
  --epochs 60 \
  --batch-size 8 \
  --workers 4
echo "Finished $(date)"
