#!/bin/bash
# COMP3710 Lab 2 Part 3.2 - reproducible Rangpur A100 training job
# Submit from lab2-pattern-recognition with: sbatch train_cifar_a100.sh

#SBATCH --job-name=cifar-55
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=cifar_55_%j.out
#SBATCH --error=cifar_55_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710/lab2-pattern-recognition"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

python pattern_recognition_demo2/part3_cifar10_resnet18.py \
  --epochs 55 \
  --batch-size 256 \
  --workers 4

echo "Finished $(date)"
