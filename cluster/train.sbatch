#!/bin/bash
#SBATCH --job-name=refract-train
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# Usage: sbatch cluster/train.sbatch --config configs/hparams/chameleon.yaml --split-idx 0

module load cuda/12.1
source ~/.bashrc
conda activate refract

mkdir -p results/logs
cd $SLURM_SUBMIT_DIR

python scripts/train.py --device cuda "$@"
