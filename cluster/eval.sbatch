#!/bin/bash
#SBATCH --job-name=refract-eval
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# Usage: sbatch cluster/eval.sbatch chameleon --n-splits 10 --n-seeds 3

module load cuda/12.1
source ~/.bashrc
conda activate refract

mkdir -p results/logs
cd $SLURM_SUBMIT_DIR

DS=$1
shift

python scripts/eval.py "$DS" \
    --config "configs/hparams/${DS}.yaml" \
    --device cuda \
    "$@"
