#!/bin/bash
#SBATCH --job-name=refract-tune-array
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-13
#SBATCH --output=results/logs/%x-%A-%a.out
#SBATCH --error=results/logs/%x-%A-%a.err

module load cuda/12.1
source ~/.bashrc
conda activate refract

mkdir -p results/logs
cd $SLURM_SUBMIT_DIR

DATASETS=(cora citeseer pubmed texas cornell wisconsin chameleon squirrel actor roman-empire amazon-ratings minesweeper tolokers questions)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python scripts/tune.py "$DS" --device cuda --n-configs 30 --n-val-splits 3
