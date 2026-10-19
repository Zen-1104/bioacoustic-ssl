#!/bin/bash
#SBATCH --job-name=bioacoustic_stress
#SBATCH --output=logs/stress_%A_%a.out
#SBATCH --error=logs/stress_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

set -euo pipefail
module load cuda/11.8
module load python/3.10
source venv/bin/activate
SEEDS=(42 123 999)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}
STRESS_FRACTION="${STRESS_FRACTION:-1.0}"
STRESS_CONFIG="${STRESS_CONFIG:-configs/stress_test.yaml}"
python -m experiments.exp3_unseen_habitat_stress_test --config "$STRESS_CONFIG" --seed "$SEED" --fraction "$STRESS_FRACTION"
