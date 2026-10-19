#!/bin/bash
#SBATCH --job-name=bioacoustic_probe
#SBATCH --output=logs/probe_%A_%a.out
#SBATCH --error=logs/probe_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-59%6

set -euo pipefail
module load cuda/11.8
module load python/3.10
source venv/bin/activate

SEEDS=(42 123 999)
FRACTIONS=(0.01 0.05 0.10 1.00)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEED_INDEX=$((TASK_ID / 20))
LOCAL_INDEX=$((TASK_ID % 20))
SEED=${SEEDS[$SEED_INDEX]}
FRACTION_INDEX=$((LOCAL_INDEX % 4))
FRACTION=${FRACTIONS[$FRACTION_INDEX]}
GROUP_INDEX=$((LOCAL_INDEX / 4))

case "$GROUP_INDEX" in
    0)
        python -m experiments.exp2_linear_probe_vs_supervised --config configs/supervised_baseline.yaml --mode supervised --variant supervised --fraction "$FRACTION" --seed "$SEED"
        ;;
    1)
        python -m experiments.exp2_linear_probe_vs_supervised --config configs/supervised_noise_baseline.yaml --mode supervised_noise --variant supervised_noise --fraction "$FRACTION" --seed "$SEED"
        ;;
    *)
        VARIANTS=(habitat standard white_noise)
        VARIANT=${VARIANTS[$((GROUP_INDEX - 2))]}
        CHECKPOINT="checkpoints/moco_${VARIANT}_s${SEED}.pt"
        python -m experiments.exp2_linear_probe_vs_supervised --config configs/linear_probe.yaml --mode probe --checkpoint "$CHECKPOINT" --variant "$VARIANT" --fraction "$FRACTION" --seed "$SEED"
        ;;
esac
