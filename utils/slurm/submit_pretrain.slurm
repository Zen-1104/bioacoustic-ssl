#!/bin/bash
#SBATCH --job-name=bioacoustic_pretrain
#SBATCH --output=logs/pretrain_%A_%a.out
#SBATCH --error=logs/pretrain_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-8%3

set -euo pipefail

module load cuda/11.8
module load python/3.10
source venv/bin/activate

SEEDS=(42 123 999)
VARIANTS=(habitat standard white_noise)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEED_INDEX=$((TASK_ID / 3))
VARIANT_INDEX=$((TASK_ID % 3))
SEED=${SEEDS[$SEED_INDEX]}
VARIANT=${VARIANTS[$VARIANT_INDEX]}
CONFIG=configs/pretrain_moco_standard.yaml
if [[ "$VARIANT" == "habitat" ]]; then CONFIG=configs/pretrain_moco_habitat.yaml; fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=$SEED

echo "Pretraining variant=$VARIANT seed=$SEED"
git rev-parse HEAD
LATEST="checkpoints/moco_${VARIANT}_s${SEED}_latest.pt"
if [[ "${RESUME_PRETRAIN:-0}" == "1" && -f "$LATEST" ]]; then
  echo "Resume requested; latest checkpoint exists: $LATEST"
  python -m experiments.exp1_pretrain_moco \
    --config "$CONFIG" --variant "$VARIANT" --seed "$SEED" --resume
else
  if [[ "${RESUME_PRETRAIN:-0}" == "1" ]]; then
    echo "Resume requested but no latest checkpoint exists for variant=$VARIANT seed=$SEED; starting this task fresh."
  fi
  python -m experiments.exp1_pretrain_moco \
    --config "$CONFIG" --variant "$VARIANT" --seed "$SEED"
fi
