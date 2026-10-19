#!/bin/bash
#SBATCH --job-name=bioacoustic_bn_ablation
#SBATCH --output=logs/bn_ablation_%A_%a.out
#SBATCH --error=logs/bn_ablation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-8%3

# Optional representation-proxy ablation: three random seeds x three BN policies.
# This runs the habitat-SSL pretraining variant only; downstream probe/soundscape
# evaluations still need to be launched separately for any policy being reported.
set -euo pipefail
module load cuda/11.8
module load python/3.10
source venv/bin/activate
mkdir -p logs checkpoints results

SEEDS=(42 123 999)
POLICIES=(copy_query ema_query frozen_key)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEED_INDEX=$((TASK_ID / 3))
POLICY_INDEX=$((TASK_ID % 3))
SEED=${SEEDS[$SEED_INDEX]}
POLICY=${POLICIES[$POLICY_INDEX]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=$SEED

ARGS=(--config configs/pretrain_moco_habitat.yaml --variant habitat --seed "$SEED" --bn-policy "$POLICY")
LATEST="checkpoints/moco_habitat_s${SEED}_bn-${POLICY}_latest.pt"
if [[ "${RESUME_BN_ABLATION:-0}" == "1" && -f "$LATEST" ]]; then
  ARGS+=(--resume)
fi
printf 'BN ablation seed=%s policy=%s latest=%s\n' "$SEED" "$POLICY" "$LATEST"
python -m experiments.exp1_pretrain_moco "${ARGS[@]}"
