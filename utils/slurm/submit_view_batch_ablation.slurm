#!/bin/bash
#SBATCH --job-name=bioacoustic_view_batch_ablation
#SBATCH --output=logs/view_batch_ablation_%A_%a.out
#SBATCH --error=logs/view_batch_ablation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%3

# Tests the BatchNorm running-statistics confound: query views handled in two
# separate forwards vs. one concatenated forward. This is a representation-proxy
# ablation only; selected policies require downstream evaluation.
set -euo pipefail
module load cuda/11.8
module load python/3.10
source venv/bin/activate
mkdir -p logs checkpoints results

SEEDS=(42 123 999)
POLICIES=(separate concatenated)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEED_INDEX=$((TASK_ID / 2))
POLICY_INDEX=$((TASK_ID % 2))
SEED=${SEEDS[$SEED_INDEX]}
POLICY=${POLICIES[$POLICY_INDEX]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=$SEED

ARGS=(--config configs/pretrain_moco_habitat.yaml --variant habitat --seed "$SEED" --view-batch-policy "$POLICY")
LATEST="checkpoints/moco_habitat_s${SEED}_views-${POLICY}_latest.pt"
if [[ "${RESUME_VIEW_BATCH_ABLATION:-0}" == "1" && -f "$LATEST" ]]; then
  ARGS+=(--resume)
fi
printf 'view-batch ablation seed=%s policy=%s latest=%s\n' "$SEED" "$POLICY" "$LATEST"
python -m experiments.exp1_pretrain_moco "${ARGS[@]}"
