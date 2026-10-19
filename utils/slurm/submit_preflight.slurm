#!/bin/bash
#SBATCH --job-name=bioacoustic_preflight
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail
module load python/3.10
source venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# All expensive scans/decoding/hash reads run as a batch job, not on the login node.
HABITAT_ARGS=(--seed 42)
if [[ "${REBUILD_HABITAT:-0}" == "1" ]]; then
  HABITAT_ARGS+=(--overwrite)
fi
python scripts/create_habitat_split.py "${HABITAT_ARGS[@]}"
PREFLIGHT_ARGS=(--audio-dir data/raw/xeno_canto \
  --cache-dir data/processed/xeno_canto_cache \
  --manifest data/processed/xeno_canto_preflight.json \
  --sample-rate 16000 --clip-sec 10 --num-crops "${XENO_CACHE_CROPS:-3}" --min-rms 0.00001 \
  --max-excluded-fraction "${PREFLIGHT_MAX_EXCLUDED_FRACTION:-0.05}" \
  --early-abort-min-files "${PREFLIGHT_EARLY_ABORT_MIN_FILES:-1000}" \
  --early-abort-check-interval "${PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL:-100}" \
  --order-seed "${PREFLIGHT_ORDER_SEED:-42}" \
  --staging-stale-hours "${PREFLIGHT_STAGING_STALE_HOURS:-72}")
if [[ "${REBUILD_PREFLIGHT:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--overwrite)
fi
python scripts/preflight_xeno_audio.py "${PREFLIGHT_ARGS[@]}"
python scripts/audit_dataset_overlap.py \
  --xeno-dir data/raw/xeno_canto \
  --birdclef-dir data/raw/birdclef \
  --output data/processed/xeno_canto_excluded.json

BIRDCLEF_PREFLIGHT_ARGS=(--audio-dir data/raw/birdclef \
  --output data/processed/birdclef_audio_preflight.json \
  --sample-rate 16000 --clip-sec 10 --minimum-rms 0.00001 \
  --max-read-error-fraction "${BIRDCLEF_MAX_READ_ERROR_FRACTION:-0.01}" \
  --max-excluded-fraction "${BIRDCLEF_MAX_EXCLUDED_FRACTION:-0.05}" --list-read-errors)
if [[ "${REBUILD_BIRDCLEF_PREFLIGHT:-0}" == "1" ]]; then
  BIRDCLEF_PREFLIGHT_ARGS+=(--overwrite)
fi
python scripts/preflight_birdclef_audio.py "${BIRDCLEF_PREFLIGHT_ARGS[@]}"

SPLIT_ARGS=(--audio-dir data/raw/birdclef --output-dir checkpoints --seeds 42 123 999 --split-seed 42 \
  --min-recordings 5 --min-groups 5 --minimum-rms 0.00001 \
  --max-read-error-fraction "${BIRDCLEF_MAX_READ_ERROR_FRACTION:-0.01}" \
  --max-excluded-fraction "${BIRDCLEF_MAX_EXCLUDED_FRACTION:-0.05}" \
  --rare-class-policy "${RARE_CLASS_POLICY:-exclude}")
if [[ "${REBUILD_SPLITS:-0}" == "1" ]]; then
  SPLIT_ARGS+=(--overwrite)
fi
python scripts/create_birdclef_splits.py "${SPLIT_ARGS[@]}"
