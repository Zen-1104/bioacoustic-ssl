#!/usr/bin/env bash
# Submit preflight, pretraining, probing, stress evaluation and aggregation as dependent SLURM stages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable; run this script on the SLURM login node." >&2
  exit 2
fi
if [[ ! -x venv/bin/python || ! -f venv/bin/activate ]]; then
  echo "ERROR: venv/ was not found. Create the Python environment and install requirements first." >&2
  exit 2
fi
# SLURM opens stdout/stderr paths before the batch script starts; create logs first.
mkdir -p logs checkpoints results data/processed data/raw

STRESS_FRACTION="${STRESS_FRACTION:-1.0}"
STRESS_CONFIG="${STRESS_CONFIG:-configs/stress_test.yaml}"
XENO_CACHE_CROPS="${XENO_CACHE_CROPS:-3}"
PREFLIGHT_MAX_EXCLUDED_FRACTION="${PREFLIGHT_MAX_EXCLUDED_FRACTION:-0.05}"
PREFLIGHT_EARLY_ABORT_MIN_FILES="${PREFLIGHT_EARLY_ABORT_MIN_FILES:-1000}"
PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL="${PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL:-100}"
PREFLIGHT_ORDER_SEED="${PREFLIGHT_ORDER_SEED:-42}"
PREFLIGHT_STAGING_STALE_HOURS="${PREFLIGHT_STAGING_STALE_HOURS:-72}"
RARE_CLASS_POLICY="${RARE_CLASS_POLICY:-exclude}"
REBUILD_SPLITS="${REBUILD_SPLITS:-0}"
REBUILD_HABITAT="${REBUILD_HABITAT:-0}"
REBUILD_PREFLIGHT="${REBUILD_PREFLIGHT:-0}"
REBUILD_BIRDCLEF_PREFLIGHT="${REBUILD_BIRDCLEF_PREFLIGHT:-0}"
RESUME_PRETRAIN="${RESUME_PRETRAIN:-0}"
PREFLIGHT_PARTITION="${PREFLIGHT_PARTITION:-}"
GPU_PARTITION="${GPU_PARTITION:-}"
for required_dir in data/raw/xeno_canto data/raw/birdclef data/raw/esc50; do
  if [[ ! -d "$required_dir" ]]; then
    echo "ERROR: required dataset directory '$required_dir' is missing. Acquire/extract all datasets before submitting." >&2
    exit 2
  fi
done
if [[ ! -f "$STRESS_CONFIG" ]]; then
  echo "ERROR: stress-test config '$STRESS_CONFIG' does not exist." >&2
  exit 2
fi
venv/bin/python - "$STRESS_FRACTION" "$STRESS_CONFIG" "$XENO_CACHE_CROPS" "$PREFLIGHT_MAX_EXCLUDED_FRACTION" "$RARE_CLASS_POLICY" "$PREFLIGHT_EARLY_ABORT_MIN_FILES" "$PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL" "$PREFLIGHT_ORDER_SEED" "$PREFLIGHT_STAGING_STALE_HOURS" <<'PYFRAC'
import math, sys, yaml
fraction, config_path, cache_crops = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
max_excluded_fraction = float(sys.argv[4])
rare_class_policy = sys.argv[5]
preflight_early_abort_min_files = int(sys.argv[6])
preflight_early_abort_check_interval = int(sys.argv[7])
preflight_order_seed = int(sys.argv[8])
staging_stale_hours = float(sys.argv[9])
if not 0.0 <= max_excluded_fraction <= 1.0:
    raise SystemExit("PREFLIGHT_MAX_EXCLUDED_FRACTION must be in [0, 1].")
if rare_class_policy not in {"error", "exclude"}:
    raise SystemExit("RARE_CLASS_POLICY must be either 'error' or 'exclude'.")
if not 0.0 < fraction <= 1.0:
    raise SystemExit("STRESS_FRACTION must be in (0, 1].")
if cache_crops < 1:
    raise SystemExit("XENO_CACHE_CROPS must be >= 1.")
if preflight_early_abort_min_files < 1:
    raise SystemExit("PREFLIGHT_EARLY_ABORT_MIN_FILES must be >= 1.")
if preflight_early_abort_check_interval < 1:
    raise SystemExit("PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL must be >= 1.")
if staging_stale_hours <= 0 or not math.isfinite(staging_stale_hours):
    raise SystemExit("PREFLIGHT_STAGING_STALE_HOURS must be a finite positive number.")
print(f"Preflight early-abort checks: min={preflight_early_abort_min_files}, interval={preflight_early_abort_check_interval}; order seed: {preflight_order_seed}; stale staging age: {staging_stale_hours:g} hours")
with open(config_path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
levels = config.get("evaluation", {}).get("snr_levels") if isinstance(config, dict) else None
if not isinstance(levels, list) or not levels:
    raise SystemExit("Stress config must define a non-empty evaluation.snr_levels list.")
labels = []
for index, value in enumerate(levels):
    if value is None:
        label = "Clean"
    elif isinstance(value, bool):
        raise SystemExit(f"evaluation.snr_levels[{index}] must be numeric or null, not boolean.")
    else:
        try: number = float(value)
        except (TypeError, ValueError): raise SystemExit(f"evaluation.snr_levels[{index}] must be numeric or null.")
        if not math.isfinite(number): raise SystemExit(f"evaluation.snr_levels[{index}] must be finite.")
        label = f"{number:g}dB"
    labels.append(label)
if len(set(labels)) != len(labels):
    raise SystemExit(f"Duplicate SNR output labels: {labels}")
print("Validated stress config; SNR conditions:", ", ".join(labels))
PYFRAC

submit_id() {
  local response
  response="$(sbatch --parsable "$@")"
  echo "${response%%;*}"
}

submit_with_partition() {
  local partition="$1"
  shift
  if [[ -n "$partition" ]]; then
    submit_id --partition="$partition" "$@"
  else
    submit_id "$@"
  fi
}

PREFLIGHT_ID="$(submit_with_partition "$PREFLIGHT_PARTITION" --export="ALL,XENO_CACHE_CROPS=${XENO_CACHE_CROPS},PREFLIGHT_MAX_EXCLUDED_FRACTION=${PREFLIGHT_MAX_EXCLUDED_FRACTION},PREFLIGHT_EARLY_ABORT_MIN_FILES=${PREFLIGHT_EARLY_ABORT_MIN_FILES},PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL=${PREFLIGHT_EARLY_ABORT_CHECK_INTERVAL},PREFLIGHT_ORDER_SEED=${PREFLIGHT_ORDER_SEED},PREFLIGHT_STAGING_STALE_HOURS=${PREFLIGHT_STAGING_STALE_HOURS},RARE_CLASS_POLICY=${RARE_CLASS_POLICY},REBUILD_SPLITS=${REBUILD_SPLITS},REBUILD_HABITAT=${REBUILD_HABITAT},REBUILD_PREFLIGHT=${REBUILD_PREFLIGHT},REBUILD_BIRDCLEF_PREFLIGHT=${REBUILD_BIRDCLEF_PREFLIGHT}" utils/slurm/submit_preflight.slurm)"
echo "Submitted data preflight/cache/overlap/split job: $PREFLIGHT_ID"
PRETRAIN_ID="$(submit_with_partition "$GPU_PARTITION" --dependency="afterok:${PREFLIGHT_ID}" --export="ALL,RESUME_PRETRAIN=${RESUME_PRETRAIN}" utils/slurm/submit_pretrain.slurm)"
echo "Submitted pretraining array after preflight: $PRETRAIN_ID"
PROBE_ID="$(submit_with_partition "$GPU_PARTITION" --dependency="afterok:${PRETRAIN_ID}" utils/slurm/submit_probes.slurm)"
echo "Submitted probing array after pretraining: $PROBE_ID"
STRESS_ID="$(submit_with_partition "$GPU_PARTITION" --dependency="afterok:${PROBE_ID}" --export="ALL,STRESS_FRACTION=${STRESS_FRACTION},STRESS_CONFIG=${STRESS_CONFIG}" utils/slurm/submit_stress.slurm)"
echo "Submitted stress-test array after probing: $STRESS_ID"
AGG_ID="$(submit_with_partition "$PREFLIGHT_PARTITION" --dependency="afterok:${STRESS_ID}" --export="ALL,STRESS_FRACTION=${STRESS_FRACTION},STRESS_CONFIG=${STRESS_CONFIG}" utils/slurm/submit_aggregate.slurm)"
echo "Submitted aggregation job after stress testing: $AGG_ID"
echo "Pipeline submitted. Monitor with: squeue -u \$USER"