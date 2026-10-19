#!/bin/bash
#SBATCH --job-name=bioacoustic_aggregate
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
module load python/3.10
source venv/bin/activate
python results/aggregate_probes.py --expected-seeds 42 123 999
STRESS_FRACTION="${STRESS_FRACTION:-1.0}"
STRESS_CONFIG="${STRESS_CONFIG:-configs/stress_test.yaml}"
python results/aggregate_results.py --config "$STRESS_CONFIG" --expected-seeds 42 123 999 --fraction "$STRESS_FRACTION"
python results/paired_tests.py --expected-seeds 42 123 999 --fraction "$STRESS_FRACTION"
