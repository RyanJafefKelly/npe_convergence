#!/bin/bash -l
#PBS -N hb_stage_final
#PBS -l walltime=04:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

: "${MODEL:?set MODEL=gnk or stereological}"
: "${METHOD:?set METHOD=flow_npe or gaussian_npe}"
: "${SEED:=0}"
: "${N_OBS:=5000}"
: "${N_SIMS:=25000000}"
: "${METRIC_SAMPLES:=2000}"
: "${EXPECTED_POSTERIOR_SAMPLES:=10000}"
: "${EXPECTED_COVERAGE_SAMPLES:=100}"
: "${STAGING_ROOT:=res/staged_high_budget}"

python npe_convergence/scripts/run_high_budget_staged.py \
  --staging-root="$STAGING_ROOT" \
  aggregate-results \
  --model="$MODEL" \
  --method="$METHOD" \
  --seed="$SEED" \
  --n-obs="$N_OBS" \
  --n-sims="$N_SIMS" \
  --metric-samples="$METRIC_SAMPLES"

verify_posterior=()
if [ "$METHOD" = "flow_npe" ] || [ "$MODEL" = "gnk" ]; then
  verify_posterior=(--require-posterior)
fi

python npe_convergence/scripts/run_high_budget_staged.py \
  --staging-root="$STAGING_ROOT" \
  verify \
  --model="$MODEL" \
  --method="$METHOD" \
  --seed="$SEED" \
  --n-obs="$N_OBS" \
  --n-sims="$N_SIMS" \
  --expected-posterior-samples="$EXPECTED_POSTERIOR_SAMPLES" \
  --expected-coverage-samples="$EXPECTED_COVERAGE_SAMPLES" \
  "${verify_posterior[@]}"

deactivate
