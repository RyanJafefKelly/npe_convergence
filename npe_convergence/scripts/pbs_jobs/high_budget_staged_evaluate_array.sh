#!/bin/bash -l
#PBS -N hb_stage_eval
#PBS -l walltime=08:00:00
#PBS -l mem=32GB
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
: "${COVERAGE_SAMPLES:=100}"
: "${COVERAGE_REPS:=10}"
: "${NUM_POSTERIOR_SAMPLES:=10000}"
: "${INCLUDE_POSTERIOR_ON_SHARD0:=1}"
: "${FORCE_EVAL:=0}"
: "${STAGING_ROOT:=res/staged_high_budget}"
: "${PBS_ARRAY_INDEX:?submit this template as a PBS array}"

posterior_flag=()
if [ "$INCLUDE_POSTERIOR_ON_SHARD0" = "1" ] && [ "$PBS_ARRAY_INDEX" = "0" ]; then
  posterior_flag=(--include-posterior)
fi

force_flag=()
if [ "$FORCE_EVAL" = "1" ]; then
  force_flag=(--force)
fi

start_rep=$((PBS_ARRAY_INDEX * COVERAGE_REPS))
remaining=$((COVERAGE_SAMPLES - start_rep))
if [ "$remaining" -le 0 ]; then
  echo "No coverage replicates assigned to PBS_ARRAY_INDEX=$PBS_ARRAY_INDEX"
  deactivate
  exit 0
fi
if [ "$remaining" -lt "$COVERAGE_REPS" ]; then
  coverage_reps_this_shard="$remaining"
else
  coverage_reps_this_shard="$COVERAGE_REPS"
fi

python npe_convergence/scripts/run_high_budget_staged.py \
  --staging-root="$STAGING_ROOT" \
  evaluate-shard \
  --model="$MODEL" \
  --method="$METHOD" \
  --seed="$SEED" \
  --n-obs="$N_OBS" \
  --n-sims="$N_SIMS" \
  --shard-index="$PBS_ARRAY_INDEX" \
  --coverage-reps="$coverage_reps_this_shard" \
  --num-posterior-samples="$NUM_POSTERIOR_SAMPLES" \
  "${force_flag[@]}" \
  "${posterior_flag[@]}"

deactivate
