#!/bin/bash -l
#PBS -N hb_stage_sim
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
: "${SHARD_SIZE:?set SHARD_SIZE}"
: "${SIM_BATCH_SIZE:=1000}"
: "${STAGING_ROOT:=res/staged_high_budget}"
: "${PBS_ARRAY_INDEX:?submit this template as a PBS array}"

export MPLCONFIGDIR="$PBS_O_WORKDIR/${STAGING_ROOT}/mplconfig/sim_${PBS_ARRAY_INDEX}"
mkdir -p "$MPLCONFIGDIR"

python npe_convergence/scripts/run_high_budget_staged.py \
  --staging-root="$STAGING_ROOT" \
  simulate-shard \
  --model="$MODEL" \
  --method="$METHOD" \
  --seed="$SEED" \
  --n-obs="$N_OBS" \
  --n-sims="$N_SIMS" \
  --shard-index="$PBS_ARRAY_INDEX" \
  --shard-size="$SHARD_SIZE" \
  --sim-batch-size="$SIM_BATCH_SIZE"

deactivate
