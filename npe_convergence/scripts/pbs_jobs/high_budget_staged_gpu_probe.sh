#!/bin/bash -l
#PBS -N hb_gpu_probe
#PBS -q gpu_batch
#PBS -l walltime=00:30:00
#PBS -l mem=16GB
#PBS -l ncpus=4
#PBS -l ngpus=1

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.8.0 || true
source .venv/bin/activate

: "${STAGING_ROOT:=res/staged_high_budget}"

python npe_convergence/scripts/run_high_budget_staged.py \
  --staging-root="$STAGING_ROOT" \
  probe-env \
  --require-gpu

deactivate
