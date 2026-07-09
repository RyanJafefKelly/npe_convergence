#!/bin/bash -l
#PBS -N gnk_robust_pilot_fin
#PBS -l walltime=03:00:00
#PBS -l mem=32GB
#PBS -l ncpus=4

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate
export JAX_ENABLE_X64=1

python npe_convergence/scripts/run_gnk_gaussian_robust_scale.py \
  --seed 50 \
  --n_obs 5000 \
  --n_sims 2000000 \
  --transform asinh \
  --sim-batch-size 1000 \
  --resume-training \
  --max-epochs 810 \
  --checkpoint-every-epochs 30 \
  --checkpoint-every-seconds 1800

deactivate
