#!/bin/bash -l
#PBS -N gnk_npe_gap_p1_val
#PBS -l walltime=04:00:00
#PBS -l mem=32GB
#PBS -l ncpus=4
#PBS -o res/overnight_20260601/npe_gap/pbs_logs/
#PBS -e res/overnight_20260601/npe_gap/pbs_logs/

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

export JAX_ENABLE_X64=1
export OUT_ROOT="${OUT_ROOT:-res/overnight_20260601/npe_gap}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/${OUT_ROOT}/mplconfig/phase1_validate"
mkdir -p "$MPLCONFIGDIR" "${OUT_ROOT}/pbs_logs"

python npe_convergence/scripts/run_gnk_gaussian_robust_scale.py \
  --seed 0 \
  --n_obs 1000 \
  --n_sims 1000 \
  --transform asinh \
  --output-root "${OUT_ROOT}/gnk_gaussian_robust_n1000/cells" \
  --v3-root res/gnk_v3_refs \
  --sim-batch-size 1000 \
  --checkpoint-every-epochs 30 \
  --checkpoint-every-seconds 1800 \
  --resume-training

deactivate
