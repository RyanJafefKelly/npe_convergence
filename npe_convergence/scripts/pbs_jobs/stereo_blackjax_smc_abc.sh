#!/bin/bash -l
#PBS -N stereo_blackjax_smc_abc
#PBS -l walltime=47:00:00
#PBS -l mem=128GB
#PBS -l ncpus=8
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"

module load python/3.11.5-gcccore-13.2.0 >/dev/null 2>&1 || true

NPE_VENV="${NPE_VENV:-/mnt/hpccs01/home/n9751041/npe_convergence/.venv}"
PYTHON="${PYTHON:-$NPE_VENV/bin/python}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${PBS_NCPUS:-8}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-8}"
export MKL_NUM_THREADS="${PBS_NCPUS:-8}"

N_OBS="${N_OBS:-1000}"
SEED="${SEED:-1}"
PARTICLES="${PARTICLES:-50000}"
REPLICATES="${REPLICATES:-3}"
REPLICATE_OFFSET="${REPLICATE_OFFSET:-0}"
PILOT_SIMS="${PILOT_SIMS:-20000}"
PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-64}"
# Tolerance ladder calibrated 2026-05-31 from local pilots (n_obs=1000, seed 1).
# Distances are standardised Mahalanobis. Sigma is fully converged by tau~2
# (sigma=1.976 at eps 2.0 vs truth 2.0); pushing below eps~1 collapses acceptance
# to ~3% and buys nothing. The old default (down to 0.25) wastes simulations and
# may not be reachable at 50k particles. Override TOLERANCES if a cell needs a
# different floor.
TOLERANCES="${TOLERANCES:-20,10,5,2,1}"
MAX_ITERS="${MAX_ITERS:-30}"
MAX_LOCS="${MAX_LOCS:-300}"
SIM_BATCH="${SIM_BATCH:-2000}"
OUTPUT_LABEL="${OUTPUT_LABEL:-}"
BLACKJAX_ROOT="${BLACKJAX_ROOT:-}"

if [ -z "$OUTPUT_LABEL" ] && [ -n "${PBS_JOBID:-}" ]; then
  OUTPUT_LABEL="job_${PBS_JOBID%%.*}_rep_${REPLICATE_OFFSET}"
fi

if [ -z "$BLACKJAX_ROOT" ]; then
  for candidate in \
    /mnt/hpccs01/home/n9751041/blackjax \
    /mnt/hpccs01/home/n9751041/python_projects/blackjax \
    "$HOME/blackjax" \
    "$HOME/python_projects/blackjax"; do
    if [ -f "$candidate/blackjax/smc/abc.py" ]; then
      BLACKJAX_ROOT="$candidate"
      break
    fi
  done
fi

if [ -z "$BLACKJAX_ROOT" ]; then
  echo "Could not find a blackjax checkout containing blackjax/smc/abc.py." >&2
  echo "Set BLACKJAX_ROOT=/absolute/path/to/RyanJafefKelly/blackjax smc_abc checkout." >&2
  exit 2
fi

"$PYTHON" npe_convergence/scripts/run_stereological_blackjax_smc_abc.py \
  --blackjax-root "$BLACKJAX_ROOT" \
  --n-obs "$N_OBS" \
  --seed "$SEED" \
  --particles "$PARTICLES" \
  --replicates "$REPLICATES" \
  --replicate-offset "$REPLICATE_OFFSET" \
  --pilot-sims "$PILOT_SIMS" \
  --pilot-batch-size "$PILOT_BATCH_SIZE" \
  --tolerances "$TOLERANCES" \
  --max-iters "$MAX_ITERS" \
  --max-locs "$MAX_LOCS" \
  --sim-batch "$SIM_BATCH" \
  --output-label "$OUTPUT_LABEL"
