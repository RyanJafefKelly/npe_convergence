#!/bin/bash -l
#PBS -N gnk_high_budget_gnpe
#PBS -J 0-200%20
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -o res/gnk_high_budget/pbs_driver_logs/
#PBS -e res/gnk_high_budget/pbs_driver_logs/

# Dry-run prepared PBS array. Do not submit unless Ryan explicitly instructs it.
# Grid: n=500, d=11, x in {25,50}, N=x*d^2*n, seeds 0:100.
# Runnable rows: 201. The x=50, seed=88 row is marked reuse and excluded.
# Resource request: 47h walltime, 64GB memory, 4 CPUs, no GPU.
# Concurrency cap: %20 in the PBS -J directive. Adjust only after review.

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

export MANIFEST_JSON="${MANIFEST_JSON:-res/gnk_high_budget/dry_run_manifest_20260426T010115Z.json}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/res/gnk_high_budget/mplconfig/${PBS_ARRAY_INDEX}"

python npe_convergence/scripts/run_gnk_high_budget_array_job.py \
  --manifest "$MANIFEST_JSON" \
  --array-index "$PBS_ARRAY_INDEX"

deactivate
