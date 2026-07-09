#!/bin/bash -l
#PBS -N gnk_dim_cells
#PBS -J 0-299%20
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -o res/gnk_dim_scaling/pbs_driver_logs/
#PBS -e res/gnk_dim_scaling/pbs_driver_logs/

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

: "${MANIFEST_CSV:?Set MANIFEST_CSV to the prepared cell manifest CSV.}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-res/gnk_dim_scaling}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/res/gnk_dim_scaling/mplconfig/cell_${PBS_ARRAY_INDEX}"

python npe_convergence/scripts/run_gnk_dim_scaling.py run-manifest-row \
  --kind cell \
  --manifest "$MANIFEST_CSV" \
  --array-index "$PBS_ARRAY_INDEX" \
  --output-root "$OUTPUT_ROOT"

deactivate
