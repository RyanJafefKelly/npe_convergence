#!/bin/bash -l
#PBS -N ma2_3p3_cells
#PBS -J 0-239%12
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -o res/overnight_20260601/dim_scaling/ma2_n1000/pbs_driver_logs/
#PBS -e res/overnight_20260601/dim_scaling/ma2_n1000/pbs_driver_logs/

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5

NPE_VENV="${NPE_VENV:-$PBS_O_WORKDIR/.venv}"
source "$NPE_VENV/bin/activate"

: "${MANIFEST_CSV:?Set MANIFEST_CSV to the prepared cell manifest CSV.}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-res/overnight_20260601/dim_scaling/ma2_n1000}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/$OUTPUT_ROOT/mplconfig/cell_${PBS_ARRAY_INDEX}"

python npe_convergence/scripts/run_ma2_dim_scaling.py run-manifest-row \
  --kind cell \
  --manifest "$MANIFEST_CSV" \
  --array-index "$PBS_ARRAY_INDEX" \
  --output-root "$OUTPUT_ROOT"

deactivate
