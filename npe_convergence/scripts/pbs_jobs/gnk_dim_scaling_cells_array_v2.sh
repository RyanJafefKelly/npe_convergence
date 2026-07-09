#!/bin/bash -l
#PBS -N gnk_dim_cells_v2
#PBS -l walltime=36:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -o res/gnk_dim_scaling/pbs_driver_logs/
#PBS -e res/gnk_dim_scaling/pbs_driver_logs/

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

: "${PBS_ARRAY_INDEX:?Submit this retry template with qsub -J <failed_indices>.}"
: "${MANIFEST_CSV:?Set MANIFEST_CSV to the prepared cell manifest CSV.}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-res/gnk_dim_scaling}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/res/gnk_dim_scaling/mplconfig/cell_retry_${PBS_ARRAY_INDEX}"

cell_dir="$(
python - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST_CSV"])
output_root = Path(os.environ.get("OUTPUT_ROOT", "res/gnk_dim_scaling"))
array_index = int(os.environ["PBS_ARRAY_INDEX"])

with manifest.open(newline="") as f:
    rows = [row for row in csv.DictReader(f) if int(row["array_index"]) == array_index]
if len(rows) != 1:
    raise SystemExit(f"array_index={array_index} matched {len(rows)} rows in {manifest}")

row = rows[0]
if row.get("kind") != "cell":
    raise SystemExit(f"array_index={array_index} is kind={row.get('kind')}, expected cell")

print(
    output_root
    / (
        f"gnk_{row['method']}_d_s_{row['d_s']}_n_obs_{row['n_obs']}_"
        f"n_sims_{row['n_sims']}_seed_{row['seed']}"
    )
)
PY
)"

if [[ -f "$cell_dir/metrics.json" ]]; then
  echo "metrics.json already exists for ${cell_dir}; skipping retry."
  deactivate
  exit 0
fi

python npe_convergence/scripts/run_gnk_dim_scaling.py run-manifest-row \
  --kind cell \
  --manifest "$MANIFEST_CSV" \
  --array-index "$PBS_ARRAY_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --force

deactivate
