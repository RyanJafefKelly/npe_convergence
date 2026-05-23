#!/bin/bash -l
#PBS -N ma2_delta1_refresh
#PBS -J 0-3231
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
#PBS -l ncpus=1

# Template only. Run one local/runtime pilot cell per method before submission.

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

MANIFEST="docs/meeting_2026_05_18/ma2_delta1_refresh_plan/ma2_delta1_refresh_dry_run_manifest.csv"
CMD=$(python - "$MANIFEST" "$PBS_ARRAY_INDEX" <<'PY'
import csv
import sys

manifest, idx = sys.argv[1], int(sys.argv[2])
with open(manifest, newline="") as handle:
    row = next(row for ii, row in enumerate(csv.DictReader(handle)) if ii == idx)
print(row["runtime_command"])
PY
)

echo "$CMD"
eval "$CMD"

deactivate
