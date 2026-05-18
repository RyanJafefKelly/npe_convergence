#!/bin/bash -l
#PBS -N hb_stage_train
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
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
: "${LEARNING_RATE:=5e-4}"
: "${TRAIN_BATCH_SIZE:=256}"
: "${MAX_EPOCHS:=2000}"
: "${EPOCHS_THIS_RUN:=100}"
: "${PATIENCE:=200}"
: "${VAL_FRAC:=0.1}"
: "${CHECKPOINT_EVERY_EPOCHS:=1}"
: "${LOG_EVERY_EPOCHS:=1}"
: "${MAX_RUNTIME_SECONDS:=162000}"
: "${WALLTIME_BUFFER_SECONDS:=1800}"
: "${REQUIRE_TRAINING_COMPLETE:=0}"
: "${STAGING_ROOT:=res/staged_high_budget}"

python npe_convergence/scripts/run_high_budget_staged.py \
  --staging-root="$STAGING_ROOT" \
  train \
  --model="$MODEL" \
  --method="$METHOD" \
  --seed="$SEED" \
  --n-obs="$N_OBS" \
  --n-sims="$N_SIMS" \
  --learning-rate="$LEARNING_RATE" \
  --train-batch-size="$TRAIN_BATCH_SIZE" \
  --max-epochs="$MAX_EPOCHS" \
  --epochs-this-run="$EPOCHS_THIS_RUN" \
  --patience="$PATIENCE" \
  --val-frac="$VAL_FRAC" \
  --checkpoint-every-epochs="$CHECKPOINT_EVERY_EPOCHS" \
  --log-every-epochs="$LOG_EVERY_EPOCHS" \
  --max-runtime-seconds="$MAX_RUNTIME_SECONDS" \
  --walltime-buffer-seconds="$WALLTIME_BUFFER_SECONDS"

if [ "$REQUIRE_TRAINING_COMPLETE" = "1" ]; then
  complete_path="${STAGING_ROOT}/${MODEL}_${METHOD}_n_obs_${N_OBS}_n_sims_${N_SIMS}_seed_${SEED}/training/${METHOD}_complete.json"
  python - "$complete_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing training completion file: {path}")
with path.open() as f:
    payload = json.load(f)
if not payload.get("complete"):
    raise SystemExit(
        "training incomplete after final scheduled train job: "
        f"epoch={payload.get('epoch')} stop_reason={payload.get('stop_reason')}"
    )
PY
fi

deactivate
