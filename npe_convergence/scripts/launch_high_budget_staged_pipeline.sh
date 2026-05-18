#!/bin/bash
# Submit or print a PBS dependency chain for one staged high-budget cell.
#
# Dry-run is the default. Add --submit to actually call qsub.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  npe_convergence/scripts/launch_high_budget_staged_pipeline.sh [--submit]

Required environment:
  MODEL=gnk|stereological
  METHOD=flow_npe|gaussian_npe

Common optional environment:
  SEED=0
  N_OBS=5000
  N_SIMS=25000000
  STAGING_ROOT=res/staged_high_budget
  SHARD_SIZE=100000
  SIM_BATCH_SIZE=1000
  COVERAGE_SAMPLES=100
  COVERAGE_REPS=10
  TRAIN_REPEATS=10
  EPOCHS_THIS_RUN=100
  SIM_WALLTIME=08:00:00
  AGGREGATE_SIMS_WALLTIME=04:00:00
  TRAIN_WALLTIME=47:00:00
  EVAL_WALLTIME=08:00:00
  FINAL_WALLTIME=04:00:00
  QUEUE=
  SIM_MEM=32gb
  AGGREGATE_SIMS_MEM=64gb
  TRAIN_MEM=64gb
  EVAL_MEM=32gb
  FINAL_MEM=64gb

By default this prints the qsub commands only. With --submit it submits:
  simulation array -> aggregate simulations -> train chain -> evaluation array -> final aggregate+verify
USAGE
}

submit=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --submit) submit=1 ;;
    --dry-run) submit=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

: "${MODEL:?set MODEL=gnk or stereological}"
: "${METHOD:?set METHOD=flow_npe or gaussian_npe}"
: "${SEED:=0}"
: "${N_OBS:=5000}"
: "${N_SIMS:=25000000}"
: "${STAGING_ROOT:=res/staged_high_budget}"
: "${SHARD_SIZE:=100000}"
: "${SIM_BATCH_SIZE:=1000}"
: "${COVERAGE_SAMPLES:=100}"
: "${COVERAGE_REPS:=10}"
: "${TRAIN_REPEATS:=10}"
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
: "${NUM_POSTERIOR_SAMPLES:=10000}"
: "${METRIC_SAMPLES:=2000}"
: "${EXPECTED_POSTERIOR_SAMPLES:=10000}"
: "${EXPECTED_COVERAGE_SAMPLES:=$COVERAGE_SAMPLES}"
: "${SIM_WALLTIME:=08:00:00}"
: "${AGGREGATE_SIMS_WALLTIME:=04:00:00}"
: "${TRAIN_WALLTIME:=47:00:00}"
: "${EVAL_WALLTIME:=08:00:00}"
: "${FINAL_WALLTIME:=04:00:00}"
: "${QUEUE:=}"
: "${SIM_MEM:=32gb}"
: "${AGGREGATE_SIMS_MEM:=64gb}"
: "${TRAIN_MEM:=64gb}"
: "${EVAL_MEM:=32gb}"
: "${FINAL_MEM:=64gb}"

if [ "$SHARD_SIZE" -le 0 ] || [ "$COVERAGE_REPS" -le 0 ] || [ "$TRAIN_REPEATS" -le 0 ]; then
  echo "SHARD_SIZE, COVERAGE_REPS, and TRAIN_REPEATS must be positive" >&2
  exit 64
fi

pbs_dir="npe_convergence/scripts/pbs_jobs"
sim_template="$pbs_dir/high_budget_staged_simulate_array.sh"
agg_template="$pbs_dir/high_budget_staged_aggregate_sims.sh"
train_template="$pbs_dir/high_budget_staged_train.sh"
eval_template="$pbs_dir/high_budget_staged_evaluate_array.sh"
final_template="$pbs_dir/high_budget_staged_aggregate_verify.sh"

if [ -z "${QSUB:-}" ]; then
  if command -v qsub >/dev/null 2>&1; then
    QSUB="$(command -v qsub)"
  elif [ -x /opt/pbs/bin/qsub ]; then
    QSUB="/opt/pbs/bin/qsub"
  else
    QSUB="qsub"
  fi
fi
queue_args=()
if [ -n "$QUEUE" ]; then
  queue_args=(-q "$QUEUE")
fi

for template in "$sim_template" "$agg_template" "$train_template" "$eval_template" "$final_template"; do
  if [ ! -f "$template" ]; then
    echo "missing PBS template: $template" >&2
    exit 66
  fi
done

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  echo $(( (numerator + denominator - 1) / denominator ))
}

sim_shards=$(ceil_div "$N_SIMS" "$SHARD_SIZE")
eval_shards=$(ceil_div "$COVERAGE_SAMPLES" "$COVERAGE_REPS")
sim_last=$((sim_shards - 1))
eval_last=$((eval_shards - 1))

common_vars="MODEL=$MODEL,METHOD=$METHOD,SEED=$SEED,N_OBS=$N_OBS,N_SIMS=$N_SIMS,STAGING_ROOT=$STAGING_ROOT"

run_or_print() {
  if [ "$submit" -eq 1 ]; then
    "$@"
  else
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  fi
}

capture_job() {
  if [ "$submit" -eq 1 ]; then
    "$@"
  else
    printf 'DRYRUN_%s' "$1"
  fi
}

echo "Staged high-budget pipeline"
echo "  model/method: $MODEL / $METHOD"
echo "  seed: $SEED"
echo "  n_obs/n_sims: $N_OBS / $N_SIMS"
echo "  simulation shards: $sim_shards (PBS array 0-$sim_last, shard_size=$SHARD_SIZE)"
echo "  train jobs: $TRAIN_REPEATS"
echo "  evaluation shards: $eval_shards (PBS array 0-$eval_last, coverage_reps=$COVERAGE_REPS)"
echo "  walltimes: sim=$SIM_WALLTIME aggregate=$AGGREGATE_SIMS_WALLTIME train=$TRAIN_WALLTIME eval=$EVAL_WALLTIME final=$FINAL_WALLTIME"
echo "  memory: sim=$SIM_MEM aggregate=$AGGREGATE_SIMS_MEM train=$TRAIN_MEM eval=$EVAL_MEM final=$FINAL_MEM"
if [ -n "$QUEUE" ]; then
  echo "  queue: $QUEUE"
fi
echo "  mode: $([ "$submit" -eq 1 ] && echo submit || echo dry-run)"

if [ "$submit" -eq 0 ]; then
  echo
  echo "Dry-run qsub commands:"
  run_or_print "$QSUB" ${queue_args[@]+"${queue_args[@]}"} -J "0-$sim_last" \
    -l "walltime=$SIM_WALLTIME" \
    -l "mem=$SIM_MEM" \
    -v "$common_vars,SHARD_SIZE=$SHARD_SIZE,SIM_BATCH_SIZE=$SIM_BATCH_SIZE" \
    "$sim_template"
  run_or_print "$QSUB" ${queue_args[@]+"${queue_args[@]}"} -W "depend=afterok:<SIM_JOB>" \
    -l "walltime=$AGGREGATE_SIMS_WALLTIME" \
    -l "mem=$AGGREGATE_SIMS_MEM" \
    -v "$common_vars,EXPECTED_SHARDS=$sim_shards" \
    "$agg_template"
  prev="<AGG_JOB>"
  i=1
  while [ "$i" -le "$TRAIN_REPEATS" ]; do
    require_complete=0
    if [ "$i" -eq "$TRAIN_REPEATS" ]; then
      require_complete=1
    fi
    run_or_print "$QSUB" ${queue_args[@]+"${queue_args[@]}"} -W "depend=afterok:$prev" \
      -l "walltime=$TRAIN_WALLTIME" \
      -l "mem=$TRAIN_MEM" \
      -v "$common_vars,LEARNING_RATE=$LEARNING_RATE,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,MAX_EPOCHS=$MAX_EPOCHS,EPOCHS_THIS_RUN=$EPOCHS_THIS_RUN,PATIENCE=$PATIENCE,VAL_FRAC=$VAL_FRAC,CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS,LOG_EVERY_EPOCHS=$LOG_EVERY_EPOCHS,MAX_RUNTIME_SECONDS=$MAX_RUNTIME_SECONDS,WALLTIME_BUFFER_SECONDS=$WALLTIME_BUFFER_SECONDS,REQUIRE_TRAINING_COMPLETE=$require_complete" \
      "$train_template"
    prev="<TRAIN_JOB_$i>"
    i=$((i + 1))
  done
  run_or_print "$QSUB" ${queue_args[@]+"${queue_args[@]}"} -J "0-$eval_last" -W "depend=afterok:$prev" \
    -l "walltime=$EVAL_WALLTIME" \
    -l "mem=$EVAL_MEM" \
    -v "$common_vars,COVERAGE_SAMPLES=$COVERAGE_SAMPLES,COVERAGE_REPS=$COVERAGE_REPS,NUM_POSTERIOR_SAMPLES=$NUM_POSTERIOR_SAMPLES,INCLUDE_POSTERIOR_ON_SHARD0=1" \
    "$eval_template"
  run_or_print "$QSUB" ${queue_args[@]+"${queue_args[@]}"} -W "depend=afterok:<EVAL_JOB>" \
    -l "walltime=$FINAL_WALLTIME" \
    -l "mem=$FINAL_MEM" \
    -v "$common_vars,METRIC_SAMPLES=$METRIC_SAMPLES,EXPECTED_POSTERIOR_SAMPLES=$EXPECTED_POSTERIOR_SAMPLES,EXPECTED_COVERAGE_SAMPLES=$EXPECTED_COVERAGE_SAMPLES" \
    "$final_template"
  exit 0
fi

sim_job=$("$QSUB" ${queue_args[@]+"${queue_args[@]}"} -J "0-$sim_last" \
  -l "walltime=$SIM_WALLTIME" \
  -l "mem=$SIM_MEM" \
  -v "$common_vars,SHARD_SIZE=$SHARD_SIZE,SIM_BATCH_SIZE=$SIM_BATCH_SIZE" \
  "$sim_template")
echo "simulation job: $sim_job"

agg_job=$("$QSUB" ${queue_args[@]+"${queue_args[@]}"} -W "depend=afterok:$sim_job" \
  -l "walltime=$AGGREGATE_SIMS_WALLTIME" \
  -l "mem=$AGGREGATE_SIMS_MEM" \
  -v "$common_vars,EXPECTED_SHARDS=$sim_shards" \
  "$agg_template")
echo "aggregate simulations job: $agg_job"

prev="$agg_job"
i=1
while [ "$i" -le "$TRAIN_REPEATS" ]; do
  require_complete=0
  if [ "$i" -eq "$TRAIN_REPEATS" ]; then
    require_complete=1
  fi
  train_job=$("$QSUB" ${queue_args[@]+"${queue_args[@]}"} -W "depend=afterok:$prev" \
    -l "walltime=$TRAIN_WALLTIME" \
    -l "mem=$TRAIN_MEM" \
    -v "$common_vars,LEARNING_RATE=$LEARNING_RATE,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,MAX_EPOCHS=$MAX_EPOCHS,EPOCHS_THIS_RUN=$EPOCHS_THIS_RUN,PATIENCE=$PATIENCE,VAL_FRAC=$VAL_FRAC,CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS,LOG_EVERY_EPOCHS=$LOG_EVERY_EPOCHS,MAX_RUNTIME_SECONDS=$MAX_RUNTIME_SECONDS,WALLTIME_BUFFER_SECONDS=$WALLTIME_BUFFER_SECONDS,REQUIRE_TRAINING_COMPLETE=$require_complete" \
    "$train_template")
  echo "train job $i: $train_job"
  prev="$train_job"
  i=$((i + 1))
done

eval_job=$("$QSUB" ${queue_args[@]+"${queue_args[@]}"} -J "0-$eval_last" -W "depend=afterok:$prev" \
  -l "walltime=$EVAL_WALLTIME" \
  -l "mem=$EVAL_MEM" \
  -v "$common_vars,COVERAGE_SAMPLES=$COVERAGE_SAMPLES,COVERAGE_REPS=$COVERAGE_REPS,NUM_POSTERIOR_SAMPLES=$NUM_POSTERIOR_SAMPLES,INCLUDE_POSTERIOR_ON_SHARD0=1" \
  "$eval_template")
echo "evaluation job: $eval_job"

final_job=$("$QSUB" ${queue_args[@]+"${queue_args[@]}"} -W "depend=afterok:$eval_job" \
  -l "walltime=$FINAL_WALLTIME" \
  -l "mem=$FINAL_MEM" \
  -v "$common_vars,METRIC_SAMPLES=$METRIC_SAMPLES,EXPECTED_POSTERIOR_SAMPLES=$EXPECTED_POSTERIOR_SAMPLES,EXPECTED_COVERAGE_SAMPLES=$EXPECTED_COVERAGE_SAMPLES" \
  "$final_template")
echo "final aggregate+verify job: $final_job"
