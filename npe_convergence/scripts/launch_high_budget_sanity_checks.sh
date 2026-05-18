#!/bin/bash
# Submit or print short PBS sanity checks for the staged high-budget pipeline.
#
# Dry-run is the default. Add --submit to call qsub. The checks are intended to
# fail fast when the environment or dependency chain is wrong, not to provide
# final-quality empirical results.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  npe_convergence/scripts/launch_high_budget_sanity_checks.sh [--submit]

Optional environment:
  STAGING_ROOT=res/staged_high_budget_sanity_YYYYmmdd_HHMMSS
  SEED=0
  N_OBS=5000
  PROFILE_N_SIMS=25000000
  RUN_CPU_PROFILES=1
  RUN_GPU_PROBE=1
  RUN_GNK_TINY=1
  RUN_STEREO_TINY=1
  CPU_SANITY_QUEUE=cpu_inter
  GPU_SANITY_QUEUE=gpu_batch

The default checks are:
  1. GNK CPU simulation profile with 1,000 simulations.
  2. Stereological CPU simulation profile with 100 simulations.
  3. GPU/JAX environment probe.
  4. Tiny GNK flow-NPE dependency chain.
  5. Tiny stereological Gaussian-NPE dependency chain.

Dry-run is the default and prints the exact qsub commands. With --submit, the
profile/probe jobs are independent, and each tiny pipeline is submitted as:
  simulation array -> aggregate simulations -> train -> evaluation array -> aggregate+verify
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

: "${STAGING_ROOT:=res/staged_high_budget_sanity_$(date +%Y%m%d_%H%M%S)}"
: "${SEED:=0}"
: "${N_OBS:=5000}"
: "${PROFILE_N_SIMS:=25000000}"
: "${RUN_CPU_PROFILES:=1}"
: "${RUN_GPU_PROBE:=1}"
: "${RUN_GNK_TINY:=1}"
: "${RUN_STEREO_TINY:=1}"
if [ -z "${CPU_SANITY_QUEUE+x}" ]; then
  CPU_SANITY_QUEUE="cpu_inter"
fi
if [ -z "${GPU_SANITY_QUEUE+x}" ]; then
  GPU_SANITY_QUEUE="gpu_batch"
fi

: "${GNK_PROFILE_METHOD:=flow_npe}"
: "${GNK_PROFILE_SIMS:=1000}"
: "${GNK_PROFILE_BATCH_SIZE:=1000}"
: "${STEREO_PROFILE_METHOD:=gaussian_npe}"
: "${STEREO_PROFILE_SIMS:=100}"
: "${STEREO_PROFILE_BATCH_SIZE:=10}"
: "${PROFILE_WALLTIME:=00:30:00}"
: "${PROFILE_MEM:=32gb}"
: "${GPU_PROBE_WALLTIME:=00:30:00}"
: "${GPU_PROBE_MEM:=16gb}"

: "${GNK_TINY_METHOD:=flow_npe}"
: "${GNK_TINY_N_SIMS:=2000}"
: "${GNK_TINY_SHARD_SIZE:=1000}"
: "${GNK_TINY_SIM_BATCH_SIZE:=1000}"
: "${STEREO_TINY_METHOD:=gaussian_npe}"
: "${STEREO_TINY_N_SIMS:=100}"
: "${STEREO_TINY_SHARD_SIZE:=50}"
: "${STEREO_TINY_SIM_BATCH_SIZE:=10}"

: "${TINY_COVERAGE_SAMPLES:=2}"
: "${TINY_COVERAGE_REPS:=1}"
: "${TINY_TRAIN_REPEATS:=1}"
: "${TINY_LEARNING_RATE:=5e-4}"
: "${TINY_TRAIN_BATCH_SIZE:=256}"
: "${TINY_MAX_EPOCHS:=1}"
: "${TINY_EPOCHS_THIS_RUN:=1}"
: "${TINY_PATIENCE:=1}"
: "${TINY_MAX_RUNTIME_SECONDS:=5400}"
: "${TINY_WALLTIME_BUFFER_SECONDS:=300}"
: "${TINY_NUM_POSTERIOR_SAMPLES:=100}"
: "${TINY_METRIC_SAMPLES:=100}"
: "${TINY_EXPECTED_POSTERIOR_SAMPLES:=100}"
: "${TINY_SIM_WALLTIME:=00:30:00}"
: "${TINY_AGGREGATE_SIMS_WALLTIME:=00:15:00}"
: "${TINY_TRAIN_WALLTIME:=02:00:00}"
: "${TINY_EVAL_WALLTIME:=00:30:00}"
: "${TINY_FINAL_WALLTIME:=00:15:00}"
: "${TINY_SIM_MEM:=32gb}"
: "${TINY_AGGREGATE_SIMS_MEM:=16gb}"
: "${TINY_TRAIN_MEM:=16gb}"
: "${TINY_EVAL_MEM:=32gb}"
: "${TINY_FINAL_MEM:=16gb}"

pbs_dir="npe_convergence/scripts/pbs_jobs"
profile_template="$pbs_dir/high_budget_staged_cpu_profile.sh"
gpu_template="$pbs_dir/high_budget_staged_gpu_probe.sh"
pipeline_launcher="npe_convergence/scripts/launch_high_budget_staged_pipeline.sh"

if [ -z "${QSUB:-}" ]; then
  if command -v qsub >/dev/null 2>&1; then
    QSUB="$(command -v qsub)"
  elif [ -x /opt/pbs/bin/qsub ]; then
    QSUB="/opt/pbs/bin/qsub"
  else
    QSUB="qsub"
  fi
fi

for required_file in "$profile_template" "$gpu_template" "$pipeline_launcher"; do
  if [ ! -f "$required_file" ]; then
    echo "missing required file: $required_file" >&2
    exit 66
  fi
done

print_qsub() {
  printf '+ %q' "$QSUB"
  printf ' %q' "$@"
  printf '\n'
}

run_qsub() {
  local label="$1"
  shift
  if [ "$submit" -eq 1 ]; then
    local job_id
    job_id=$("$QSUB" "$@")
    echo "$label job: $job_id"
  else
    print_qsub "$@"
  fi
}

run_tiny_pipeline() {
  local label="$1"
  local model="$2"
  local method="$3"
  local n_sims="$4"
  local shard_size="$5"
  local sim_batch_size="$6"
  local mode_arg="--dry-run"

  if [ "$submit" -eq 1 ]; then
    mode_arg="--submit"
  fi

  echo
  echo "$label tiny dependency chain"
  env \
    MODEL="$model" \
    METHOD="$method" \
    SEED="$SEED" \
    N_OBS="$N_OBS" \
    N_SIMS="$n_sims" \
    STAGING_ROOT="$STAGING_ROOT" \
    QUEUE="$CPU_SANITY_QUEUE" \
    SHARD_SIZE="$shard_size" \
    SIM_BATCH_SIZE="$sim_batch_size" \
    COVERAGE_SAMPLES="$TINY_COVERAGE_SAMPLES" \
    COVERAGE_REPS="$TINY_COVERAGE_REPS" \
    TRAIN_REPEATS="$TINY_TRAIN_REPEATS" \
    LEARNING_RATE="$TINY_LEARNING_RATE" \
    TRAIN_BATCH_SIZE="$TINY_TRAIN_BATCH_SIZE" \
    MAX_EPOCHS="$TINY_MAX_EPOCHS" \
    EPOCHS_THIS_RUN="$TINY_EPOCHS_THIS_RUN" \
    PATIENCE="$TINY_PATIENCE" \
    MAX_RUNTIME_SECONDS="$TINY_MAX_RUNTIME_SECONDS" \
    WALLTIME_BUFFER_SECONDS="$TINY_WALLTIME_BUFFER_SECONDS" \
    NUM_POSTERIOR_SAMPLES="$TINY_NUM_POSTERIOR_SAMPLES" \
    METRIC_SAMPLES="$TINY_METRIC_SAMPLES" \
    EXPECTED_POSTERIOR_SAMPLES="$TINY_EXPECTED_POSTERIOR_SAMPLES" \
    SIM_WALLTIME="$TINY_SIM_WALLTIME" \
    AGGREGATE_SIMS_WALLTIME="$TINY_AGGREGATE_SIMS_WALLTIME" \
    TRAIN_WALLTIME="$TINY_TRAIN_WALLTIME" \
    EVAL_WALLTIME="$TINY_EVAL_WALLTIME" \
    FINAL_WALLTIME="$TINY_FINAL_WALLTIME" \
    SIM_MEM="$TINY_SIM_MEM" \
    AGGREGATE_SIMS_MEM="$TINY_AGGREGATE_SIMS_MEM" \
    TRAIN_MEM="$TINY_TRAIN_MEM" \
    EVAL_MEM="$TINY_EVAL_MEM" \
    FINAL_MEM="$TINY_FINAL_MEM" \
    bash "$pipeline_launcher" "$mode_arg"
}

echo "High-budget staged sanity checks"
echo "  staging root: $STAGING_ROOT"
echo "  seed/n_obs: $SEED / $N_OBS"
echo "  profile target n_sims: $PROFILE_N_SIMS"
echo "  CPU sanity queue: ${CPU_SANITY_QUEUE:-default}"
echo "  GPU sanity queue: ${GPU_SANITY_QUEUE:-template/default}"
echo "  mode: $([ "$submit" -eq 1 ] && echo submit || echo dry-run)"

if [ "$RUN_CPU_PROFILES" -eq 1 ]; then
  echo
  echo "CPU profile jobs"
  cpu_queue_args=()
  if [ -n "$CPU_SANITY_QUEUE" ]; then
    cpu_queue_args=(-q "$CPU_SANITY_QUEUE")
  fi
  run_qsub "GNK profile" \
    ${cpu_queue_args[@]+"${cpu_queue_args[@]}"} \
    -l "walltime=$PROFILE_WALLTIME" \
    -l "mem=$PROFILE_MEM" \
    -v "MODEL=gnk,METHOD=$GNK_PROFILE_METHOD,SEED=$SEED,N_OBS=$N_OBS,N_SIMS=$PROFILE_N_SIMS,PROFILE_SIMS=$GNK_PROFILE_SIMS,SIM_BATCH_SIZE=$GNK_PROFILE_BATCH_SIZE,STAGING_ROOT=$STAGING_ROOT" \
    "$profile_template"
  run_qsub "stereological profile" \
    ${cpu_queue_args[@]+"${cpu_queue_args[@]}"} \
    -l "walltime=$PROFILE_WALLTIME" \
    -l "mem=$PROFILE_MEM" \
    -v "MODEL=stereological,METHOD=$STEREO_PROFILE_METHOD,SEED=$SEED,N_OBS=$N_OBS,N_SIMS=$PROFILE_N_SIMS,PROFILE_SIMS=$STEREO_PROFILE_SIMS,SIM_BATCH_SIZE=$STEREO_PROFILE_BATCH_SIZE,STAGING_ROOT=$STAGING_ROOT" \
    "$profile_template"
fi

if [ "$RUN_GPU_PROBE" -eq 1 ]; then
  echo
  echo "GPU probe job"
  gpu_queue_args=()
  if [ -n "$GPU_SANITY_QUEUE" ]; then
    gpu_queue_args=(-q "$GPU_SANITY_QUEUE")
  fi
  run_qsub "GPU probe" \
    ${gpu_queue_args[@]+"${gpu_queue_args[@]}"} \
    -l "walltime=$GPU_PROBE_WALLTIME" \
    -l "mem=$GPU_PROBE_MEM" \
    -v "STAGING_ROOT=$STAGING_ROOT" \
    "$gpu_template"
fi

if [ "$RUN_GNK_TINY" -eq 1 ]; then
  run_tiny_pipeline "GNK" "gnk" "$GNK_TINY_METHOD" "$GNK_TINY_N_SIMS" "$GNK_TINY_SHARD_SIZE" "$GNK_TINY_SIM_BATCH_SIZE"
fi

if [ "$RUN_STEREO_TINY" -eq 1 ]; then
  run_tiny_pipeline "stereological" "stereological" "$STEREO_TINY_METHOD" "$STEREO_TINY_N_SIMS" "$STEREO_TINY_SHARD_SIZE" "$STEREO_TINY_SIM_BATCH_SIZE"
fi
