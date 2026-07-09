#!/bin/bash -l
#PBS -N gnk_npe_gap_p2
#PBS -J 0-29%5
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
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
export MANIFEST_CSV="${MANIFEST_CSV:-${OUT_ROOT}/manifests/phase2_flow_manifest.csv}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/${OUT_ROOT}/mplconfig/phase2_${PBS_ARRAY_INDEX}"
mkdir -p "$MPLCONFIGDIR" "${OUT_ROOT}/pbs_logs"

row="$(awk -F, -v idx="$PBS_ARRAY_INDEX" 'NR == idx + 2 {print; exit}' "$MANIFEST_CSV")"
if [[ -z "$row" ]]; then
  echo "No manifest row for PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX}" >&2
  exit 2
fi

IFS=, read -r task_id method standardisation seed n_obs n_sims transform <<< "$row"
cell_dir="${OUT_ROOT}/gnk_gaussian_robust_n1000/cells/flow_npe_n_obs_${n_obs}_n_sims_${n_sims}_seed_${seed}_transform_${transform}"
if [[ -s "${cell_dir}/metrics.json" ]]; then
  echo "Skipping completed cell ${cell_dir}"
  deactivate
  exit 0
fi

python npe_convergence/scripts/run_gnk_flow_robust_scale.py \
  --seed "$seed" \
  --n_obs "$n_obs" \
  --n_sims "$n_sims" \
  --transform "$transform" \
  --output-root "${OUT_ROOT}/gnk_gaussian_robust_n1000/cells" \
  --v3-root res/gnk_v3_refs \
  --sim-batch-size 1000

deactivate
