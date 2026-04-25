#!/bin/bash -l
#PBS -N gnk_mvn_ctl_n500_x50_s88
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -o res/gnk_model_control/gnk_asymptotic_mvn_gaussian_npe_n500_x50_seed88_20260426T000000Z/logs/stdout.log
#PBS -e res/gnk_model_control/gnk_asymptotic_mvn_gaussian_npe_n500_x50_seed88_20260426T000000Z/logs/stderr.log

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

export MPLCONFIGDIR="$PBS_O_WORKDIR/res/gnk_model_control/gnk_asymptotic_mvn_gaussian_npe_n500_x50_seed88_20260426T000000Z/mplconfig"

python npe_convergence/scripts/run_gnk_model_control_pilot.py \
  --config res/gnk_model_control/gnk_asymptotic_mvn_gaussian_npe_n500_x50_seed88_20260426T000000Z/config.yaml

deactivate
