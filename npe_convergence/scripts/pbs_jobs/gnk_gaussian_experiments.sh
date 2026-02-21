#!/bin/bash -l
#PBS -N gnk_gaussian_experiments
#PBS -J 0-100
#PBS -l walltime=48:00:00
#PBS -l mem=64GB
#PBS -l ncpus=1
cd $PBS_O_WORKDIR
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate
python npe_convergence/scripts/run_gnk_gaussian_experiments.py --seed=$PBS_ARRAY_INDEX
deactivate
