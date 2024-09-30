#!/bin/bash -l
#PBS -N mak_experiments 
#PBS -l walltime=120:00:00
#PBS -l mem=64GB
#PBS -l ncpus=1
cd $PBS_O_WORKDIR
module load python/3.11.5-gcccore-13.2.0
source .venv/bin/activate
python npe_convergence/scripts/run_ma12_experiments.py --seed=$seed
deactivate
