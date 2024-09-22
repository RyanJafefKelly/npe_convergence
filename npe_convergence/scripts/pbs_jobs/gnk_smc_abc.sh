#!/bin/bash -l
#PBS -N gnk_smc_abc 
#PBS -l walltime=96:00:00
#PBS -l mem=64GB
#PBS -l ncpus=1
cd $PBS_O_WORKDIR
module load python/3.11.5-gcccore-13.2.0
source .venv/bin/activate
pip install elfi
python npe_convergence/scripts/run_gnk_smc_abc.py --seed=0
deactivate