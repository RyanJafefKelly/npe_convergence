#!/bin/bash -l
#PBS -N stereo_experiments 
#PBS -l walltime=144:00:00
#PBS -l mem=64GB
#PBS -l ncpus=1
cd $PBS_O_WORKDIR
module load python/3.11.5-gcccore-13.2.0
source .venv/bin/activate
python npe_convergence/scripts/run_stereological_experiments.py --seed=$seed
deactivate
