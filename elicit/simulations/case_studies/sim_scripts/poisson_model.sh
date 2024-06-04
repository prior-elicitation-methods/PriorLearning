#!/bin/bash -l
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH --job-name=Poisson
#SBATCH --output=/work/mflobock/elicit/simulations/case_studies/sim_scripts/poisson.out.txt
#SBATCH --mem=60000

module load python

srun python elicit/simulations/case_studies/sim_scripts/poisson_model.py $SLURM_ARRAY_JOB_ID
