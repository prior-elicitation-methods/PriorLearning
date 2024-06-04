#!/bin/bash -l
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --job-name=Binomialmodel
#SBATCH --output=/work/mflobock/elicit/simulations/parametric_prior/binom.out.txt
#SBATCH --mem=60000

QUANTILES=("3" "5" "9")

QUANTS=${QUANTILES[$SLURM_ARRAY_TASK_ID]}

module load python

srun python elicit/simulations/parametric_prior/sim_binom.py $SLURM_ARRAY_JOB_ID $QUANTS
