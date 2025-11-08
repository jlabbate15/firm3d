#!/bin/bash

#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J orbit_class
#SBATCH --mail-user=jal2351@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4680
#SBATCH -t 0:30:0

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application: 
srun -n 128 -c 16 --cpu_bind=cores python fusion_distribution_classification.py