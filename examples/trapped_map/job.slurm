#!/bin/bash

#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J poincare_gen
#SBATCH -A m4680
#SBATCH -t 0:30:0

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application: 
srun -n 128 -c 2 --cpu_bind=cores python trapped_map.py