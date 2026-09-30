#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/test_points.out
#SBATCH -p cosma8-serial
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

#export OMP_NUM_THREADS=30
source /cosma/apps/do012/dc-mcdo1/lightcone_env/bin/activate

#mpirun -- python3 -m mpi4py examples/plot_test_points_in_beam_parallel.py

#
#wait
#
#python3 examples/make_map_from_particles.py
python3 examples/make_map_from_snapshot.py

#export OMP_NUM_THREADS=30
#source /cosma/apps/do012/dc-mcdo1/jupyter_venv/bin/activate
#python3 examples/gas_map_comp.py

