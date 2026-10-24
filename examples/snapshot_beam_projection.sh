#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -J snapshot_beam_zoom_on_haloes
#SBATCH -o ./logs/snapshot_beam_zoom_on_haloes.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 06:00:00
########################

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# CHANGE ALL PATHS AS NEEDED

# virtual environment made with venv_scripts/make_cosma_env.sh
source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate"

# use the cores given to the job for the numba threads
export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}

python3 -u ./examples/snapshot_beam_projection.py
