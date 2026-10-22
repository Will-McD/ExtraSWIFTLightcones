#!/bin/bash -l
#
# Run snapshot_beam_zoom_on_haloes.py on COSMA. Submit from the top directory of the repository, after making
# the log directory (SLURM does not make it):
#
#   mkdir -p ./logs
#   sbatch ./examples/snapshot_beam_zoom_on_haloes.sh
#
# The script runs in serial (no MPI). The snapshot coordinates are re-oriented with numba threads, up to
# SWIFTLET_MAX_THREADS, so one task with several cores is requested.
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

# virtual environment made with venv_scripts/make_cosma_env.sh
source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate"

# use the cores given to the job for the numba threads
export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}

# the simulation, number of haloes, halo mass and map resolution are set in the __main__ block of the python script.
# It reads the snapshots and SOAP catalogues of L1000N1800/HYDRO_FIDUCIAL from /cosma8
python3 -u ./examples/snapshot_beam_zoom_on_haloes.py