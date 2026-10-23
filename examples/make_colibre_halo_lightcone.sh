#!/bin/bash -l
#
#SBATCH --nodes=2
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -J Thermal
#SBATCH -o ./logs/colibre_halo_lightcones.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
########################

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# CHANGE AS NEEDED

# virtual environment made with venv_scripts/make_cosma_env.sh
venv_name="/cosma/apps/dp004/${USER}/swiftlet_env"
source "${venv_name}/bin/activate"


# one numba thread per MPI rank, so the ranks don't compete for the cores
export SWIFTLET_MAX_THREADS=1

# COLIBRE L400_m7, and the simulation name from the job name (Thermal when not run with sbatch)
box_res="L0400N3008"
sim_name="${SLURM_JOB_NAME:-Thermal}"
output_dir="./example_outputs/colibre_halo_lightcones/${box_res}/${sim_name}"

mpirun -- python3 -m mpi4py ./examples/make_colibre_halo_lightcone.py \
    --box_res="${box_res}" --sim_name="${sim_name}" --zmin=0.001 --zmax=0.1 --output_dir="${output_dir}"
