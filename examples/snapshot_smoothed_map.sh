#!/bin/bash -l
#
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -J smoothed_map
#SBATCH -o ./logs/smoothed_map.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 06:00:00
########################

###############################################
#
# CHANGE AS NEEDED.
# This example is written to run on COSMA8, 
# please update the paths for: 
#   - the modules, 
#   - the slurm job, 
#   - virtual environment 
#
###############################################

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# virtual environment made with venv_scripts/make_cosma_env.sh
venv_name="/cosma/apps/dp004/${USER}/swiftlet_env"
source "${venv_name}/bin/activate"

# one numba thread per MPI rank, so the ranks don't compete for the cores
export SWIFTLET_MAX_THREADS=1

# redshift range of the lightcone shell, Nside of the maps and the orientation of the box tiles
zmin=0.05
zmax=0.10
nside=2048
orientation_lock="cube"

# box size and resolution label and name of each simulation
for simulation in "L1000N1800 HYDRO_FIDUCIAL" "L0400N3008 Thermal"; do
    read -r box_res sim_name <<< "${simulation}"
    output_dir=./example_outputs/smoothed_maps/${box_res}/${sim_name}
    mkdir -p "${output_dir}"

    mpirun -- python3 -m mpi4py ./examples/snapshot_smoothed_map.py \
        --box_res="${box_res}" \
        --sim_name="${sim_name}" \
        --zmin=${zmin} \
        --zmax=${zmax} \
        --nside=${nside} \
        --orientation_lock="${orientation_lock}" \
        --output_dir="${output_dir}"

done

