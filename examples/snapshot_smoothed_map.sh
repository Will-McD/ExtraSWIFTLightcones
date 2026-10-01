#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -J HYDRO_FIDUCIAL
#SBATCH -o ./logs/smoothed_gas_mass_map.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 06:00:00
########################

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# virtual environment made with venv_scripts/make_cosma_env.sh
venv_name="/cosma/apps/dp004/${USER}/extra_swift_lightcones_env"
source "${venv_name}/bin/activate"

# simulations resolution
box_res="L1000N1800"
# use job name as FLAMINGO simulations name
sim_name="${SLURM_JOB_NAME}"

# redshift range of the lightcone shell
zmin=0.05
zmax=0.1

# Nside of the output map
nside=2048

# directory to write the map to
output_dir=./example_outputs/smoothed_maps/${box_res}/${sim_name}

mkdir -p "${output_dir}"

mpirun -np 12 -- python3 -m mpi4py ./examples/make_smoothed_map_from_snapshot.py \
    --box_res="${box_res}" \
    --sim_name="${sim_name}" \
    --zmin=${zmin} \
    --zmax=${zmax} \
    --nside=${nside} \
    --output_dir="${output_dir}"