#!/bin/bash -l
#
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -J HYDRO_FIDUCIAL
#SBATCH -o ./logs/L2800N5040/bin_halo_mask_M500c.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
########################


module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4


source "/cosma/apps/dp004/${USER}/extra_swift_lightcones_env/bin/activate"

# use job array as the lightcone number

# simulations resolution
box_res="L1000N1800"
# use job name as FLAMINGO simulations name
sim_name="${SLURM_JOB_NAME}"

# directories with halo lightcones and soap catalogues
halo_lc_dir="/cosma8/data/dp004/flamingo/Runs/${box_res}/${SLURM_JOB_NAME}/sorted_hbt_lightcone_halos"
soap_dir="/cosma8/data/dp004/flamingo/Runs/${box_res}/${SLURM_JOB_NAME}/SOAP-HBT"

# mask filename
mask_filename="M500c_bin_mask.hdf5"

# directory to write mask too
output_dir=./example_outputs/${box_res}/${SLURM_JOB_NAME}/

# Nside of the output binary mask
nside=1024

# Only make masks for the shells 0:nshell
nshell=2

# binning method
bin_method="both"

mpirun -- python3 -m mpi4py ./examples/M500crit_binary_mask.py "${halo_lc_dir}" "${soap_dir}" "${mask_filename}" "${output_dir}" ${SLURM_ARRAY_TASK_ID} \
    --nside=${nside} \
    --nshell=${nshell} \
    --bin_method="${bin_method}" \
