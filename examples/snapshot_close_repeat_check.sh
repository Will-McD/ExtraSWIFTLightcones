
#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -J snapshot_close_repeat_check
#SBATCH -o ./logs/snapshot_close_repeat_check.%j.out
#SBATCH -p cosma8-serial
#SBATCH -A dp004
#SBATCH -t 12:00:00
########################

# This example was written for use on COSMA:
# CHANGE ALL PATHS AS NEEDED

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

## virtual environment made with venv_scripts/make_cosma_env.sh
#source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate"
#
## use the cores given to the job for the numba threads
#export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}

source /cosma/apps/do012/dc-mcdo1/swiftlet_env2/bin/activate
export SWIFTLET_MAX_THREADS=30


# the simulations to check, as "box_res sim_name zmax", and the shell
# set zmax so the comoving volume of the lightcone is approx 5 times larger than a snapshot box.

simulations=("L0400N3008 Thermal 0.1" "L1000N1800 HYDRO_FIDUCIAL 0.25")
zmin=0.001
seeds="13 27 54 6 19"

for simulation in "${simulations[@]}"; do
    read -r box_res sim_name zmax <<< "${simulation}"
    output_dir=./example_outputs/close_repeat_check/${box_res}/${sim_name}
    mkdir -p "${output_dir}"

    python3 -u ./examples/snapshot_close_repeat_check.py \
        --box_res="${box_res}" \
        --sim_name="${sim_name}" \
        --zmin=${zmin} \
        --zmax=${zmax} \
        --seeds ${seeds} \
        --output_dir="${output_dir}"

done