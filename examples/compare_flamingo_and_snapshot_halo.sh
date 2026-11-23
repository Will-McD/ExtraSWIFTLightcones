#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -J compare_flamingo_and_snapshot_halo
#SBATCH -o ./logs/compare_flamingo_and_snapshot_halo.out
#SBATCH -p cosma8-serial
#SBATCH -A dp004
#SBATCH -t 03:00:00
########################

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# CHANGE ALL PATHS AS NEEDED. This example is built to run on COSMA

# virtual environment made with venv_scripts/make_cosma_env.sh
source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate"

# use the cores given to the job for the numba threads
export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}

python3 -u ./examples/compare_flamingo_and_snapshot_halo.py --box_res L2800N5040 --sim_name HYDRO_FIDUCIAL \
    --lightcone_nr 6 --shell_nr 1 --log_mass 14.5 --log_mass_min 13.5 \
    --flamingo_dir "FLAMINGO/L2p8_m9/L2p8_m9" --output_dir ./example_outputs/compare_flamingo_and_snapshot_halo \
    --nside 16384 
