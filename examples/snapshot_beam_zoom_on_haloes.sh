#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -J snapshot_beam_projections
#SBATCH -o ./logs/snapshot_beam_projections.out
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

# zoom on haloes of FLAMINGO L1_m9 and of COLIBRE L400_m7, reading their snapshots and SOAP catalogues from /cosma8.
# The number of haloes, halo mass, largest redshift and pixel size are options of the python script, see --help
n_haloes=3
log_mass=14.0
output_dir=./example_outputs/snapshot_beam_zoom_on_haloes

# COLIBRE L400_m7: the haloes must be inside its 400 Mpc box tile, so they are searched for out to z ~ 0.08
python3 -u ./examples/snapshot_beam_zoom_on_haloes.py --box_res L0400N3008 --sim_name Thermal \
    --n_haloes ${n_haloes} --log_mass ${log_mass} --output_dir "${output_dir}"

# FLAMINGO L1_m9
python3 -u ./examples/snapshot_beam_zoom_on_haloes.py --box_res L1000N1800 --sim_name HYDRO_FIDUCIAL \
    --n_haloes ${n_haloes} --log_mass ${log_mass} --output_dir "${output_dir}"


