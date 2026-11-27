#!/bin/bash -l


# This example was written for use on COSMA: 
# CHANGE ALL PATHS AS NEEDED

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4


# virtual environment made with venv_scripts/make_cosma_env.sh
source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate"

# use the cores given to the job for the numba threads
export SWIFTLET_MAX_THREADS=1


# the simulation, and its directory's name on the hdfstream server (L1_m9 for L1000N1800/HYDRO_FIDUCIAL)
box_res="L1000N1800"
sim_name="L1_m9"


# FOR CIRCULAR BEAM 

output_dir=./example_outputs/snapshot_lightcone
mkdir -p "${output_dir}"

python3 -u ./examples/snapshot_beam_halo_placement.py \
    --box_res="${box_res}" \
    --sim_name="${sim_name}" \
    --output_dir="${output_dir}" \

# FOR SQUARE BEAM 

output_dir=./example_outputs/snapshot_lightcone
mkdir -p "${output_dir}"

python3 -u ./examples/snapshot_square_beam_halo_placement.py \
    --box_res="${box_res}" \
    --sim_name="${sim_name}" \
    --output_dir="${output_dir}"