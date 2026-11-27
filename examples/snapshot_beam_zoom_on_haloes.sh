#!/bin/bash -l


module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4


# virtual environment made with venv_scripts/make_cosma_env.sh
source "/cosma/apps/dp004/${USER}/swiftlet_env/bin/activate" # CHANGE AS NEEDED 


# zoom on haloes of FLAMINGO L1_m9 and of COLIBRE L400_m7, reading their snapshots and SOAP catalogues from /cosma8.
# The number of haloes, halo mass, largest redshift and pixel size are options of the python script, see --help
n_haloes=3
log_mass=14.0

# FOR CIRCULAR BEAM
output_dir=./example_outputs/snapshot_beam_zoom_on_haloes

# COLIBRE L400_m7: the haloes must be inside its 400 Mpc box tile, so they are searched for out to z ~ 0.08
python3 -u ./examples/snapshot_beam_zoom_on_haloes.py --box_res L0400N3008 --sim_name Thermal \
    --n_haloes ${n_haloes} --log_mass ${log_mass} --output_dir "${output_dir}"

# FLAMINGO L1_m9
python3 -u ./examples/snapshot_beam_zoom_on_haloes.py --box_res L1000N1800 --sim_name HYDRO_FIDUCIAL \
    --n_haloes ${n_haloes} --log_mass ${log_mass} --output_dir "${output_dir}"


# REPEAT FOR SQUARE BEAM

output_dir=./example_outputs/snapshot_square_beam_zoom_on_haloes

# COLIBRE L400_m7: the haloes must be inside its 400 Mpc box tile, so they are searched for out to z ~ 0.08
python3 -u ./examples/snapshot_square_beam_zoom_on_haloes.py --box_res L0400N3008 --sim_name Thermal \
    --n_haloes ${n_haloes} --log_mass ${log_mass} --output_dir "${output_dir}"

# FLAMINGO L1_m9
python3 -u ./examples/snapshot_square_beam_zoom_on_haloes.py --box_res L1000N1800 --sim_name HYDRO_FIDUCIAL \
    --n_haloes ${n_haloes} --log_mass ${log_mass} --output_dir "${output_dir}"
