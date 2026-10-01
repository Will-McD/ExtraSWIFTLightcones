#!/bin/bash
#
# Make a virtual environment for ExtraSWIFTLightcones, with the package, all the dependencies of the
# package, examples and tests, and the FLAMINGO lightcone shell redshift files.
#
# Run from anywhere, optionally giving the path of the environment to make:
#   bash venv_scripts/make_venv.sh [/path/to/environment]
#
# On COSMA use make_cosma_env.sh instead, which installs mpi4py and h5py built for COSMA's MPI and parallel HDF5.
#

set -e

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# path of the new environment
venv_name="${1:-${repo_dir}/extra_swift_lightcones_env}"

# create the environment and activate it
python3 -m venv "${venv_name}"
source "${venv_name}/bin/activate"
python -m pip install --upgrade pip

# install ExtraSWIFTLightcones in editable mode, with the dependencies of the examples and tests
pip install -e "${repo_dir}[all]"

# download the shell redshift files into the environment, and export their paths when it is activated
extra_swift_lightcones-configure

echo "Made ${venv_name}, activate it with: source ${venv_name}/bin/activate"