#!/bin/bash
#
# The same as found in LightconeIO
#
# Set up a virtual env on COSMA for ExtraSWIFTLightcones, using openmpi
# and parallel HDF5.
#
# Run from anywhere: bash scripts/virtual_env/make_cosma_env.sh
#

set -e

ompi_version=5.0.3
hdf5_version=1.12.3

module purge
module load python/3.12.4

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "${repo_dir}"

# Pip can't distinguish between incompatible builds of the same module,
# so we need to clear any cached wheels before we start.
python -m pip cache purge

# Location of local wheels
WHEEL_DIR=/cosma/local/python-wheels/3.12.4/openmpi-${ompi_version}-hdf5-${hdf5_version}

# Name of the new venv to create
venv_name="/cosma/apps/dp004/${USER}/extra_swift_lightcones_env"

# Create an empty venv and activate it
python -m venv "${venv_name}"
source "${venv_name}/bin/activate"

# Install modules with tricky dependencies from locally built wheels.
# These must go in before anything else, or pip will pull in generic
# PyPI builds of mpi4py/h5py that don't use COSMA's MPI and parallel HDF5.
pip install ${WHEEL_DIR}/mpi4py-3.1.6-cp312-cp312-linux_x86_64.whl
pip install ${WHEEL_DIR}/h5py-3.11.0-cp312-cp312-linux_x86_64.whl

# Install ExtraSWIFTLightcones in editable mode. This also installs
# lightcone_io and the other dependencies listed in pyproject.toml.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "${repo_dir}"
pip install -e "${repo_dir}"

cd "${repo_dir}"

pip install swiftsimio
pip install unyt
pip install scipy
pip install matplotlib
pip install argparse
pip install healpy
pip install virgodc
pip install cmasher # extra colourmaps

