#!/bin/bash
#
# Set up a virtual env on COSMA for ExtraSWIFTLightcones, using openmpi
# and parallel HDF5.
# The same as done for in LightconeIO with additional packages included 
# for projections and generating new lightcones 


set -e

ompi_version=5.0.3
hdf5_version=1.12.3

module purge
module load python/3.12.4

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# clear any cached wheels
python -m pip cache purge

# Location of local wheels
WHEEL_DIR=/cosma/local/python-wheels/3.12.4/openmpi-${ompi_version}-hdf5-${hdf5_version}

# Name of the new venv to create, can be given as the first argument
venv_name="${1:-/cosma/apps/do012/${USER}/extra_swift_lightcones_env}"

# Create an empty venv and activate it
python -m venv "${venv_name}"
source "${venv_name}/bin/activate"

# Install modules with tricky dependencies from locally built wheels.
# These must go in before anything else, or pip will pull in generic
# PyPI builds of mpi4py/h5py that don't use COSMA's MPI and parallel HDF5.
pip install ${WHEEL_DIR}/mpi4py-3.1.6-cp312-cp312-linux_x86_64.whl
pip install ${WHEEL_DIR}/h5py-3.11.0-cp312-cp312-linux_x86_64.whl

# Install ExtraSWIFTLightcones in editable mode, with the dependencies of the examples and tests.
# This also installs lightcone_io and the other dependencies listed in pyproject.toml.
pip install -e "${repo_dir}[all]"

# Download the FLAMINGO lightcone shell redshift files into the venv,
# and export their paths when it is activated
extra_swift_lightcones-configure