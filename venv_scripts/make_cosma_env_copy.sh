#!/bin/bash
#
# Set up a virtual env on COSMA for ExtraSWIFTLightcones, using openmpi
# and parallel HDF5.
# The same as done for in LightconeIO with additional packages included 
# for projections and generating new lightcones 


module purge 
module load python3
venv_name="./extra_swift_lightcones_env"

set -e

# Create an empty venv and activate it
python3 -m venv "${venv_name}"
source "${venv_name}/bin/activate"

pip cache purge

pip install --no-binary mpi4py mpi4py

export CC="`which mpicc`"
export HDF5_MPI="ON"
##export HDF5_DIR=<path to mpi enabled hdf5 installation>


pip install setuptools cython numpy pkgconfig
pip install --no-binary h5py --no-build-isolation h5py

pip install -e .[mpi]
pip install lightcone_io
pip install swiftsimio
pip install unyt
pip install scipy
pip install matplotlib
pip install argparse
pip install healpy
pip install psutil
pip install datetime 
pip install virgodc
pip install cmasher # extra colourmaps used in examples

# Install lightcone shell redshifts 

BASE_URL="https://dataweb.cosma.dur.ac.uk:8443/hdfstream/download/FLAMINGO"
# Repo root = one level up from this script, wherever it is run from
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
dest_dir="${1:-${repo_dir}/data/redshifts}"

L1_REDSHIFTS_FILENAME="${dest_dir}/L1_shell_redshifts_z3.txt"
L2P8_REDSHIFTS_FILENAME="${dest_dir}/L2p8_shell_redshifts_z5.txt"

# make output ./data directory 
mkdir -p "${dest_dir}"

# download URL TARGET
download() {
    local url="$1"
    local target="$2"
    local tmp="${target}.part"

    if [ -s "${target}" ]; then
        echo "Already downloaded: ${target}"
        return 0
    fi

    echo "Downloading ${url}"
    if ! curl -fL --retry 3 -o "${tmp}" "${url}"; then
        rm -f "${tmp}"
        echo "Error: could not download ${url}" >&2
        exit 1
    fi

    # Should be plain text, not an HTML error/login page
    if [ ! -s "${tmp}" ] || grep -qi "<html" "${tmp}"; then
        rm -f "${tmp}"
        echo "Error: ${url} did not return a text file" >&2
        exit 1
    fi

    mv "${tmp}" "${target}"
    echo "Saved ${target}"
}

download "${BASE_URL}/L1_m9/L1_m9/shell_redshifts_z3.txt"  "${L1_REDSHIFTS_FILENAME}"
download "${BASE_URL}/L2p8_m9/L2p8_m9/shell_redshifts.txt" "${L2P8_REDSHIFTS_FILENAME}"

# Add the paths to the venv activate script
if [ -z "${venv_name}" ]; then
    echo "No venv active and none given: not updating an activate script."
    echo "Add these to your job scripts instead:"
    echo "  export L1_REDSHIFTS_FILENAME=\"${L1_REDSHIFTS_FILENAME}\""
    echo "  export L2P8_REDSHIFTS_FILENAME=\"${L2P8_REDSHIFTS_FILENAME}\""
    exit 0
fi

activate_script="${venv_name}/bin/activate"
if ! grep -q "L1_REDSHIFTS_FILENAME" "${activate_script}"; then
    echo "export L1_REDSHIFTS_FILENAME=\"${L1_REDSHIFTS_FILENAME}\"" >> "${activate_script}"
fi
if ! grep -q "L2P8_REDSHIFTS_FILENAME" "${activate_script}"; then
    echo "export L2P8_REDSHIFTS_FILENAME=\"${L2P8_REDSHIFTS_FILENAME}\"" >> "${activate_script}"
fi

echo "L1_REDSHIFTS_FILENAME and L2P8_REDSHIFTS_FILENAME set in ${activate_script}"

