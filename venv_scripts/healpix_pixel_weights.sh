#!/bin/bash -l

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4


venv_name="/cosma/apps/dp004/${USER}/extra_swift_lightcones_env"
echo ${venv_name}
source "${venv_name}/bin/activate"

set -e

DATAURL="https://github.com/healpy/healpy-data"

dest_dir="${1:-${HEALPY_PIXEL_WEIGHTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/healpy-data}}"

mkdir -p "${dest_dir}"

echo "destination of pixel weights: ${dest_dir}"

if [ -d "${dest_dir}/.git" ]; then
    echo "healpy-data already cloned in ${dest_dir}, updating"
    git -C "${dest_dir}" pull --ff-only
else
    git clone --depth 1 "${DATAURL}" "${dest_dir}"
fi
cd "${dest_dir}"
bash download_weights_8192.sh

# add to venv activate/bin so weights are loaded on source

# Make activating the venv export HEALPY_PIXEL_WEIGHTS (only add the line once)
activate_script="${venv_name}/bin/activate"
if ! grep -q "HEALPY_PIXEL_WEIGHTS" "${activate_script}"; then
    #echo "" >> "${activate_script}"
    echo "export HEALPY_PIXEL_WEIGHTS=\"${dest_dir}\"" >> "${activate_script}"
fi