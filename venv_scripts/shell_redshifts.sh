#!/bin/bash
#
# Download the FLAMINGO lightcone shell redshift files into an existing ExtraSWIFTLightcones environment,
# and export their paths (L1_REDSHIFTS_FILENAME and L2P8_REDSHIFTS_FILENAME) when it is activated.
# This is already done by make_venv.sh and make_cosma_env.sh.
#
#   bash venv_scripts/shell_redshifts.sh [/path/to/environment] [swiftlet-configure options]
#

set -e

venv_name="${1:-/cosma/apps/dp004/${USER}/extra_swift_lightcones_env}"
echo "${venv_name}"
source "${venv_name}/bin/activate"

swiftlet-configure "${@:2}"