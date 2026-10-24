#!/bin/env python
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_snapshot_lightcone_placement as placement  # noqa: E402

"""
Place the haloes of the fake SOAP catalogues into the lightcone shell and then run under mpiexec by
test_snapshot_lightcone_placement.test_mpi_haloes_match_serial. 
Each rank saves the haloes it placed to <output>.<rank>.npz.

The json arguments are run_dir, beam_vector (null for all-sky), ang_radius_deg, redshift_range, orientation_lock,
soap_chunk_size, redistribute and output.

"""

def main(args):
    """
    Place the haloes on every rank and save them.
    :param  args:   arguments, see the module docstring
    :type   args:   dict
    """
    comm = MPI.COMM_WORLD
    patch = pytest.MonkeyPatch()
    run_dir = Path(args["run_dir"])
    placement.patch_snapshot_lightcone(patch, run_dir)

    lightcone = placement.new_lightcone(placement.HALO_SIM, args["beam_vector"], args["orientation_lock"])
    halos = lightcone.place_halos_in_shell(placement.soap_format(run_dir), tuple(args["redshift_range"]),
                                           soap_chunk_size=args["soap_chunk_size"], comm=comm,
                                           redistribute_halos=args["redistribute"],
                                           **placement.shell_args(lightcone, args["ang_radius_deg"]))

    np.savez(f"{args['output']}.{comm.Get_rank()}.npz",
             snap_nr=halos["Lightcone/SnapshotNumber"].value.astype(int),
             index=halos["InputHalos/HaloCatalogueIndex"].value.astype(int),
             soap_index=halos["InputHalos/SOAPIndex"].value.astype(int),
             centre=halos["Lightcone/HaloCentre"].to_value("Mpc"),
             a=halos["Lightcone/ExpansionFactor"].value)
    patch.undo()


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))