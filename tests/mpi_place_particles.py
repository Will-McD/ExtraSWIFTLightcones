#!/bin/env python
"""
Place the particles of a fake simulation snapshot into a lightcone shell in parallel, run under mpiexec by
test_snapshot_lightcone_placement.test_mpi_matches_serial. 

Each rank saves the particles it placed to <output>.<rank>.npz.

    mpiexec -n 2 python mpi_place_particles.py '<json arguments>'

The json arguments are run_dir, sim_name, beam_vector (null for all-sky), ang_radius_deg, redshift_range,
orientation_lock, redistribute and output.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_snapshot_lightcone_placement as placement  # noqa: E402


def main(args):
    """
    Place the particles on every rank and save them.

    :param  args:   arguments, see the module docstring
    :type   args:   dict
    """
    comm = MPI.COMM_WORLD
    patch = pytest.MonkeyPatch()
    placement.patch_snapshot_lightcone(patch, Path(args["run_dir"]))

    lightcone = placement.new_lightcone(args["sim_name"], args["beam_vector"], args["orientation_lock"])
    kwargs = {"comm": comm}
    if args["beam_vector"] is not None:
        kwargs["ang_radius_deg"] = args["ang_radius_deg"]
        kwargs["redistibute_particles"] = args["redistribute"]
    particles = lightcone.place_snapshot_particles_in_shell(tuple(args["redshift_range"]), property_names=["ParticleIDs", "Coordinates"],
                                                            particle_types=placement.PTYPE, **kwargs)[placement.PTYPE]

    if particles and particles.get("Coordinates") is not None and len(particles["Coordinates"]) > 0:
        snap_nr, ids, coords = placement.particle_rows(particles)
    else:
        snap_nr, ids, coords = np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 3))
    np.savez(f"{args['output']}.{comm.Get_rank()}.npz", snap_nr=snap_nr, ids=ids, coords=coords)
    patch.undo()


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))