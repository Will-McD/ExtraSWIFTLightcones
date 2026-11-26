#!/bin/env python
"""
Place the particles of a fake snapshot in a lightcone shell in parallel, reading remotely (through a
FakeRemoteDirectory on each rank) or from disk, with a cap on the read requests in flight; run under mpiexec by
test_remote_reading.test_mpi_requests_capped.

Rank 0 writes, to <output>.json, one entry per rank: the most requests for the remote data
(max_requests_in_flight), the (start, end) times of the rank's data requests (for particles and SOAP catalogues) and
the particles placed by every rank (SnapshotNumber, TileIndex, ParticleIDs and Coordinates [Mpc]).

    mpiexec -n 3 python mpi_remote_reads.py '<json arguments>'

"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_snapshot_lightcone_placement as placement
from lightcone_helpers import remote_simulation

from swiftlet import snapshot_lightcone
from swiftlet.simulation_suites import Simulation


def main(args):
    """
    Place the particles on every rank and write what each read, and every rank's particles, from rank 0.

    :param  args:   arguments, see the module docstring
    :type   args:   dict
    """
    comm = MPI.COMM_WORLD
    run_dir = Path(args["run_dir"])
    patch = pytest.MonkeyPatch()
    placement.patch_snapshot_lightcone(patch, run_dir)
    simulation = Simulation.from_label(placement.BOX_RES, placement.HALO_SIM)
    if args["remote"]:
        simulation = remote_simulation(simulation, run_dir, delay=args["delay"])

    if args["beam_vector"] is None:
        lightcone = snapshot_lightcone.SnapshotAllSky(simulation, verbose=0, orientation_lock=args["orientation_lock"])
    else:
        lightcone = snapshot_lightcone.SnapshotBeam(
            simulation,
            beam_vector=args["beam_vector"],
            ang_radius_deg=args["ang_radius_deg"],
            verbose=0,
            orientation_lock=args["orientation_lock"],
        )

    if args.get("haloes"):
        place_haloes(args, simulation, lightcone, comm, run_dir)
        patch.undo()
        return

    particles = lightcone.place_snapshot_particles_in_shell(
        tuple(args["redshift_range"]),
        property_names=["ParticleIDs", "Coordinates", "TileIndex"],
        particle_types=placement.PTYPE,
        comm=comm,
        max_particles_per_read=args.get("max_particles_per_read"),
        **cap_of(args),
    )[placement.PTYPE]

    requests = []
    if args["remote"]:
        requests = [
            (r.start, r.end) for r in simulation.remote.log.requests if r.dataset.startswith(f"/{placement.PTYPE}/")
        ]
    entry = cap_entry(lightcone.last_shell, requests)
    entries = comm.gather(entry, root=0)

    columns = {
        "SnapshotNumber": particles["SnapshotNumber"].value.astype(int),
        "TileIndex": particles["TileIndex"].value.astype(int),
        "ParticleIDs": particles["ParticleIDs"].value.astype(int),
        "Coordinates": particles["Coordinates"].to_value("Mpc"),
    }

    gathered = {name: comm.gather(values, root=0) for name, values in columns.items()}
    if comm.Get_rank() == 0:
        Path(f"{args['output']}.json").write_text(json.dumps(entries))
        np.savez(f"{args['output']}.npz", **{name: np.concatenate(parts) for name, parts in gathered.items()})

    patch.undo()


def cap_of(args):
    """The max_concurrent_requests argument to pass: none for "default"."""
    cap = args.get("max_concurrent_requests", "default")
    return {} if cap == "default" else {"max_concurrent_requests": cap}


def cap_entry(shell, requests):
    """What a rank writes: the most requests in flight the cap's counter saw, and the rank's requests."""
    return {"max_requests_in_flight": shell.max_requests_in_flight, "particle_requests": requests}


def place_haloes(args, simulation, lightcone, comm, run_dir):
    """Place the haloes on every rank and write what each read, and every rank's haloes, from rank 0."""
    halo_format = placement.soap_format(run_dir)

    if args["remote"]:
        halo_format = os.path.relpath(halo_format, run_dir)

    haloes = lightcone.place_halos_in_shell(
        tuple(args["redshift_range"]),
        soap_chunk_size=args.get("soap_chunk_size"),
        comm=comm,
        **cap_of(args),
        halo_format=halo_format,
    )
    requests = []

    if args["remote"]:
        requests = [(r.start, r.end) for r in simulation.remote.log.requests if r.dataset.startswith("/InputHalos/")]

    entry = cap_entry(lightcone.last_shell, requests)
    entries = comm.gather(entry, root=0)
    columns = {
        "SnapshotNumber": haloes["Lightcone/SnapshotNumber"].value.astype(int),
        "HaloCatalogueIndex": haloes["InputHalos/HaloCatalogueIndex"].value.astype(int),
        "SOAPIndex": haloes["InputHalos/SOAPIndex"].value.astype(int),
        "HaloCentre": haloes["Lightcone/HaloCentre"].to_value("Mpc"),
    }

    gathered = {name: comm.gather(values, root=0) for name, values in columns.items()}
    if comm.Get_rank() == 0:
        Path(f"{args['output']}.json").write_text(json.dumps(entries))
        np.savez(f"{args['output']}.npz", **{name: np.concatenate(parts) for name, parts in gathered.items()})


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
