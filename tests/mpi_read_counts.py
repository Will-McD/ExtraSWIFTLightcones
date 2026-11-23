#!/bin/env python
"""
Place the particles of a fake snapshot in a lightcone shell in parallel while counting what each rank reads, run under
mpiexec by test_mpi_file_by_file.test_mpi_reads_each_file_once.

Rank 0 writes, to <output>.json, one entry per rank: the snapshot files it read particles from, the files it was
assigned, its particles read (rows of Coordinates, and rows read more than once), the opens of its particle files, its
reads of cell metadata, and the particles it kept.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_snapshot_lightcone_placement as placement  # noqa: E402
from read_counter import ReadCounter  # noqa: E402


def main(args):
    """
    Place the particles on every rank, counting each rank's reads, and write them from rank 0.

    :param  args:   arguments, see the module docstring
    :type   args:   dict
    """
    comm = MPI.COMM_WORLD
    patch = pytest.MonkeyPatch()

    placement.patch_snapshot_lightcone(patch, Path(args["run_dir"]))

    lightcone = placement.new_lightcone(
        placement.HALO_SIM, args["beam_vector"], args["orientation_lock"], ang_radius_deg=args["ang_radius_deg"]
    )
    if args.get("rank_0_gathers_first") and comm.Get_rank() == 0:
        lightcone.gather_files(placement.PTYPE, tuple(args["redshift_range"]))
    with ReadCounter() as counter:
        particles = lightcone.place_snapshot_particles_in_shell(
            tuple(args["redshift_range"]),
            property_names=["ParticleIDs", "Coordinates"],
            particle_types=placement.PTYPE,
            comm=comm,
            max_particles_per_read=args.get("max_particles_per_read"),
        )[placement.PTYPE]

    plans = lightcone.last_shell.plans
    assigned = lightcone._assign_files_to_ranks([plan.nr_placed for plan in plans], comm.Get_size())[comm.Get_rank()]
    particle_rows = counter.rows(ReadCounter.particle_datasets)
    coordinates = [row for row in particle_rows if row["dataset"].endswith("/Coordinates")]
    particle_files = sorted({row["file"] for row in particle_rows})
    cell_files = {row["file"] for row in counter.rows(ReadCounter.cell_datasets)}

    entry = {
        "files_read": particle_files,
        "files_assigned": [
            os.path.realpath(lightcone.simulation.snapshot_file(plans[i].snap_nr, plans[i].file_nr)) for i in assigned
        ],
        "expected_to_read": sum(plans[i].nr_particles for i in assigned),
        "expected_to_place": sum(plans[i].nr_placed for i in assigned),
        "all_expected_to_place": sum(plan.nr_placed for plan in plans),
        "coordinate_rows_read": sum(row["rows_read"] for row in coordinates),
        "rows_read_again": sum(row["rows_read_again"] for row in particle_rows),
        "particle_file_opens": [counter.opens[name] - (name in cell_files) for name in particle_files],
        "cell_metadata_requests": counter.totals(ReadCounter.cell_datasets)["requests"],
        "kept": len(particles["ParticleIDs"]),
    }

    entries = comm.gather(entry, root=0)

    if comm.Get_rank() == 0:
        Path(f"{args['output']}.json").write_text(json.dumps(entries))

    patch.undo()


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
