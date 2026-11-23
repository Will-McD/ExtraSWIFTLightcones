#!/usr/bin/env python
r"""
Count what particles are read from the snapshots when placing them in a lightcone and ouptut a markdown doc with the 
results shown in a table.

For each lightcone asked for, places one particle type in a shell while ReadCounter (tests/read_counter.py) records
every snapshot file opened and every dataset read. The report gives, per lightcone, the files read, how often they
were opened, the read requests, and the particle rows read, distinct, and read more than once; with --per-dataset,
also a table per (file, dataset). It is the baseline (and later the check) for reading snapshot files once each.

Usage:

    # the fake simulation of the tests: a 5 deg beam, a 20 deg beam and the all-sky shell, written to a temporary
    # directory
    python tools/count_reads.py --fake

    # a FLAMINGO simulation (on COSMA): a 2 deg beam along (1, 2, 3) and the all-sky shell of 0.1 < z < 0.15
    python tools/count_reads.py --box_res L1000N0900 --sim_name HYDRO_FIDUCIAL --ptype PartType1 \
        --beam 1 2 3 2.0 --all_sky --z 0.1 0.15 --output reads.md
"""

import argparse
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO))

from read_counter import ReadCounter, _merged_length  # noqa: E402

FAKE_CASES = [
    ("beam 5 deg along (1, 2, 3)", (1, 2, 3), 5.0, (0.05, 0.25)),
    ("beam 20 deg along (0, 0, 1)", (0, 0, 1), 20.0, (0.05, 0.25)),
    ("all-sky", None, None, (0.1, 0.2)),
]


def batch_size(text):
    """Returns a --max_particles_per_read value: an int, or None for "none" (whole files)."""
    return None if text.lower() == "none" else int(text)


def parse_args(argv=None):
    """Returns the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fake", action="store_true", help="use the fake simulation of the tests")
    source.add_argument("--box_res", help="box size and resolution of a simulation, e.g. L1000N0900")
    parser.add_argument("--sim_name", help="name of the simulation, e.g. HYDRO_FIDUCIAL (with --box_res)")
    parser.add_argument(
        "--beam",
        nargs=4,
        type=float,
        action="append",
        metavar=("X", "Y", "Z", "RADIUS_DEG"),
        help="a beam along (X, Y, Z) of this angular radius; can be given more than once",
    )
    parser.add_argument("--all_sky", action="store_true", help="also the all-sky lightcone")
    parser.add_argument("--z", nargs=2, type=float, metavar=("Z_MIN", "Z_MAX"), help="the shell's redshift range")
    parser.add_argument("--lock", default="cube", help='orientation lock: "cube", "sphere" or "none" (default cube)')
    parser.add_argument("--ptype", default="PartType1", help="particle type to place (default PartType1)")
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["Coordinates", "ParticleIDs"],
        help="properties to read (default Coordinates ParticleIDs)",
    )
    parser.add_argument(
        "--max_particles_per_read",
        nargs="+",
        type=batch_size,
        default=[None],
        metavar="N",
        help='batch sizes to place each lightcone with, "none" for whole files (default none)',
    )
    parser.add_argument("--memory", action="store_true", help="also measure the peak memory (slower)")
    parser.add_argument("--per_dataset", action="store_true", help="also a table per (file, dataset)")
    parser.add_argument("--output", help="write the report to this file as well as printing it")

    args = parser.parse_args(argv)

    if args.box_res and not args.sim_name:
        parser.error("--box_res needs --sim_name")

    if args.box_res and not (args.beam or args.all_sky):
        parser.error("give --beam and/or --all_sky")

    if args.box_res and not args.z:
        parser.error("give the shell's redshift range with --z")

    return args


def lightcone_cases(args):
    """Returns (name, beam vector or None, radius [deg] or None, redshift range) for each lightcone to count."""
    if args.fake and not (args.beam or args.all_sky):
        return [(name, vector, radius, args.z or z) for name, vector, radius, z in FAKE_CASES]
    cases = []
    for x, y, z, radius in args.beam or []:
        cases.append((f"beam {radius:g} deg along ({x:g}, {y:g}, {z:g})", (x, y, z), radius, tuple(args.z)))
    if args.all_sky:
        cases.append(("all-sky", None, None, tuple(args.z)))
    return cases


def make_lightcone(simulation, beam_vector, radius, lock):
    """Returns a SnapshotBeam, or a SnapshotAllSky if beam_vector is None."""
    from swiftlet import snapshot_lightcone

    lock = None if lock == "none" else lock

    if beam_vector is None:
        return snapshot_lightcone.SnapshotAllSky(simulation, verbose=0, orientation_lock=lock)

    return snapshot_lightcone.SnapshotBeam(
        simulation, beam_vector=beam_vector, ang_radius_deg=radius, verbose=0, orientation_lock=lock
    )


def count_case(simulation, case, max_particles_per_read, args, comm=None):
    """
    Place the particles of one lightcone while counting the reads (and, with --memory, the peak memory) of this rank.

    Returns (summary dict of this rank, ReadCounter).
    """
    name, vector, radius, shell_z = case
    lightcone = make_lightcone(simulation, vector, radius, args.lock)
    if comm is not None:
        comm.Barrier()
    if args.memory:
        tracemalloc.start()
    start = time.perf_counter()
    batches = {} if max_particles_per_read is None else {"max_particles_per_read": max_particles_per_read}
    with ReadCounter() as counter:
        particles = lightcone.place_snapshot_particles_in_shell(
            shell_z, args.properties, args.ptype, comm=comm, **batches
        )[args.ptype]
    seconds = time.perf_counter() - start
    peak = None
    if args.memory:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    files, to_read, to_place = planned(lightcone.last_shell)
    placed = particles.get("Coordinates")
    totals = counter.totals(ReadCounter.particle_datasets)
    row = {
        **totals,  # first, so "files" below is the files planned, not those this rank read
        "lightcone": name,
        "shell": f"{shell_z[0]:g} < z < {shell_z[1]:g}",
        "batch": "whole files" if max_particles_per_read is None else f"{max_particles_per_read}",
        "files": files,
        "to_read": to_read,
        "to_place": to_place,
        "kept": 0 if placed is None else len(placed),
        "seconds": seconds,
        "peak_mb": None if peak is None else peak / 2**20,
        "cell_requests": counter.totals(ReadCounter.cell_datasets)["requests"],
    }
    plans = getattr(lightcone.last_shell, "plans", None)
    if comm is not None and plans is not None:
        rank_files = lightcone._assign_files_to_ranks([plan.nr_placed for plan in plans], comm.Get_size())
        mine = rank_files[comm.Get_rank()]
        row.update(
            rank=comm.Get_rank(),
            rank_files=len(mine),
            rank_to_read=sum(plans[i].nr_particles for i in mine),
            rank_to_place=sum(plans[i].nr_placed for i in mine),
        )
    elif comm is not None:  # code from before files were shared whole: the rank's share isn't known in advance
        row.update(rank=comm.Get_rank(), rank_files="", rank_to_read="", rank_to_place="")
    return row, counter


def planned(shell):
    """
    Returns the shell's (files, particles to read, particles to place): each cell once, and once per tile using it.

    From the file plans, or for code from before them (to measure a baseline with the same tool) from its
    per-(snapshot, tile, file) reads.

    :param  shell:  the shell gathered (last_shell)
    :type   shell:  swiftlet.snapshot_lightcone.ShellRead
    """
    plans = getattr(shell, "plans", None)
    if plans is not None:
        return len(plans), sum(plan.nr_particles for plan in plans), sum(plan.nr_placed for plan in plans)
    ranges = {}
    for spec in shell.files:
        ranges.setdefault((spec.snap_nr, spec.file_nr), []).extend(
            (int(o), int(o + n)) for o, n in zip(spec.offsets, spec.lengths) if n > 0
        )
    to_read = sum(_merged_length(file_ranges) for file_ranges in ranges.values())
    to_place = sum(int(sum(spec.lengths)) for spec in shell.files)
    return len(ranges), to_read, to_place


def megabytes(value):
    """Returns a peak memory for the report."""
    return "" if value is None else f"{value:.1f}"


def report(rows, counters, rank_rows, root, args, title):
    """Returns the markdown report."""
    lines = [
        f"# Particle reads: {title}",
        "",
        "Files are the snapshot files planned for the shell. Opens are of the files the particles were read from",
        "(any reason, including cell metadata); requests and rows are of their particle datasets. Rows read again =",
        "rows read beyond the first time; read factor = rows read / distinct rows. To read and to place are the",
        "particles of the planned cells, each cell once and once per tile using it. Peak memory (tracemalloc) is of",
        "the placing call.",
        "",
        "| lightcone | shell | batch | files | to read | to place | opens | requests | rows read | distinct rows "
        "| rows read again | read factor | cell metadata requests | particles kept | peak memory [MB] | time [s] |",
        "| --- | --- | --- " + "| ---: " * 13 + "|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lightcone']} | {row['shell']} | {row['batch']} | {row['files']} | {row['to_read']} "
            f"| {row['to_place']} | {row['opens']} | {row['requests']} | {row['rows_read']} | {row['unique_rows']} "
            f"| {row['rows_read_again']} | {row['reread_factor']:.2f} | {row['cell_requests']} | {row['kept']} "
            f"| {megabytes(row['peak_mb'])} | {row['seconds']:.1f} |"
        )
    for ranks in rank_rows:
        first = ranks[0]
        lines += [
            "",
            f"## {first['lightcone']}, {first['batch']}: {len(ranks)} MPI ranks",
            "",
            "| rank | files | to read | to place | opens | rows read | rows read again | cell metadata requests "
            "| particles kept | peak memory [MB] | time [s] |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for rank in ranks:
            lines.append(
                f"| {rank['rank']} | {rank['rank_files']} | {rank['rank_to_read']} | {rank['rank_to_place']} "
                f"| {rank['opens']} | {rank['rows_read']} | {rank['rows_read_again']} | {rank['cell_requests']} "
                f"| {rank['kept']} | {megabytes(rank['peak_mb'])} | {rank['seconds']:.1f} |"
            )
    if args.per_dataset:
        for row, counter in zip(rows, counters):
            lines += [
                "",
                f"## {row['lightcone']}, {row['batch']}",
                "",
                counter.format_table(ReadCounter.particle_datasets, root),
            ]
    return "\n".join(lines) + "\n"


def combine_ranks(ranks):
    """Returns the row of a lightcone over all ranks: the totals of reads and particles, the largest peak memory."""
    row = dict(ranks[0])
    for name in ("opens", "requests", "rows_read", "unique_rows", "rows_read_again", "cell_requests", "kept"):
        row[name] = sum(rank[name] for rank in ranks)
    row["reread_factor"] = row["rows_read"] / row["unique_rows"] if row["unique_rows"] else 1.0
    row["seconds"] = max(rank["seconds"] for rank in ranks)
    peaks = [rank["peak_mb"] for rank in ranks if rank["peak_mb"] is not None]
    row["peak_mb"] = max(peaks) if peaks else None
    return row


def mpi_comm():
    """Returns MPI.COMM_WORLD if run on more than one MPI rank, otherwise None."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None


def main(argv=None):
    """Count the reads of each lightcone asked for, and print (and write) the report."""
    # import args
    args = parse_args(argv)

    # import needed inspect and swiftlet classes for lightcone making
    import inspect
    from swiftlet.simulation_suites import Simulation
    from swiftlet.snapshot_lightcone import SnapshotLightcone

    placing = inspect.signature(SnapshotLightcone.place_snapshot_particles_in_shell).parameters
    if "max_particles_per_read" not in placing and any(n is not None for n in args.max_particles_per_read):
        raise SystemExit(
            "this SWIFTLET has no max_particles_per_read (it is from before reading in batches): measure whole files "
            "only, i.e. leave out --max_particles_per_read"
        )

    comm = mpi_comm()
    rank = 0 if comm is None else comm.Get_rank()
    patch, root = None, None
    if args.fake:
        import pytest
        import test_snapshot_lightcone_placement as fake

        if rank == 0:
            root = Path(tempfile.mkdtemp(prefix="swiftlet_fake_runs_"))
            print(f"writing the fake simulation to {root}", file=sys.stderr)
            fake.write_fake_runs(root)
        if comm is not None:
            root = comm.bcast(root, root=0)
        patch = pytest.MonkeyPatch()
        fake.patch_snapshot_lightcone(patch, root)
        simulation = Simulation.from_label(fake.BOX_RES, fake.HALO_SIM)
        title = f"fake simulation {fake.BOX_RES}/{fake.HALO_SIM}"
    else:
        simulation = Simulation.from_label(args.box_res, args.sim_name)
        title = f"{args.box_res}/{args.sim_name}"
    title += f", {args.ptype}, lock {args.lock}" + ("" if comm is None else f", {comm.Get_size()} MPI ranks")

    try:
        rows, counters, rank_rows = [], [], []
        for case in lightcone_cases(args):
            for max_particles_per_read in args.max_particles_per_read:
                if rank == 0:
                    print(f"counting {case[0]}, max_particles_per_read={max_particles_per_read}...", file=sys.stderr)
                row, counter = count_case(simulation, case, max_particles_per_read, args, comm)
                if comm is None:
                    rows.append(row)
                    counters.append(counter)
                    continue
                ranks = comm.gather(row, root=0)
                if rank == 0:
                    rows.append(combine_ranks(ranks))
                    rank_rows.append(ranks)
                    counters.append(counter)  # rank 0's, for --per_dataset
    finally:
        if patch is not None:
            patch.undo()

    if rank != 0:
        return
    text = report(rows, counters, rank_rows, root, args, title)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
