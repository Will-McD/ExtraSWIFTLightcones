#!/bin/env python
"""
Test sharing snapshot files between MPI ranks: whole files, balanced by their expected placed particles, each read once
by one rank, with the cell metadata read on rank 0 only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_snapshot_lightcone_placement import (  # noqa: F401 the fixtures, used by name
    fake_run_dir,
    fake_runs,
    mpiexec_command,
)

from swiftlet.snapshot_lightcone import SnapshotLightcone

assign = SnapshotLightcone._assign_files_to_ranks


def test_assign_files_by_hand():
    """Largest first, each to the least loaded rank (ties to the lowest), each rank's files in file order."""
    weights = [5, 9, 2, 7, 7, 1]
    ranks = assign(weights, 3)
    # 9 -> rank 0; 7 (file 3) -> rank 1; 7 (file 4) -> rank 2; 5 -> rank 1 (7); 2 -> rank 2 (7 < 12); 1 -> rank 0
    assert [r.tolist() for r in ranks] == [[1, 5], [0, 3], [2, 4]]
    assert [sum(weights[i] for i in r) for r in ranks] == [10, 12, 9]


def test_assign_files_edge_cases():
    """More ranks than files leaves ranks empty; one rank takes every file; no files, no work; all files assigned."""
    assert [r.tolist() for r in assign([4, 8], 4)] == [[1], [0], [], []]
    assert [r.tolist() for r in assign([3, 1, 2], 1)] == [[0, 1, 2]]
    assert [r.tolist() for r in assign([], 2)] == [[], []]
    rng = np.random.default_rng(3)
    weights = rng.integers(1, 10_000, size=200)
    ranks = assign(weights, 7)
    assert sorted(np.concatenate(ranks).tolist()) == list(range(200))
    totals = np.array([weights[r].sum() for r in ranks])
    # greedy largest first: no rank more than the largest file above the least loaded one
    assert totals.max() - totals.min() <= weights.max()
    assert totals.max() < 1.01 * totals.mean()  # many unequal files: within about a per cent


def run_read_counts(fake_run_dir, tmp_path, nr_ranks, **args):
    """
    Run mpi_read_counts.py on nr_ranks ranks with the given arguments.
    """
    pytest.importorskip("mpi4py")
    mpiexec = mpiexec_command()

    if mpiexec is None:
        pytest.skip("mpiexec is not available")

    args = dict(args, run_dir=str(fake_run_dir), output=str(tmp_path / "reads"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])

    run = subprocess.run(
        [
            mpiexec,
            "-n",
            str(nr_ranks),
            sys.executable,
            str(Path(__file__).parent / "mpi_read_counts.py"),
            json.dumps(args),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    ranks = json.loads((tmp_path / "reads.json").read_text())
    assert len(ranks) == nr_ranks

    return ranks


CASES = [
    pytest.param(None, None, (0.1, 0.2), "cube", None, id="all_sky"),
    pytest.param(None, None, (0.1, 0.2), "sphere", 200, id="all_sky_sphere_batches"),
    pytest.param((1, 2, 3), 20.0, (0.05, 0.3), "cube", None, id="beam"),
]


@pytest.mark.parametrize("nr_ranks", [2, 3, 4])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range, orientation_lock, max_particles_per_read", CASES)
def test_mpi_reads_each_file_once(
    fake_run_dir,
    tmp_path,
    nr_ranks,
    beam_vector,
    ang_radius_deg,
    redshift_range,
    orientation_lock,
    max_particles_per_read,
):
    """
    Test with several ranks, each snapshot file is read by exactly the rank it was assigned to, opened once, and
    no particle row is read twice, each rank reads exactly its files' expected particles, the cell metadata is read on
    rank 0 only, and every file is read by some rank.
    """
    pytest.importorskip("mpi4py")
    mpiexec = mpiexec_command()
    if mpiexec is None:
        pytest.skip("mpiexec is not available")

    ranks = run_read_counts(
        fake_run_dir,
        tmp_path,
        nr_ranks,
        beam_vector=beam_vector,
        ang_radius_deg=ang_radius_deg,
        redshift_range=redshift_range,
        orientation_lock=orientation_lock,
        max_particles_per_read=max_particles_per_read,
    )

    read_by = {}

    for rank, entry in enumerate(ranks):
        assert sorted(entry["files_read"]) == sorted(entry["files_assigned"]), rank

        for filename in entry["files_read"]:
            assert filename not in read_by, f"{filename} read by ranks {read_by[filename]} and {rank}"
            read_by[filename] = rank

        assert set(entry["particle_file_opens"]) <= {1}, rank
        assert entry["rows_read_again"] == 0, rank
        assert entry["coordinate_rows_read"] == entry["expected_to_read"], rank
        assert (entry["cell_metadata_requests"] > 0) == (rank == 0), rank

    # every planned file was read, and the files were shared by their expected placed particles
    assert sum(len(entry["files_assigned"]) for entry in ranks) == len(read_by)

    placed = [entry["expected_to_place"] for entry in ranks]
    assert sum(placed) == ranks[0]["all_expected_to_place"]
    assert sum(entry["kept"] for entry in ranks) > 0
    assert sum(p > 0 for p in placed) == min(nr_ranks, len(read_by))


def test_mpi_ranks_with_different_cell_caches(fake_run_dir, tmp_path):
    """
    Check that after rank 0 gathered, placing particles on 3 ranks doesn't hang or read the cell metadata again.
    """
    ranks = run_read_counts(
        fake_run_dir,
        tmp_path,
        3,
        beam_vector=None,
        ang_radius_deg=None,
        redshift_range=(0.1, 0.2),
        orientation_lock="cube",
        rank_0_gathers_first=True,
    )
    assert all(entry["cell_metadata_requests"] == 0 for entry in ranks)
    assert all(entry["rows_read_again"] == 0 for entry in ranks)
    assert sum(entry["kept"] for entry in ranks) > 0
