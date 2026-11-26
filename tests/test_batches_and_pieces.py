#!/bin/env python
"""
Test reading a snapshot file in batches of cells given by  max_particles_per_read.
"""

import tracemalloc

import numpy as np
import pytest

from read_counter import ReadCounter
from lightcone_helpers import assert_same_rows, rows_of
from test_read_file_by_file import PROPERTIES
from test_snapshot_lightcone_placement import (  # noqa: F401 the fixtures, used by name
    HALO_SIM,
    LIGHTCONES,
    PTYPE,
    fake_run_dir,
    fake_runs,
    new_lightcone,
)

from swiftlet.snapshot_lightcone import SnapshotLightcone

# test the batches


def test_cell_batches_by_hand():
    """Test Cells are batched in order, never split, a cell over the limit is a batch on its own, None is one batch."""
    lengths = [5, 3, 0, 8, 2, 20, 1]
    batches = SnapshotLightcone._cell_batches(lengths, 8)

    assert [b.tolist() for b in batches] == [[0, 1, 2], [3], [4], [5], [6]]
    assert [b.tolist() for b in SnapshotLightcone._cell_batches(lengths, 10)] == [[0, 1, 2], [3, 4], [5], [6]]
    assert [b.tolist() for b in SnapshotLightcone._cell_batches(lengths, None)] == [list(range(7))]
    assert [b.tolist() for b in SnapshotLightcone._cell_batches(lengths, 1000)] == [list(range(7))]
    assert [b.tolist() for b in SnapshotLightcone._cell_batches([], 8)] == [[]]

    with pytest.raises(ValueError, match="at least 1"):
        SnapshotLightcone._cell_batches(lengths, 0)


# test batches place the same particles


@pytest.mark.parametrize("orientation_lock", [None, "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_batch_size_changes_nothing_placed(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """Every batch size, from one cell at a time to whole files, places the same particles in the same tiles."""
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg)

    whole = rows_of(lightcone.place_snapshot_particles_in_shell(redshift_range, PROPERTIES, PTYPE)[PTYPE])

    for max_particles_per_read in (1, 50, 1000):
        batched = lightcone.place_snapshot_particles_in_shell(
            redshift_range, PROPERTIES, PTYPE, max_particles_per_read=max_particles_per_read
        )[PTYPE]
        assert_same_rows(rows_of(batched), whole)


def test_batches_still_read_each_cell_once(fake_runs):
    """
    Ensure that when reading in batches that each file is still opened once and no particle row read twice.
    """
    lightcone = new_lightcone(HALO_SIM, None, "cube")

    with ReadCounter() as whole:
        lightcone.place_snapshot_particles_in_shell((0.1, 0.2), ["ParticleIDs", "Coordinates"], PTYPE)

    lightcone.clear_cache()  # the same cell metadata reads as the first call, so the opens compare

    with ReadCounter() as batched:
        lightcone.place_snapshot_particles_in_shell(
            (0.1, 0.2), ["ParticleIDs", "Coordinates"], PTYPE, max_particles_per_read=100
        )
    whole_totals = whole.totals(ReadCounter.particle_datasets)
    batched_totals = batched.totals(ReadCounter.particle_datasets)

    assert batched_totals["rows_read_again"] == 0
    assert batched_totals["opens"] == whole_totals["opens"]
    assert batched_totals["requests"] > whole_totals["requests"]

    coordinates = batched.totals(lambda name, dataset: dataset.endswith("/Coordinates"))
    assert coordinates["rows_read"] == whole.totals(lambda name, dataset: dataset.endswith("/Coordinates"))["rows_read"]


def test_batches_lower_the_peak_memory(fake_runs):
    """Check that placing the largest file of the all-sky shell a quarter at a time takes less memory at its peak."""
    lightcone = new_lightcone(HALO_SIM, None, "cube")

    _, plans = lightcone.gather_files(PTYPE, (0.1, 0.2))
    file_number, plan = max(enumerate(plans), key=lambda item: item[1].nr_placed)
    assert plan.nr_placed > plan.nr_particles  # shared cells: copies in several tiles

    def peak(max_particles_per_read):
        tracemalloc.start()
        kept = 0
        for piece in lightcone.iter_file_pieces(
            file_number, PTYPE, ["ParticleIDs", "Coordinates"], max_particles_per_read=max_particles_per_read
        ):
            kept += len(piece["ParticleIDs"])  # written out or mapped here, then dropped
            del piece
        peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak_bytes, kept

    peak(None)  # the first read sets up the units
    whole, kept_whole = peak(None)
    quarter, kept_quarter = peak(plan.nr_particles // 4)

    assert kept_quarter == kept_whole > 0
    assert quarter < 0.75 * whole, (quarter, whole)


def test_mpi_batches_on_one_rank(fake_runs):
    """With MPI (one rank here; several in test_mpi_matches_serial), batches place the same particles as serial."""
    MPI = pytest.importorskip("mpi4py.MPI")
    lightcone = new_lightcone(HALO_SIM, None, "cube")
    serial = lightcone.place_snapshot_particles_in_shell((0.1, 0.2), PROPERTIES, PTYPE)[PTYPE]
    batched = lightcone.place_snapshot_particles_in_shell(
        (0.1, 0.2), PROPERTIES, PTYPE, comm=MPI.COMM_SELF, max_particles_per_read=100
    )[PTYPE]
    assert_same_rows(rows_of(batched), rows_of(serial))


# test with pieces


@pytest.mark.parametrize("max_particles_per_read", [None, 100])
def test_pieces_are_the_file(fake_runs, max_particles_per_read):
    """
    Check that the pieces of each file are the particles of place_file_in_shell where each piece is one tile from a
    batch and only pieces keeping particles are yielded.
    """
    lightcone = new_lightcone(HALO_SIM, None, "sphere")
    nr_files, plans = lightcone.gather_files(PTYPE, (0.1, 0.2))
    nr_batches = 0

    for file_number, plan in enumerate(plans):
        whole = lightcone.place_file_in_shell(file_number, PTYPE, PROPERTIES)
        pieces = list(
            lightcone.iter_file_pieces(file_number, PTYPE, PROPERTIES, max_particles_per_read=max_particles_per_read)
        )

        if len(whole.get("ParticleIDs", [])) == 0:
            assert not pieces
            continue

        tiles = {use.tile for use in plan.uses}

        for piece in pieces:
            meta = piece.metadata
            assert (meta["snapshot_number"], meta["file_number"]) == (plan.snap_nr, plan.file_nr)
            assert meta["tile"] in tiles
            assert len(piece["ParticleIDs"]) > 0 and "ExpansionFactors" in piece

            np.testing.assert_array_equal(piece["TileIndex"].value, np.tile(meta["tile"], (len(piece["TileIndex"]), 1)))

            if max_particles_per_read is None:
                assert meta["batch"] == 0

        nr_batches = max(nr_batches, 1 + max(piece.metadata["batch"] for piece in pieces))
        joined = {name: np.concatenate([piece[name] for piece in pieces]) for name in whole}
        assert_same_rows(rows_of(joined), rows_of(whole))

        np.testing.assert_allclose(
            np.sort(joined["ExpansionFactors"].value), np.sort(whole["ExpansionFactors"].value), rtol=1e-12
        )

    if max_particles_per_read is not None:
        assert nr_batches > 1


def test_pieces_of_one_tile(fake_runs):
    """Test that for a tile that a file doesn't use yields none."""
    lightcone = new_lightcone(HALO_SIM, None, "cube")
    _, plans = lightcone.gather_files(PTYPE, (0.1, 0.2))
    file_number, plan = max(enumerate(plans), key=lambda item: len(item[1].uses))
    tile = plan.uses[0].tile

    pieces = list(lightcone.iter_file_pieces(file_number, PTYPE, PROPERTIES, tile=tile, max_particles_per_read=50))
    assert pieces and all(piece.metadata["tile"] == tile for piece in pieces)

    one = lightcone.place_file_in_shell(file_number, PTYPE, PROPERTIES, tile=tile)
    joined = {name: np.concatenate([piece[name] for piece in pieces]) for name in one}
    assert_same_rows(rows_of(joined), rows_of(one))

    shell_tiles = {piece.tile for piece in lightcone.last_shell.pieces}

    unused = [
        (number, tile) for number, other in enumerate(plans) for tile in shell_tiles - {u.tile for u in other.uses}
    ]
    assert unused

    number, tile = unused[0]
    assert list(lightcone.iter_file_pieces(number, PTYPE, PROPERTIES, tile=tile)) == []
