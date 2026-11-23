"""
Helpers shared by the tests of reading snapshot files (test_read_plan, test_read_file_by_file, test_batches_and_pieces,
test_read_counts).
"""

from typing import NamedTuple

import numpy as np


class TileFilePiece(NamedTuple):
    """The cells of one (snapshot, tile) piece of a shell that are in one snapshot file."""

    snap_nr: int
    tile: tuple
    file_nr: int
    offsets: np.ndarray
    lengths: np.ndarray


def tile_file_pieces(shell):
    """
    Returns the (snapshot, tile, file) pieces of a gathered shell, found independently of the read plans.

    For each (snapshot, tile) piece of the shell (last_shell.pieces), the cells of its tile (last_shell.tile_cells)
    grouped by the file that holds them, in the order they were found.
    Files whose cells have no particles are left out.

    :param  shell:  the shell gathered, e.g. lightcone.last_shell
    :type   shell:  swiftlet.snapshot_lightcone.ShellRead
    """
    ptype = shell.ptype
    found = []
    for piece in shell.pieces:
        cell_data = shell.cell_data[int(piece.snap_nr)]

        if ptype not in cell_data["cells_in_file"]:
            continue

        cells = np.asarray(shell.tile_cells[(piece.snap_nr, piece.tile)], dtype=np.int64)
        files = cell_data["cells_in_file"][ptype][cells]

        for file_nr in np.unique(files):
            in_file = cells[files == file_nr]
            lengths = cell_data["lengths"][ptype][in_file]

            if np.sum(lengths) > 0:
                offsets = cell_data["offsets"][ptype][in_file]
                found.append(TileFilePiece(int(piece.snap_nr), piece.tile, int(file_nr), offsets, lengths))

    return found


def rows_of(particles):
    """
    Return rows of (snapshot, tile, particle ID, position [Mpc]) of placed particles, sorted, so two sets of particles
    can be compared.
    """
    snap_nr = particles["SnapshotNumber"].value.astype(int)
    tiles = particles["TileIndex"].value.astype(int)
    ids = particles["ParticleIDs"].value.astype(int)
    coords = particles["Coordinates"].to_value("Mpc")
    keys = [coords[:, 2], coords[:, 1], coords[:, 0], ids] + [tiles[:, i] for i in reversed(range(tiles.shape[1]))]
    order = np.lexsort(keys + [snap_nr])
    return snap_nr[order], tiles[order], ids[order], coords[order]


def assert_same_rows(rows, other):
    """Check that two sets of rows are the same particles, in the same tiles, at the same positions."""
    assert len(rows[0]) == len(other[0]) > 0

    for mine, theirs in zip(rows[:3], other[:3]):
        np.testing.assert_array_equal(mine, theirs)

    np.testing.assert_allclose(rows[3], other[3], atol=1e-6)
