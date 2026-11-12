#!/bin/env python
"""
Identify and report replicas of the same SWIFT snapshot cells in a SnapshotLightcone.

To generate a report of closeley repeated copies of cells in a lightcone::

    from swiftlet import Simulation, SnapshotAllSky
    from swiftlet.repeat_check import find_close_cell_repeats

    lightcone = SnapshotAllSky(Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL"), orientation_lock="cube")
    report = find_close_cell_repeats(lightcone, (0.0, 0.1))
    print(report.summary())

"""

import warnings
from dataclasses import dataclass, field
import numpy as np


class CloseRepeatWarning(UserWarning):
    """
    Warning for when copies of the same snapshot cell lie next to each other, or are close together,
    but in different tiles along a line of sight.
    """


@dataclass
class CloseRepeatReport:
    """
    Class to report the result of find_close_cell_repeats.

    :ivar max_snapshot_difference:      The most the snapshot numbers of two copies paired differ by: 0 for copies from
                                        the same snapshot (the default), None for copies from any snapshots
    :vartype max_snapshot_difference:   int | None
    :ivar nr_tiles:             number of box tiles the shell reaches
    :vartype nr_tiles:          int
    :ivar nr_cells:             number of (snapshot, cell) in the shell, each counted once however many copies it has
    :vartype nr_cells:          int
    :ivar nr_cell_copies:       number of cell copies in the shell (cells overlapping a tile's part of the shell)
    :vartype nr_cell_copies:    int
    :ivar nr_cells_repeated:    number of (snapshot, cell) with more than one copy in the shell
    :vartype nr_cells_repeated: int
    :ivar nr_cell_pairs:        number of pairs of copies of the same (snapshot, cell): every pair that could be a close
                                    repeat, of which the clashing and line of sight pairs are a part
    :vartype nr_cell_pairs:     int
    :ivar nr_touching:          number of pairs of copies of the same cell that share a face, counted once
    :vartype nr_touching:       int
    :ivar nr_line_of_sight:     number of pairs of copies of the same cell within angle_deg of each other on the sky and
                                    closer than max_los_separation along the line of sight
    :vartype nr_line_of_sight:  int
    :ivar closest_separation:   smallest distance [Mpc] between two copies of the same cell (inf if none is repeated).
                                With find_closest_copy=False, the smallest among the pairs found within the search
                                radius (inf if none)
    :vartype closest_separation: float
    :ivar cell_size:            side length [Mpc] of a cell
    :vartype cell_size:         float
    :ivar boxsize:              side length [Mpc] of the snapshot box
    :vartype boxsize:           float
    :ivar angle_deg:            angle [deg] used for the line of sight pairs
    :vartype angle_deg:         float
    :ivar max_los_separation:   distance [Mpc] along the line of sight used for the line of sight pairs
    :vartype max_los_separation: float
    :ivar pairs:                the pairs reported (clashing, along a line of sight, or closer than max_separation), a
                                dict of arrays: "snapshot", "cell", "tile_a", "tile_b", "centre_a", "centre_b"[Mpc],
                                "separation" [Mpc], "angle_deg", "los_separation" [Mpc], "clashing" and
                                "line_of_sight" (bool)
    :vartype pairs:             dict
    :ivar cells:                every repeated (snapshot, cell): "snapshot", "cell", "nr_copies" and
                                "closest_separation" [Mpc] to its own copies (with find_closest_copy=False, only
                                if a copy is within the search radius, otherwise inf)
    :vartype cells:             dict
    """

    max_snapshot_difference: object
    nr_tiles: int
    nr_cells: int
    nr_cell_copies: int
    nr_cells_repeated: int
    nr_cell_pairs: int
    nr_touching: int
    nr_line_of_sight: int
    closest_separation: float
    cell_size: float
    boxsize: float
    angle_deg: float
    max_los_separation: float
    pairs: dict = field(repr=False)
    cells: dict = field(repr=False)

    def summary(self):
        """Returns a summary message describing the close repeats found."""
        # check if same cell for same snapshot or just any snapshot
        same_cell = (
            "the same cell"
            if self.max_snapshot_difference == 0
            else "the same cell (from any snapshots)"
            if self.max_snapshot_difference is None
            else f"the same cell (from snapshots up to {self.max_snapshot_difference} apart)"
        )
        summary_message = (
            f"{self.nr_touching} pairs of {same_cell} sharing a face\n{self.nr_line_of_sight} pairs are "
            f"within {self.angle_deg:g} deg on the sky and closer than {self.max_los_separation:g} Mpc along line of "
            f"sight\n{self.nr_cells_repeated} of the cells are repeated, from {self.nr_cell_copies} cell copies; "
            f"closest copies {self.closest_separation:.1f} Mpc apart, cells {self.cell_size:g} Mpc"
        )
        return summary_message


def _tile_label(tile):
    """A tile as 4 integers: (nx, ny, nz, layer), layer -1 if the tile has none (skip orientation_lock="sphere")."""
    tile = [int(n) for n in tile]
    return tile + [-1] * (4 - len(tile))


def _cells_in_shell(lightcone, ang_radius_deg, records, tile_cells=None, cell_range=None):
    """
    Find overlapping cells int the lightcone shell.

    Overlap every cell centre from each tile with the last gathered pieces and then keep the cells overlapping
    with the tile's part of the lightcone .

    Returns arrays:

        for the kept cells:
            - snapshot number
            - cell index
            - tile label (N, 4)
            - centre [Mpc] (N, 3)

        - cell size  [Mpc]
        - box size   [Mpc]

    :param  lightcone:      the lightcone, after gather_files
    :type   lightcone:      SnapshotLightcone
    :param  ang_radius_deg: angular radius [deg] of the beam, None for all-sky
    :type   ang_radius_deg: float
    :param  records:        the pieces recorded by lightcone.last_snapshot_read
    :type   records:        list
    :param  tile_cells:     the cells of each piece, {(snapshot, tile): cell indices, sorted}, or None for every cell
    :type   tile_cells:     dict
    :param  cell_range:     if given, only the cells numbered from cell_range[0] to cell_range[1] - 1
    :type   cell_range:     tuple
    """
    snap_nr, cell, tiles, orientation, centres = [], [], [], [], []
    orientation_ids = {}
    cell_size = boxsize = None

    beam_vec = None if ang_radius_deg is None else np.asarray(lightcone.beam_vec, dtype=float)

    # for record in lightcone.last_snapshot_read:
    for record in records:
        cell_data = lightcone._cell_data_cache[record.snap_nr]
        box_centres = cell_data["cell_centres"].to_value("Mpc")
        cell_size = float(np.max(cell_data["cell_size"].to_value("Mpc")))
        boxsize = float(np.max(cell_data["snap_boxsize"].to_value("Mpc")))
        half_diagonal = 0.5 * np.sqrt(3.0) * cell_size

        # the cells gather_files found,
        # tests them with a 1 cell buffer
        if tile_cells is None:
            candidates = np.arange(len(box_centres))
        else:
            candidates = tile_cells.get((record.snap_nr, record.tile), np.zeros(0, dtype=np.int32)).astype(np.int64)

        if cell_range is not None:
            candidates = candidates[
                np.searchsorted(candidates, cell_range[0]) : np.searchsorted(candidates, cell_range[1])
            ]
        if len(candidates) == 0:
            continue
        coords = lightcone.Snapshot2Lightcone(record.snap_nr, box_centres[candidates], tile=record.tile)[record.tile]

        # get lightcone coords from last run
        r = np.linalg.norm(coords, axis=1)
        r_in, r_out = lightcone.cosmo.comoving_distance(list(record.z_updated)).to_value("Mpc")

        # make mask to keep cells close to the lightcones footprint
        keep = (r >= r_in - half_diagonal) & (r <= r_out + half_diagonal)

        if beam_vec is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                angle = np.arccos(np.clip(coords @ beam_vec / r, -1.0, 1.0))
                buffer = np.arcsin(np.clip(half_diagonal / r, 0.0, 1.0))
            keep &= (r <= half_diagonal) | (angle <= np.deg2rad(ang_radius_deg) + buffer)

        n = int(np.sum(keep))
        # add kept cells
        snap_nr.append(np.full(n, record.snap_nr, dtype=np.int64))
        cell.append(candidates[keep].astype(np.int64))
        tiles.append(np.tile(np.asarray(_tile_label(record.tile), dtype=np.int64), (n, 1)))
        key = tuple(tuple(int(x) for x in np.ravel(part)) for part in lightcone._snapshot_reorientation(record.tile))
        orientation.append(np.full(n, orientation_ids.setdefault(key, len(orientation_ids)), dtype=np.int64))
        centres.append(coords[keep])

    if not snap_nr:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros((0, 4), dtype=np.int64), empty, np.zeros((0, 3)), cell_size, boxsize

    snap_nr, cell, tiles = np.concatenate(snap_nr), np.concatenate(cell), np.concatenate(tiles)
    orientation, centres = np.concatenate(orientation), np.concatenate(centres)

    if lightcone.orientation_lock != "sphere":
        return snap_nr, cell, tiles, orientation, centres, cell_size, boxsize

    # with orientation_lock="sphere" a tile crossing a layer sphere is used once for each layer, so a cell on the
    # sphere can be kept by both, at the same place with the same orientation: the same copy, kept once
    position = np.rint(centres / cell_size * 1e6).astype(np.int64)
    _, first = np.unique(np.column_stack([snap_nr, cell, orientation, position]), axis=0, return_index=True)
    first = np.sort(first)
    return snap_nr[first], cell[first], tiles[first], orientation[first], centres[first], cell_size, boxsize


def find_close_cell_repeats(
    lightcone,
    redshift_range,
    angle_deg=1.0,
    max_los_separation=None,
    max_separation=None,
    output_file=None,
    warn=True,
    copies_per_block=4_000_000,
    max_snapshot_difference=1,
    find_closest_copy=False,
    ptype="PartType1",
):
    """
    Identify where replicas of the same snapshot cells in adjacent tiles that are either clashing or have a small
    angular separation on the sky.

    Construct the lightcone at a cell level (no particle data is read) and identify, at the edges of
    neighbouring tiles in the lightcone, if replicas of any cell are either in contact or within a given distance
    (in general, max_separation, or along a given line of sight, max_los_separation).
    This function works for any COLIBRE or FLAMINGO simulation and all orientation tables (orientation lock and seeds
    included).

    Returns the separation between pairs of repeated cells as a CloseRepeatReport.

    :param  lightcone:                  the lightcone (SnapshotAllSky or SnapshotBeam), without an MPI communicator
    :type   lightcone:                  SnapshotLightcone
    :param  redshift_range:             minimum and maximum redshift of the shell [z_min, z_max]
    :type   redshift_range:             tuple
    :param  angle_deg:                  two copies of the same cell within this angle [deg] on the sky are considered to
                                        be in the same line of sight.
    :type   angle_deg:                  float
    :param  max_los_separation:         copies on the same line of sight closer than this [Mpc] along it are reported.
                                        If None, half the box size
    :type   max_los_separation:         float
    :param  max_separation:             if given, also report every pair of copies closer than this [Mpc]
    :type   max_separation:             float
    :param  output_file:                if given, write the pairs and the closest separation of every repeated cell to
                                        this HDF5 file
    :type   output_file:                str
    :param  warn:                       if True, warn (CloseRepeatWarning) if touching or line of sight pairs are found
    :type   warn:                       bool
    :param  copies_per_block:           most cell copies compared at once, to limit the memory needed
    :type   copies_per_block:           int
    :param  max_snapshot_difference:    the most the snapshot numbers of two copies of a cell paired may differ by.
                                        If 0, only use pairs from the same snapshot (i.e. the same structures at
                                        the same time).
                                        Default is 1, which includes pairs in the same and adjacent snapshots. If None,
                                        pair copies of the same cell from any snapshots, this is a very computationally
                                        expensive task.
    :type   max_snapshot_difference:    int or None
    :param  find_closest_copy:          If True, find the closest copy of every repeated cell, however far apart,
                                        with a nearest neighbour search of every copy (about a third of the time spent
                                        on the pairs). If False, skip that search: each cell's closest separation is the
                                        smallest among the pairs already found within the search radius (the larger of
                                        max_los_separation plus a cell diagonal, and max_separation), and inf for a cell
                                        with no copy that close. The pairs reported are the same either way.
    :type   find_closest_copy:          bool
    :param  ptype:                      particle type whose cells are used, e.g. "PartType1"
    :type   ptype:                      str
    """
    from scipy.spatial import cKDTree
    from .threads import thread_limit

    # find the pieces of the shell, and the cells of each piece that overlap it
    beam_args = {}
    lightcone._tile_cells = {}
    # attempt to gather files
    try:
        lightcone.gather_files(ptype, redshift_range, **beam_args)
        tile_cells = lightcone._tile_cells
    finally:
        lightcone._tile_cells = None

    records_of_snapshot = {}
    for record in lightcone.last_snapshot_read or []:
        records_of_snapshot.setdefault(int(record.snap_nr), []).append(record)

    row_property_names = (
        "snapshot",
        "snapshot_b",
        "cell",
        "tile_a",
        "tile_b",
        "centre_a",
        "centre_b",
        "separation",
        "angle_deg",
        "los_separation",
        "touching",
        "line_of_sight",
    )

    # set up data structure
    rows = {name: [] for name in row_property_names}
    cell_rows = {name: [] for name in ("snapshot", "cell", "nr_copies", "closest_separation")}

    nr_cell_copies = nr_cells_in_shell = nr_cell_pairs = nr_line_of_sight = 0
    nr_tiles = len(
        {tuple(_tile_label(record.tile)[:3]) for records in records_of_snapshot.values() for record in records}
    )
    cell_size = boxsize = None
    workers = thread_limit() or -1

    # the copies paired: for each snapshot on its own, or for every snapshot together (labelled -1), keeping pairs of
    # copies from snapshots at most max_snapshot_difference apart
    window = None if max_snapshot_difference is None else int(max_snapshot_difference)  # snapshot window

    if window is not None and window < 0:
        raise ValueError(f"max_snapshot_difference must be 0 or more, or None, not {max_snapshot_difference}")
    if window == 0:
        groups = records_of_snapshot
    else:
        groups = {-1: [record for snapshot in sorted(records_of_snapshot) for record in records_of_snapshot[snapshot]]}
        cells_per_snapshot = {
            len(lightcone._cell_data_cache[snapshot]["cell_centres"]) for snapshot in records_of_snapshot
        }
        if len(cells_per_snapshot) > 1:
            raise ValueError(
                "max_snapshot_difference other than 0 needs every snapshot to have the same cells, not "
                f"{cells_per_snapshot}"
            )

    # iterate through snapshots one block of cells at a time: the cells numbered from first to last - 1
    # Max number of copies per block = copies_per_block copies (estimated from the cells gather_files found)

    blocks = []

    for snapshot in sorted(groups):
        selected = [
            tile_cells.get((record.snap_nr, record.tile), np.zeros(0, dtype=np.int64)) for record in groups[snapshot]
        ]
        nr_cells = len(lightcone._cell_data_cache[int(groups[snapshot][0].snap_nr)]["cell_centres"])
        copies_of_cell = np.bincount(np.concatenate(selected), minlength=nr_cells) if selected else np.zeros(nr_cells)

        edges = np.searchsorted(
            np.cumsum(copies_of_cell), np.arange(copies_per_block, copies_of_cell.sum(), copies_per_block)
        )

        bounds = np.unique(np.concatenate([[0], edges, [nr_cells]]))
        blocks += [(snapshot, (int(first), int(last))) for first, last in zip(bounds[:-1], bounds[1:])]

    for snapshot, cell_range in blocks:
        snap_nr, cell, tiles, orientation, centres, size, box = _cells_in_shell(
            lightcone, lightcone.ang_radius_deg, groups[snapshot], tile_cells, cell_range
        )

        cell_size, boxsize = (size, box) if size is not None else (cell_size, boxsize)
        nr_cell_copies += len(cell)
        if len(cell) == 0:
            continue

        if max_los_separation is None:
            max_los_separation = boxsize

        half_diagonal = 0.5 * np.sqrt(3.0) * cell_size

        # the copies of each cell, and every pair of copies of a cell (the cells of a block are of one snapshot, or of
        # every snapshot, and in no other block), from snapshots at most window apart
        cells_here, copy_of, copies = np.unique(cell, return_inverse=True, return_counts=True)

        nr_cells_in_shell += len(cells_here)

        if window in (0, None):
            pairs_of_cell = copies * (copies - 1) // 2
        else:
            snapshots_here, snapshot_of = np.unique(snap_nr, return_inverse=True)
            per_snapshot = np.zeros((len(cells_here), len(snapshots_here)), dtype=np.int64)
            np.add.at(per_snapshot, (copy_of, snapshot_of), 1)
            pairs_of_cell = np.sum(per_snapshot * (per_snapshot - 1) // 2, axis=1)
            for i, j in zip(
                *np.nonzero(np.triu(np.abs(snapshots_here[:, None] - snapshots_here[None, :]) <= window, 1))
            ):
                pairs_of_cell += per_snapshot[:, i] * per_snapshot[:, j]
        nr_cell_pairs += int(np.sum(pairs_of_cell))

        repeated = pairs_of_cell > 0
        if not np.any(repeated):
            continue

        # copies of different cells are kept far apart by a fourth coordinate, the cell's number times a spacing larger
        # than any distance between copies, so neighbours in the tree are always copies of the same cell
        search_radius = max(max_los_separation + 2.0 * half_diagonal, 1.000001 * cell_size, max_separation or 0.0)
        r = np.linalg.norm(centres, axis=1)
        spacing = 4.0 * float(r.max()) + 2.0 * search_radius + 1.0
        tree = cKDTree(np.column_stack([centres, copy_of * spacing]))

        # copies of a cell at the same place, from two snapshots, are one piece of space: a cell on the hand-over
        # between the snapshots placed by both, each filling only its own side. They are not pairs of copies
        if window != 0:
            same_place = tree.query_pairs(1e-6 * cell_size, output_type="ndarray")
            if window is not None and len(same_place):
                same_place = same_place[np.abs(snap_nr[same_place[:, 0]] - snap_nr[same_place[:, 1]]) <= window]
            if len(same_place):
                pairs_of_cell = pairs_of_cell - np.bincount(copy_of[same_place[:, 0]], minlength=len(cells_here))
                nr_cell_pairs -= len(same_place)
                repeated = pairs_of_cell > 0

        # every pair of copies of a cell within the search radius
        pairs_here = tree.query_pairs(search_radius, output_type="ndarray")
        if window not in (0, None) and len(pairs_here):
            pairs_here = pairs_here[np.abs(snap_nr[pairs_here[:, 0]] - snap_nr[pairs_here[:, 1]]) <= window]

        a, b = pairs_here[:, 0], pairs_here[:, 1]
        offset = centres[b] - centres[a]
        separation = np.linalg.norm(offset, axis=1)

        closest = np.full(len(cells_here), np.inf)

        if find_closest_copy:
            # the closest copy of each cell. Across snapshots, the nearest of its 16 nearest copies at another place,
            # from a snapshot in the window

            k = 2 if window == 0 else min(17, len(cell))
            distance, nearest = tree.query(np.column_stack([centres, copy_of * spacing]), k=k, workers=workers)
            valid = (copy_of[nearest[:, 1:]] == copy_of[:, None]) & np.isfinite(distance[:, 1:])
            if window is not None and window > 0:
                valid &= np.abs(snap_nr[np.minimum(nearest[:, 1:], len(cell) - 1)] - snap_nr[:, None]) <= window

            # not a copy at the same place, see above
            valid &= distance[:, 1:] > 1e-6 * cell_size
            np.minimum.at(closest, copy_of, np.where(valid, distance[:, 1:], np.inf).min(axis=1))

        elif len(pairs_here):
            # the closest of the pairs already found, not counting copies at the same place (see above)
            apart = separation > 1e-6 * cell_size
            np.minimum.at(closest, copy_of[a[apart]], separation[apart])

        # update rows
        cell_rows["snapshot"].append(np.full(int(np.sum(repeated)), snapshot, dtype=np.int64))
        cell_rows["cell"].append(cells_here[repeated])
        cell_rows["nr_copies"].append(copies[repeated])
        cell_rows["closest_separation"].append(closest[repeated])

        # no need to keep going if no pairs found in snapshot window
        if len(pairs_here) == 0:
            continue

        # check separation of cell a and b in each pair
        a, b = pairs_here[:, 0], pairs_here[:, 1]
        offset = centres[b] - centres[a]
        separation = np.linalg.norm(offset, axis=1)
        direction = centres / np.where(r > 0.0, r, 1.0)[:, None]
        cell_angle = np.arcsin(np.clip(half_diagonal / np.where(r > 0.0, r, np.inf), 0.0, 1.0))
        centre_angle = np.arccos(np.clip(np.sum(direction[a] * direction[b], axis=1), -1.0, 1.0))

        # copies with the same orientation a whole number of box lengths apart: the box repeated periodically
        periodic = (orientation[a] == orientation[b]) & np.all(
            np.abs(offset / boxsize - np.rint(offset / boxsize)) < 1e-6, axis=1
        )

        touching = np.isclose(separation, cell_size, rtol=1e-6, atol=0.0)

        # could the two copies hold copies of a particle within angle_deg and closer than max_los_separation
        line_of_sight = (
            ~periodic
            & (centre_angle <= np.deg2rad(angle_deg) + cell_angle[a] + cell_angle[b])
            & (separation < max_los_separation + 2.0 * half_diagonal)
        )

        report = touching | line_of_sight
        if max_separation is not None:
            report |= separation < max_separation
        nr_line_of_sight += int(np.sum(line_of_sight))

        # update the rows for pairs being reported
        a, b = a[report], b[report]

        rows["snapshot"].append(snap_nr[a])
        rows["snapshot_b"].append(snap_nr[b])
        rows["cell"].append(cell[a])
        rows["tile_a"].append(tiles[a])
        rows["tile_b"].append(tiles[b])
        rows["centre_a"].append(centres[a])
        rows["centre_b"].append(centres[b])
        rows["separation"].append(separation[report])
        rows["angle_deg"].append(np.rad2deg(centre_angle[report]))
        rows["los_separation"].append(np.abs(r[b] - r[a]))
        rows["touching"].append(touching[report])
        rows["line_of_sight"].append(line_of_sight[report])

    if max_los_separation is None:
        max_los_separation = boxsize if boxsize is not None else 0.0

    def joined(arrays, dtype, shape=(0,)):
        return np.concatenate(arrays).astype(dtype) if arrays else np.zeros(shape, dtype=dtype)

    # store all pairs
    pairs = {
        "snapshot": joined(rows["snapshot"], np.int64),
        "snapshot_b": joined(rows["snapshot_b"], np.int64),
        "cell": joined(rows["cell"], np.int64),
        "tile_a": joined(rows["tile_a"], np.int64, (0, 4)),
        "tile_b": joined(rows["tile_b"], np.int64, (0, 4)),
        "centre_a": joined(rows["centre_a"], float, (0, 3)),
        "centre_b": joined(rows["centre_b"], float, (0, 3)),
        "separation": joined(rows["separation"], float),
        "angle_deg": joined(rows["angle_deg"], float),
        "los_separation": joined(rows["los_separation"], float),
        "touching": joined(rows["touching"], bool),
        "line_of_sight": joined(rows["line_of_sight"], bool),
    }

    # store cell information for the pairs
    cells = {
        "snapshot": joined(cell_rows["snapshot"], np.int64),
        "cell": joined(cell_rows["cell"], np.int64),
        "nr_copies": joined(cell_rows["nr_copies"], np.int64),
        "closest_separation": joined(cell_rows["closest_separation"], float),
    }

    # a touching pair whose cells straddle a snapshot hand-over is kept by each snapshot: count it once, by its cell
    # and its two tiles
    touching = pairs["touching"]
    nr_touching = len(
        {
            (int(c), frozenset([tuple(a), tuple(b)]))
            for c, a, b in zip(pairs["cell"][touching], pairs["tile_a"][touching], pairs["tile_b"][touching])
        }
    )
    report = CloseRepeatReport(
        max_snapshot_difference=window,
        nr_tiles=nr_tiles,
        nr_cells=nr_cells_in_shell,
        nr_cell_copies=nr_cell_copies,
        nr_cell_pairs=nr_cell_pairs,
        nr_cells_repeated=len(cells["cell"]),
        nr_touching=nr_touching,
        nr_line_of_sight=nr_line_of_sight,
        closest_separation=float(cells["closest_separation"].min()) if len(cells["cell"]) else np.inf,
        cell_size=cell_size if cell_size is not None else np.nan,
        boxsize=boxsize if boxsize is not None else np.nan,
        angle_deg=float(angle_deg),
        max_los_separation=float(max_los_separation),
        pairs=pairs,
        cells=cells,
    )

    if output_file is not None:
        write_close_repeats(output_file, report, lightcone, redshift_range, lightcone.ang_radius_deg)

    if warn and (nr_touching > 0 or nr_line_of_sight > 0):
        warnings.warn(
            f"close repeats in the lightcone (orientation_lock={lightcone.orientation_lock!r}, "
            f"orientation_seed={lightcone.orientation_seed}): {report.summary()}",
            CloseRepeatWarning,
            stacklevel=2,
        )
    return report


def write_close_repeats(filename, report, lightcone, redshift_range, ang_radius_deg=None):
    """
    Write a report on the number of repeated or clashing cell pairs (CloseRepeatReport) to a HDF5 file.

    In the output file:

    -   The "pairs" and "cells" groups contain datasets for every pair of repeated cells and corresponding cell flagged
        in the report, written directly from the report.pairs and report.cells of CloseRepeatReport.
    -   The file's attributes store the counts, the settings of the check, and the lightcone's metadata
        (simulation, orientation seed, lock and table).

    :param  filename:       path of the HDF5 file
    :type   filename:       str
    :param  report:         the report from find_close_cell_repeats
    :type   report:         CloseRepeatReport
    :param  lightcone:      the lightcone checked
    :type   lightcone:      SnapshotLightcone
    :param  redshift_range: minimum and maximum redshift of the shell
    :type   redshift_range: tuple
    :param  ang_radius_deg: angular radius [deg] of the beam, None for all-sky
    :type   ang_radius_deg: float
    """
    import h5py
    from .snapshot_lightcone import write_lightcone_metadata

    with h5py.File(filename, "w") as f:
        for group_name, arrays in (("pairs", report.pairs), ("cells", report.cells)):
            group = f.create_group(group_name)
            for name, values in arrays.items():
                group.create_dataset(name, data=values)

        f["pairs"].attrs["description"] = (
            "pairs of copies of the same cell that share a face (touching), lie on the "
            "same line of sight closer than max_los_separation (line_of_sight), or are "
            "closer than max_separation; distances in comoving Mpc"
        )

        f["cells"].attrs["description"] = (
            "every repeated (snapshot, cell): number of copies and closest separation [Mpc]"
        )

        attrs_properties = (
            "nr_tiles",
            "nr_cells",
            "nr_cell_copies",
            "nr_cell_pairs",
            "nr_cells_repeated",
            "nr_touching",
            "nr_line_of_sight",
            "closest_separation",
            "cell_size",
            "boxsize",
            "angle_deg",
            "max_los_separation",
        )

        for name in attrs_properties:
            f.attrs[name] = getattr(report, name)

        # include the max snapshot difference
        if report.max_snapshot_difference is None:
            max_diff_attr = -1
        else:
            max_diff_attr = report.max_snapshot_difference

        f.attrs["max_snapshot_difference"] = max_diff_attr

        write_lightcone_metadata(f.create_group("lightcone"), lightcone.lightcone_metadata(redshift_range))
