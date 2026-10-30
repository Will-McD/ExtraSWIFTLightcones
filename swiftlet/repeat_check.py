#!/bin/env python
import warnings
from dataclasses import dataclass, field
import numpy as np

class CloseRepeatWarning(UserWarning):
    """
    Warning for when copies of the same snapshot cell lie next to each other, or are close together, 
    but in different tile layers along a line of sight, in a lightcone. 
    """

@dataclass
class CloseRepeatReport:
    """
    Class to report the result of find_close_cell_repeats.

    nr_cell_copies:         number of cell copies in the shell (cells overlapping a tile's part of the shell)
    nr_cells_repeated:      number of (snapshot, cell) with more than one copy in the shell
    nr_touching:            number of pairs of copies of the same cell that share a face, counted once
    nr_line_of_sight:       number of pairs of copies of the same cell within angle_deg of each other on the sky and
                                closer than max_los_separation along the line of sight
    closest_separation:     smallest distance [Mpc] between two copies of the same cell (inf if none is repeated)
    cell_size:              side length [Mpc] of a cell
    boxsize:                side length [Mpc] of the snapshot box
    angle_deg:              angle [deg] used for the line of sight pairs
    max_los_separation:     distance [Mpc] along the line of sight used for the line of sight pairs
    pairs:                  the pairs reported (touching, along a line of sight, or closer than max_separation), a
                                dict of arrays: "snapshot", "cell", "tile_a", "tile_b" (N, 4; the fourth entry is
                                the layer for orientation_lock="sphere", -1 otherwise), "centre_a", "centre_b"
                                (N, 3) [Mpc], "separation" [Mpc], "angle_deg", "los_separation" [Mpc], "touching"
                                and "line_of_sight" (bool)
    cells:                  every repeated (snapshot, cell): "snapshot", "cell", "nr_copies" and
                                "closest_separation" [Mpc] to its own copies
    """
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
        """
        Summary mesage describing the close repeats found.
        """
        summary_message = f"{self.nr_touching} pairs of cell replicas sharing a face" # main message
        summary_message+= f"\n{self.nr_line_of_sight} pairs are with {self.angle_deg:g} deg on the sky" # closeness as an angular seperation 
        summary_message+= f" and closer than {self.max_los_separation:g} Mpc along line of sight" # closeness in comoving distance 
        summary_message+= f"\n{self.nr_cells_repeated} of the cells are repeated, from {self.nr_cell_copies} cell;" # numb repeats 
        summary_message+= f" closest copies {self.closest_separation:.1f} Mpc apart, cells {self.cell_size:g} Mpc" #how close repeats are

        return (summary_message)


def _tile_label(tile):
    """
    A tile as 4 integers: (nx, ny, nz, layer), with layer -1 if the tile has none (all but orientation_lock="sphere").
    """
    tile = [int(n) for n in tile]
    return tile + [-1] * (4 - len(tile))


def _cells_in_shell(lightcone, ang_radius_deg, records, tile_cells=None, cell_range=None):

    """
    Overlap every cell centre from each tile with the last gathered pieces and then keep the cells overlaping 
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

    #for record in lightcone.last_snapshot_read:
    for record in records:
        cell_data = lightcone._cell_data_cache[record.snap_nr]
        box_centres = cell_data["cell_centres"].to_value("Mpc")
        cell_size = float(np.max(cell_data["cell_size"].to_value("Mpc")))
        boxsize = float(np.max(cell_data["snap_boxsize"].to_value("Mpc")))
        half_diagonal = 0.5 * np.sqrt(3.) * cell_size

        # the cells gather_files found,
        # tests them with a 1 cell buffer
        if tile_cells is None:
            candidates=(np.arange(len(box_centres)))
        else:
             candidates = (tile_cells.get((record.snap_nr, record.tile), np.zeros(0, dtype=np.int32)).astype(np.int64))

        if cell_range is not None:
            candidates = candidates[np.searchsorted(candidates, cell_range[0]):np.searchsorted(candidates, cell_range[1])]
        if len(candidates) == 0:
            continue
        coords = lightcone.Snapshot2Lightcone(record.snap_nr, box_centres[candidates], tile=record.tile)[record.tile]

        # get lightcone coords from last run 
        #coords = lightcone.Snapshot2Lightcone(record.snap_nr, box_centres, tile=record.tile)[record.tile]
        r = np.linalg.norm(coords, axis=1)
        r_in, r_out = lightcone.cosmo.comoving_distance(list(record.z_updated)).to_value("Mpc")
        
        # make mask to keep cells close to the lightcones footprint
        keep = (r >= r_in - half_diagonal) & (r <= r_out + half_diagonal)

        if beam_vec is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                angle = np.arccos(np.clip(coords @ beam_vec / r, -1., 1.))
                buffer = np.arcsin(np.clip(half_diagonal / r, 0., 1.))
            keep &= (r <= half_diagonal) | (angle <= np.deg2rad(ang_radius_deg) + buffer)

        n = int(np.sum(keep))
        # add kept cells 
        snap_nr.append(np.full(n, record.snap_nr, dtype=np.int64))
        #cell.append(np.flatnonzero(keep).astype(np.int64))
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


def find_close_cell_repeats(lightcone, redshift_range, ang_radius_deg=None, ptype="PartType1", angle_deg=1.,
                            max_los_separation=None, max_separation=None, output_file=None, warn=True,
                            copies_per_block=4_000_000):
    """
    Find copies of the same cell of a snapshot that lie next to each other (i.e. share a face), or are close together
    along a line of sight.  
    Computed using only the cell centres (i.e. no particles are read). 

    Works for any simulation, orientation table, orientation_lock and orientation_seed, but is serial 
    only (a lightcone without a communicator).

    Returns a CloseRepeatReport.

    :param  lightcone:          the lightcone (SnapshotAllSky or SnapshotBeam), without an MPI communicator
    :type   lightcone:          SnapshotLightcone
    :param  redshift_range:     minimum and maximum redshift of the shell [z_min, z_max]
    :type   redshift_range:     tuple
    :param  ang_radius_deg:     angular radius [deg] of the beam, for a SnapshotBeam; None for SnapshotAllSky
    :type   ang_radius_deg:     float
    :param  ptype:              particle type whose cells are used, e.g. "PartType1"
    :type   ptype:              str
    :param  angle_deg:          two copies within this angle [deg] on the sky are on the same line of sight
    :type   angle_deg:          float
    :param  max_los_separation: copies on the same line of sight closer than this [Mpc] along it are reported. If
                                    None, half the box size
    :type   max_los_separation: float
    :param  max_separation:     if given, also report every pair of copies closer than this [Mpc]
    :type   max_separation:     float
    :param  output_file:        if given, write the pairs and the closest separation of every repeated cell to this
                                    HDF5 file
    :type   output_file:        str
    :param  warn:               if True, warn (CloseRepeatWarning) if touching or line of sight pairs are found
    :type   warn:               boolean
    :param  copies_per_block:   most cell copies compared at once, to limit the memory needed
    :type   copies_per_block:   int
    """
    
    from scipy.spatial import cKDTree
    from .threads import thread_limit
    
    # find the pieces of the shell, and the cells of each piece that overlap it
    beam_args = {} if ang_radius_deg is None else {"ang_radius_deg": ang_radius_deg}
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
        "snapshot", "cell", 
        "tile_a", "tile_b", 
        "centre_a", "centre_b", 
        "separation", "angle_deg", 
        "los_separation", "touching", 
        "line_of_sight"
    )
    
    rows = {name: [] for name in row_property_names}
    cell_rows = {name: [] for name in ("snapshot", "cell", "nr_copies", "closest_separation")}
    
    #nr_cell_copies = nr_line_of_sight = 0
    nr_cell_copies = nr_cells_in_shell = nr_cell_pairs = nr_line_of_sight = 0
    nr_tiles = len({tuple(_tile_label(record.tile)[:3]) for records in records_of_snapshot.values() for record in records})
    cell_size = boxsize = None
    workers = thread_limit() or -1

    # a block of each snapshot's cells at a time: the cells numbered from first to last - 1, at most copies_per_block
    # copies (estimated from the cells gather_files found)
    blocks = []
    for snapshot in sorted(records_of_snapshot):
        selected = [tile_cells.get((record.snap_nr, record.tile), np.zeros(0, dtype=np.int64))
                    for record in records_of_snapshot[snapshot]]
        nr_cells = len(lightcone._cell_data_cache[snapshot]["cell_centres"])
        copies_of_cell = np.bincount(np.concatenate(selected), minlength=nr_cells) if selected else np.zeros(nr_cells)
        edges = np.searchsorted(np.cumsum(copies_of_cell), np.arange(copies_per_block, copies_of_cell.sum(),
                                                                      copies_per_block))
        bounds = np.unique(np.concatenate([[0], edges, [nr_cells]]))
        blocks += [(snapshot, (int(first), int(last))) for first, last in zip(bounds[:-1], bounds[1:])]

    for snapshot, cell_range in blocks:
        snap_nr, cell, tiles, orientation, centres, size, box = _cells_in_shell(
            lightcone, ang_radius_deg, records_of_snapshot[snapshot], tile_cells, cell_range)
        cell_size, boxsize = (size, box) if size is not None else (cell_size, boxsize)
        nr_cell_copies += len(cell)
        #if len(cell) < 2:
        if len(cell) == 0:
            continue
        if max_los_separation is None:
            max_los_separation = boxsize
        half_diagonal = 0.5 * np.sqrt(3.) * cell_size

        # the copies of each cell
        cells_here, copy_of, copies = np.unique(cell, return_inverse=True, return_counts=True)
        
        nr_cells_in_shell += len(cells_here)
        nr_cell_pairs += int(np.sum(copies * (copies - 1) // 2))

        repeated = copies > 1
        if not np.any(repeated):
            continue

        # copies of different cells are kept far apart by a fourth coordinate, the cell's number times a spacing larger
        # than any distance between copies, so neighbours in the tree are always copies of the same cell
        search_radius = max(max_los_separation + 2. * half_diagonal, 1.000001 * cell_size, max_separation or 0.)
        r = np.linalg.norm(centres, axis=1)
        spacing = 4. * float(r.max()) + 2. * search_radius + 1.
        tree = cKDTree(np.column_stack([centres, copy_of * spacing]))

        # the closest copy of each copy: its nearest neighbour, a copy of the same cell if the cell has more than one
        distance, nearest = tree.query(np.column_stack([centres, copy_of * spacing]), k=2, workers=workers)
        same = copy_of[nearest[:, 1]] == copy_of
        closest = np.full(len(cells_here), np.inf)
        np.minimum.at(closest, copy_of[same], distance[same, 1])
        cell_rows["snapshot"].append(np.full(int(np.sum(repeated)), snapshot, dtype=np.int64))
        cell_rows["cell"].append(cells_here[repeated])
        cell_rows["nr_copies"].append(copies[repeated])
        cell_rows["closest_separation"].append(closest[repeated])

        # every pair of copies of a cell within the search radius
        pairs_here = tree.query_pairs(search_radius, output_type="ndarray")
        if len(pairs_here) == 0:
            continue
        a, b = pairs_here[:, 0], pairs_here[:, 1]
        offset = centres[b] - centres[a]
        separation = np.linalg.norm(offset, axis=1)
        direction = centres / np.where(r > 0., r, 1.)[:, None]
        cell_angle = np.arcsin(np.clip(half_diagonal / np.where(r > 0., r, np.inf), 0., 1.))
        centre_angle = np.arccos(np.clip(np.sum(direction[a] * direction[b], axis=1), -1., 1.))
        # copies with the same orientation a whole number of box lengths apart: the box repeated periodically
        periodic = ((orientation[a] == orientation[b])
                    & np.all(np.abs(offset / boxsize - np.rint(offset / boxsize)) < 1e-6, axis=1))
        touching = np.isclose(separation, cell_size, rtol=1e-6, atol=0.)
        # could the two copies hold copies of a particle within angle_deg and closer than max_los_separation
        line_of_sight = (~periodic & (centre_angle <= np.deg2rad(angle_deg) + cell_angle[a] + cell_angle[b])
                         & (separation < max_los_separation + 2. * half_diagonal))
        report = touching | line_of_sight
        if max_separation is not None:
            report |= separation < max_separation
        nr_line_of_sight += int(np.sum(line_of_sight))
        a, b = a[report], b[report]
        rows["snapshot"].append(snap_nr[a])
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
        max_los_separation = boxsize if boxsize is not None else 0.

    def joined(arrays, dtype, shape=(0,)):
        return np.concatenate(arrays).astype(dtype) if arrays else np.zeros(shape, dtype=dtype)

    pairs = {
        "snapshot": joined(rows["snapshot"], np.int64), "cell": joined(rows["cell"], np.int64),
        "tile_a": joined(rows["tile_a"], np.int64, (0, 4)), "tile_b": joined(rows["tile_b"], np.int64, (0, 4)),
        "centre_a": joined(rows["centre_a"], float, (0, 3)), "centre_b": joined(rows["centre_b"], float, (0, 3)),
        "separation": joined(rows["separation"], float), "angle_deg": joined(rows["angle_deg"], float),
        "los_separation": joined(rows["los_separation"], float),
        "touching": joined(rows["touching"], bool), "line_of_sight": joined(rows["line_of_sight"], bool),
    }
    cells = {
        "snapshot": joined(cell_rows["snapshot"], np.int64), "cell": joined(cell_rows["cell"], np.int64),
        "nr_copies": joined(cell_rows["nr_copies"], np.int64),
        "closest_separation": joined(cell_rows["closest_separation"], float),

    }
    # a touching pair whose cells straddle a snapshot hand-over is kept by each snapshot: count it once, by its cell
    # and its two tiles
    touching = pairs["touching"]
    nr_touching = len({(int(c), frozenset([tuple(a), tuple(b)]))
                       for c, a, b in zip(pairs["cell"][touching], pairs["tile_a"][touching], pairs["tile_b"][touching])})
    report = CloseRepeatReport(
        nr_tiles=nr_tiles, nr_cells=nr_cells_in_shell, nr_cell_copies=nr_cell_copies, nr_cell_pairs=nr_cell_pairs,
        nr_cells_repeated=len(cells["cell"]), nr_touching=nr_touching,
        nr_line_of_sight=nr_line_of_sight,
        closest_separation=float(cells["closest_separation"].min()) if len(cells["cell"]) else np.inf,
        cell_size=cell_size if cell_size is not None else np.nan, boxsize=boxsize if boxsize is not None else np.nan,
        angle_deg=float(angle_deg), max_los_separation=float(max_los_separation), pairs=pairs, cells=cells)

    if output_file is not None:
        write_close_repeats(output_file, report, lightcone, redshift_range, ang_radius_deg)
    if warn and (nr_touching > 0 or nr_line_of_sight > 0):
        warnings.warn(f"close repeats in the lightcone (orientation_lock={lightcone.orientation_lock!r}, "
                      f"orientation_seed={lightcone.orientation_seed}): {report.summary()}", CloseRepeatWarning,
                      stacklevel=2)
    return report


def write_close_repeats(filename, report, lightcone, redshift_range, ang_radius_deg=None):
    """
    Write a report on the number of touching and close cell pairs (CloseRepeatReport) to a HDF5 file. 
    
        - The "pairs" and "cells" groups hold arrays of report.pairs and report.cells. 
        - The file's attributes store the counts, the settings of the check, and the 
            lightcone's metadata (simulation, orientation seed, lock and table)

    :param  filename:       path of the HDF5 file
    :type   filename:       str
    :param  report:         the report of find_close_cell_repeats
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
        
        f["pairs"].attrs["description"] = ("pairs of copies of the same cell that share a face (touching), lie on the "
                                           "same line of sight closer than max_los_separation (line_of_sight), or are "
                                           "closer than max_separation; distances in comoving Mpc")
        
        f["cells"].attrs["description"] = "every repeated (snapshot, cell): number of copies and closest separation [Mpc]"

        attrs_properties=("nr_tiles", "nr_cells", "nr_cell_copies", "nr_cell_pairs", "nr_cells_repeated", 
                "nr_touching", "nr_line_of_sight", "closest_separation", "cell_size", "boxsize", "angle_deg", "max_los_separation")
        
        for name in attrs_properties:
            f.attrs[name] = getattr(report, name)

        write_lightcone_metadata(f.create_group("lightcone"), lightcone.lightcone_metadata(redshift_range, ang_radius_deg))

