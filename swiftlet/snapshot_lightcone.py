#!/bin/env python
"""
Build lightcones from SWIFT snapshots and halo catalogues.


The construction of all lightcones from the snapshot or halo catlogues is carried out through
:class:`~swiftlet.snapshot_lightcone.SnapshotBeam` and :class:`~swiftlet.snapshot_lightcone.SnapshotAllSky`, which share
the same pipeline given by :class:`~swiftlet.snapshot_lightcone.SnapshotLightcone`.

For example, the dark matter particles of a shell of an all-sky lightcone::

    from swiftlet import Simulation, SnapshotAllSky

    simulation = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
    lightcone = SnapshotAllSky(simulation, orientation_lock="cube")
    redshift_range= (0.05, 0.1) # redshift range of the lightcone
    particles = lightcone.place_snapshot_particles_in_shell(redshift_range, ["Masses"], particle_types=["PartType1"])
    particles["PartType1"]["Coordinates"]  # positions in the lightcone, with the observer at the origin

"""

import numpy as np
import os
import unyt
import h5py
import healpy as hp
from mpi4py import MPI
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple  # from collections import namedtuple
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from lightcone_io.particle_reader import merge_cells
from . import snapshot_orientation as box_structure
from . import snapshot_units as sw_units
from .simulation_suites import Simulation
from .snapshot_cosmology import SnapshotCosmology, as_astropy_cosmology
from .soap_reader import soap_catalogue_length, read_soap_rows, read_soap_in_chunks, soap_unit_attributes  # noqa: F401
from .soap_reader import comoving as soap_comoving
from . import snapshot_redshifts
import datetime as dt
import warnings


try:
    from numba import njit, prange

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# define simple printout functions
def separator_str(n=35, line_separator="~"):
    """
    Make fancy line separation.

    Returns a newline followed by the separator line.
    :param  n:              number of characters in the line
    :type   n:              int
    :param  line_separator: character used to draw the line
    :type   line_separator: str
    """
    sep_str = line_separator * n
    return "\n" + sep_str


def message(rank, m, show_time_date=False):
    """
    Print a new message if serial or on the zero'th rank.

    :param  rank:               MPI rank. If None, always print
    :type   rank:               int
    :param  m:                  message to print
    :type   m:                  str
    :param  show_time_date:   If True, prefix the message with the current time
    :type   show_time_date:   bool
    """
    if show_time_date:
        current_time = dt.datetime.now()
        m = f"[{current_time.strftime('%H:%M:%S')}]\t" + m

    if rank is None or rank == 0:
        print(m)


def rank_message(rank, m):
    """
    Print a new message with a timestamp and the rank number on each rank.

    :param  rank:   MPI rank
    :type   rank:   int
    :param  m:      message to print
    :type   m:      str
    """
    current_time = dt.datetime.now()
    time_str = current_time.strftime("%H:%M:%S")
    # +\n, end="" stops multiple ranks printing on top of the same line
    print(f"\t[Rank {rank:03d}, @{time_str}]\t" + m + "\n", end="")


def print_coord_ranges(x, indent=0, rank=None):
    """
    Print the minimum and maximum of the coordinates along each axis.

    :param  x:      coordinates, shape (N, 3)
    :type   x:      numpy.ndarray or unyt.array.unyt_array
    :param  indent: number of tabs to indent each line by
    :type   indent: int
    :param  rank:   MPI rank, only rank 0 prints. If None, always print
    :type   rank:   int
    """
    indent = "\t" * max(int(indent), 0)
    units = None
    if isinstance(x, unyt.unyt_array):
        try:
            x = x.to_value("Mpc")
        except unyt.exceptions.UnitConversionError:
            x, units = x.value, str(x.units)
    if units is None:
        units = "Mpc"
    x = np.asarray(x)
    ranges = ", ".join(f"{axis}: {np.min(x[:, i]):.2f} - {np.max(x[:, i]):.2f}" for i, axis in enumerate("xyz"))
    coords_str = f"{indent}" + f"{ranges} [{units}]"
    message(rank, coords_str)


# classes for holding the internal state of SnapshotLightcone
# changed from named tuples to improve the documentations clarity


class SnapshotReadRecord(NamedTuple):
    """
    A (snapshot, tile) piece of the lightcone, with the redshift range it fills.

    The gather step keeps them in ``last_shell.pieces``, e.g. to find the tiles of a snapshot in the last shell.

    :ivar snap_nr:      snapshot number
    :vartype snap_nr:   int
    :ivar tile:         index of the periodic replica tile
    :vartype tile:      tuple
    :ivar z_updated:    minimum and maximum redshift of the shell filled by this snapshot tile
    :vartype z_updated: tuple
    """

    snap_nr: int
    tile: tuple
    z_updated: tuple


class HaloCatalogueSpec(NamedTuple):
    """
    The SOAP catalogue to read for one tile of the lightcone.

    Its haloes are moved into the tile and kept if they are in the tile's redshift range of the shell.

    :ivar snap_nr:      snapshot number
    :vartype snap_nr:   int
    :ivar tile:         index of the periodic replica tile
    :vartype tile:      tuple
    :ivar filename:     name of the SOAP catalogue file
    :vartype filename:  str
    :ivar z_range:      minimum and maximum redshift of the shell filled by this tile
    :vartype z_range:   tuple
    """

    snap_nr: int
    tile: tuple
    filename: str
    z_range: tuple


class HaloReadChunk(NamedTuple):
    """
    The rows [start, stop) of a snapshot's SOAP catalogue, read by one rank.

    :ivar snap_nr:      snapshot number
    :vartype snap_nr:   int
    :ivar start:        first row to read
    :vartype start:     int
    :ivar stop:         row after the last row to read
    :vartype stop:      int
    """

    snap_nr: int
    start: int
    stop: int


# planning classes and function for snapshot files
class TileUse(NamedTuple):
    """
    A tile (and, with the "sphere" lock, a layer) that needs some of the file's planned cells.

    :ivar tile:         index of the periodic replica tile, as the gather step gives it: (nx, ny, nz), or
                        (nx, ny, nz, layer) with the "sphere" lock
    :vartype tile:      tuple
    :ivar z_range:      minimum and maximum redshift of the shell filled by this tile (the piece's z_updated)
    :vartype z_range:   tuple
    :ivar cells:        positions in the FilePlan's cells of the cells this use needs, in the order the gather step
                        found them
    :vartype cells:     numpy.ndarray
    """

    tile: tuple
    z_range: tuple
    cells: np.ndarray


class FilePlan(NamedTuple):
    """
    What to read from one snapshot file for every tile that uses the cells stored in it.

    Each cell needed by any tile is. A tile that uses the file is a TileUse, naming its cells by their
    positions in the plan, so a cell shared by several tiles is read once and placed once per tile. The gather step
    makes one plan per file with particles in the selected cells, and keeps them in ``last_shell.plans``.

    :ivar snap_nr:      snapshot number
    :vartype snap_nr:   int
    :ivar file_nr:      number of the snapshot file
    :vartype file_nr:   int
    :ivar cells:        indices in the snapshot of the cells needed by any use, each once, in the order of their
                        offsets in the file
    :vartype cells:     numpy.ndarray
    :ivar offsets:      offset in the file of the first particle of each cell
    :vartype offsets:   numpy.ndarray
    :ivar lengths:      number of particles of each cell
    :vartype lengths:   numpy.ndarray
    :ivar uses:         the tiles that take particles from the file, a TileUse each, in the order the gather step found
                        them
    :vartype uses:      tuple
    """

    snap_nr: int
    file_nr: int
    cells: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    uses: tuple

    @property
    def nr_particles(self):
        """Number of particles to read from the file."""
        return int(np.sum(self.lengths))

    @property
    def nr_placed(self):
        """Number of particles to place from the file, includes copies of each cell in the lightcone."""
        return int(sum(np.sum(self.lengths[use.cells]) for use in self.uses))


def _make_file_plan(snap_nr, file_nr, uses, offsets, lengths):
    """
    Returns the FilePlan of one snapshot file, from the cells each tile needs from it.

    :param  snap_nr:    snapshot number
    :type   snap_nr:    int
    :param  file_nr:    number of the snapshot file
    :type   file_nr:    int
    :param  uses:       (tile, z_range, cell indices in the snapshot) of each tile that uses the file, in order
    :type   uses:       list
    :param  offsets:    offset in its file of each cell of the snapshot, for the particle type
    :type   offsets:    numpy.ndarray
    :param  lengths:    number of particles of each cell of the snapshot, for the particle type
    :type   lengths:    numpy.ndarray
    """
    use_cells = [np.asarray(cells, dtype=np.int64) for _, _, cells in uses]  # all cells to read from file
    cells = np.unique(np.concatenate(use_cells))  # get only the unique cells to read
    cells = cells[np.argsort(offsets[cells], kind="stable")]  # in the order of the particles in the file
    by_index = np.argsort(cells)

    # find the cells that are needed from each file to build the lightcone
    plan_uses = tuple(
        TileUse(tile, tuple(z_range), by_index[np.searchsorted(cells[by_index], tile_cells)])
        for (tile, z_range, _), tile_cells in zip(uses, use_cells)
    )

    return FilePlan(int(snap_nr), int(file_nr), cells, offsets[cells], lengths[cells], plan_uses)


# repot classes: objects holding the information about the lightcone that are reported by the gather step and used in
# the read and placing of particles.
@dataclass
class ShellRead:
    """
    What the gather step found for one particle type of a lightcone and everything the read step needs.

    Each gather step makes a new one, :meth:`~SnapshotLightcone.gather_files` or
    :meth:`~SnapshotLightcone.place_snapshot_particles_in_shell` for each particle type, and keeps it as ``last_shell``,
    so nothing is carried from one call to the next.

    :ivar shell_z:                  minimum and maximum redshift of the shell
    :vartype shell_z:               tuple
    :ivar ptype:                    particle type, e.g. "PartType1"
    :vartype ptype:                 str
    :ivar snapshots:                snapshot numbers used, from high to low redshift
    :vartype snapshots:             numpy.ndarray
    :ivar z_ranges:                 redshift range each snapshot fills, {snapshot number: (z_min, z_max)}, wider where
                                    it fills in for a missing neighbour. A snapshot not in it fills its own range
    :vartype z_ranges:              dict
    :ivar pieces:                   the (snapshot, tile) pieces of the shell, with the redshift range each fills, as
                                    SnapshotReadRecord
    :vartype pieces:                list
    :ivar plans:                    the files to read, each planned once: a FilePlan per (snapshot, file) with
                                    particles in the selected cells, with the tiles that use it, in the order of the
                                    snapshots, then of the file numbers
    :vartype plans:                 list
    :ivar tile_cells:               the cells of each piece that overlap the shell, {(snapshot, tile): cell indices}
    :vartype tile_cells:            dict
    :ivar cell_data:                the cell metadata of the call's snapshots, {snapshot number: cell metadata}. Shared
                                    by the particle types of one call, and emptied when the call ends
    :vartype cell_data:             dict
    """

    # properties tracked by the internal shell read states are now be recorded here instead
    shell_z: tuple
    ptype: str
    snapshots: np.ndarray
    z_ranges: dict
    pieces: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    tile_cells: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)


@dataclass
class HaloShellRead:
    """
    What the gather step (:meth:`~swiftlet.snapshot_lightcone.SnapshotLightcone.gather_halo_catalogues`) found for a
    lightcone, the halo equivalent of ShellRead.

    Kept as ``last_shell`` and reused by :meth:`~swiftlet.snapshot_lightcone.SnapshotLightcone.gather_halo_catalogues`
    when it is called with the same settings.

    :ivar settings:                 the arguments of gather_halo_catalogues, compared to reuse it
    :vartype settings:              tuple
    :ivar shell_z:                  minimum and maximum redshift of the shell
    :vartype shell_z:               tuple
    :ivar snapshots:                snapshot numbers used, from high to low redshift
    :vartype snapshots:             numpy.ndarray
    :ivar z_ranges:                 redshift range each snapshot fills, as in ShellRead
    :vartype z_ranges:              dict
    :ivar pieces:                   the (snapshot, tile) pieces of the shell, as SnapshotReadRecord
    :vartype pieces:                list
    :ivar catalogues:               the SOAP catalogue to read for each piece, as HaloCatalogueSpec
    :vartype catalogues:            list
    :ivar catalogue_lengths:        number of haloes in each catalogue, {snapshot number: rows}
    :vartype catalogue_lengths:     dict
    :ivar cell_snapshot:            the snapshot the box and cell metadata are read from
    :vartype cell_snapshot:         int
    :ivar cell_data:                its cell metadata, {cell_snapshot: cell metadata}
    :vartype cell_data:             dict
    """

    # properties tracked by the internal shell read states are now be recorded here instead
    settings: tuple
    shell_z: tuple
    snapshots: np.ndarray
    z_ranges: dict
    pieces: list = field(default_factory=list)
    catalogues: list = field(default_factory=list)
    catalogue_lengths: dict = field(default_factory=dict)
    cell_snapshot: int = None
    cell_data: dict = field(default_factory=dict)

    @staticmethod
    def settings_of(halo_format, shell_z, ang_radius_deg, use_snapshots, snapshot_number, observer_position=None):
        """
        Returns the arguments of :meth:`~SnapshotLightcone.gather_halo_catalogues` as a tuple.

        :param  halo_format:        see gather_halo_catalogues
        :type   halo_format:        str
        :param  shell_z:            minimum and maximum redshift of the shell
        :type   shell_z:            sequence of two floats
        :param  ang_radius_deg:     angular radius [deg] of the beam, None for all-sky
        :type   ang_radius_deg:     float
        :param  use_snapshots:      see gather_halo_catalogues
        :type   use_snapshots:      list or numpy.ndarray
        :param  snapshot_number:    see gather_halo_catalogues
        :type   snapshot_number:    int
        :param  observer_position:  the observer position [Mpc] in the snapshot box, or None
        :type   observer_position:  array-like, shape (3,)
        """
        # halo_settings internal read state now given here
        return (
            str(halo_format),
            tuple(float(z) for z in shell_z),
            ang_radius_deg,
            None if use_snapshots is None else tuple(int(n) for n in np.atleast_1d(use_snapshots)),
            None if snapshot_number is None else int(snapshot_number),
            None if observer_position is None else tuple(float(x) for x in observer_position),
        )


# numba optimised function for flagging cells and particles in a wedge
if _HAVE_NUMBA:
    # setting parallel=False when using MPI parallel processing
    @njit(parallel=False, fastmath=True, cache=True)
    def _classify_wedge_numba(coords, axis, chi_inner, chi_outer, half_angle, buffer_radius, mask):
        """
        Flag points inside a cone about the axis that lie between two comoving distances.

        Returns the updated mask, True where point lies within the cone and comoving distances.

        :param  coords:         comoving coordinates [Mpc] relative to the observer, shape (N, 3)
        :type   coords:         numpy.ndarray
        :param  axis:           unit vector along the axis of the cone
        :type   axis:           numpy.ndarray
        :param  chi_inner:      inner comoving distance [Mpc] of the shell
        :type   chi_inner:      float
        :param  chi_outer:      outer comoving distance [Mpc] of the shell
        :type   chi_outer:      float
        :param  half_angle:     angular radius [rad] of the cone
        :type   half_angle:     float
        :param  buffer_radius:  distance [Mpc] to extend the boundaries of the shell by
        :type   buffer_radius:  float
        :param  mask:           output boolean buffer, shape (N,)
        :type   mask:           numpy.ndarray
        """
        n = coords.shape[0]
        ax, ay, az = axis[0], axis[1], axis[2]
        for i in prange(n):
            x = coords[i, 0]
            y = coords[i, 1]
            z = coords[i, 2]

            r = np.sqrt(x * x + y * y + z * z)
            along = x * ax + y * ay + z * az
            px = x - along * ax
            py = y - along * ay
            pz = z - along * az
            r_perp = np.sqrt(px * px + py * py + pz * pz)
            angle = np.arctan2(r_perp, along)

            if r > buffer_radius:
                ratio = buffer_radius / r
                if ratio > 1.0:
                    ratio = 1.0
                angle_buffer = np.arcsin(ratio)
            else:
                angle_buffer = np.pi

            mask[i] = (
                (r > chi_inner - buffer_radius)
                and (r <= chi_outer + buffer_radius)
                and (along > -buffer_radius)
                and (angle <= half_angle + angle_buffer)
            )
        return mask


# set up redshift-comoving dist interpolation grid
def define_redshift_at_comoving_distance_function(cosmo, zmin=0.0001, zmax=5.05, n_grid=3000, method="precise"):
    """
    Returns a CubicSpline function giving the redshift at a comoving distance [Mpc].

    :param  cosmo:  cosmology of the simulation
    :type   cosmo:  SnapshotCosmology, astropy.cosmology.w0waCDM or str
    :param  zmin:   minimum redshift of the grid
    :type   zmin:   float
    :param  zmax:   maximum redshift of the grid
    :type   zmax:   float
    :param  n_grid: number of points in the grid
    :type   n_grid: int
    :param  method: how the grid is spaced:

                        - 'fast' (default): a uniform grid in redshift, its comoving distances found in
                            one call. With 5000 points out to z = 5 the redshift is within 1e-12 of the
                            exact one, in ~2 ms
                        - 'precise': a uniform grid in comoving distance, each point's redshift found by
                            root finding. As accurate, but ~8 s for 5000 points
    :type   method: str
    """
    # ensure cosmology is correct format, use wrapper to read all used formats across astropy, lightcone_io, swiftsim_io
    cosmo = as_astropy_cosmology(cosmo)

    if method == "fast":
        # fastes method to build grid
        z_grid = np.linspace(zmin, zmax, n_grid)
        r_grid = unyt.unyt_array.from_astropy(cosmo.comoving_distance(z_grid)).to_value("Mpc")
    elif method == "precise":
        # uniform grid in comoving dist gives more precise interp function than grid from z
        r_range = unyt.unyt_array.from_astropy(cosmo.comoving_distance([zmin, zmax])).to_value("Mpc")
        r_grid = np.linspace(r_range[0], r_range[1], n_grid)

        def r_from_z(z):
            return unyt.unyt_quantity.from_astropy(cosmo.comoving_distance(z)).to_value("Mpc")

        z_grid = np.array([brentq(lambda z: r_from_z(z) - r, 1e-10, zmax) for r in r_grid])
    else:
        raise ValueError(f"grid building method {method} not recognised, use 'fast' or 'precise'")

    return CubicSpline(r_grid, z_grid)


# maximum angular radius of beam, larger beams become far slower than utilising a full sky map due to searching boundary
# conditions.
MAX_BEAM_ANG_RADIUS_DEG = 60.01

# random primes used for offsets of selecting rotation, reflection and periodic shifts about each axis when creating
# new snapshot boxes
TILE_HASH_PRIMES = (374761393, 2147483647, 668265263)
TILE_HASH_SEED_PRIME = 2246822519

# ways of locking the orientation of the box tiles into layers
ORIENTATION_LOCK_OPTIONS = (None, "none", "cube", "sphere")


# warnings for bad spacing between snapshots of particles and halo catalogues
class SnapshotGapWarning(UserWarning):
    """
    Warning that snapshots (or SOAP catalogues) a lightcone shell needs are missing, and that the available snapshots
    will fill the missing redshifts instead (see ``SnapshotLightcone.snapshots_for_shell``).
    """


######################################################
# observer position / repositioning functions and global variables


def _check_observer_position(observer_position):
    """
    Returns an observer position as a numpy.ndarray of 3 floats [Mpc], or raises ValueError.

    :param  observer_position:  (x, y, z) comoving position in the snapshot box, [Mpc] if not a unyt array
    :type   observer_position:  array-like or unyt.unyt_array, shape (3,)
    """
    value = observer_position
    if isinstance(value, unyt.unyt_array):
        value = value.to_value("Mpc")
    try:
        position = np.array(value, dtype=np.float64)
    except (TypeError, ValueError):
        position = None
    if position is None or position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            f"observer_position must be 3 finite numbers (x, y, z) [Mpc] or None, not {observer_position!r}"
        )
    return position


######################################################
# Metadata class and related functions

# the settings recorded with every lightcone output .metadata, what is needed to exactly reproduce the same lightcone
LIGHTCONE_METADATA_NAMES = (
    "lightcone_type",
    "boxsize_resolution",
    "simulation_name",
    "simulation_suite_name",  # simulation info
    "orientation_seed",
    "orientation_lock",  # geometry info
    "orientation_table",
    "orientation_table_fingerprint",  # geometry table info
    "beam_vector",
    "ang_radius_deg",
    "redshift_range",  # beam info
    "observer_position",
    "observer_corner",
    "observer_cell_shift",  # observer: position given [Mpc], cell corner used [Mpc], shift of every tile [cells]
    "version",  # the SWIFTLET version
)


class LightconeOutput(dict):
    """
    The output dict of particles or haloes placed in a lightcone ({name: array}), the lightcones settings are stored
    as ``.metadata``.
    """

    def __init__(self, data, metadata):  # noqa: D107
        """
        :param  data:       the particle or halo data placed in the lightcone, {name: array}
        :type   data:       dict
        :param  metadata:   the settings used to make the lightcone, kept as .metadata
        :type   metadata:   dict
        """
        super().__init__(data)
        self.metadata = dict(metadata)


def write_lightcone_metadata(hdf5_object, metadata):
    """
    Write the settings that made a lightcone as attributes of an HDF5 file, group or dataset.

    None is written as "none".

    :param  hdf5_object:    h5py file, group or dataset to add the attributes to
    :type   hdf5_object:    h5py.File, h5py.Group or h5py.Dataset
    :param  metadata:       the settings, see SnapshotLightcone.lightcone_metadata
    :type   metadata:       dict
    """
    for name, value in metadata.items():
        if value is None:
            value = "none"
        elif isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value)
        hdf5_object.attrs[name] = value


def read_lightcone_metadata(hdf5_object):
    """
    Read the settings written by write_lightcone_metadata.

    Returns a dict of the settings, with "none" read back as None, except orientation_lock, where "none" is one of
    its values and is kept as the string.

    :param  hdf5_object:    h5py file, group or dataset holding the attributes
    :type   hdf5_object:    h5py.File, h5py.Group or h5py.Dataset
    """
    metadata = {}
    for name in LIGHTCONE_METADATA_NAMES:
        if name in hdf5_object.attrs:
            value = hdf5_object.attrs[name]
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            metadata[name] = None if value == "none" and name != "orientation_lock" else value
    return metadata


def write_halo_lightcone_file(filename, haloes, soap_filename, properties, metadata, nside=16, order="nest"):
    """
    Write the halo lightcone data to a .hdf5 file.

    Each property of the lightcone's haloes is written as a dataset, in the format of the FLAMINGO halo lightcones of
    lightcone_io, so the file can be read with lightcone_io.halo_reader.HaloLightconeFile.

    :param  filename:       path of the HDF5 file to write, e.g. lightcone_haloes_0127.hdf5
    :type   filename:       str
    :param  haloes:         haloes placed by :func:`swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell`,
                            all of them on this rank
    :type   haloes:         dict
    :param  soap_filename:  SOAP catalogue of the haloes' snapshot, for the unit attributes and the Cosmology and
                            Units groups
    :type   soap_filename:  str
    :param  properties:     {name: values} , additional SOAP properties (not returned by place_halos_in_shell) of each
                            halo to write e.g. {"InputHalos/IsCentral": ...}
    :type   properties:     dict
    :param  metadata:       .metadata of the haloes placed, see
                            :func:`swiftlet.snapshot_lightcone.SnapshotLightcone.lightcone_metadata`
    :type   metadata:       dict
    :param  nside:          HEALPix resolution of the index, as lightcone_io uses (16)
    :type   nside:          int
    :param  order:          HEALPix pixel ordering of the index, "nest" or "ring"
    :type   order:          str
    """
    # remove units and convert to regular numpy arrays.
    centre = haloes["Lightcone/HaloCentre"].to_value("Mpc")
    expansion_factor = haloes["Lightcone/ExpansionFactor"].value
    # find pixels that each halo is found in low nside map
    pixel = (
        hp.vec2pix(nside, centre[:, 0], centre[:, 1], centre[:, 2], nest=(order == "nest"))
        if len(centre)
        else np.zeros(0, dtype=int)
    )
    by_pixel = np.argsort(pixel, kind="stable")
    num_per_pixel = np.bincount(pixel, minlength=hp.nside2npix(nside)).astype(np.int64)
    first_in_pixel = (np.cumsum(num_per_pixel) - num_per_pixel).astype(np.int64)

    # add unit conversion info as found in LightconeIO
    dimensionless = {
        "Conversion factor to CGS (not including cosmological corrections)": [1.0],
        "Conversion factor to physical CGS (including cosmological corrections)": [1.0],
        **{f"{dim} exponent": [0.0] for dim in ("U_I", "U_L", "U_M", "U_T", "U_t", "a-scale", "h-scale")},
    }

    with h5py.File(soap_filename, "r") as soap, h5py.File(filename, "w") as outfile:
        for group_name in ("Cosmology", "Units"):
            if group_name in soap:
                group = outfile.create_group(group_name)
                for name, value in soap[group_name].attrs.items():
                    group.attrs[name] = value
        index = outfile.create_group("Index")
        index.attrs["nside"] = nside
        index.attrs["order"] = order
        index.create_dataset("FirstHaloInPixel", data=first_in_pixel)
        index.create_dataset("NumHalosPerPixel", data=num_per_pixel)

        def write(name, values, attrs, description):
            dataset = outfile.create_dataset(
                name, data=np.asarray(values)[by_pixel], compression="gzip", compression_opts=6
            )
            for attr_name, attr_value in attrs.items():
                dataset.attrs[attr_name] = attr_value
            dataset.attrs["Description"] = description

        # the halo centres in the units of SOAP's InputHalos/HaloCentre (comoving), with its attributes
        centre_attrs = soap_unit_attributes(soap, "InputHalos/HaloCentre")
        centre_units = soap_comoving(
            read_soap_rows(soap_filename, ["InputHalos/HaloCentre"], 0, 0)["InputHalos/HaloCentre"]
        ).units
        file_units_in_mpc = unyt.unyt_quantity(1.0, centre_units).to_value("Mpc")
        write(
            "Lightcone/HaloCentre",
            centre / file_units_in_mpc,
            centre_attrs,
            "Comoving position of the halo centre relative to the observer, in the lightcone",
        )
        write(
            "Lightcone/Redshift",
            1.0 / expansion_factor - 1.0,
            dimensionless,
            "Redshift of the halo at its comoving distance from the observer",
        )
        write(
            "Lightcone/SnapshotNumber",
            haloes["Lightcone/SnapshotNumber"].value.astype(np.int64),
            dimensionless,
            "Snapshot the halo was placed from",
        )
        write(
            "InputHalos/SOAPIndex",
            haloes["InputHalos/SOAPIndex"].value.astype(np.int64),
            dimensionless,
            "Index of the halo in the input SOAP catalogue",
        )
        write(
            "InputHalos/HaloCatalogueIndex",
            haloes["InputHalos/HaloCatalogueIndex"].value.astype(np.int64),
            soap_unit_attributes(soap, "InputHalos/HaloCatalogueIndex"),
            "Index of the halo in the input halo catalogue",
        )
        # store attrs names
        if properties is None:
            lightcone = outfile["Lightcone"]
            lightcone.attrs["property_names"] = [
                "Lightcone/HaloCentre",
                "Lightcone/Redshift",
                "Lightcone/SnapshotNumber",
                "InputHalos/SOAPIndex",
                "InputHalos/HaloCatalogueIndex",
            ]
        else:
            for name, values in properties.items():
                write(
                    name,
                    np.asarray(values),
                    soap_unit_attributes(soap, name),
                    str(soap[name].attrs.get("Description", name)),
                )

            lightcone = outfile["Lightcone"]
            lightcone.attrs["property_names"] = [
                "Lightcone/HaloCentre",
                "Lightcone/Redshift",
                "Lightcone/SnapshotNumber",
                "InputHalos/SOAPIndex",
                "InputHalos/HaloCatalogueIndex",
                *properties,
            ]

        # write the meta data
        write_lightcone_metadata(lightcone, metadata)


######################################################
# SWIFT Simulation class functions


def check_simulation(simulation, lightcone_type="SnapshotLightcone"):
    """
    Return the Simulation object, or raise TypeError if it is not one.

    :param  simulation:         the simulation
    :type   simulation:         Simulation
    :param  lightcone_type:     name of the lightcone class, for the error message
    :type   lightcone_type:     str
    """
    if isinstance(simulation, Simulation):
        return simulation
    if isinstance(simulation, str):
        raise TypeError(
            f"{lightcone_type} takes a Simulation, not the label {simulation!r}: "
            f"{lightcone_type}(Simulation.from_label({simulation!r}, simulation_name), ...)"
        )
    raise TypeError(f"{lightcone_type} takes a swiftlet.Simulation, not {type(simulation).__name__}")


######################################################


class SnapshotLightcone:
    """
    Construct lightcones from snapshots and halo catalogues, the base of the SnapshotBeam and SnapshotAllSky classes.

    This class handles:

        1. the slicing of snapshots and their reorientation to avoid exact replication of structure
        2. reading and placing particles and haloes from the snapshots and SOAP catalogues into the lightcone, in
           serial or shared between MPI ranks (the communicator is given to each call, not kept)
        3. the cosmology of the snapshots and the lightcone

    What each gather step finds is kept as ``last_shell`` (a ShellRead or HaloShellRead) until the next call.

    The SnapshotBeam and SnapshotAllSky sub classes define the geometry: which particles are inside the lightcone, and
    which tiles and cells the lightcone crosses
    """

    def __init__(
        self,
        simulation,
        beam_vector=None,
        ang_radius_deg=None,
        orientation_seed=0,
        orientation_lock=None,
        orientation_table=None,
        verbose=1,
        redshift_interp_method="fast",
        redshift_interp_grid_size=3000,
        observer_position=None,
    ):
        """
        :param  simulation:                 the Simulation (e.g. Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")),
                                            which sets where its snapshots, SOAP catalogues and output list are.
        :type   simulation:                 Simulation
        :param  beam_vector:                direction vector of the line of sight as an array of 3 floats. None for
                                            all-sky lightcone
        :type   beam_vector:                numpy.ndarray
        :param  ang_radius_deg:             angular radius [deg] of the beam. None for all-sky lightcone.
        :type   ang_radius_deg:             float
        :param  orientation_seed:           seed used to select the rotation, reflection and periodic shift of each box
                                            tile
        :type   orientation_seed:           int
        :param  orientation_lock:           how the orientations of the box tiles are shared:

                                            - "none": every box tile has its own orientation.
                                            - "cube": every box tile in a cube shell of boxes around the observer's box
                                              shares one orientation, so each shell is periodically continuous and the
                                              orientation only changes on the faces between shells.
                                            - "sphere": everything in a spherical shell, between (n-1/2) and (n+1/2) box
                                              sidelengths from the observer, shares one orientation, so the orientation
                                              only changes at those distances. A box crossing a sphere is used once for
                                              each shell, and tiles are labelled (nx, ny, nz, layer).
        :type   orientation_lock:           str
        :param  orientation_table:          path of a file of orientation tables (see
                                            snapshot_orientation.write_orientation_tables) to take the quarter turns,
                                            reflections and cell shifts of the box tiles from. If None, use the default
                                            tables
        :type   orientation_table:          str
        :param  verbose:                    amount printed. 0: nothing, only warnings. 1: a summary of each call. 2:
                                            details of every snapshot, box tile and file
        :type   verbose:                    int
        :param  redshift_interp_method:     how the comoving distance to redshift interpolation is made: "fast" or
                                            "precise" (see define_redshift_at_comoving_distance_function)
        :type   redshift_interp_method:     str
        :param  redshift_interp_grid_size:  number of points in the grid used to interpolate redshifts from comoving
                                            distances
        :type   redshift_interp_grid_size:  int
        :param  observer_position:          (x, y, z) comoving position [Mpc] in the snapshot box to put the observer
                                            at: the nearest cell corner is used, and every tile is periodically shifted
                                            by the same whole number of cells to move it to the observer. It can be
                                            replaced by each gather or placing call. If None, the observer is at the
                                            centre of its tile, as before.
        :type   observer_position:          array-like, shape (3,)
        """
        # Check simluation passed is a Simulation object
        self.simulation = check_simulation(simulation, lightcone_type="SnapshotLightcone")

        # Define all related sim names and paths
        self.box_res = self.simulation.box_res
        self.sim_name = self.simulation.name
        self.simulation_suite_name = self.simulation.suite.name
        self.simulation_dir = self.simulation.directory
        self.snapshot_format = self.simulation.snapshot_format

        # How much to print the individual outputs
        self.verbose = int(verbose)

        # Define the cosmology from the snapshots
        self.snapshot_cosmology = SnapshotCosmology.from_simulation(self.simulation)
        self.cosmo = self.snapshot_cosmology.COSMO

        # define the beam vector (direction of line of sight)
        if beam_vector is None:
            self.beam_vec = None
        else:
            self.beam_vec = self.__define_beam_vector(beam_vector)

        # The angular radius of the beam (around the beams vector), checked by the subclass for its geometry
        self._validate_ang_radius_deg(ang_radius_deg)
        self.ang_radius_deg = None if ang_radius_deg is None else float(ang_radius_deg)

        # Define the orientation seed: order of reading the table of orientation boxes
        self.orientation_seed = int(orientation_seed)
        if orientation_lock not in ORIENTATION_LOCK_OPTIONS:
            raise ValueError(f"orientation_lock must be one of {ORIENTATION_LOCK_OPTIONS}, not {orientation_lock!r}")

        # lock="none" is the same as lock=None
        self.orientation_lock = None if orientation_lock == "none" else orientation_lock

        # Define where to read orientation instructions from: default or user defined table...
        # quarter turns, reflections and cell shifts of the box tiles: from a file, or the default tables
        if orientation_table is None or str(orientation_table) == "default":
            orientation_table = box_structure.DEFAULT_ORIENTATION_TABLE

        if str(orientation_table) in box_structure.INSTALLED_ORIENTATION_TABLES:
            # Tables installed with SWIFTLET, recorded by name
            self.orientation_table = str(orientation_table)
            self._orientation_tables = box_structure.default_orientation_tables(self.orientation_table)
        else:
            self.orientation_table = os.path.abspath(str(orientation_table))
            self._orientation_tables = box_structure.read_orientation_tables(self.orientation_table)

        # Read the tables finger print so it can be recovered as needed.
        self.orientation_table_fingerprint = box_structure.orientation_tables_fingerprint(self._orientation_tables)

        # The observer position in the snapshot box, and the corresponding cell corner and periodic shift [cells]
        # The corner needs the cell grid, so it is found with the first cell metadata read
        self.__observer_snapshot_coords = None
        self.__observer_corner = None
        self.__observer_cell_shift = None
        self.__observer_grid = None  # (cells per axis, box size) the corner was found for
        self.__observer_shifts = {}  # (cells per axis, box size): (corner, shift)
        if observer_position is not None:
            self.__observer_snapshot_coords = _check_observer_position(observer_position)

        # The snapshots' units, the same for every snapshot of the simulation, read and set once
        self.unit_registry = None  # place holder

        # Record what the last gather step found: ShellRead for particles and HaloShellRead for haloes
        # Each call makes its own records, so nothing is carried from one call to the next
        self.last_shell = None

        self._halo_shell = None  # place_halos_in_shell reuses the last HaloShellRead when called with the same settings

        # the cell metadata of the snapshots of the last shell gathered, {snapshot number: cell metadata}, kept across
        # calls so consecutive shells, particle types and haloes reuse it; see _keep_cell_data and clear_cache
        self._cell_cache = {}

        self._log_rank = None  # MPI rank of the current call, only rank 0 prints. None in serial

        # Define grid to interpolate redshift from comoving distance.
        if redshift_interp_method not in ("fast", "precise"):
            raise ValueError(f"redshift_interp_method must be 'fast' or 'precise', not {redshift_interp_method!r}")
        self.__redshift_from_comoving_distance(
            [0.0005, 5.0], n_grid=redshift_interp_grid_size, method=redshift_interp_method
        )

        # once, from MPI rank 0, as every rank makes its own lightcone
        if MPI.COMM_WORLD.Get_rank() == 0:
            self._log(self.__summary(), level=1)

    def __summary(self):
        """Returns a one line str describing the lightcone: its simulation, geometry and box tile orientations."""
        # geometry info
        if self.beam_vec is None:
            geometry = "all-sky"
        else:
            vector = ", ".join(f"{x:.4f}" for x in np.ravel(self.beam_vec))
            geometry = f"beam along ({vector}), {self.ang_radius_deg:g} deg radius"

        # table info
        table = (
            self.orientation_table
            if self.orientation_table in box_structure.INSTALLED_ORIENTATION_TABLES
            else os.path.basename(self.orientation_table)
        )

        # suite / sim info
        suite = self.simulation_suite_name.upper()

        summary = (
            f"{type(self).__name__}: {suite} {self.box_res}/{self.sim_name}, geometry: {geometry}, "
            f"orientation table: {table}, lock: {self.orientation_lock or 'none'}, seed: {self.orientation_seed}"
        )
        # add new observer coords to the summary
        if self.__observer_snapshot_coords is not None:
            position = ", ".join(f"{x:g}" for x in self.__observer_snapshot_coords)
            summary += f", observer position: ({position}) Mpc"

        return summary

    def lightcone_metadata(self, redshift_range=None):
        """
        Returns the lightcone metadata, a dict with the names LIGHTCONE_METADATA_NAMES.

        The metadata records the settings that made the lightcone:

            - the orientation seed and lock,
            - the orientation table (its file and a fingerprint of its contents) that set the orientation of every box
              tile,
            - the simulation,
            - the beam (its vector and angular radius)
            - the redshift range.
            - the observer position, the cell corner it was put at and the periodic shift [cells] of every tile

        Given the same settings and snapshots, the same exact lightcone can be made again.

        :param  redshift_range: minimum and maximum redshift of the shell placed
        :type   redshift_range: sequence of two floats
        """
        from . import __version__ as package_version

        metadata_dict = {
            "lightcone_type": type(self).__name__,
            "simulation_suite_name": self.simulation_suite_name,
            "boxsize_resolution": self.box_res,
            "simulation_name": self.sim_name,
            "orientation_seed": self.orientation_seed,
            "orientation_lock": "none" if self.orientation_lock is None else self.orientation_lock,
            "orientation_table": self.orientation_table,
            "orientation_table_fingerprint": self.orientation_table_fingerprint,
            "beam_vector": None if self.beam_vec is None else [float(x) for x in np.ravel(self.beam_vec)],
            "ang_radius_deg": None if self.ang_radius_deg is None else float(self.ang_radius_deg),
            "redshift_range": None if redshift_range is None else [float(z) for z in redshift_range],
            "observer_position": None,
            "observer_corner": None,
            "observer_cell_shift": None,
            "version": package_version,
        }

        # include observer position information
        position, corner = self.observer_snapshot_coords
        if position is not None:
            metadata_dict["observer_position"] = list(position)
            metadata_dict["observer_corner"] = None if corner is None else list(corner)
            shift = self.__observer_cell_shift
            metadata_dict["observer_cell_shift"] = None if shift is None else [int(k) for k in shift]

        # remove excess metadata names
        for metadata_prop in list(metadata_dict):
            if metadata_prop not in LIGHTCONE_METADATA_NAMES:
                del metadata_dict[metadata_prop]

        # ensure all extra metadata names are added
        for required_prop in LIGHTCONE_METADATA_NAMES:
            if required_prop not in metadata_dict:
                metadata_dict[required_prop] = None

        return metadata_dict

    @property
    def observer_snapshot_coords(self):
        """
        Returns a tuple (observer_position, observer_corner), the observer position [Mpc] given in the snapshot box and
        the cell corner the observer is placed at [Mpc].

        Without an observer position returns (None, None). The corner needs the snapshots' cell grid, so it is None
        until the first gather or placing call (or Snapshot2Lightcone) reads cell metadata.
        """
        position = self.__observer_snapshot_coords
        return (None if position is None else tuple(float(x) for x in position), self.__observer_corner)

    @contextmanager
    def _mpi_call(self, comm) -> Iterator[None]:
        """
        Print the messages of a call from rank 0, for the length of the call.

        The comm itself is passed to the methods that need it and not kept on the lightcone, so the next call never uses
        it.

        :param  comm:   MPI communicator of the call. If None, serial
        :type   comm:   mpi4py.MPI.Comm
        :yield:         nothing: the call runs inside the with block
        """
        previous = self._log_rank
        self._log_rank = None if comm is None else comm.Get_rank()
        try:
            yield
        finally:
            self._log_rank = previous

    def _cell_data(self, snap_nr, cache):
        """
        Returns a dict with the cell metadata of a snapshot (__get_cell_data), read once and kept in the cache

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        :param  cache:      the cell metadata kept, {snapshot number: cell metadata}, e.g. ShellRead.cell_data
        :type   cache:      dict
        """
        snap_nr = int(snap_nr)
        if snap_nr not in cache:
            cache[snap_nr] = self.__get_cell_data(snap_nr)
        return cache[snap_nr]

    def _keep_cell_data(self, snapshots):
        """
        Returns the lightcone's cell metadata cache, keeping only the snapshots given.

        Each gather step keeps the cell metadata of its shell's snapshots that is already cached and drops the rest.
        The cache never holds more than one shell's snapshots.

        :param  snapshots:  the snapshot numbers of the shell being gathered
        :type   snapshots:  iterable of int
        """
        keep = {int(n) for n in snapshots}
        for snap_nr in [n for n in self._cell_cache if n not in keep]:
            del self._cell_cache[snap_nr]
        return self._cell_cache

    def clear_cache(self):
        """
        Free the cell metadata kept from earlier calls.

        The lightcone keeps the cell metadata of the snapshots of the last shell it gathered (particles or haloes), so
        the next shell, particle type or halo call reuses what it shares with it and never holds more than one shell's
        snapshots.
        """
        self._cell_cache.clear()

    @staticmethod
    def __define_beam_vector(beam_vector):
        """
        Check the beam vector and normalise it.

        Returns the beam vector as a unit vector.

        :param  beam_vector:    direction vector as an array of 3 floats
        :type   beam_vector:    numpy.ndarray
        """
        beam_vector = np.asarray(beam_vector, dtype=float)
        if beam_vector.shape != (3,):  # test for vector shape
            raise ValueError("beam_vector must be a vector with shape (3,)")
        norm = np.linalg.norm(beam_vector)
        if norm == 0:  # test for non-zero vector
            raise ValueError("beam_vector must be a nonzero vector")
        return beam_vector / norm  # ensure unit vector

    @staticmethod
    def _with_required_properties(ptype, property_names):
        """
        Returns the properties to read: ParticleIDs, Coordinates and SmoothingLengths (except for dark matter).

        :param  ptype:          particle type, e.g. "PartType1"
        :type   ptype:          str
        :param  property_names: particle properties asked for
        :type   property_names: list
        """
        property_names = list(property_names)
        needed = ["ParticleIDs", "Coordinates"] + ([] if ptype == "PartType1" else ["SmoothingLengths"])
        return property_names + [name for name in needed if name not in property_names]

    @staticmethod
    def _observer_metadata(metadata, shell):
        """
        Returns the metadata with the observer of the shell placed, as recorded by its gather step.

        :param  metadata:   lightcone metadata, see lightcone_metadata
        :type   metadata:   dict
        :param  shell:      the shell placed
        :type   shell:      ShellRead or HaloShellRead
        """
        metadata = dict(metadata)
        metadata["observer_position"] = None if shell.observer_position is None else list(shell.observer_position)
        metadata["observer_corner"] = None if shell.observer_corner is None else list(shell.observer_corner)
        metadata["observer_cell_shift"] = None if shell.observer_cell_shift is None else list(shell.observer_cell_shift)
        return metadata

    @staticmethod
    def _read_slices(dataset, starts, lengths, dtype=None):
        """
        Returns the rows [start, start + length) of a dataset for each range given, joined in the order given.

        Every read of particle data goes through here as it determines what is used differently for different file
        locations.

        :param  dataset:    the dataset to read, along its first axis
        :type   dataset:    h5py.Dataset
        :param  starts:     first row of each range
        :type   starts:     numpy.ndarray
        :param  lengths:    number of rows of each range
        :type   lengths:    numpy.ndarray
        :param  dtype:      dtype of the array returned. If None, the dataset's
        :type   dtype:      numpy.dtype
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        values = np.empty((int(np.sum(lengths)),) + dataset.shape[1:], dtype=dtype or dataset.dtype)
        position = 0
        for start, length in zip(np.asarray(starts, dtype=np.int64), lengths):
            if length > 0:
                values[position : position + length] = dataset[start : start + length]
                position += length
        return values

    @staticmethod
    def _cell_batches(lengths, max_particles_per_read=None):
        """
        Returns the cells split, in order, into batches of at most max_particles_per_read particles.

        The batches are a list of index arrays into lengths. A cell is never split: a cell with more particles than
        the limit is a batch on its own.

        :param  lengths:                number of particles of each cell, in file order
        :type   lengths:                numpy.ndarray
        :param  max_particles_per_read: largest number of particles in a batch. If None, one batch of all the cells
        :type   max_particles_per_read: int
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        if max_particles_per_read is None or len(lengths) == 0:
            return [np.arange(len(lengths))]
        if max_particles_per_read < 1:
            raise ValueError(f"max_particles_per_read must be at least 1, not {max_particles_per_read}")
        batches, first, in_batch = [], 0, 0
        for cell, length in enumerate(lengths):
            if in_batch > 0 and in_batch + length > max_particles_per_read:
                batches.append(np.arange(first, cell))
                first, in_batch = cell, 0
            in_batch += length
        batches.append(np.arange(first, len(lengths)))
        return batches

    @staticmethod
    def _needed_cells(plan, uses=None):
        """
        Returns the positions in the plan of the cells needed (all if None) in file order.

        :param  plan:   the file's plan
        :type   plan:   FilePlan
        :param  uses:   the uses (TileUse). If None, all of the plan's
        :type   uses:   list
        """
        uses = plan.uses if uses is None else uses
        if not uses:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate([use.cells for use in uses]))

    def __read_cell_ranges(self, infile, ptype, prop, starts, lengths):
        """
        Returns one property of the particles in ranges of rows of an open snapshot file, as a unyt array in the
        snapshot's units (without the expansion factor of comoving properties).

        :param  infile:     the open snapshot file
        :type   infile:     h5py.File
        :param  ptype:      particle type, e.g. "PartType1"
        :type   ptype:      str
        :param  prop:       the property, e.g. "Coordinates"
        :type   prop:       str
        :param  starts:     first row of each range, e.g. of each set of adjacent cells
        :type   starts:     numpy.ndarray
        :param  lengths:    number of rows of each range
        :type   lengths:    numpy.ndarray
        """
        if prop not in infile[ptype]:
            raise ValueError(f"{prop} not found")
        dataset = infile[ptype][prop]
        values = self._read_slices(dataset, starts, lengths, np.int64 if prop == "ParticleIDs" else None)
        units = sw_units.units_from_attributes(dict(dataset.attrs), self.unit_registry)
        values = unyt.unyt_array(values, units)
        if "a*" in str(units) or "*a" in str(units):
            values = sw_units.drop_a_from_comoving_property(values)
        return values

    def __get_cell_data(self, snap_nr):
        """
        Read the cell metadata of a snapshot.

        IMPORTANT: ALL CELL METADATA IS IN SNAPSHOT UNITS

        Returns a dict of the number of cells, cell size, cell centres, box size, number of files, the file, offsets
        and lengths of each cell per particle type, and the snapshot's unit metadata.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        """
        cell_data = {}

        # with h5py.File(self.snapshot_format.format(snap_nr=snap_nr, file_nr=0), "r") as f:
        with h5py.File(self.simulation.snapshot_file(snap_nr, 0), "r") as f:
            # IMPORTANT: CELL METADATA IS IN SNAPSHOT UNITS
            # The cell sizes, cell centres and box size are in the snapshot's length unit.
            # The same as the particle coordinates.
            # Therefore, cells have the same units as coords before they are converted to Mpc exactly and points on the
            # faces of cells and the snapshot box stay on them when the snapshot is re-oriented

            # the snapshot's units, also for its particles (see _set_unit_registry)
            cell_data["unit_metadata"] = sw_units.snapshot_unit_metadata(f)
            snap_length_registry = sw_units.unit_registry_from_metadata(cell_data["unit_metadata"])

            def snap_length(values):
                return unyt.unyt_array(values, "snap_length", registry=snap_length_registry)

            cell_data["nr_cells"] = f["/Cells/Meta-data"].attrs["nr_cells"][0]  # total number of cells
            cell_data["nr_cells_axis"] = f["/Cells/Meta-data"].attrs["dimension"][:]  # number of cells per axes
            cell_data["cell_size"] = snap_length(f["/Cells/Meta-data"].attrs["size"])  # side length of cell
            cell_data["cell_centres"] = snap_length(f["/Cells/Centres"][:, :])  # centre of each cell in snapshot coords
            cell_data["snap_boxsize"] = snap_length(f["Header"].attrs["BoxSize"])  # box sidelength
            cell_data["nr_files"] = f["Header"].attrs["NumFilesPerSnapshot"][0]  # num of subfiles

            cells_in_file_dict = {}
            snap_offsets = {}
            snap_lengths = {}

            for ptype_i in range(7):
                if f["Header"].attrs["NumPart_Total"][ptype_i] > 0:
                    cells_in_file_dict[f"PartType{ptype_i}"] = f[f"Cells/Files/PartType{ptype_i}"][...]

                    snap_offsets[f"PartType{ptype_i}"] = f[f"/Cells/OffsetsInFile/PartType{ptype_i}"][...]
                    snap_lengths[f"PartType{ptype_i}"] = f[f"/Cells/Counts/PartType{ptype_i}"][...]

                    assert snap_offsets[f"PartType{ptype_i}"].shape == snap_lengths[f"PartType{ptype_i}"].shape
                    assert np.shape(snap_offsets[f"PartType{ptype_i}"])[0] == np.shape(cell_data["cell_centres"])[0]

            cell_data["cells_in_file"] = cells_in_file_dict
            cell_data["offsets"] = snap_offsets
            cell_data["lengths"] = snap_lengths

        return cell_data

    def _set_unit_registry(self, unit_metadata):
        """
        Make the lightcone's unit registry from a snapshot's unit metadata.

        :param  unit_metadata:  the snapshot's unit metadata, e.g. the cell metadata's "unit_metadata"
        :type   unit_metadata:  dict
        """
        if self.unit_registry is None:
            self.unit_registry = sw_units.unit_registry_from_metadata(unit_metadata)

    def __comoving_distance_to_scalefactor(self, r, cosmo=None):
        """
        Compute the scale factor of particles based on comoving distance from observer.

        :param  r:      comoving distance from the observer [Mpc]
        :type   r:      float, numpy.ndarray or unyt.array.unyt_array
        :param  cosmo:  cosmology. If None, use the snapshot cosmology
        :type   cosmo:  astropy.cosmology.w0waCDM
        """
        cosmo = self.cosmo if cosmo is None else as_astropy_cosmology(cosmo)

        if self.interp_redshift_from_comoving_distance is None:
            raise ValueError("interpolation function not defined")

        r = sw_units.apply_expected_units(r, unyt.Mpc)  # place distance in terms of Mpc
        a = 1.0 / (self.interp_redshift_from_comoving_distance(r.to_value("Mpc")) + 1.0)

        return a

    def __redshift_from_comoving_distance(self, shell_z, n_grid, method="fast"):
        """
        Define the interpolation function for the redshift at a given comoving distance over the redshift range.

        :param  shell_z:    minimum and maximum redshift [z_min, z_max]
        :type   shell_z:    sequence of two floats
        :param  n_grid:     number of points in the interpolation grid
        :type   n_grid:     int
        :param  method:     'fast' or 'precise', see docstring for define_redshift_at_comoving_distance_function
        :type   method:     str
        """
        zmin = shell_z[0] - 0.01 if shell_z[0] > 0.01 else shell_z[0]
        zmax = shell_z[1] + 0.01
        self.interp_redshift_from_comoving_distance = define_redshift_at_comoving_distance_function(
            self.cosmo, zmin, zmax, n_grid, method=method
        )

    def __snapshot_z_range(self, snapshot_number, z_ranges):
        """
        Returns the redshift range a snapshot fills, (z_min, z_max).

        The range increases when a snpahot fills in for its neighbours.

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  z_ranges:           the ranges of the snapshots filling in for missing ones, see _shell_snapshots
        :type   z_ranges:           dict
        """
        if int(snapshot_number) in z_ranges:
            return z_ranges[int(snapshot_number)]
        return self.simulation.snapshot_redshift_range(snapshot_number)

    def _available_snapshots(self, halo_format=None):
        """
        Returns the simulation's snapshot number - output redshift table that are on disk, in increasing order.

        Looked up once per lightcone and stored.

        :param  halo_format:    see gather_halo_catalogues. If None, look for the snapshot files
        :type   halo_format:    str
        """
        if not hasattr(self, "_available_snapshot_cache"):
            self._available_snapshot_cache = {}
        if halo_format not in self._available_snapshot_cache:
            nr_snapshots = len(self.simulation.output_redshifts)
            if halo_format is None:
                paths = [self.simulation.snapshot_file(n, 0) for n in range(nr_snapshots)]
            else:
                paths = [halo_format.format(snap_nr=n) for n in range(nr_snapshots)]
            self._available_snapshot_cache[halo_format] = np.array(
                [n for n, path in enumerate(paths) if os.path.exists(path)], dtype=int
            )
        return self._available_snapshot_cache[halo_format]

    def snapshots_for_shell(self, shell_z, halo_format=None, range_tol=0.0):
        """
        Returns the snapshot numbers for the redshift range of the lightcone shell, from high to low redshift.

        Read ahead and find which snapshots (and SOAP catalogues) are available on disk, then identify which snapshots
        fill the lightcones redshift range. If any snapshots are missing, fill the gaps in redshift with the available
        neighbouring snapshots and raise a SnapshotGapWarning.

        Raise a ValueError if the redshift spacing between the snapshots is greater than MAX_SNAPSHOT_SPACING.

        :param  shell_z:        minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  halo_format:    see gather_halo_catalogues. If None, the snapshot files are looked for
        :type   halo_format:    str
        :param  range_tol:      a snapshot edge within range_tol of the shell counts as touching it
        :type   range_tol:      float
        """
        return self.__snapshot_coverage(shell_z, halo_format, range_tol)[0]

    def __snapshot_coverage(self, shell_z, halo_format, range_tol, stacklevel=4):
        """
        The snapshots that fill a shell and the redshift range each fills, missing ones filled by their neighbours.

        Returns a tuple (snapshot numbers from high to low redshift, {snapshot number: (z_min, z_max)}).

        :param  shell_z:        minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  halo_format:    see gather_halo_catalogues. If None, the snapshot files are looked for
        :type   halo_format:    str
        :param  range_tol:      a snapshot edge within range_tol of the shell counts as touching it
        :type   range_tol:      float
        :param  stacklevel:     stacklevel of the SnapshotGapWarning, to point at the user's call
        :type   stacklevel:     int
        """
        data_type = "snapshot files" if halo_format is None else "SOAP catalogues"
        label = f"{self.simulation.box_res}/{self.simulation.name}"
        try:
            coverage = snapshot_redshifts.snapshot_coverage(
                self.simulation.output_redshifts, shell_z, self._available_snapshots(halo_format), range_tol=range_tol
            )
        except ValueError as error:
            raise ValueError(f"{label}, {data_type}: {error}") from None

        # if missing / filling gaps then give warning
        if coverage.missing and MPI.COMM_WORLD.Get_rank() == 0:
            # snapshots filling the gaps
            filled = "; ".join(
                f"snapshot {n} by {' and '.join(str(m) for m in by)}" for n, by in coverage.filled_by.items()
            )
            # warning message: explain gaps + what snapshots fill gaps.
            warning_str = (
                f"{label}: the {data_type} of snapshots {coverage.missing} are missing. Filling the gaps with"
                f" neighbouring snapshots: {filled}"
            )
            warnings.warn(warning_str, SnapshotGapWarning, stacklevel=stacklevel)

        return np.asarray(coverage.snapshots, dtype=int)[::-1], dict(coverage.ranges)

    def _shell_snapshots(self, shell_z, use_snapshots=None, halo_format=None, snapshot_range_tol=1e-5):
        """
        Choose the snapshots of a shell, the same way for particles and haloes.

        Without use_snapshots, every snapshot whose redshift range overlaps or touches the shell (within
        snapshot_range_tol), with missing ones filled by their nearest neighbours. Snapshots given by use_snapshots that
        leave gaps in the ligthcone are warned about (SnapshotGapWarning) but not filled.

        Returns a tuple (
            snapshot numbers [in order from high to low redshift unless given with use_snapshots],
            {snapshot number: (z_min, z_max)} of the snapshots that fill in for a missing neighbour
            )

        :param  shell_z:            minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:            sequence of two floats
        :param  use_snapshots:      snapshot numbers to use instead of the ones selected from shell_z
        :type   use_snapshots:      list or numpy.ndarray
        :param  halo_format:        see gather_halo_catalogues. If None, the snapshot files are used
        :type   halo_format:        str
        :param  snapshot_range_tol: a snapshot whose redshift range ends within this of the shell's edge counts as
                                    touching it, for particles and haloes. Default is 1e-5
        :type   snapshot_range_tol: float
        """
        if use_snapshots is None:
            snapshots, z_ranges = self.__snapshot_coverage(shell_z, halo_format, snapshot_range_tol)
        else:
            snapshots, z_ranges = np.asarray(use_snapshots, dtype=int).reshape(-1), {}

        # only the snapshots overlapping the shell: catches edge cases and rounding errors that slip through
        in_shell = []
        for snap_nr in snapshots:
            zmin, zmax = self.__snapshot_z_range(int(snap_nr), z_ranges)
            if (zmin <= shell_z[1]) and (zmax >= shell_z[0]):
                in_shell.append(int(snap_nr))

        snapshots = np.asarray(in_shell, dtype=int)

        if use_snapshots is not None:
            self.__check_snapshots_exist(snapshots, halo_format=halo_format)
            self.__warn_use_snapshots_gaps(shell_z, snapshots)

        return snapshots, z_ranges

    def __check_snapshots_exist(self, snapshot_numbers, halo_format=None):
        """
        Check the snapshot files or SOAP catalogues are on disk.

        Raises FileNotFoundError and lists the missing files.

        :param  snapshot_numbers:   snapshot numbers
        :type   snapshot_numbers:   sequence of int
        :param  halo_format:        see gather_halo_catalogues. If None, look for the snapshot files
        :type   halo_format:        str
        """
        available = set(self._available_snapshots(halo_format).tolist())
        missing = sorted(int(n) for n in snapshot_numbers if int(n) not in available)
        if missing:
            example = (
                self.simulation.snapshot_file(missing[0], 0)
                if halo_format is None
                else halo_format.format(snap_nr=missing[0])
            )
            what = "snapshot files" if halo_format is None else "SOAP catalogues"
            raise FileNotFoundError(
                (
                    f"{self.simulation.box_res}/{self.simulation.name}: the {what} of snapshots {missing} are missing"
                    f" (e.g. {example})"
                )
            )

    def __warn_use_snapshots_gaps(self, shell_z, used):
        """
        Warn (SnapshotGapWarning) if the snapshots given leave parts of the shell unfilled.

        Each snapshot fills only its own redshift range, so the gaps are not filled.

        :param  shell_z:    minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:    sequence of two floats
        :param  used:       the snapshots given
        :type   used:       sequence of int
        """
        used = set(int(n) for n in used)
        unfilled = []
        for n in self.simulation.snapshots_in_range(shell_z, range_tol=0.0):
            lo, hi = self.simulation.snapshot_redshift_range(n)
            lo, hi = max(lo, shell_z[0]), min(hi, shell_z[1])
            if int(n) not in used and hi > lo:
                unfilled.append(f"{lo:.4g} < z < {hi:.4g} (snapshot {int(n)})")
        if unfilled and MPI.COMM_WORLD.Get_rank() == 0:
            label = f"{self.simulation.box_res}/{self.simulation.name}"
            warnings.warn(
                (
                    f"{label}: the snapshots given by use_snapshots leaves gaps for redshifts "
                    f"{shell_z[0]} < z < {shell_z[1]}, which are not filled: {', '.join(unfilled)}"
                ),
                SnapshotGapWarning,
                stacklevel=5,
            )

    def _log(self, m, level=1, every_rank=False, show_time_date=False):
        """
        Print a message if self.verbose is at least the given level.

        The message is printed by MPI rank 0 only, unless every_rank is True.

        :param  m:                  message to print
        :type   m:                  str
        :param  level:              1 for a summary, 2 for details
        :type   level:              int
        :param  every_rank:         If True, print the message from every MPI rank, prefixed with its rank
        :type   every_rank:         bool
        :param  show_time_date:     If True, prefix the message with the current time
        :type   show_time_date:     bool
        """
        if self.verbose < level:
            return
        if every_rank:
            rank_message(self._log_rank, m)
        else:
            message(self._log_rank, m, show_time_date=show_time_date)

    def _log_coord_ranges(self, title, coords, indent=2):
        """
        Print the minimum and maximum of the coordinates along each axis, if self.verbose == 2.

        :param  title:  line printed before the coordinate ranges
        :type   title:  str
        :param  coords: coordinates, shape (N, 3)
        :type   coords: numpy.ndarray or unyt.array.unyt_array
        :param  indent: number of tabs to indent each line by
        :type   indent: int
        """
        if self.verbose >= 2:
            message(self._log_rank, f"\t{title}:")
            print_coord_ranges(coords, indent=indent, rank=self._log_rank)

    def __log_snapshot_shell(self, snapshot_number, snap_shell_z_range, target):
        """
        Print the redshift and comoving distance range a snapshot populates, if self.verbose == 2.

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  snap_shell_z_range: redshift range of the shell populated by the snapshot [z_min, z_max]
        :type   snap_shell_z_range: tuple
        :param  target:               what is added, e.g. "particles" or "haloes"
        :type   target:               str
        """
        if self.verbose < 2:
            return

        # give read out of max beam diameter for snapshots redshift range in shell
        r_shell = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")
        self._log(
            (
                f"\tadd {target} to lightcone at {snap_shell_z_range[0]:.4f} < z < {snap_shell_z_range[1]:.4f} ("
                f"{r_shell[0]:.4f} - {r_shell[1]:.4f} [Mpc])"
            ),
            level=2,
        )

    def __log_observer(self, position, corner, shift, nr_cells_axis, boxsize):
        """
        Print the cell corner the observer was placed at, and warn if the position was outside the snapshot box.

        :param  position:       the observer position given [Mpc]
        :type   position:       numpy.ndarray
        :param  corner:         the nearest cell corner [Mpc]
        :type   corner:         numpy.ndarray
        :param  shift:          the periodic shift [cells] of every tile
        :type   shift:          numpy.ndarray
        :param  nr_cells_axis:  number of cells along each axis
        :type   nr_cells_axis:  numpy.ndarray
        :param  boxsize:        side length [Mpc] of the box along each axis
        :type   boxsize:        numpy.ndarray
        """
        position = np.asarray(position, dtype=np.float64)

        if np.any(position < 0.0) or np.any(position >= boxsize):
            warnings.warn(
                f"observer_position: {tuple(float(x) for x in position)} [Mpc] is outside the snapshot box "
                f"[0, {float(np.max(boxsize)):g}) [Mpc]: the nearest cell corner is used",
                stacklevel=4,
            )

        def coords(values):
            return ", ".join(f"{float(x):.4f}" for x in values)

        cell_size = boxsize / nr_cells_axis
        self._log(
            (
                f"Observer: requested ({coords(position)}) [Mpc], placed at the nearest cell corner ({coords(corner)}) "
                f"[Mpc] (cell size {float(np.max(cell_size)):g} [Mpc], {int(np.max(nr_cells_axis))} cells per axis).\nA"
                f" periodic shift of ({', '.join(str(int(k)) for k in shift)}) cells added to every tile"
            ),
            level=1,
        )

    def _snapshot_tile_idx(self, snap_shell_z_range, cell_data):
        """
        Find every tile a snapshot's assigned redshift range touches and the snapshot's full (unsplit) redshift range.

        Returns a list of tuples: (tile, z_min, z_max).

        :param  snap_shell_z_range: redshift range of the shell populated by the snapshot [z_min, z_max]
        :type   snap_shell_z_range: tuple
        :param  cell_data:          cell metadata of the snapshot, for its box size
        :type   cell_data:          dict
        """
        r_min, r_max = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")

        boxsize = cell_data["snap_boxsize"].to_value("Mpc")[0]

        tiles = self._tiles_intersecting_shell(r_min, r_max, boxsize)
        if self.orientation_lock == "sphere":
            return self._split_tiles_into_spherical_layers(tiles, snap_shell_z_range, boxsize)

        return [(tile, snap_shell_z_range[0], snap_shell_z_range[1]) for tile in tiles]

    def _split_tiles_into_spherical_layers(self, tiles, snap_shell_z_range, boxsize):
        """
        Split box tiles into the spherical shells they cross.

        A box crossing the sphere between two layers is used once for each layer,
        each copy supplying only the redshift range of its own layer.

        Returns a list of tuples, ((nx, ny, nz, layer), z_min, z_max).

        :param  tiles:              box tiles (nx, ny, nz) intersecting the snapshot's shell
        :type   tiles:              list of tuple
        :param  snap_shell_z_range: redshift range of the shell populated by the snapshot [z_min, z_max]
        :type   snap_shell_z_range: tuple
        :param  boxsize:            snapshot box side length [Mpc] (box assumed cubic)
        :type   boxsize:            float
        """
        z_min, z_max = snap_shell_z_range
        r_min, r_max = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")

        # redshift range of each layer within the snapshot's range, and the comoving distances
        # the particle selection will use for it. Neighbouring layers share the redshift of the
        # sphere between them, so they meet exactly

        layers = {}
        for layer in range(int(np.floor(r_min / boxsize + 0.5)), int(np.floor(r_max / boxsize + 0.5)) + 1):
            r_inner, r_outer = (layer - 0.5) * boxsize, (layer + 0.5) * boxsize
            z_inner = z_min if r_inner <= r_min else float(self.interp_redshift_from_comoving_distance(r_inner))
            z_outer = z_max if r_outer >= r_max else float(self.interp_redshift_from_comoving_distance(r_outer))
            if z_outer <= z_inner:
                continue
            layer_r = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance([z_inner, z_outer])).to_value("Mpc")
            layers[layer] = (z_inner, z_outer, layer_r[0], layer_r[1])

        split_tiles = []
        sidelengths = np.full(3, boxsize)
        for tile in tiles:
            centre = self.get_snapshot_reposition_coords(tile, sidelengths) + 0.5 * sidelengths
            # closest and furthest distance of the box from the observer
            near = np.linalg.norm(np.maximum(np.abs(centre) - 0.5 * boxsize, 0.0))
            far = np.linalg.norm(np.abs(centre) + 0.5 * boxsize)
            for layer, (z_inner, z_outer, layer_r_inner, layer_r_outer) in layers.items():
                if near <= layer_r_outer and far >= layer_r_inner:
                    split_tiles.append((tuple(int(n) for n in tile[:3]) + (layer,), z_inner, z_outer))

        # order by layer
        return sorted(split_tiles, key=lambda split: split[0][3])

    def _orientation_layer(self, tile):
        """
        Returns the layer number used for identifying the box tiles sharing one orientation.

        :param  tile:   periodic replica tile index
        :type   tile:   tuple
        """
        if self.orientation_lock == "sphere":
            return int(tile[3])
        return max(abs(int(n)) for n in tile[:3])

    def _lattice_tile(self, tile):
        """
        Returns the position (nx, ny, nz) of a tile in the regular lattice along the x, y and z axes, in units of tile
        numbers.

        :param  tile:   periodic replica tile index
        :type   tile:   tuple
        """
        return tuple(int(n) for n in tile[:3])

    def _snapshot_reorientation(self, tile):
        """
        Orientation (rotation, reflection, periodic shift) for snapshot 'tile' (nx, ny, n_los).

        Create a new permutation of the snapshot if the comoving distance to this redshift
        is > box sidelength or in a set orientation lock layer.
        With orientation_lock="cube" or "sphere" every tile in the same layer takes the orientation
        of tile (0, 0, layer), so the boxes in a layer are periodically continuous.

        Returns a tuple, (rotation angles [deg], reflections, periodic cell shifts)

        :param  tile:   periodic replica tile index
        :type   tile:   tuple
        """
        tables = self._orientation_tables
        n_ang = tables["QuarterTurns"].shape[0]
        n_ref = tables["Reflections"].shape[0]
        n_shift = tables["CellShifts"].shape[0]

        if self.orientation_lock is not None:
            nx, ny, nz = 0, 0, self._orientation_layer(tile)
        else:
            nx, ny, nz = self._lattice_tile(tile)

        p1, p2, p3 = TILE_HASH_PRIMES
        tile_hash = nx * p1 + ny * p2 + nz * p3 + self.orientation_seed * TILE_HASH_SEED_PRIME

        orientation_idx = (tile_hash % n_ang, tile_hash % n_ref, tile_hash % n_shift)

        ang = tables["QuarterTurns"][orientation_idx[0]] * 90  # degrees
        ref = tables["Reflections"][orientation_idx[1]]
        shift = tables["CellShifts"][orientation_idx[2]]

        return ang, ref, shift

    def _observer_cell_shift(self, nr_cells_axis, boxsize):
        """
        Returns the periodic cell shift [cells] added to every tile for the observer position, zero without an observer
        position.

        The first time the corner is found it is kept for observer_snapshot_coords and printed.

        :param  nr_cells_axis:  number of cells along each axis of the snapshot
        :type   nr_cells_axis:  int or array-like of int, shape (3,)
        :param  boxsize:        side length [Mpc] of the snapshot box
        :type   boxsize:        float or array-like, shape (3,)
        """
        position = self.__observer_snapshot_coords
        if position is None:
            return np.zeros(3, dtype=np.int64)

        n = np.broadcast_to(np.asarray(nr_cells_axis, dtype=np.int64), (3,))
        boxsize = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
        key = (tuple(int(k) for k in n), tuple(float(x) for x in boxsize))

        if key not in self.__observer_shifts:
            # the observer's tile takes its own shift from its row of the tables: (0, 0, 0) in layer 0 for every lock,
            # which the observer's shift makes up for, so the corner is at the observer for any seed and table
            _, _, tile_shift = self._snapshot_reorientation((0, 0, 0, 0))
            tile_shift = box_structure.cell_shifts_in_cells(tile_shift, n)
            self.__observer_shifts[key] = box_structure.observer_corner_and_shift(
                position, boxsize, n, tile_cell_shift=tile_shift
            )
        corner, shift = self.__observer_shifts[key]

        if key != self.__observer_grid:
            self.__observer_grid = key
            self.__observer_corner = tuple(float(x) for x in corner)
            self.__observer_cell_shift = tuple(int(k) for k in shift)
            self.__log_observer(position, corner, shift, n, boxsize)

        return shift

    def _find_observer_corner(self, cell_data):
        """
        Find and print the cell corner the observer is placed at, from a snapshot's cell metadata.

        :param  cell_data:  cell metadata of a snapshot, or None if none was read
        :type   cell_data:  dict
        """
        if cell_data is not None and self.__observer_snapshot_coords is not None:
            self._observer_cell_shift(cell_data["nr_cells_axis"], cell_data["snap_boxsize"].to_value("Mpc"))

    def _cell_shift_to_vector(self, cell_shift, n_cells_per_axis, cell_sidelength, snapshot_sidelength):
        """
        Convert cell shift to a comoving distance vector.

        Returns the shift vector as an numpy.ndarray, shape (3,).

        :param  cell_shift:             shift along x, y, z. If None, no shift
        :type   cell_shift:             array-like of int, shape (3,)
        :param  n_cells_per_axis:       number of cells per sidelength
        :type   n_cells_per_axis:       int or array-like, shape (3,)
        :param  cell_sidelength:        length along each axis of a cell
        :type   cell_sidelength:        float or array-like, shape (3,)
        :param  snapshot_sidelength:    length along each axis of snapshot
        :type   snapshot_sidelength:    float or array-like, shape (3,)
        """
        if cell_shift is None:
            cell_shift = np.zeros(3, dtype=np.int64)

        # check params have correct shapes
        cell_shift = np.asarray(cell_shift, dtype=np.int64)
        if cell_shift.shape != (3,):
            raise ValueError("cell_shift must have shape (3,)")

        n_cells_per_axis = np.broadcast_to(np.asarray(n_cells_per_axis), (3,)).astype(np.int64)

        # the shift in whole cells, from the unit of the tables
        cell_shift = box_structure.cell_shifts_in_cells(cell_shift, n_cells_per_axis)

        cell_sidelength = np.broadcast_to(np.asarray(cell_sidelength, dtype=np.float64), (3,))
        snapshot_sidelength = np.broadcast_to(np.asarray(snapshot_sidelength, dtype=np.float64), (3,))

        expected_snapshot_sidelength = n_cells_per_axis * cell_sidelength

        # check cells sum to boxsize
        if not np.allclose(expected_snapshot_sidelength, snapshot_sidelength, rtol=1e-5):
            raise ValueError(
                (
                    f"n_cells_per_axis * cell_sidelength ({expected_snapshot_sidelength}) doesn't "
                    f"match snapshot_sidelength ({snapshot_sidelength})"
                )
            )

        # the observer's periodic shift, the same for every tile, added before the tile's rotation and reflection
        cell_shift += self._observer_cell_shift(n_cells_per_axis, snapshot_sidelength)

        # wrap around the box
        wrapped_cell_shift = np.mod(cell_shift, n_cells_per_axis)

        return wrapped_cell_shift.astype(np.float64) * cell_sidelength

    def _transform_snapshot_coordinates(
        self,
        coords,
        rot_angles,
        cell_data,
        reflections=None,
        periodic_cell_shift=None,
        order="xyz",
        degrees=False,
        out=None,
        inplace=False,
    ):
        """
        For each axis (x,y,z), rotate, reflect and warp (about the periodic boundaries) the snapshot coordinates.

        Generate a new set of coordinates making a new orientation of the snapshot
        through rotations, reflections and wrapping coordinates about the periodic boundaries.

        Returns the coordinates for the new snapshot orientation.

        :param  coords:                 coordinates, shape (N, 3)
        :type   coords:                 numpy.ndarray
        :param  rot_angles:             (angle_x, angle_y, angle_z), radians unless degrees=True
        :type   rot_angles:             array-like, shape (3,)
        :param  cell_data:              snapshot cell metadata
        :type   cell_data:              dict
        :param  reflections:            per-axis reflection signs (+1/-1). Pass None for no reflection
        :type   reflections:            array-like, shape (3,)
        :param  periodic_cell_shift:    number of cells to shift along x, y, z axes
        :type   periodic_cell_shift:    array-like of int, shape (3,)
        :param  order:                  rotation composition order, a permutation of "x", "y", "z"
        :type   order:                  str
        :param  degrees:                If True, then angles are given in degrees
        :type   degrees:                bool
        :param  out:                    preallocated output buffer, shape (N, 3) (avoids reallocation on repeated calls)
        :type   out:                    numpy.ndarray
        :param  inplace:                If True, overwrite coords in place instead of allocating new memory
        :type   inplace:                bool
        """
        nr_cell_axis = cell_data["nr_cells_axis"]

        # convert to Mpc when performing orientation related functions
        cell_sidelength = cell_data["cell_size"].to_value("Mpc")
        snapshot_sidelength = cell_data["snap_boxsize"].to_value("Mpc")

        # get shift vecttor
        shift_vector = self._cell_shift_to_vector(
            periodic_cell_shift, nr_cell_axis, cell_sidelength, snapshot_sidelength
        )

        return box_structure.rotate_coords_cartesian(
            coords,
            rot_angles,
            reflections=reflections,
            periodic_shift=shift_vector,
            sidelengths=snapshot_sidelength,
            order=order,
            degrees=degrees,
            out=out,
            inplace=inplace,
        )

    def _transform_snapshot_vectors(self, vectors, tile):
        """
        Re-orient vector snapshot properties (e.g., particle velocities) the same as the coordinates by applying the
        the tile's reflection and quarter turns, without its periodic shift or offset.

        Returns the re-oriented vectors

        :param  vectors:    vectors in the snapshot's frame, shape (N, 3)
        :type   vectors:    numpy.ndarray
        :param  tile:       the box tile, as in _snapshot_reorientation
        :type   tile:       tuple
        """
        rot_angles, reflections, _ = self._snapshot_reorientation(tile)
        vectors = np.asarray(vectors)
        return box_structure.rotate_coords_cartesian(
            vectors, rot_angles, reflections=reflections, order="xyz", degrees=True
        ).astype(vectors.dtype, copy=False)

    def _validate_ang_radius_deg(self, ang_radius_deg):
        """
        Check the angular radius given to the subclasses particle and halo placement methods.

        Each subclass overrides this for their corresponding geometry.
        SnapshotBeam: requires a radius and places an upper bound on ang_radius_deg
        SnapshotAllSky: this doesn't need validating and ang_radius_deg is None

        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        """
        # place holder for full-sky, only needed for beam and then it uses to sub-classes own method
        pass

    def _attach_expansion_factors(self, particle_data):
        """
        Compute the scale factors (``a``; :math:`a=1/(1+z)`) of the particles inside the beam from their comoving
        distance.

        Stored as the ``ExpansionFactors`` property of particle_data.

        :param  particle_data:  the particles placed, with their Coordinates
        :type   particle_data:  dict
        """
        coords = particle_data["Coordinates"]
        npart_in_beam = coords.shape[0]

        if npart_in_beam > 0:
            self._log_coord_ranges(f"coordinates range of the {npart_in_beam} particles in the lightcone", coords[:])

        # compute distance from observer
        r_comoving = unyt.unyt_array(np.zeros(npart_in_beam), units=coords.units)
        r_comoving[:] = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2)

        # determine scale factor of particles in the beam
        particle_data["ExpansionFactors"] = unyt.unyt_array(np.zeros(npart_in_beam), units=unyt.dimensionless)
        particle_data["ExpansionFactors"][:] = self.__comoving_distance_to_scalefactor(r_comoving) * unyt.dimensionless

    def _gather_files_for_ptype(self, current_ptype, shell_z, use_snapshots=None, comm=None):
        """
        Plan out every snapshot file needed to populate a lightcone for a single particle type and the tiles each file
        is mapped too.

        Without reading any particle data record the pieces of the shell and the file plans in a new ShellRead, kept
        as self.last_shell.

        Returns the ShellRead.

        :param  current_ptype:  particle type to gather, e.g. "PartType1"
        :type   current_ptype:  str
        :param  shell_z:        minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or numpy.ndarray
        :param  comm:           MPI communicator: the cell metadata is read on rank 0 and given to every rank. If None,
                                read where it is needed
        :type   comm:           mpi4py.MPI.Comm
        """
        self.last_shell = None  # until this shell is gathered
        snapshots, z_ranges = self._shell_snapshots(shell_z, use_snapshots)
        shell = ShellRead(
            shell_z=tuple(shell_z),
            ptype=current_ptype,
            snapshots=snapshots,
            z_ranges=z_ranges,
            cell_data=self._keep_cell_data(snapshots),  # reusing what earlier calls read for these snapshots
        )

        if comm is not None:
            self.__broadcast_cell_data(shell.snapshots, shell.cell_data, comm)

        # each snapshot's tile pieces (snapshot, tile) and their files (a single snapshot may need more than one piece)
        for snap_nr in shell.snapshots:
            self.__gather_cells_and_tiles(shell, int(snap_nr))

        # the observer's cell corner, from the cell grid (the same for every snapshot)
        self._find_observer_corner(next(iter(shell.cell_data.values()), None))

        # sanity check a particle types that are not in the snapshots. These are is most likely a typo, e.g. "PartType9"
        in_snapshots = [int(n) for n in shell.snapshots]
        if in_snapshots and not any(current_ptype in shell.cell_data[n]["cells_in_file"] for n in in_snapshots):
            types = sorted(shell.cell_data[in_snapshots[0]]["cells_in_file"])
            raise ValueError(
                f"{self.simulation.box_res}/{self.simulation.name}: no snapshots {in_snapshots} have {current_ptype}, "
                f"they have {', '.join(types)}"
            )

        self.last_shell = shell

        # check for no particles existing in identified cells
        if not shell.plans:
            self._log(f"\n{current_ptype}: no particles in the selected cells\n", level=1)
            return shell

        snapshots_used = sorted({int(plan.snap_nr) for plan in shell.plans})
        npart_to_read = sum(plan.nr_particles for plan in shell.plans)  # total numb of particles in file to read
        npart_to_place = sum(plan.nr_placed for plan in shell.plans)  # total numb of particles in files, cells to read

        self._log(
            (
                f"\n{current_ptype}: {len(shell.plans)} files to read for snapshots {snapshots_used}, with "
                f"{npart_to_read:.3e} particles in the selected cells and {npart_to_place} particles to place in the"
                " lightcone."
            ),
            level=1,
        )

        # print of all files to read from
        for plan in shell.plans:
            self._log(
                (
                    f"\tsnap: {plan.snap_nr:<2},\tfile: {plan.file_nr:<3},\tcells: {len(plan.cells):3e},\ttiles: "
                    f"{len(plan.uses)}\tnumb part to read:{plan.nr_particles:.3e},\tnumb part to place: "
                    f"{plan.nr_placed:.3e}\nuse tiles: {[use.tile for use in plan.uses]}"
                ),
                level=2,
            )
        return shell

    def __populate_lightcone_with_ptype(
        self,
        particle_type,
        property_names,
        shell_z,
        use_snapshots=None,
        comm=None,
        redistribute=False,
        max_particles_per_read=None,
    ):
        """
        Read and place all particles of one type into the lightcone shell.

        Returns a dict of particle properties, empty if no particles are found.

        :param  particle_type:  particle type to place, e.g. "PartType1"
        :type   particle_type:  str
        :param  property_names: particle properties to read, required properties are added if missing
        :type   property_names: list
        :param  shell_z:        minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or numpy.ndarray
        :param  comm:           MPI communicator. If None, read in serial
        :type   comm:           mpi4py.MPI.Comm
        :param  redistribute:   If True, redistribute particles evenly across MPI ranks
        :type   redistribute:   bool
        :param  max_particles_per_read: largest number of particles read from a file at once. If None, all
                                        of a file's needed cells at once
        :type   max_particles_per_read: int
        """
        shell = self._gather_files_for_ptype(particle_type, shell_z, use_snapshots, comm=comm)

        # check for no particles existing in identified cells
        if not shell.plans:
            return {}

        property_names = self._with_required_properties(particle_type, property_names)

        # select method for reading files
        method = "serial" if comm is None else "parallel"

        if method == "serial":
            # each file read once, its particles placed in every tile that uses it
            particle_data = self.__read_and_place_plans(
                shell, [(plan, None) for plan in shell.plans], property_names, max_particles_per_read
            )
            self._attach_expansion_factors(particle_data)
        elif method == "parallel":
            particle_data = self.__parallel_read_and_place_particles(
                shell, property_names, comm, max_particles_per_read
            )
            if redistribute:
                # evenly distribute particles across all ranks
                self.__redistribute_particles_evenly(particle_data, property_names, comm)
        else:
            raise ValueError("method not recognised, comm is not None or a recognised MPI communicator")

        return particle_data

    def __snapshot_shell_z_range(self, snapshot_number, shell_z, z_ranges):
        """
        Returns the redshift range (z_min, z_max) of the lightcone that a snapshot covers.

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  shell_z:            minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:            sequence of two floats
        :param  z_ranges:           the ranges of the snapshots filling in for missing ones, see _shell_snapshots
        :type   z_ranges:           dict
        """
        # get snapshot redshift range
        snapshot_z_range = self.__snapshot_z_range(snapshot_number, z_ranges)

        # print out of the snapshot redshift info
        snapshot_to_shell_str = (
            f"\nsnapshot: {snapshot_number},\tz={self.simulation.snapshot_redshift(snapshot_number)} can fill redshifts"
            f" {snapshot_z_range[0]:.4f} < z < {snapshot_z_range[1]:.4f}"
        )
        self._log(snapshot_to_shell_str, level=2)

        return (max(snapshot_z_range[0], shell_z[0]), min(snapshot_z_range[1], shell_z[1]))

    def __gather_cells_and_tiles(self, shell, snapshot_number):
        """
        Find the cells and files of a snapshot, for every tile, that fall within the lightcone shell.

        Use the file plan to add the snapshot's pieces (snapshot number, tile) and cells to the ShellRead.

        :param  shell:              the shell being gathered
        :type   shell:              ShellRead
        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        """
        # read in snapshot redshift range in the lightcone and cell metadata.
        snap_shell_z_range = self.__snapshot_shell_z_range(snapshot_number, shell.shell_z, shell.z_ranges)
        cell_data = self._cell_data(snapshot_number, shell.cell_data)

        # give read out of max beam diameter for snapshots redshift range in shell
        self.__log_snapshot_shell(snapshot_number, snap_shell_z_range, "particles")

        # a snapshot can have no particles of a type, e.g. no stars at high redshift: its tiles are filled with none
        ptype = shell.ptype
        has_ptype = ptype in cell_data["cells_in_file"]
        if not has_ptype:
            self._log(f"\tno {ptype} particles in snapshot {snapshot_number}", level=2)

        #  for its plan each file uses, {file number: [(tile, z range, cells of the tile in the file), ...]}
        uses_of_file = {}

        # iterate through tiles pieces and add transverse tiles when the beam's diameter exceeds one box sidelength.
        for tile, z_sub_min, z_sub_max in self._snapshot_tile_idx(snap_shell_z_range, cell_data):
            # the cells of the snapshot in the tile's part of the shell
            beam_idx = self.__cells_in_tile(cell_data, tile, z_sub_min, z_sub_max)
            shell.tile_cells[(snapshot_number, tile)] = np.asarray(beam_idx, dtype=np.int32)

            # the snapshot's full shell range for the tile, so the read step uses the same (wider) radial bound
            shell.pieces.append(SnapshotReadRecord(snapshot_number, tile, (z_sub_min, z_sub_max)))

            # check for no particles in file
            if not has_ptype:
                continue

            # one file to read per file holding particles of the cells, i.e. use unique file numbers
            files_of_cells = cell_data["cells_in_file"][ptype][beam_idx]  # all files to read
            for file_nr in np.unique(files_of_cells):  # iterate through unique files only
                infile_cell_idx = beam_idx[files_of_cells == file_nr]
                lengths = cell_data["lengths"][ptype][infile_cell_idx]  # numb of particles in cells to read per file
                if np.sum(lengths) == 0:  # no particles in the cells
                    continue

                uses_of_file.setdefault(int(file_nr), []).append((tile, (z_sub_min, z_sub_max), infile_cell_idx))

        # each file needs a plan for how to use the cells for all the tiles they are foudn in
        # make plans in order of the file numbers
        for file_nr in sorted(uses_of_file):
            shell.plans.append(
                _make_file_plan(
                    snapshot_number,
                    file_nr,
                    uses_of_file[file_nr],
                    cell_data["offsets"][ptype],
                    cell_data["lengths"][ptype],
                )
            )

    def __cells_in_tile(self, cell_data, tile, z_sub_min, z_sub_max):
        """
        Identify the snapshot cells within a tile.

        Find the cells of the current snapshot (after the snapshot is moved into the lightcones tile) that overlap the
        tile's part of the lightcone.
        The cell centres are tested with a buffer of one cell length, so every point of the snapshot inside
        the shell is in one of the cells found.

        Returns the indices of the cells, into the snapshot's cell metadata.

        :param  cell_data:          cell metadata of the snapshot
        :type   cell_data:          dict
        :param  tile:               periodic replica tile index
        :type   tile:               tuple
        :param  z_sub_min:          minimum redshift of the tile's part of the shell
        :type   z_sub_min:          float
        :param  z_sub_max:          maximum redshift of the tile's part of the shell
        :type   z_sub_max:          float
        """
        # read in the instructions for how to reconstruct the snapshot box
        snapshot_rotation_angles, snapshot_rotation_reflections, snapshot_periodic_shifts = (
            self._snapshot_reorientation(tile)
        )
        orientation_str = (
            f"\ttile {tile}, layer {self._orientation_layer(tile)}\n\t\trotation: "
            f"{np.asarray(snapshot_rotation_angles).tolist()} [deg], reflection: "
            f"{np.asarray(snapshot_rotation_reflections).tolist()}, periodic shift: "
            f"{np.asarray(snapshot_periodic_shifts).tolist()}"
        )
        self._log(orientation_str, level=2)

        # array to update with the new lightcone coordinates
        repositioned_cell_centres = np.empty(np.shape(cell_data["cell_centres"]))

        # transform coordinates from snapshot to lightcone frame
        __ = self._transform_snapshot_coordinates(
            coords=cell_data["cell_centres"].to_value("Mpc"),  # convert to Mpc when doing re-orientation
            rot_angles=snapshot_rotation_angles,
            reflections=snapshot_rotation_reflections,
            periodic_cell_shift=snapshot_periodic_shifts,
            cell_data=cell_data,
            order="xyz",
            degrees=True,
            out=repositioned_cell_centres,
            inplace=False,
        )

        # move tile to position in lattice
        repositioned_cell_centres += self.get_snapshot_reposition_coords(
            tile, cell_data["snap_boxsize"].to_value("Mpc")
        )

        # check that some part of the cells is in the lightcone
        beam_idx, __ = self._in_shell(
            coords=repositioned_cell_centres,
            z_min=z_sub_min,
            z_max=z_sub_max,
            buffer_length=cell_data["cell_size"][0].to_value("Mpc"),
            buffer_shape="cube",
            return_bool=False,
        )

        # give printout of number of cells in the lightcone for the tile
        self._log(f"\t\t{np.shape(beam_idx)[0]} cells in the lightcone", level=2)

        # give range of cell centres on each axis
        if len(beam_idx) > 0:
            self._log_coord_ranges("\tcell centres in the lightcone", repositioned_cell_centres[beam_idx, :], indent=3)

        return beam_idx

    def __iter_file_pieces(self, shell, plan, property_names, uses=None, max_particles_per_read=None):
        """
        Read one snapshot file once and yield its particles placed in each tile that uses it, a batch of cells at a
        time.

        The cells needed by the uses are split, in file order, into batches of at most max_particles_per_read
        particles (all in one if None). The file is opened once; for each batch, the coordinates of its cells are read
        once, each use gets its own copy of its cells' particles, moved into its tile and cut to its part of the shell
        (in the order the gather step found its cells), and the other properties are read once for the cells holding
        a particle kept by any use.

        Yields a tuple (batch number, use, particles kept: a dict of arrays per property, with SnapshotNumber, and
        TileIndex if it is in property_names) per (batch, use) with cells in the batch.

        :param  shell:                  the shell the file is planned for
        :type   shell:                  ShellRead
        :param  plan:                   the file's plan, from shell.plans
        :type   plan:                   FilePlan
        :param  property_names:         particle properties to read (TileIndex is filled in, not read)
        :type   property_names:         list
        :param  uses:                   the uses (TileUse) of the plan to place. If None, all of them
        :type   uses:                   list
        :param  max_particles_per_read: largest number of particles read at once (whole cells). If None, all the
                                        needed cells of the file at once
        :type   max_particles_per_read: int
        """
        uses = list(plan.uses if uses is None else uses)

        needed_all = self._needed_cells(plan, uses)
        if len(needed_all) == 0:
            return

        file_properties = [prop for prop in property_names if prop not in ("Coordinates", "TileIndex")]

        kept_parts = {prop: [] for prop in property_names}
        kept_parts["SnapshotNumber"] = []
        if not uses:
            return kept_parts, 0, 0

        cell_data = self._cell_data(plan.snap_nr, shell.cell_data)
        self._set_unit_registry(cell_data["unit_metadata"])

        with h5py.File(self.simulation.snapshot_file(plan.snap_nr, plan.file_nr), "r") as infile:
            nr_in_file = int(np.sum(infile["Header"].attrs["NumPart_ThisFile"][:]))
            if np.any(plan.offsets[needed_all] + plan.lengths[needed_all] > nr_in_file):
                raise ValueError(
                    f"{infile.filename}: the cell metadata asks for more rows than particles in the file"
                    f"({plan.offsets[needed_all] + plan.lengths[needed_all]} > {nr_in_file})"
                )

            for batch_number, batch in enumerate(self._cell_batches(plan.lengths[needed_all], max_particles_per_read)):
                # the batch's cells, each once, read as ranges of adjacent cells
                needed = needed_all[batch]
                range_offsets, range_lengths = merge_cells(plan.offsets[needed].copy(), plan.lengths[needed].copy())

                # the coordinates, read once; each use's copy of its cells moved into its tile and cut to the shell
                coordinates = self.__read_cell_ranges(infile, shell.ptype, "Coordinates", range_offsets, range_lengths)
                coordinate_units = coordinates.units
                placed = self.__place_batch_in_tiles(
                    plan, uses, needed, coordinates.to_value("Mpc"), cell_data, batch_number
                )
                del coordinates

                # the other properties, read once for the ranges holding a particle kept by any use
                other, indices = self.__read_kept_ranges(
                    infile, shell.ptype, file_properties, range_offsets, range_lengths, [rows for _, rows, _ in placed]
                )

                # each batches copies of its kept particles, hand out one batch at a time
                for (use, _, new_coords), index in zip(placed, indices):
                    yield (
                        batch_number,
                        use,
                        self.__piece_data(
                            plan.snap_nr, use, new_coords, coordinate_units, other, index, property_names
                        ),
                    )
                del other, placed

    def __place_batch_in_tiles(self, plan, uses, needed, coordinates_mpc, cell_data, batch_number):
        """
        Place each batch of cells copy of the particles in to the corresponding tile, cut to its part of the shell.


        Returns a list (use, rows of the batch kept, their coordinates in the lightcone [Mpc]), for each use with
        cells in the batch. The rows are in the order that the gather step found the cells.

        :param  plan:               the file's plan
        :type   plan:               FilePlan
        :param  uses:               the uses (TileUse) to place
        :type   uses:               list
        :param  needed:             positions in the plan of the batch's cells, in file order
        :type   needed:             numpy.ndarray
        :param  coordinates_mpc:    coordinates of the batch's particles [Mpc], its cells one after the other
        :type   coordinates_mpc:    numpy.ndarray
        :param  cell_data:          the snapshot's cell metadata
        :type   cell_data:          dict
        :param  batch_number:       number of the batch in the file, for the log
        :type   batch_number:       int
        """
        lengths = plan.lengths[needed]
        cell_block_start = np.cumsum(lengths) - lengths  # where each cell starts in the batch
        boxsize = cell_data["snap_boxsize"].to_value("Mpc")

        placed = []
        for use in uses:
            use_cells = use.cells[np.isin(use.cells, needed)]  # in the order the gather step found them

            if len(use_cells) == 0:
                continue

            cell = np.searchsorted(needed, use_cells)
            cell_lengths = lengths[cell]

            rows = np.repeat(
                cell_block_start[cell] - (np.cumsum(cell_lengths) - cell_lengths), cell_lengths
            ) + np.arange(int(np.sum(cell_lengths)))

            # get instructions on how to re-orient the tiles
            angles, reflections, cell_shifts = self._snapshot_reorientation(use.tile)

            new_coords = np.empty((len(rows), 3))  # empty array to update with new lightcone coords

            # apply coords transformation for tile re-orientation
            self._transform_snapshot_coordinates(
                coordinates_mpc[rows],
                angles,
                reflections=reflections,
                periodic_cell_shift=cell_shifts,
                cell_data=cell_data,
                order="xyz",
                degrees=True,
                out=new_coords,
                inplace=False,
            )
            new_coords += self.get_snapshot_reposition_coords(use.tile, boxsize)  # add tile location to coords

            # printout update on new coords
            self._log_coord_ranges(
                f"snap: {plan.snap_nr},\ttile:{use.tile},\tfile:{plan.file_nr},\tbatch:{batch_number}\n\t\t"
                "coordinates of the particles read from file",
                new_coords,
                indent=3,
            )

            # particles kept in lightcone
            kept, __ = self.__select_particles_in_shell(
                new_coords, zmin=use.z_range[0], zmax=use.z_range[1], method="exact", r_tol=0 * unyt.Mpc
            )

            # particles placed in lightcone
            placed.append((use, rows[kept], new_coords[kept]))

        return placed

    def __read_kept_ranges(self, infile, ptype, properties, range_offsets, range_lengths, kept_rows):
        """
        Read properties once for the ranges of a batch that hold a kept particle.

        Returns a tuple (a dict of the values read per property, the ranges used one after the other; for each set of
        kept rows, where its particles are in those values).

        :param  infile:         the open snapshot file
        :type   infile:         h5py.File
        :param  ptype:          particle type, e.g. "PartType1"
        :type   ptype:          str
        :param  properties:     the properties to read
        :type   properties:     list
        :param  range_offsets:  first row in the file of each range of the batch
        :type   range_offsets:  numpy.ndarray
        :param  range_lengths:  number of rows of each range of the batch
        :type   range_lengths:  numpy.ndarray
        :param  kept_rows:      rows of the batch kept, a set per use
        :type   kept_rows:      list
        """
        range_block_start = np.cumsum(range_lengths) - range_lengths  # where each range starts in the batch
        all_kept = np.concatenate(kept_rows) if kept_rows else np.zeros(0, dtype=np.int64)

        used = np.unique(np.searchsorted(range_block_start, all_kept, side="right") - 1)
        used_block_start = np.cumsum(range_lengths[used]) - range_lengths[used]
        # dict to fill with particle properties
        values = {
            prop: self.__read_cell_ranges(infile, ptype, prop, range_offsets[used], range_lengths[used])
            for prop in properties
        }

        # iterate through kept particles
        indices = []
        for rows in kept_rows:
            in_range = np.searchsorted(range_block_start, rows, side="right") - 1
            indices.append(rows - range_block_start[in_range] + used_block_start[np.searchsorted(used, in_range)])

        return values, indices

    def __piece_data(self, snap_nr, use, new_coords, coordinate_units, other, index, property_names):
        """
        Returns the particles one batch of cells kept as a dict of arrays per property, with SnapshotNumber.

        :param  snap_nr:            snapshot number
        :type   snap_nr:            int
        :param  use:                the use (TileUse)
        :type   use:                TileUse
        :param  new_coords:         coordinates of the kept particles in the lightcone [Mpc]
        :type   new_coords:         numpy.ndarray
        :param  coordinate_units:   units of the coordinates in the snapshot, to give the coordinates in
        :type   coordinate_units:   unyt.Unit
        :param  other:              the other properties of the batch, see __read_kept_ranges
        :type   other:              dict
        :param  index:              where the kept particles are in other
        :type   index:              numpy.ndarray
        :param  property_names:     particle properties to return (TileIndex is filled in, not read)
        :type   property_names:     list
        """
        data = {}
        for prop in property_names:
            if prop == "Coordinates":
                data[prop] = (new_coords * unyt.Mpc).to(coordinate_units)

            elif prop == "TileIndex":
                data[prop] = unyt.unyt_array(
                    np.tile(np.asarray(use.tile, dtype=np.int32), (len(new_coords), 1)), "dimensionless"
                )

            elif prop == "Velocities":  # the same rotation and reflection as the coordinates, no shift
                values = other[prop][index]
                data[prop] = unyt.unyt_array(self._transform_snapshot_vectors(values.value, use.tile), values.units)

            else:
                data[prop] = other[prop][index]

        # add snapshot number of particle can be traced back to the redshift it was found at in the simulation
        data["SnapshotNumber"] = unyt.unyt_array(np.full(len(new_coords), snap_nr, dtype=int), units=unyt.dimensionless)

        return data

    def __read_and_place_plans(self, shell, plans, property_names, max_particles_per_read=None, comm=None):
        """
        Read and place the planned files, each once, in serial.

        Returns a dict of the particles kept per property (None for a property if no file was read), with their
        coordinates in the lightcone and initial snapshot number (SnapshotNumber).

        :param  shell:                  the shell the files are planned for
        :type   shell:                  ShellRead
        :param  plans:                  the files to read and place, each with the uses to place, as
                                        (FilePlan, uses or None)
        :type   plans:                  list
        :param  property_names:         particle properties to read
        :type   property_names:         list
        :param  max_particles_per_read: largest number of particles read at once from a file (whole cells). If None,
                                        all the needed cells of a file at once
        :type   max_particles_per_read: int
        :param  comm:                   MPI communicator, to add up the numbers of particles of every rank for the log
        :type   comm:                   mpi4py.MPI.Comm
        """
        kept_parts = {prop: [] for prop in property_names}
        kept_parts["SnapshotNumber"] = []
        nr_read, nr_kept = 0, 0

        for plan, uses in plans:
            needed = self._needed_cells(plan, uses)
            file_read = int(np.sum(plan.lengths[needed]))
            file_kept, nr_batches = 0, 0

            for batch_number, _, data in self.__iter_file_pieces(
                shell, plan, property_names, uses, max_particles_per_read
            ):
                for prop, values in data.items():
                    kept_parts[prop].append(values)

                file_kept += len(data["SnapshotNumber"])
                nr_batches = batch_number + 1

            nr_uses = len(plan.uses if uses is None else uses)

            self._log(
                f"\t\tsnap: {plan.snap_nr}, file: {plan.file_nr}, {nr_uses} tiles, {nr_batches} batches: {file_read} "
                f"particles read, {file_kept} kept in the lightcone",
                level=2,
            )

            nr_read += file_read
            nr_kept += file_kept

        if comm is not None:
            self._log(f"{shell.ptype}: {nr_read} particles read, {nr_kept} kept on this rank", level=2, every_rank=True)
            nr_read, nr_kept = comm.allreduce(nr_read), comm.allreduce(nr_kept)

        if nr_read > 0:
            self._log(
                (
                    f"\n{shell.ptype}: {nr_read} particles read, {nr_kept} kept in the lightcone "
                    f"({nr_kept / nr_read * 100:.3f}%)"
                ),
                level=1,
            )
        else:
            self._log("no particles found in selected cells", level=1)

        particle_data = {name: None for name in property_names}

        for prop, parts in kept_parts.items():
            if parts:
                particle_data[prop] = np.concatenate(parts)

        return particle_data

    @staticmethod
    def _assign_files_to_ranks(weights, nr_ranks):
        """
        Returns the files each MPI rank reads, sharing them so every rank has about the same total weight.

        The files are taken by decreasing weight (ties in file order) and each is given to the
        rank with the least total weight so far (ties to the lowest rank). Every rank computes the same assignment.
        Each rank's files are returned in file order, as a list of index arrays, one per rank.

        :param  weights:    weight of each file, e.g. its expected placed particles (FilePlan.nr_placed)
        :type   weights:    sequence of int
        :param  nr_ranks:   number of MPI ranks
        :type   nr_ranks:   int
        """
        # determine the current particle type being used and the snapshots redshift range in the lightcone.
        weights = np.asarray(weights, dtype=np.int64)
        totals = np.zeros(nr_ranks, dtype=np.int64)
        files = [[] for _ in range(nr_ranks)]

        for file_index in np.argsort(-weights, kind="stable"):
            rank = int(np.argmin(totals))  # the first of the least loaded ranks
            files[rank].append(int(file_index))
            totals[rank] += weights[file_index]

        return [np.sort(np.asarray(rank_files, dtype=np.int64)) for rank_files in files]

    def __broadcast_cell_data(self, snapshots, cache, comm):
        """
        Read the cell metadata of the snapshots on rank 0 only and give it to every rank, adding it to the cache.

        :param  snapshots:  snapshot numbers
        :type   snapshots:  sequence of int
        :param  cache:      the cell metadata kept, {snapshot number: cell metadata}, filled in place
        :type   cache:      dict
        :param  comm:       MPI communicator
        :type   comm:       mpi4py.MPI.Comm
        """
        missing_here = [int(n) for n in snapshots if int(n) not in cache]
        missing = sorted({n for rank_missing in comm.allgather(missing_here) for n in rank_missing})

        if not missing:
            return

        data, error, failure = None, None, None

        if comm.Get_rank() == 0:
            try:
                data = {snap_nr: self._cell_data(snap_nr, cache) for snap_nr in missing}
            except Exception as exc:  # raised again below, on every rank
                failure = exc
                error = f"{type(exc).__name__}: {exc}"
        data, error = comm.bcast((data, error), root=0)

        if failure is not None:
            raise failure

        if error is not None:
            raise RuntimeError(f"rank 0 could not read the cell metadata of snapshots {missing}: {error}")

        cache.update(data)

    def __parallel_read_and_place_particles(self, shell, property_names, comm, max_particles_per_read=None):
        """
        The snapshot files are shared between the ranks, never split: each is read by one rank, which places its
        particles in every tile that uses it. Files are shared so each rank has about the same number of particles to
        place (FilePlan.nr_placed: each cell's particles once per tile that uses it), see _assign_files_to_ranks.

        :param  shell:                  the shell gathered
        :type   shell:                  ShellRead
        :param  property_names:         particle properties to read
        :type   property_names:         list
        :param  comm:                   MPI communicator
        :type   comm:                   mpi4py.MPI.Comm
        :param  max_particles_per_read: largest number of particles read from a file at once (whole cells). If
                                        None, all of a file's needed cells at once
        :type   max_particles_per_read: int
        """
        rank = comm.Get_rank()
        assigned = self._assign_files_to_ranks([plan.nr_placed for plan in shell.plans], comm.Get_size())[rank]
        my_plans = [shell.plans[i] for i in assigned]

        self._log(
            (
                f"{shell.ptype}: reading {len(my_plans)} / {len(shell.plans)} files, "
                f"{sum(plan.nr_particles for plan in my_plans)} particles to read and "
                f"{sum(plan.nr_placed for plan in my_plans)} to place"
            ),
            level=1,
            every_rank=True,
        )

        particle_data = self.__read_and_place_plans(
            shell, [(plan, None) for plan in my_plans], property_names, max_particles_per_read, comm=comm
        )

        # Fill space for when rank has no files assigned.
        self.__fill_missing_particle_data(shell, particle_data, property_names, comm)

        # compute lightcone redshift expansion factors (1/(redshift+1)) for each particle based on its comoving distance
        self._attach_expansion_factors(particle_data)

        return particle_data

    def __fill_missing_particle_data(self, shell, particle_data, property_names, comm):
        """
        Fill in a correct zero-length unyt_array for a rank assigned zero files.

        Usse each property's dtype + units from a rank that did read something, with the units of the shell's snapshots
        (every rank has their cell metadata).

        Note: SnapshotNumber is always included in particle properties.

        :param  shell:          the shell placed
        :type   shell:          ShellRead
        :param  particle_data:  the particles this rank placed, filled in place
        :type   particle_data:  dict
        :param  property_names: particle properties read
        :type   property_names: list
        :param  comm:           MPI communicator
        :type   comm:           mpi4py.MPI.Comm
        """
        # call unit registry
        self._set_unit_registry(shell.cell_data[int(shell.plans[0].snap_nr)]["unit_metadata"])

        for prop in list(property_names) + ["SnapshotNumber"]:
            arr = particle_data.get(prop)
            info = None if arr is None else (arr.dtype, arr.shape[1:], str(arr.units))
            all_info = comm.allgather(info)

            if arr is None:
                for other in all_info:
                    if other is not None:
                        dtype, trailing_shape, units_str = other
                        particle_data[prop] = unyt.unyt_array(
                            np.zeros((0,) + tuple(trailing_shape), dtype=dtype), units_str, registry=self.unit_registry
                        )
                        break
                else:
                    raise RuntimeError(f"no rank read any {prop} particles; cannot determine its dtype/units")

    def __select_particles_in_shell(self, coords, zmin, zmax, method="exact", r_tol=2 * unyt.Mpc):
        """
        Select the particles inside the lightcone shell.

        Returns a tuple of (indices inside the shell, indices outside the shell).

        :param  coords:         comoving lightcone coordinates [Mpc], shape (N, 3)
        :type   coords:         numpy.ndarray
        :param  zmin:           minimum redshift of the shell
        :type   zmin:           float
        :param  zmax:           maximum redshift of the shell
        :type   zmax:           float
        :param  method:         'exact', no buffer. 'approx', extend the shell by r_tol. 'all', keep every particle
        :type   method:         str
        :param  r_tol:          buffer distance used with the 'approx' method (units of Mpc or equivalent)
        :type   r_tol:          float or unyt.array.unyt_quantity
        """
        if method == "approx":
            buffer_length = sw_units.apply_expected_units(r_tol, unyt.Mpc).to_value("Mpc")
            buffer_shape = "sphere"

        elif method == "exact":
            buffer_length = sw_units.apply_expected_units(0.0, unyt.Mpc).to_value("Mpc")
            buffer_shape = "sphere"

        elif method == "all":
            assert np.ndim(coords) == 2
            assert np.shape(coords)[-1] == 3
            return (np.arange(np.shape(coords)[0]), None)

        return self._in_shell(
            coords,
            z_min=zmin,
            z_max=zmax,
            buffer_length=buffer_length,
            buffer_shape=buffer_shape,
            return_bool=False,
        )

    def __redistribute_particles_evenly(self, particle_data, property_names, comm):
        """
        Redistribute particles evenly across all ranks.

        Every rank should then hold roughly equal amounts of particles.

        :param  particle_data:  the particles this rank placed, redistributed in place
        :type   particle_data:  dict
        :param  property_names: particle properties to redistribute, ExpansionFactors is always included
        :type   property_names: list
        :param  comm:           MPI communicator
        :type   comm:           mpi4py.MPI.Comm
        """
        all_props = list(property_names)
        for prop in ("ExpansionFactors", "SnapshotNumber"):
            if prop not in all_props:
                all_props.append(prop)

        # local and global numb of particles
        n_local = particle_data[all_props[0]].shape[0]
        ntot = sum(comm.allgather(n_local))

        size = comm.Get_size()  # numb of ranks

        # get numb of particles that each rank wants to hold after redistribution
        ndesired = np.full(size, ntot // size, dtype=np.int64)
        ndesired[: ntot % size] += 1

        for prop in all_props:
            arr = particle_data[prop]
            units = arr.units
            new_values = psort.repartition(arr.to_value(units), ndesired, comm=comm)
            particle_data[prop] = unyt.unyt_array(new_values, units, registry=self.unit_registry)

        # give update to show redistribution has happened.
        self._log(f"redistributed {ntot} particles evenly across {size} ranks", level=1)

    def gather_files(self, current_ptype, shell_z, use_snapshots=None):
        """
        Determine the snapshot files containing the particles needed to populate the lightcone.

        Each snapshot file is found and a plan (FilePlan) is created to map each tile the cells in the file fill.
        The plan is kept in ``last_shell.plans`` and placed by place_file_in_shell.

        Returns a tuple (number of files found, a list FilePlan entries).

        :param  current_ptype:  particle type to gather, e.g. "PartType1"
        :type   current_ptype:  str
        :param  shell_z:        minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or numpy.ndarray
        """
        shell = self._gather_files_for_ptype(current_ptype, shell_z, use_snapshots)
        return len(shell.plans), shell.plans

    def __file_and_uses(self, file_number, current_ptype, tile):
        """
        Returns the shell gathered, the plan of one of its files, the uses to place (None for all) and the tile as a
        tuple (or None), checking that gather_files found that particle type and, if given, the tile.

        :param  file_number:    index of the files found by gather_files
        :type   file_number:    int
        :param  current_ptype:  particle type being placed, e.g. "PartType1"
        :type   current_ptype:  str
        :param  tile:           (nx, ny, nz) or (nx, ny, nz, layer), or None for every tile that uses the file
        :type   tile:           tuple
        """
        # read data from the last lightcone shell gathered
        shell = self.last_shell

        # sanity check files have been gathered and particle types have been found
        if not isinstance(shell, ShellRead):
            raise ValueError("must call gather_files before place_file_in_shell")

        if current_ptype != shell.ptype:
            raise ValueError(f"gather_files found the files of {shell.ptype}, not {current_ptype}")

        # get property names and populate lightcone

        plan = shell.plans[file_number]
        if tile is None:
            return shell, plan, None, None
        tile = tuple(int(i) for i in tile)
        uses = [use for use in plan.uses if use.tile == tile or use.tile[: len(tile)] == tile]

        if not uses:
            shell_tiles = sorted({piece.tile for piece in shell.pieces})

            if not any(piece_tile[: len(tile)] == tile for piece_tile in shell_tiles):
                raise ValueError(f"tile {tile} is not in the shell; its tiles are {shell_tiles}")

        return shell, plan, uses, tile

    def place_file_in_shell(self, file_number, current_ptype, property_names, tile=None, max_particles_per_read=None):
        """
        Read one snapshot file, selected by index, and place its particles in the lightcone.

        The file is read once: a cell used by several tiles is read once and its particles placed in each of them.
        With tile, only the particles placed in that tile are read and returned. The file is placed with the observer
        position of the gather_files call that found it (kept in ``last_shell``), even if the lightcone's has changed
        since.

        Returns the same per property particle data as place_snapshot_particles_in_shell, but containing only the
        particles kept from the selected file (and tile), with the snapshot, file number and tile in its metadata
        (``snapshot_number``, ``file_number``, ``tile``). It is empty if no particles of the file (or tile) are inside
        the lightcone.

        Calling it once per tile reads the cells those tiles share once for each: use tile=None (the default) to read
        each file once, e.g. from remote files. To hand the particles out a batch and a tile at a time instead of
        returning them all, see iter_file_pieces.

        :param  file_number:            index of the files found by gather_files
        :type   file_number:            int
        :param  current_ptype:          particle type being placed, e.g. "PartType1"
                                            (must match the ptype passed to gather_files)
        :type   current_ptype:          str
        :param  property_names:         particle properties to read, e.g. ["ParticleIDs", "Coordinates"]; "TileIndex"
                                        adds each particle's tile, (nx, ny, nz), or (nx, ny, nz, layer) with the
                                        "sphere" lock
        :type   property_names:         list
        :param  tile:                   the tile to place the file's particles in, (nx, ny, nz) for every layer of it
                                        with the "sphere" lock, or (nx, ny, nz, layer) for one. If None, every tile
                                        that uses the file
        :type   tile:                   tuple
        :param  max_particles_per_read: largest number of particles read from the file at once (whole cells, so a
                                        cell with more is read on its own); lowers the peak memory, the particles
                                        placed are the same. If None, all the file's needed cells at once
        :type   max_particles_per_read: int
        """
        shell, plan, uses, tile = self.__file_and_uses(file_number, current_ptype, tile)

        # record metadata
        metadata = self.lightcone_metadata(shell.shell_z)
        metadata.update(snapshot_number=int(plan.snap_nr), file_number=int(plan.file_nr), tile=tile)

        if uses == []:
            self._log(f"\nfile {file_number} has no particles for tile {tile}\n", level=2)
            return LightconeOutput({}, metadata)

        # get property names and populate lightcone
        property_names = self._with_required_properties(current_ptype, property_names)
        particle_data = self.__read_and_place_plans(shell, [(plan, uses)], property_names, max_particles_per_read)

        if particle_data["Coordinates"] is None:
            self._log(f"\nNo particles kept from file {file_number}\n", level=2)
            return LightconeOutput({}, metadata)  # return empty lightcone with associated metadata.

        # compute lightcone redshift expansion factors (1/(redshift+1)) for each particle based on its comoving distance
        self._attach_expansion_factors(particle_data)

        return LightconeOutput(particle_data, metadata)  # return lightcone with metadata.

    def iter_file_pieces(self, file_number, current_ptype, property_names, tile=None, max_particles_per_read=None):
        """
        Read one snapshot file, selected by index, and yield its particles placed in the lightcone a piece at a time.

        A piece is the particles of one batch of the file's cells for one tile. The file is read once, a batch
        of at most max_particles_per_read particles at a time, and each batch's particles are handed out tile by tile
        as soon as they are placed, so they can be written out and dropped before the next read is made. Together, the
        pieces of a file are the particles of place_file_in_shell with the same arguments.

        Yields a LightconeOutput per (batch, tile) keeping particles, with the snapshot, file number, tile and batch
        number in its metadata (``snapshot_number``, ``file_number``, ``tile``, ``batch``).

        :param  file_number:            index of the files found by gather_files
        :type   file_number:            int
        :param  current_ptype:          particle type being placed (must match the ptype passed to gather_files)
        :type   current_ptype:          str
        :param  property_names:         particle properties to read, e.g. ["ParticleIDs", "Coordinates"]; "TileIndex"
                                        adds each particle's tile
        :type   property_names:         list
        :param  tile:                   the tile to place the file's particles in, as for place_file_in_shell. If
                                        None, every tile that uses the file
        :type   tile:                   tuple
        :param  max_particles_per_read: largest number of particles read from the file at once (whole cells). If
                                        None, all the file's needed cells at once (one batch)
        :type   max_particles_per_read: int
        """
        shell, plan, uses, tile = self.__file_and_uses(file_number, current_ptype, tile)
        if uses == []:
            return

        property_names = self._with_required_properties(current_ptype, property_names)
        metadata = self.lightcone_metadata(shell.shell_z)

        for batch_number, use, data in self.__iter_file_pieces(
            shell, plan, property_names, uses, max_particles_per_read
        ):
            if len(data["SnapshotNumber"]) == 0:
                continue

            self._attach_expansion_factors(data)

            piece_metadata = dict(
                metadata, snapshot_number=int(plan.snap_nr), file_number=int(plan.file_nr), tile=use.tile
            )

            piece_metadata["batch"] = int(batch_number)

            yield LightconeOutput(data, piece_metadata)

    def place_snapshot_particles_in_shell(
        self,
        lightcone_redshift_range,
        property_names,
        particle_types=["PartType0", "PartType1", "PartType5"],
        use_snapshots=None,
        comm=None,
        redistribute_particles=False,
        max_particles_per_read=None,
    ):
        """
        Read and place every particle of the given type(s) into a lightcone shell.

        Returns a dict of particle data per particle type.

        :param  lightcone_redshift_range:   minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   lightcone_redshift_range:   sequence of two floats
        :param  property_names:             particle properties to read, e.g. ["ParticleIDs", "Coordinates"]
        :type   property_names:             list
        :param  particle_types:             particle type(s) to place, e.g. "PartType1"
        :type   particle_types:             str or list
        :param  use_snapshots:              snapshot numbers to use instead of the ones selected from the redshift range
        :type   use_snapshots:              list or numpy.ndarray
        :param  comm:                       MPI communicator. If None, read in serial
        :type   comm:                       mpi4py.MPI.Comm
        :param  redistribute_particles:     If True, redistribute the particles evenly across the MPI ranks
        :type   redistribute_particles:     bool
        :param  max_particles_per_read:     largest number of particles read from a snapshot file at once (whole
                                            cells). This can lower the peak memory of reading  particle data. If None,
                                            all of a file's needed cells at once. Serial only for now
        :type   max_particles_per_read:     int
        """
        if property_names is None:
            raise TypeError('place_snapshot_particles_in_shell needs property_names, e.g. ["Coordinates"]')

        # make sure particle types is iterable if a list is given
        particle_types = [particle_types] if isinstance(particle_types, str) else list(particle_types)

        with self._mpi_call(comm):
            if comm is not None:
                self._log(f"\nreading snapshot files in parallel on {comm.Get_size()} MPI ranks\n", level=1)

            combined_dset = {}

            # iterate through particle types,
            for ptype in particle_types:
                # a line between the messages of each particle type when there is more than one
                if len(particle_types) > 1:
                    self._log(separator_str(), level=1)

                self._log(
                    f"{ptype}: placing in {lightcone_redshift_range[0]} < z < {lightcone_redshift_range[1]}", level=1
                )

                combined_dset[ptype] = self.__populate_lightcone_with_ptype(
                    particle_type=ptype,
                    property_names=property_names,
                    shell_z=lightcone_redshift_range,
                    use_snapshots=use_snapshots,
                    comm=comm,
                    redistribute=redistribute_particles,
                    max_particles_per_read=max_particles_per_read,
                )

            # record the settings that made the lightcone as metadata
            metadata = self.lightcone_metadata(lightcone_redshift_range)

        # Return the lightcone and its metadata
        return LightconeOutput(
            {ptype: LightconeOutput(data, metadata) for ptype, data in combined_dset.items()}, metadata
        )

    def Snapshot2Lightcone(self, snapshot_number, coords, tile):
        """
        Move coordinates from a snapshot's box into a tile of the lightcone, as the particles of the snapshot are moved.

        The tile's rotation, reflection and periodic shifts (with the shift of the lightcone's observer position) are
        applied, then the box is moved to the tile's position. A snapshot can fill more than one tile of a shell, each
        oriented differently, so the tile must always be specified (i.e. a tile of ``last_shell.pieces``).

        Returns the lightcone coordinates [Mpc]

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  coords:             comoving snapshot-frame (box) coordinates [Mpc]
        :type   coords:             numpy.ndarray
        :param  tile:               tile index: (nx, ny, n_los) for SnapshotBeam, (nx, ny, nz) for
                                    SnapshotAllSky, with the spherical layer last for orientation_lock="sphere"
        :type   tile:               tuple
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be a vector with shape (N, 3)")

        # the cell metadata kept (the snapshots of the last shell gathered), otherwise read without keeping it
        cell_data = self._cell_cache.get(int(snapshot_number))

        if cell_data is None:
            cell_data = self.__get_cell_data(snapshot_number)

        return self._to_lightcone(coords, tuple(tile), cell_data)

    def _to_lightcone(self, coords, tile, cell_data):
        """
        Move coordinates from the snapshot box into a tile of the lightcone, as the particles and haloes are moved.

        Returns the lightcone coordinates [Mpc].

        :param  coords:     comoving coordinates in the snapshot box [Mpc]
        :type   coords:     numpy.ndarray
        :param  tile:       periodic replica tile index
        :type   tile:       tuple
        :param  cell_data:  cell metadata of the snapshot, for its box and cells
        :type   cell_data:  dict
        """
        rot_angles, reflections, periodic_shift = self._snapshot_reorientation(tile)
        oriented = self._transform_snapshot_coordinates(
            coords,
            rot_angles,
            cell_data,
            reflections=reflections,
            periodic_cell_shift=periodic_shift,
            order="xyz",
            degrees=True,
            inplace=False,
        )
        return oriented + self.get_snapshot_reposition_coords(tile, cell_data["snap_boxsize"].to_value("Mpc"))

    def gather_halo_catalogues(self, halo_format, shell_z, use_snapshots=None, snapshot_number=None):
        """
        Halo counterpart of gather_files.

        For every snapshot that populates the shell, find the tiles and snapshot it's moved into, as done for the
        particles.
        Note that SOAP catalogues are not sorted by cells like the snapshot particle data, so unlike for particles no
        cells are selected and every halo of a catalogue is tested.
        Each SOAP catalogue to read is recorded with its tile and the tile's redshift range as a HaloCatalogueSpec.
        As done for particles, the tiles are also recorded and kept as ``last_shell`` in a HaloShellRead object.

        Returns the list of HaloCatalogueSpec.

        :param  halo_format:        format string of the SOAP catalogue filenames, using {snap_nr}
        :type   halo_format:        str
        :param  shell_z:            minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:            sequence of two floats
        :param  use_snapshots:      snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:      list or numpy.ndarray
        :param  snapshot_number:    the snapshot to read the box and cell metadata from. If None, the lowest redshift
                                        snapshot used
        :type   snapshot_number:    int
        """
        self.last_shell = None  # None until a shell is gathered

        # the snapshots, chosen as for the particles, with missing SOAP catalogues filled by the available ones
        snapshots, z_ranges = self._shell_snapshots(shell_z, use_snapshots, halo_format=halo_format)

        # read in halo shell, with settings
        shell = HaloShellRead(
            settings=HaloShellRead.settings_of(
                halo_format, shell_z, self.ang_radius_deg, use_snapshots, snapshot_number
            ),
            shell_z=tuple(shell_z),
            snapshots=snapshots,
            z_ranges=z_ranges,
        )

        if len(shell.snapshots) == 0:
            self._log("Haloes: no snapshots in the shell", level=1)
            self.last_shell = self._halo_shell = shell
            return shell.catalogues

        # the box and cell grid, from one snapshot only
        if snapshot_number is None:
            with_files = [n for n in shell.snapshots[::-1] if int(n) in self._available_snapshots()]

            if not with_files:
                with_files = self._available_snapshots()[::-1]

            if len(with_files) == 0:  # raise error if file not found
                raise FileNotFoundError(
                    f"no snapshot files found ({self.simulation.snapshot_file(0, 0)}, ...) to read the box and cell "
                    "metadata from"
                )

            shell.cell_snapshot = int(with_files[0])

        else:
            self.__check_snapshots_exist([snapshot_number])
            shell.cell_snapshot = int(snapshot_number)

        # get cell metadata
        shell.cell_data = self._keep_cell_data(list(shell.snapshots) + [shell.cell_snapshot])
        cell_data = self._cell_data(shell.cell_snapshot, shell.cell_data)

        # the observer's cell corner, from the cell grid
        self._find_observer_corner(cell_data)

        for snap_nr in shell.snapshots:
            snap_nr = int(snap_nr)
            snap_shell_z_range = self.__snapshot_shell_z_range(snap_nr, shell.shell_z, shell.z_ranges)

            if snap_shell_z_range[0] > snap_shell_z_range[1]:
                # the snapshot only touches the shell
                continue

            self.__log_snapshot_shell(snap_nr, snap_shell_z_range, "haloes")  # update printout on snapshots being read

            filename = halo_format.format(snap_nr=snap_nr)

            # iterate through pieces of the tile
            tiles = self._snapshot_tile_idx(snap_shell_z_range, cell_data)
            for tile, z_sub_min, z_sub_max in tiles:
                shell.pieces.append(SnapshotReadRecord(snap_nr, tile, (z_sub_min, z_sub_max)))
                shell.catalogues.append(HaloCatalogueSpec(snap_nr, tile, filename, (z_sub_min, z_sub_max)))

            if tiles and snap_nr not in shell.catalogue_lengths:
                shell.catalogue_lengths[snap_nr] = soap_catalogue_length(filename)

        # update last lightcone shell used
        self.last_shell = self._halo_shell = shell

        snapshots_used = sorted(shell.catalogue_lengths)
        self._log(
            (f"\nHaloes: {len(shell.catalogues)} SOAP catalogue tiles to read for snapshots {snapshots_used}"),
            level=1,
        )
        for spec in shell.catalogues:
            self._log(
                (
                    f"\tsnap: {spec.snap_nr:<2},\ttile: {str(spec.tile):<10}, "
                    f"\tredshift: {spec.z_range[0]:.4f} - {spec.z_range[1]:.4f}"
                ),
                level=2,
            )

        return shell.catalogues

    def __halo_read_chunks(self, shell, chunk_size, nr_chunks_min=1):
        """
        Split the SOAP catalogues to read into chunks of rows, in the order the haloes are placed.

        Returns a list of HaloReadChunk.

        :param  shell:          the lightcone redshift shell gathered
        :type   shell:          HaloShellRead
        :param  chunk_size:     number of rows per piece. If None, each catalogue is split into nr_chunks_min pieces
        :type   chunk_size:     int
        :param  nr_chunks_min:  number of pieces each catalogue is split into when chunk_size is None
        :type   nr_chunks_min:  int
        """
        chunks = []
        # iterate through the SOAP catalogues, at each snapshots number
        for snap_nr in dict.fromkeys(spec.snap_nr for spec in shell.catalogues):
            nr_rows = shell.catalogue_lengths[snap_nr]  # numb haloes to read

            if chunk_size is None:
                step = max(-(-nr_rows // nr_chunks_min), 1)
            else:
                step = int(chunk_size)
                if step < 1:
                    raise ValueError(f"soap_chunk_size must be at least 1, not {chunk_size}")
            for start in range(0, nr_rows, step):
                # add new chunks
                chunks.append(HaloReadChunk(snap_nr, start, min(start + step, nr_rows)))

        return chunks

    def __read_and_place_halos(self, shell, chunks):
        """
        Read the chunks of the SOAP catalogues and place their haloes in the lightcone.

        Every halo is moved into each tile populated by its snapshot and kept if it is in the tile's part of the shell,
        the same as the particles.
        The haloes are in the order of the catalogues to read, then of the rows in each catalogue.

        Returns a tuple (halo centres [Mpc], snapshot numbers, catalogue indices (HaloCatalogueIndex),
        SOAP rows (SOAPIndex), number of haloes read)

        :param  shell:          the lightcone redshift shell gathered
        :type   shell:          HaloShellRead
        :param  chunks:         chunks of the catalogues to read
        :type   chunks:         list of HaloReadChunk
        """
        if not chunks:
            return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), 0

        to_read = ["InputHalos/HaloCentre", "InputHalos/HaloCatalogueIndex"]

        specs_of_snapshot = {}
        for spec_nr, spec in enumerate(shell.catalogues):
            specs_of_snapshot.setdefault(spec.snap_nr, []).append((spec_nr, spec))

        kept = {spec_nr: [] for spec_nr in range(len(shell.catalogues))}
        cell_data = self._cell_data(shell.cell_snapshot, shell.cell_data)  # read again if a later call dropped it
        nr_read = 0

        # iterate through chunks of the snapshot
        for chunk in chunks:
            specs = specs_of_snapshot[chunk.snap_nr]
            halo_data = read_soap_rows(specs[0][1].filename, to_read, chunk.start, chunk.stop)
            halo_index = halo_data["InputHalos/HaloCatalogueIndex"].value.astype(int)

            if halo_index.shape[0] == 0:
                continue

            nr_read += halo_index.shape[0]

            # get halo centres in snapshot frame
            halo_centre_box = sw_units.drop_a_from_comoving_property(halo_data["InputHalos/HaloCentre"]).to_value("Mpc")

            for spec_nr, spec in specs:
                # transform halo centres from snapshot frame to lightcone frame
                lightcone_coords = self._to_lightcone(halo_centre_box, spec.tile, cell_data)

                # identify which haloes to keep
                keep_idx, __ = self.__select_particles_in_shell(
                    lightcone_coords,
                    zmin=spec.z_range[0],
                    zmax=spec.z_range[1],
                    method="exact",
                    r_tol=0 * unyt.Mpc,
                )

                if len(keep_idx) == 0:  # skip if no haloes kept
                    continue

                # add selected haloes to keep dict
                kept[spec_nr].append((lightcone_coords[keep_idx], halo_index[keep_idx], chunk.start + keep_idx))

        # joined catalogue tile by catalogue tile, so the order is the same however the catalogues are read
        joined = [chunk for spec_nr in sorted(kept) for chunk in kept[spec_nr]]
        if not joined:
            return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), nr_read

        # join the 'kept' haloes by catalogue tile, so the order is the same however the catalogues are read
        centres, snap_nr, index, rows = [], [], [], []
        for spec_nr in sorted(kept):
            # the snapshot of this catalogue tile, given to each of its haloes
            spec_snap_nr = shell.catalogues[spec_nr].snap_nr

            for chunk_centres, chunk_index, chunk_rows in kept[spec_nr]:
                centres.append(chunk_centres)
                snap_nr.append(np.full(len(chunk_index), spec_snap_nr, dtype=int))
                index.append(chunk_index)
                rows.append(chunk_rows)

        # return empty structure if there are no haloes left
        if not centres:
            return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), nr_read

        # build output by concatenating values for the kept haloes
        # centres [Mpc], snapshot numbers, catalogue indices (HaloCatalogueIndex), SOAP rows (SOAPIndex) and the
        # number of haloes read
        return (
            np.concatenate(centres, axis=0),
            np.concatenate(snap_nr),
            np.concatenate(index),
            np.concatenate(rows),
            nr_read,
        )

    def place_halos_in_shell(
        self,
        halo_format,
        lightcone_redshift_range,
        use_snapshots=None,
        soap_chunk_size=None,
        comm=None,
        redistribute_halos=False,
        snapshot_number=None,
    ):
        """
        Halo counterpart of place_snapshot_particles_in_shell.

        Place the haloes of SOAP catalogues in a lightcone, exactly as done for the particles.
        The catalogues are gathered (as done by gather_halo_catalogues), read, and their haloes moved into the
        lightcone exactly as particles are, keeping only those inside the lightcone. If gather_halo_catalogues was
        already called with the same arguments, what it found (a HaloShellRead) is reused, even if particles were placed
        in between.

        How the SOAP catalogues are read:

        -   in serial (no comm, one MPI rank):  soap_chunk_size rows at a time, or each catalogue as a whole if None.
        -   with MPI:   the catalogues are split into chunks of soap_chunk_size rows
            (if None, each catalogue into one chunk per rank), shared between the ranks as particle files are.
            Each rank reads only its own chunk, and returns the haloes it placed. With redistribute_halos, the haloes
            are then shared evenly between the ranks.

        The haloes placed are the same whichever way the catalogues are read (serial or MPI).

        Tracks the fewest properties needed to look up any other property of a halo in the SOAP catalogues later:

        -   ``Lightcone/HaloCentre``: the halo's centre, transformed into the lightcone frame exactly like a particle's
            Coordinates.
        -   ``Lightcone/ExpansionFactor``: the expansion factor at the halo's lightcone comoving distance.
        -   ``Lightcone/SnapshotNumber``: the snapshot the halo came from.
        -   ``InputHalos/HaloCatalogueIndex``: the halo's index in that snapshot's SOAP catalogue, as written in it.
        -   ``InputHalos/SOAPIndex``: the row of the halo in that snapshot's SOAP catalogue.

        Returns a :class:`LightconeOutput` of these properties, one row per halo placed.

        :param  halo_format:                format string for SOAP catalogue filenames (using {snap_nr})
        :type   halo_format:                str
        :param  lightcone_redshift_range:   minimum and maximum redshift of the shell to populate [z_min, z_max]
        :type   lightcone_redshift_range:   sequence of two floats
        :param  use_snapshots:              snapshot numbers to use, as in gather_files, bypassing the usual
                                            redshift-range lookup
        :type   use_snapshots:              list or numpy.ndarray
        :param  soap_chunk_size:            number of rows of a SOAP catalogue read at a time. If None, see above
        :type   soap_chunk_size:            int
        :param  comm:                       MPI communicator. If None, MPI.COMM_WORLD when more than one rank is
                                            running, otherwise serial
        :type   comm:                       mpi4py.MPI.Comm
        :param  redistribute_halos:         If True, with MPI, share the haloes placed evenly between the ranks
        :type   redistribute_halos:         bool
        :param  snapshot_number:            snapshot to read the box and cell metadata from, see gather_halo_catalogues
        :type   snapshot_number:            int
        """
        if comm is None and MPI.COMM_WORLD.Get_size() > 1:
            comm = MPI.COMM_WORLD

        with self._mpi_call(comm):
            return self.__place_halos(
                halo_format,
                lightcone_redshift_range,
                use_snapshots,
                soap_chunk_size,
                comm,
                redistribute_halos,
                snapshot_number,
            )

    def __place_halos(
        self,
        halo_format,
        lightcone_redshift_range,
        use_snapshots,
        soap_chunk_size,
        comm,
        redistribute_halos,
        snapshot_number,
    ):
        """
        Returns a LightconeOutput, as the inner layer of place_halos_in_shell.

        :param  halo_format:                see place_halos_in_shell
        :type   halo_format:                str
        :param  lightcone_redshift_range:   see place_halos_in_shell
        :type   lightcone_redshift_range:   sequence of two floats
        :param  use_snapshots:              see place_halos_in_shell
        :type   use_snapshots:              list or numpy.ndarray
        :param  soap_chunk_size:            see place_halos_in_shell
        :type   soap_chunk_size:            int
        :param  comm:                       MPI communicator, or None for serial
        :type   comm:                       mpi4py.MPI.Comm
        :param  redistribute_halos:         see place_halos_in_shell
        :type   redistribute_halos:         bool
        :param  snapshot_number:            see place_halos_in_shell
        :type   snapshot_number:            int
        """
        settings = HaloShellRead.settings_of(
            halo_format, lightcone_redshift_range, self.ang_radius_deg, use_snapshots, snapshot_number
        )
        if self._halo_shell is None or self._halo_shell.settings != settings:
            self.gather_halo_catalogues(
                halo_format,
                lightcone_redshift_range,
                use_snapshots=use_snapshots,
                snapshot_number=snapshot_number,
            )

        shell = self.last_shell = self._halo_shell
        metadata = self.lightcone_metadata(lightcone_redshift_range)

        if comm is None:
            my_chunks = self.__halo_read_chunks(shell, soap_chunk_size)
        else:
            # share the chunks between the ranks, as the particle files are
            size, rank = comm.Get_size(), comm.Get_rank()  # numb of ranks and local rank
            chunks = self.__halo_read_chunks(shell, soap_chunk_size, nr_chunks_min=size)
            chunks_on_rank = phdf5.assign_files(len(chunks), size)  # local ranks chunks of a tile
            first = (np.cumsum(chunks_on_rank) - chunks_on_rank)[rank]  # first halo on rank
            my_chunks = chunks[first : first + chunks_on_rank[rank]]  # ranks own parts of the lightcone

            update_str = (
                f"Haloes: reading {len(my_chunks)}/{len(chunks)} SOAP catalogue chunks, "
                f"{sum(piece.stop - piece.start for piece in my_chunks)} haloes"
            )
            self._log(update_str, level=1, every_rank=True)

        centres, snap_nr, halo_index, soap_index, nr_read = self.__read_and_place_halos(shell, my_chunks)

        nr_kept = len(snap_nr)
        if comm is not None:
            nr_read = comm.allreduce(nr_read)
            nr_kept = comm.allreduce(nr_kept)

        if nr_read > 0:
            self._log(
                f"Haloes: {nr_read} read, {nr_kept} kept in the lightcone ({nr_kept / nr_read * 100:.3f}%)", level=1
            )
        else:
            self._log("Haloes: 0 read", level=1)

        halos = {
            "Lightcone/HaloCentre": centres,
            "Lightcone/SnapshotNumber": snap_nr,
            "InputHalos/HaloCatalogueIndex": halo_index,
            "InputHalos/SOAPIndex": soap_index,
        }

        # if parallel and redistribute is given, then evenly share haloes across all ranks
        if comm is not None and redistribute_halos:
            size = comm.Get_size()
            ntot = comm.allreduce(len(snap_nr))
            ndesired = np.full(size, ntot // size, dtype=np.int64)
            ndesired[: ntot % size] += 1
            halos = {name: psort.repartition(values, ndesired, comm=comm) for name, values in halos.items()}
            self._log(f"redistributed {ntot} haloes evenly across {size} ranks", level=1)

        r_comoving = np.linalg.norm(halos["Lightcone/HaloCentre"], axis=1)
        expansion_factor = self.__comoving_distance_to_scalefactor(r_comoving) if len(r_comoving) > 0 else np.zeros(0)

        # output halo lightcone and metadata
        return LightconeOutput(
            {
                "Lightcone/HaloCentre": unyt.unyt_array(halos["Lightcone/HaloCentre"], unyt.Mpc),
                "Lightcone/SnapshotNumber": unyt.unyt_array(halos["Lightcone/SnapshotNumber"], unyt.dimensionless),
                "InputHalos/HaloCatalogueIndex": unyt.unyt_array(
                    halos["InputHalos/HaloCatalogueIndex"], unyt.dimensionless
                ),
                "InputHalos/SOAPIndex": unyt.unyt_array(halos["InputHalos/SOAPIndex"], unyt.dimensionless),
                "Lightcone/ExpansionFactor": unyt.unyt_array(expansion_factor, unyt.dimensionless),
            },
            metadata,
        )


class SnapshotBeam(SnapshotLightcone):
    """Construct a beam lightcone, about a single line of sight, from snapshots."""

    def __init__(
        self,
        simulation,
        beam_vector=None,
        ang_radius_deg=None,
        orientation_seed=0,
        orientation_lock=None,
        orientation_table=None,
        verbose=1,
        redshift_interp_method="fast",
        redshift_interp_grid_size=3000,
        observer_position=None,
    ):
        """

        :param  simulation:                 the Simulation (e.g. Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")),
                                            which sets where its snapshots, SOAP catalogues and output list are.
        :type   simulation:                 Simulation
        :param  beam_vector:                direction vector of the line of sight as an array of 3 floats
        :type   beam_vector:                numpy.ndarray
        :param  ang_radius_deg:             angular radius [deg] of the beam.
        :type   ang_radius_deg:             float
        :param  orientation_seed:           seed used to select the rotation, reflection and periodic shift of each box
                                            tile
        :type   orientation_seed:           int
        :param  orientation_lock:           how the orientations of the box tiles are shared:

                                            - "none": every box tile has its own orientation.
                                            - "cube": every box tile in a cube shell of boxes around the observer's box
                                              shares one orientation, so each shell is periodically continuous and the
                                              orientation only changes on the faces between shells.
                                            - "sphere": everything in a spherical shell, between (n-1/2) and (n+1/2) box
                                              sidelengths from the observer, shares one orientation, so the orientation
                                              only changes at those distances. A box crossing a sphere is used once for
                                              each shell, and tiles are labelled (nx, ny, nz, layer).
        :type   orientation_lock:           str
        :param  orientation_table:          path of a file of orientation tables (see
                                            snapshot_orientation.write_orientation_tables) to take the quarter turns,
                                            reflections and cell shifts of the box tiles from. If None, use the default
                                            tables
        :type   orientation_table:          str
        :param  verbose:                    amount printed. 0: nothing, only warnings. 1: a summary of each call. 2:
                                            details of every snapshot, box tile and file
        :type   verbose:                    int
        :param  redshift_interp_method:     how the comoving distance to redshift interpolation is made: "fast" or
                                            "precise" (see define_redshift_at_comoving_distance_function)
        :type   redshift_interp_method:     str
        :param  redshift_interp_grid_size:  number of points in the grid used to interpolate redshifts from comoving
                                            distances
        :type   redshift_interp_grid_size:  int
        :param  observer_position:          (x, y, z) comoving position [Mpc] in the snapshot box to put the observer
                                            at: the nearest cell corner is used, and every tile is periodically shifted
                                            by the same whole number of cells to move it to the observer. It can be
                                            replaced by each gather or placing call. If None, the observer is at the
                                            centre of its tile, as before.
        :type   observer_position:          array-like, shape (3,)
        """
        # check beam vector and angular radius is passed, the angular radius will be checked by _validate_ang_radius_deg
        if beam_vector is None:
            raise TypeError("SnapshotBeam needs a beam_vector, the direction of the line of sight")
        if ang_radius_deg is None:
            raise TypeError("SnapshotBeam needs an angular radius (ang_radius_deg)")

        # pass to SnapshotLightcone class
        super().__init__(
            simulation=check_simulation(simulation, lightcone_type="SnapshotBeam"),
            beam_vector=beam_vector,
            ang_radius_deg=ang_radius_deg,
            orientation_seed=orientation_seed,
            orientation_lock=orientation_lock,
            orientation_table=orientation_table,
            verbose=verbose,
            redshift_interp_method=redshift_interp_method,
            redshift_interp_grid_size=redshift_interp_grid_size,
            observer_position=observer_position,
        )

        # set up orthonormal basis perpendicular to beam_vec, used to place tiles transverse to the line of sight once
        # the beam's diameter exceeds one box sidelength.
        self.beam_transverse_e1, self.beam_transverse_e2 = self.__transverse_basis(self.beam_vec)

        self.__init_tile_frame()

    def __init_tile_frame(self):
        """
        Sets the tile frame as self._tile_frame, which is a permutation matrix mapping lattice positions to tiles.

        Tiles are placed on in a lattice using offsets equal to the snapshot box sidelengths
        (L=1000Mpc for L1_m9 FLAMINGO) along x, y and z, so they fill the beam exactly once for any beam direction.

        Tiles are labelled as (nx, ny, n_los), which is its lattice position along the x, y or z axis closest to e1, e2
        and beam_vec (with the same sign).

        """
        frame = np.array([self.beam_transverse_e1, self.beam_transverse_e2, self.beam_vec])
        tile_frame = np.zeros((3, 3), dtype=np.int64)
        free_axes = [0, 1, 2]
        # assign the beam's axis first, so n_los follows the line of sight
        for row in (2, 0, 1):
            axis = max(free_axes, key=lambda ax: abs(frame[row, ax]))
            tile_frame[row, axis] = 1 if frame[row, axis] >= 0 else -1
            free_axes.remove(axis)
        self._tile_frame = tile_frame

    def _validate_ang_radius_deg(self, ang_radius_deg):
        """
        Raise a TypeError if no angular radius is given and raise a ValueError if it exceeds MAX_BEAM_ANG_RADIUS_DEG.

        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        """
        # no radius given
        if ang_radius_deg is None:
            raise TypeError("SnapshotBeam needs an angular radius (ang_radius_deg)")

        # radius is too large
        if ang_radius_deg > MAX_BEAM_ANG_RADIUS_DEG:
            radius_error_str = (
                f"angular radius ({ang_radius_deg} [deg]) exceeds the maximum radius of {MAX_BEAM_ANG_RADIUS_DEG} [deg]"
            )
            raise ValueError(radius_error_str)

    @staticmethod
    def __transverse_basis(beam_vector):
        """
        Orthonormal basis (e1, e2) perpendicular to beam_vector, used to place tiles transverse to the line of sight.

        :param  beam_vector:    unit direction vector of the line of sight
        :type   beam_vector:    numpy.ndarray
        """
        reference = np.array([1.0, 0.0, 0.0]) if abs(beam_vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(beam_vector, reference)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(beam_vector, e1)
        return e1, e2

    def _tiles_intersecting_shell(self, r_min, r_max, boxsize):
        """
        Find every tile that the beam crosses between two comoving distances.

        Every integer tile index (nx, ny, nz) whose periodic box replica intersects the beam from the observer at
        the origin.
        Each box is centred at (nx, ny, nz)*boxsize and has a length of
        [n_i*boxsize - boxsize/2, n_i*boxsize + boxsize/2) along each axis.

        Returns a list of tuples [(nx, ny, n_los)]
            n_los:      is the replica index along beam_vec (how many replications)
            (nx, ny):   is the replica offset along the two transverse axes.

        :param  r_min:          minimum comoving distance [Mpc]
        :type   r_min:          float
        :param  r_max:          maximum comoving distance [Mpc]
        :type   r_max:          float
        :param  boxsize:        snapshot box side length [Mpc] (box assumed cubic)
        :type   boxsize:        float
        """
        half_angle = np.deg2rad(self.ang_radius_deg)
        half_L = 0.5 * boxsize

        along_min = r_min * np.cos(half_angle)

        w_max = r_max * np.sin(half_angle)
        perp_extent = w_max * np.sqrt(np.clip(1.0 - self.beam_vec**2, 0.0, None))
        ends = np.array([along_min * self.beam_vec, r_max * self.beam_vec])
        lo = ends.min(axis=0) - perp_extent
        hi = ends.max(axis=0) + perp_extent

        # lattice positions of the boxes (centred on m*boxsize) overlapping the bounding box
        m_lo = np.floor(lo / boxsize + 0.5).astype(int)
        m_hi = np.floor(hi / boxsize + 0.5).astype(int)

        # radius of the sphere bounding a box
        box_radius = np.sqrt(3.0) * half_L

        tiles = []
        for mx in range(m_lo[0], m_hi[0] + 1):
            for my in range(m_lo[1], m_hi[1] + 1):
                for mz in range(m_lo[2], m_hi[2] + 1):
                    m = np.array([mx, my, mz])
                    centre = m * boxsize

                    # closest and furthest distance of the box from the observer
                    near = np.linalg.norm(np.maximum(np.abs(centre) - half_L, 0.0))
                    far = np.linalg.norm(np.abs(centre) + half_L)
                    if near > r_max or far < r_min:
                        continue

                    # the box's bounding sphere must reach inside the cone
                    dist = np.linalg.norm(centre)
                    if dist > box_radius:
                        angle = np.arccos(np.clip(centre @ self.beam_vec / dist, -1.0, 1.0))
                        if angle > half_angle + np.arcsin(box_radius / dist):
                            continue

                    tiles.append(tuple(int(n) for n in self._tile_frame @ m))

        # order along the line of sight
        return sorted(tiles, key=lambda tile: (tile[2], tile[0], tile[1]))

    def _in_shell(
        self,
        coords,
        z_min,
        z_max,
        buffer_length=0.0,
        buffer_shape="cube",
        return_bool=False,
        cosmo=None,
    ):
        """
        Determine which points fall inside the lightcone shell defined by an angular radius and the redshift range.

        Returns a tuple, (indices inside the shell, indices outside the shell).
        If return_bool is True, returns (boolean mask, None), where it is True for points inside the lightcone wedge.

        :param  coords:         Cartesian comoving lightcone coordinates [Mpc], shape (N, 3)
        :type   coords:         numpy.ndarray
        :param  z_min:          minimum redshift of the lightcone slice
        :type   z_min:          float
        :param  z_max:          maximum redshift of the lightcone slice
        :type   z_max:          float
        :param  buffer_length:  size [Mpc] of the buffer added to the boundaries of the shell
        :type   buffer_length:  float
        :param  buffer_shape:   'cube', buffer_length is a cube sidelength. 'sphere', buffer_length is a sphere radius
        :type   buffer_shape:   str
        :param  return_bool:    If True, return a boolean mask instead of indices
        :type   return_bool:    bool
        :param  cosmo:          A cosmology object. If None, use the snapshot cosmology.
                                    If not None, then either an astropy of SnapshotCosmology object
                                    (the swiftlet class object, i.e. a wrapper for astropy.cosmology objects)
                                    or a path to the snapshot from which the cosmology info can be extracted
        :type   cosmo:          astropy.cosmology.w0waCDM, SnapshotCosmology object (a swiftlet class) or str
        """
        if cosmo is None:
            cosmo = self.cosmo
        else:
            cosmo = as_astropy_cosmology(cosmo)

        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 3:  # test for coords shape
            raise ValueError("coords must be a vector with shape (N, 3)")

        axis = self.beam_vec
        half_angle = np.deg2rad(self.ang_radius_deg)
        shell_inner = unyt.unyt_array.from_astropy(cosmo.comoving_distance(z_min)).to_value("Mpc")
        shell_outer = unyt.unyt_array.from_astropy(cosmo.comoving_distance(z_max)).to_value("Mpc")

        # Resolve the buffer length + shape into an equivalent buffer radius
        if buffer_length > 0:
            if buffer_shape == "cube":
                # buffer_length = cube sidelength; radius = half the space diagonal,
                # the largest possible distance from the cube's centre to a corner.
                buffer_radius = (np.sqrt(3.0) / 2.0) * buffer_length
            elif buffer_shape == "sphere":
                # buffer_length = sphere radius.
                buffer_radius = buffer_length
            else:
                raise ValueError(f"buffer shape '{buffer_shape}' not recognised")
        else:
            buffer_radius = 0.0

        m = np.empty(coords.shape[0], dtype=np.bool_)

        if _HAVE_NUMBA:
            # use faster identification method
            _classify_wedge_numba(coords, axis, shell_inner, shell_outer, half_angle, buffer_radius, m)

        else:
            r = np.linalg.norm(coords, axis=1)
            along_axis = coords @ axis
            perp_vec = coords - along_axis[:, None] * axis
            r_perp = np.linalg.norm(perp_vec, axis=1)
            angle_from_axis = np.arctan2(r_perp, along_axis)

            with np.errstate(invalid="ignore"):
                angle_buffer = np.where(
                    r > buffer_radius,
                    np.arcsin(np.clip(buffer_radius / np.where(r > 0, r, np.inf), 0.0, 1.0)),
                    np.pi,
                )

            m[:] = (
                (r > shell_inner - buffer_radius)
                & (r <= shell_outer + buffer_radius)  # dist from observer check
                & (along_axis > -buffer_radius)  # direction check
                & (angle_from_axis <= half_angle + angle_buffer)  # perpendicular distance check
            )

        if return_bool:
            return m, None
        else:
            return np.flatnonzero(m), np.flatnonzero(np.invert(m))

    def _diameter_at_redshift(self, z, cosmo=None, use_comoving_dist=False):
        """
        Returns the diameter [Mpc] of the beam's cone at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  cosmo:              cosmology. If given, replaces the stored cosmology
        :type   cosmo:              astropy.cosmology, SnapshotCosmology (swiftlet class object), or str
        :param  use_comoving_dist:  If True, then z is passed as a comoving distance from the observer
        :type   use_comoving_dist:  bool
        """
        # set cosmology
        if cosmo is not None:
            self.cosmo = as_astropy_cosmology(cosmo)

        half_angle = np.deg2rad(self.ang_radius_deg)
        if use_comoving_dist:
            h = z
        else:
            h = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(z)).to_value("Mpc")
        return h * np.tan(half_angle) * 2

    def beam_diameter(self, z, cosmo=None, use_comoving_dist=False):
        """
        Returns the diameter [Mpc] of the beam's cone at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  cosmo:              cosmology to use. If given, it also replaces the lightcone's stored cosmology
        :type   cosmo:              astropy.cosmology.w0waCDM, SnapshotCosmology or str
        :param  use_comoving_dist:  if True, z is a comoving distance [Mpc] from the observer, not a redshift
        :type   use_comoving_dist:  bool
        """
        return self._diameter_at_redshift(z, cosmo=cosmo, use_comoving_dist=use_comoving_dist)

    def _lattice_tile(self, tile):
        """
        Transform the position of the tile from the beams coordnates (nx, ny, n_los) into the coordaintes of the regular
        lattice of tiles along x, y and z, (nx, ny, nz).

        This function over-rides the version in SnapshotLightcone as it uses the Beams specific geometry. This is
        necessary for the Beam to be be able to reproduce the same off-axis subvolumes of the lightcone that are found
        in the all-sky lightcones. A tiles orientation is hashed from this function , so a tile now takes the same
        orientation as in a SnapshotAllSky.

        Returns a tuple of 3 ints.

        :param  tile:   (nx, ny, n_los), the tile's label in the beam's frame
        :type   tile:   tuple
        """
        return tuple(int(n) for n in self._tile_frame.T @ np.asarray(tile[:3], dtype=np.int64))

    def get_snapshot_reposition_coords(self, tile, snapshot_sidelengths):
        """
        Returns the offset of a snapshots geometric centre at its position on the lattice of tiles.

        :param  tile:                   (nx, ny, n_los), the tile's lattice position along the x, y or z axis
                                        closest to e1, e2 and the beam vector
        :type   tile:                   tuple
        :param  snapshot_sidelengths:   side lengths [Mpc] of the snapshot box
        :type   snapshot_sidelengths:   numpy.ndarray
        """
        # lattice position of the tile along x, y, z
        lattice_position = np.asarray(self._lattice_tile(tile), dtype=np.int64)

        snapshot_sidelengths = np.asarray(snapshot_sidelengths, dtype=float)
        return -0.5 * snapshot_sidelengths + lattice_position * snapshot_sidelengths


class SnapshotAllSky(SnapshotLightcone):
    """Construct an all-sky lightcone from snapshots (a SnapshotLightcone with no beam vector)."""

    def __init__(
        self,
        simulation,
        orientation_seed=0,
        orientation_lock=None,
        orientation_table=None,
        verbose=1,
        redshift_interp_method="fast",
        redshift_interp_grid_size=3000,
        observer_position=None,
    ):
        """

        :param  simulation:                 the Simulation (e.g. Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")),
                                            which sets where its snapshots, SOAP catalogues and output list are.
        :type   simulation:                 Simulation
        :param  orientation_seed:           seed used to select the rotation, reflection and periodic shift of each box
                                            tile
        :type   orientation_seed:           int
        :param  orientation_lock:           how the orientations of the box tiles are shared:

                                            - "none": every box tile has its own orientation.
                                            - "cube": every box tile in a cube shell of boxes around the observer's box
                                              shares one orientation, so each shell is periodically continuous and the
                                              orientation only changes on the faces between shells.
                                            - "sphere": everything in a spherical shell, between (n-1/2) and (n+1/2) box
                                              sidelengths from the observer, shares one orientation, so the orientation
                                              only changes at those distances. A box crossing a sphere is used once for
                                              each shell, and tiles are labelled (nx, ny, nz, layer).
        :type   orientation_lock:           str
        :param  orientation_table:          path of a file of orientation tables (see
                                            snapshot_orientation.write_orientation_tables) to take the quarter turns,
                                            reflections and cell shifts of the box tiles from. If None, use the default
                                            tables
        :type   orientation_table:          str
        :param  verbose:                    amount printed. 0: nothing, only warnings. 1: a summary of each call. 2:
                                            details of every snapshot, box tile and file
        :type   verbose:                    int
        :param  redshift_interp_method:     how the comoving distance to redshift interpolation is made: "fast" or
                                            "precise" (see define_redshift_at_comoving_distance_function)
        :type   redshift_interp_method:     str
        :param  redshift_interp_grid_size:  number of points in the grid used to interpolate redshifts from comoving
                                            distances
        :type   redshift_interp_grid_size:  int
        :param  observer_position:          (x, y, z) comoving position [Mpc] in the snapshot box to put the observer
                                            at: the nearest cell corner is used, and every tile is periodically shifted
                                            by the same whole number of cells to move it to the observer. It can be
                                            replaced by each gather or placing call. If None, the observer is at the
                                            centre of its tile, as before.
        :type   observer_position:          array-like, shape (3,)
        """
        # pass all info to SnapshotLightcone class.
        super().__init__(
            simulation=check_simulation(simulation, lightcone_type="SnapshotAllSky"),
            beam_vector=None,  # not required for all sky, ensure no information is passed for beam vectors
            ang_radius_deg=None,  # not required for all sky, ensure no information is passed for angular radius
            orientation_seed=orientation_seed,
            orientation_lock=orientation_lock,
            orientation_table=orientation_table,
            verbose=verbose,
            redshift_interp_method=redshift_interp_method,
            redshift_interp_grid_size=redshift_interp_grid_size,
            observer_position=observer_position,
        )

    def _validate_ang_radius_deg(self, ang_radius_deg):
        """
        Raise TypeError if an angular radius is given, as its not needed for full-sky.

        :param  ang_radius_deg: must be None
        :type   ang_radius_deg: None
        """
        # given a radius when none is needed
        if ang_radius_deg is not None:
            raise TypeError("SnapshotAllSky doesn't need an angular radius")

    def _tiles_intersecting_shell(self, r_min, r_max, boxsize):
        """
        Find every tile that the lightcone crosses between two comoving distances.

        Every integer tile index (nx, ny, nz) whose periodic box replica intersects the spherical annulus [r_min, r_max]
        around the observer at the origin.
        Each box is centred at (nx, ny, nz)*boxsize and has a length of
        [n_i*boxsize - boxsize/2, n_i*boxsize + boxsize/2) along each axis.

        Returns a list of (nx, ny, nz) int tuples.

        :param  r_min:          minimum comoving distance [Mpc]
        :type   r_min:          float
        :param  r_max:          maximum comoving distance [Mpc]
        :type   r_max:          float
        :param  boxsize:        snapshot box side length [Mpc] (box assumed cubic)
        :type   boxsize:        float
        """
        half_L = 0.5 * boxsize
        n_max = int(np.ceil(r_max / boxsize + 0.5)) + 1

        tiles = []
        for nx in range(-n_max, n_max + 1):
            cx_lo, cx_hi = nx * boxsize - half_L, nx * boxsize + half_L
            near_x = 0.0 if cx_lo <= 0.0 <= cx_hi else min(abs(cx_lo), abs(cx_hi))
            far_x = max(abs(cx_lo), abs(cx_hi))

            for ny in range(-n_max, n_max + 1):
                cy_lo, cy_hi = ny * boxsize - half_L, ny * boxsize + half_L
                near_y = 0.0 if cy_lo <= 0.0 <= cy_hi else min(abs(cy_lo), abs(cy_hi))
                far_y = max(abs(cy_lo), abs(cy_hi))

                for nz in range(-n_max, n_max + 1):
                    cz_lo, cz_hi = nz * boxsize - half_L, nz * boxsize + half_L
                    near_z = 0.0 if cz_lo <= 0.0 <= cz_hi else min(abs(cz_lo), abs(cz_hi))

                    near_dist = np.sqrt(near_x**2 + near_y**2 + near_z**2)
                    if near_dist > r_max:
                        continue

                    far_z = max(abs(cz_lo), abs(cz_hi))
                    far_dist = np.sqrt(far_x**2 + far_y**2 + far_z**2)
                    if far_dist < r_min:
                        continue

                    tiles.append((nx, ny, nz))

        return tiles

    def _in_shell(
        self,
        coords,
        z_min,
        z_max,
        buffer_length=0.0,
        buffer_shape="cube",
        return_bool=False,
        cosmo=None,
    ):
        """
        Determine which points fall inside the lightcone shell defined by a redshift range.

        Returns a tuple of (indices inside the shell, indices outside the shell). If return_bool is True,
        returns (boolean mask, None).

        :param  coords:         Cartesian comoving lightcone coordinates [Mpc], shape (N, 3)
        :type   coords:         numpy.ndarray
        :param  z_min:          minimum redshift of the lightcone shell
        :type   z_min:          float
        :param  z_max:          maximum redshift of the lightcone shell
        :type   z_max:          float
        :param  buffer_length:  size [Mpc] of the buffer added to the boundaries of the shell
        :type   buffer_length:  float
        :param  buffer_shape:   'cube', buffer_length is a cube sidelength. 'sphere', buffer_length is a sphere radius
        :type   buffer_shape:   str
        :param  return_bool:    If True, return a boolean mask instead of indices
        :type   return_bool:    bool
        :param  cosmo:          A cosmology object. If None, use the snapshot cosmology.
                                    If not None, then either an astropy of SnapshotCosmology object
                                    (the swiftlet class object, i.e. a wrapper for astropy.cosmology objects)
                                    or a path to the snapshot from which the cosmology info can be extracted.
        :type   cosmo:          astropy.cosmology.w0waCDM, SnapshotCosmology object (a swiftlet class object) or str
        """
        # update to a new cosmology if one is given
        if cosmo is None:
            cosmo = self.cosmo
        else:
            cosmo = as_astropy_cosmology(cosmo)

        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be a vector with shape (N, 3)")

        shell_inner = unyt.unyt_array.from_astropy(cosmo.comoving_distance(z_min)).to_value("Mpc")
        shell_outer = unyt.unyt_array.from_astropy(cosmo.comoving_distance(z_max)).to_value("Mpc")

        if buffer_length > 0:
            if buffer_shape == "cube":
                buffer_radius = (np.sqrt(3.0) / 2.0) * buffer_length
            elif buffer_shape == "sphere":
                buffer_radius = buffer_length
            else:
                raise ValueError(f"buffer shape '{buffer_shape}' not recognised")
        else:
            buffer_radius = 0.0

        r = np.linalg.norm(coords, axis=1)
        m = (r >= shell_inner - buffer_radius) & (r <= shell_outer + buffer_radius)

        if return_bool:
            return m, None
        else:
            return np.flatnonzero(m), np.flatnonzero(np.invert(m))

    def _diameter_at_redshift(self, z, cosmo=None, use_comoving_dist=False):
        """
        Returns the diameter [Mpc] of the past lightcone at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  cosmo:              cosmology. If given, replaces the stored cosmology.
        :type   cosmo:              astropy.cosmology.w0waCDM, SnapshotCosmology object or str
        :param  use_comoving_dist:  If True, then redshift is passed as a comoving distance from the observer
        :type   use_comoving_dist:  bool
        """
        # update to a new cosmology if one is given
        if cosmo is not None:
            self.cosmo = as_astropy_cosmology(cosmo)

        # get height (h), now the radius for full-sky, of the ligthcones beam
        if use_comoving_dist:
            h = z
        else:
            h = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(z)).to_value("Mpc")
        return h * 2

    def lightcone_diameter(self, z, cosmo=None, use_comoving_dist=False):
        """
        Returns the diameter [Mpc] of the past lightcone at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  cosmo:              cosmology to use. If given, it also replaces the lightcone's stored cosmology
        :type   cosmo:              astropy.cosmology.w0waCDM, SnapshotCosmology or str
        :param  use_comoving_dist:  if True, z is a comoving distance [Mpc] from the observer, not a redshift
        :type   use_comoving_dist:  bool
        """
        return self._diameter_at_redshift(z, cosmo=cosmo, use_comoving_dist=use_comoving_dist)

    def get_snapshot_reposition_coords(self, tile, snapshot_sidelengths):
        """
        Returns the coordinate offset of a snapshot box from the observer.

        :param  tile:                   (nx, ny, nz) tile index
        :type   tile:                   tuple
        :param  snapshot_sidelengths:   side lengths [Mpc] of the snapshot box
        :type   snapshot_sidelengths:   numpy.ndarray
        """
        tile_arr = np.asarray(tile[:3], dtype=float)
        snapshot_sidelengths = np.asarray(snapshot_sidelengths, dtype=float)
        return -0.5 * snapshot_sidelengths + tile_arr * snapshot_sidelengths
