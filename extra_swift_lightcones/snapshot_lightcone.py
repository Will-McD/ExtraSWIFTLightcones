#!/bin/env python
import os
import sys
import numpy as np
import unyt
import h5py
import argparse
import healpy as hp
import glob
import re
from collections import namedtuple
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.mpi.util import MPIArgumentParser
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from lightcone_io.particle_reader import merge_cells
import lightcone_io.halo_catalogue as hc

#import snapshot_orientation as box_structure
#import snapshot_units as sw_units
#import swift_snapshot_redshift_conversion as nz
from . import snapshot_orientation as box_structure
from . import snapshot_units as sw_units
from . import  swift_snapshot_redshift_conversion as nz
import datetime as dt

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

def seperator_str(n=35,line_seperator="~"):
    sep_str=line_seperator * n
    return "\n"+sep_str

def message(rank, m, time_date_update=False):
    """
    Print a new message if on the prime (zero) rank. 
    """
    if rank is None:
        if time_date_update:
            current_time=dt.datetime.now()
            time_str=current_time.strftime("%H:%M:%S")
            m = '[@{print_time}]\t'.format(print_time=time_str) + m
        print(m)
    else:
        if rank == 0:
            if time_date_update:
                current_time=dt.datetime.now()
                time_str=current_time.strftime("%H:%M:%S")
                m = '[@{print_time}]\t'.format(print_time=time_str) + m
            print(m)

def rank_message(rank, m):
    """
    Print a new message on each rank.
    """
    current_time=dt.datetime.now()
    time_str=current_time.strftime("%H:%M:%S")
    print('\t[Rank {rank_nr:03d}] [@{print_time}]\t'.format(rank_nr=rank,print_time=time_str) + m)

def print_coord_ranges(x, indent=0, rank=None):
    if indent >0:
        indent = "\t"*indent
    else:
        indent=""
    for i in range(3):
        coords_str = indent+ f"{i}, {np.min(x[:,i]):.3f} -> {np.max(x[:,i]):.3f}"
        message(rank, coords_str)


FileReadSpec = namedtuple("FileReadSpec", ["snap_nr", "tile", "file_nr", "offsets", "lengths"])
SnapshotReadRecord = namedtuple("SnapshotReadRecord", ["snap_nr", "tile", "z_updated"])


# numba optimised function for flagging cells and particles in a wedge 
if _HAVE_NUMBA:
    #print(f"defining numba enhanced functions")
    # setting parallel=False when using MPI parallel processing
    @njit(parallel=False, fastmath=True, cache=True)
    def _classify_wedge_numba(coords, axis, chi_inner, chi_outer, half_angle,
                               buffer_radius, mask):
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
                (r > chi_inner - buffer_radius) and (r <= chi_outer + buffer_radius)
                and (along > -buffer_radius)
                and (angle <= half_angle + angle_buffer)
            )
        return mask

def define_redshift_at_comoving_distance_function(cosmo, zmin=0.0001, zmax=5.05, n_grid=3000, method="fast"):

    if method=="fast":
        # fastes method to build grid
        z_grid = np.linspace(zmin, zmax, n_grid)
        r_grid = unyt.unyt_array.from_astropy(cosmo.comoving_distance(z_grid)).to_value("Mpc")
    elif method=="precise":
        # uniform grid in comoving dist gives more precise interp function than grid from z
        r_range = unyt.unyt_array.from_astropy(cosmo.comoving_distance([zmin, zmax])).to_value("Mpc")
        r_grid = np.linspace(r_range[0], r_range[1], n_grid)
        def r_from_z(z):
            return unyt.unyt_quantity.from_astropy(cosmo.comoving_distance(z)).to_value("Mpc")
        z_grid = np.array([
                brentq(lambda z: r_from_z(z) - r, 1e-10, zmax) for r in r_grid
            ])
    else:
        raise ValueError("grid building method {method} not recognised, use 'fast' or 'precise'")
    
    return CubicSpline(r_grid, z_grid)

# maximum angular radius of beam, larger beams become far slower than utilising a full sky map due to searching boundary conditions. 
MAX_BEAM_ANG_RADIUS_DEG=60.01

# random primes used for offsets of selecting rotation, reflection and periodic shifts about each axis when creating new snapshot boxes 
TILE_HASH_PRIMES = (374761393, 2147483647, 668265263) 
TILE_HASH_SEED_PRIME = 2246822519

class SnapshotLightcone():
    """
    Fundamental code used to constuct lightcones from snapshots 
    with SnapshotBeam and SnapshotAllSky sub classes. 

    This class tracks:
        1. snapshot cosmology 
        2. MPI ranks
        3. periodic replications of snapshots and their reorientation
            to avoid exact replication of structures. 
        4. reading and placing particles from snapshots into 
            the lightcone

    The exact methods of identifying particles within the lightcones footprint 
    and how the snapshot box replication occurs is handled in 
    the SnapshotBeam and SnapshotAllSky sub classes. 
    """

    def __init__(self, boxsize_resolution, simulation_name, beam_vector, orientation_seed=0):
        """
            Define the lightcones cosmology, units and vector + radius (where applicable)
        """

        # simulation values
        self.box_res=boxsize_resolution
        self.sim_name=simulation_name
        simulation_dir="/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}".format(box_res=self.box_res, sim_name=self.sim_name)
        self.snapshot_format = simulation_dir+'/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5'

        # define the cosmology from the snapshots
        self.__define_snapshot_cosmology(int(nz.snapshot_number_redshifts(snapshot_number=0, boxsize_resolution=self.box_res, inverse=True)), 0)

        #self.__clear_all_internal_values()

        # define the beam vector (direction of line of sight)
        if beam_vector is None:
            self.beam_vec = None
        else:
            self.beam_vec=self.__define_beam_vector(beam_vector)
        
        self.orientation_seed = int(orientation_seed)
        
        # values & caches to compute once and reuse for every call
        self.unit_registry=None
        self._unit_metadata=None
        self._cell_data_cache={} # track cell data per snapshot
        self._ptype_in_file_cache={} # track ptype in  data per snapshot

        self.__clear_shell_read_state()
        self._define_mpi_mode(None, False)

        # define grid to interpolate redshift from comoving distance
        self.__redshift_from_comoving_distance([0.0005, 5.0], n_grid=5000, method="precise")

    def __clear_shell_read_state(self, ):
        """
        Reset the data built up while reading one particle type of a ligthcone shell.
        Call when starting to build a new shell, so repeated calls on the same
        instance never leak between calls.
        """
        self.__clear_snapshot_read_state()
        self.snapshots_to_read=None
        self.snap_file_offset_lengths=None
        self.npart_per_file=None
        self.last_snapshot_read=None
        self.particle_data=None

        self.npart_kept_per_file=None
        self._file_order_index=None

    def __clear_snapshot_read_state(self, ):
        # snapshot specific values
        # points to state of current snapshot and cell data
        self.__snap_cell_data=None
        self.__cell_data_snap_nr=None
        self.__snap_redshift_range_to_populate=None

    def __reset_cell_read_state(self, snap_nr):

        if snap_nr not in self._cell_data_cache:
            self._cell_data_cache[snap_nr] = self.__get_cell_data(snap_nr)
        self.__snap_cell_data = self._cell_data_cache[snap_nr] # current snapshot cell data
        self.__cell_data_snap_nr=snap_nr # current snapshot number of cell data

    def __last_snapshot_read(self, snap_nr, tile, z_updated):

        if self.last_snapshot_read is None:
            self.last_snapshot_read=[]
        self.last_snapshot_read.append(SnapshotReadRecord(snap_nr, tile, z_updated))

    def __last_snapshot_index(self, snap_nr, tile):
        for idx, record in enumerate(self.last_snapshot_read):
            if record.snap_nr == snap_nr and record.tile == tile:
                return idx
        raise ValueError(f"snapshot {snap_nr}, tile {tile} not found in last_snap records")

    @staticmethod
    def __define_beam_vector(beam_vector):
        beam_vector = np.asarray(beam_vector, dtype=float)
        if beam_vector.shape != (3,): # test for vector shape
            raise ValueError("beam_vector must be a vector with shape (3,)")
        norm = np.linalg.norm(beam_vector)
        if norm == 0: # test for non-zero vector
            raise ValueError("beam_vector must be a nonzero vector")
        return beam_vector / norm # ensure unit vector

    def __get_cell_data(self, snap_nr):
        cell_data={}
        with h5py.File(self.snapshot_format.format(snap_nr=snap_nr, file_nr=0), "r") as f:
            cell_data["nr_cells"]=f["/Cells/Meta-data"].attrs["nr_cells"][0] # total number of cells
            cell_data["nr_cells_axis"]=f["/Cells/Meta-data"].attrs['dimension'][:] # number of cells per axes
            cell_data["cell_size"]=f["/Cells/Meta-data"].attrs["size"] # side length of cell
            cell_data["cell_centres"]=f["/Cells/Centres"][:,:] # centre of each cell in snapshot coords
            cell_data["snap_boxsize"]=f['Header'].attrs['BoxSize'] # box sidelength
            cell_data["nr_files"]=f['Header'].attrs['NumFilesPerSnapshot'][0] # num of subfiles

            cells_in_file_dict={}
            snap_offsets={}
            snap_lengths={}

            ptype_in_file_boolean = f["Header"].attrs["NumPart_ThisFile"][:]>0

            for i in range(7):
                if f["Header"].attrs['NumPart_Total'][i]>0:
                    cells_in_file_dict[f"PartType{i}"] = f[f"Cells/Files/PartType{i}"][...]
                    snap_offsets[f"PartType{i}"] = f[f"/Cells/OffsetsInFile/PartType{i}"][...]
                    snap_lengths[f"PartType{i}"] = f[f"/Cells/Counts/PartType{i}"][...]
                    assert snap_offsets[f"PartType{i}"].shape == snap_lengths[f"PartType{i}"].shape
                    assert np.shape(snap_offsets[f"PartType{i}"])[0] == np.shape(cell_data["cell_centres"])[0]

            cell_data["cells_in_file"]=cells_in_file_dict
            cell_data["offsets"]=snap_offsets
            cell_data["lengths"]=snap_lengths

        return cell_data

    def __files_with_ptype(self, snap_nr, numb_files):

        # if snapshot is already stored in cache return cache values
        if snap_nr in self._ptype_in_file_cache:
            return self._ptype_in_file_cache[snap_nr]

        infile_dict = {f"PartType{i}":[] for i in np.arange(8)}

        for ii in range(numb_files):
            with h5py.File(self.snapshot_format.format(snap_nr=snap_nr, file_nr=ii), "r") as f:
                m=f["Header"].attrs["NumPart_ThisFile"][:]>0
                for jj in range(7):
                    if m[jj]:
                        infile_dict[f"PartType{jj}"].append(ii)

        self._ptype_in_file_cache[snap_nr] = infile_dict # update cache
        return infile_dict

    def __define_snapshot_cosmology(self, snapshot_number, file_number):
        self.cosmo = Snapshot_Cosmology_For_Lightcone(self.snapshot_format.format(snap_nr=snapshot_number, file_nr=file_number)).COSMO

    def __comoving_distance_to_scalefactor(self, r, cosmo=None):
        """
        Compute the scale factor of particles based on comoving distance from observer
        """
        if cosmo is None:
            cosmo=self.cosmo
        if self.interp_redshift_from_comoving_distance is None:
            raise ValueError("interpolation function not defined")

        r = apply_expected_units(r, unyt.Mpc) # place distance in terms of Mpc
        a = 1./(self.interp_redshift_from_comoving_distance(r.to_value("Mpc"))+1.)

        return a

    def __redshift_from_comoving_distance(self, shell_z, n_grid, method="precise"):
        zmin=shell_z[0]-0.01 if shell_z[0]>0.01 else shell_z[0]
        zmax=shell_z[1]+0.01
        self.interp_redshift_from_comoving_distance = define_redshift_at_comoving_distance_function(self.cosmo, zmin, zmax, n_grid, method="precise")

    def __remove_snapshots_outside_shell(self, shell_z):
        """
        Remove snapshots to read that do not fit within the bounds of the shell.
        """
        m = np.ones(len(self.snapshots_to_read), dtype=bool)
        for ii, snap_nr in enumerate(self.snapshots_to_read):
            snapshot_z_range = nz.snapshot_redshift_range(snap_nr, self.box_res)
            if snapshot_z_range[0] > shell_z[1]:
                m[ii]=0
            elif  snapshot_z_range[1] < shell_z[0]:
                m[ii]=0
        self.snapshots_to_read=self.snapshots_to_read[m]

    def _define_mpi_mode(self, comm, redistibute_particles):
        """
        Enable MPI mode for this output. In MPI mode each MPI rank reads a
        subset of the selected particles.

        :param comm: MPI communicator
        :type  comm: mpi4py.MPI.Comm
        """
        self.comm = comm

        if self.comm is None:
            self.comm_rank=None
            self.comm_size=None
            self.redistibute_particles=False
        else:
            self.comm_rank = comm.Get_rank()
            self.comm_size = comm.Get_size()
            self.redistibute_particles=redistibute_particles

    def _snapshot_tile_idx(self, snap_shell_z_range, ang_radius_deg=None):
        """
        Every 3D tile a snapshot's assigned redshift range touches,
        paired with that snapshot's full (unsplit) z range -- see the
        class docstring for why all-sky doesn't split by z the way the
        pencil beam splits along a single line of sight.

        ang_radius_deg is accepted (and unused) purely for call-site
        compatibility with the shared __gather_cells_and_tiles engine,
        which now passes it to support SnapshotBeam's transverse tiling
        -- meaningless here, since every direction is already covered by
        _tiles_intersecting_shell regardless of any angle.
        """
        r_min, r_max = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")

        boxsize = self.__snap_cell_data["snap_boxsize"][0]

        tiles = self._tiles_intersecting_shell(r_min, r_max, boxsize, ang_radius_deg)
        return [(tile, snap_shell_z_range[0], snap_shell_z_range[1]) for tile in tiles]

    def _snapshot_reorientation(self, tile):
        """
        Orientation (rotation, reflection, periodic shift) for box tile
        (nx, ny, n_los). If the comoving distance to this redshift is
        > box sidelength, create a new permutation of the snapshot to
        avoid periodic replication. The hash collapses to exactly
        n_los when nx=ny=0, so existing (LOS-only) orientations are
        unaffected by transverse tiling.
        """
        n_ang = box_structure.SnapshotBeamAngles_quaters.shape[0]
        n_ref = box_structure.SnapshotBeamAngles_reflections.shape[0]
        n_shift = box_structure.SnapshotBeamAngles_shifts.shape[0]

        nx, ny, nz = tile
        p1, p2, p3 = TILE_HASH_PRIMES
        tile_hash = nx * p1 + ny * p2 + nz * p3 + self.orientation_seed * TILE_HASH_SEED_PRIME

        orientation_idx = (tile_hash % n_ang, tile_hash % n_ref, tile_hash % n_shift)
        ang = box_structure.SnapshotBeamAngles_quaters[orientation_idx[0]] * 90  # degrees
        ref = box_structure.SnapshotBeamAngles_reflections[orientation_idx[1]]
        shift = box_structure.SnapshotBeamAngles_shifts[orientation_idx[2]]

        return ang, ref, shift

    def _cell_shift_to_vector(self, cell_shift, n_cells_per_axis, cell_sidelength,
                              snapshot_sidelength):
        """
        convert cell shift to a comoving distance vector

        Params
            n_cells_per_axis: (3,) array, number of cells per sidelength
            cell_sidelength: (3, ) array, length along each axis a cell
            snapshot_sidelength: (3, ) array, length along each axis of snapshot
        """

        if cell_shift is None:
            cell_shift=np.zeros(3, dtype=np.int64)
        # check params have correct shapes
        cell_shift = np.asarray(cell_shift, dtype=np.int64)
        if cell_shift.shape != (3,):
            raise ValueError("cell_shift must have shape (3,)")

        n_cells_per_axis = np.broadcast_to(np.asarray(n_cells_per_axis), (3,)).astype(np.int64)
        cell_sidelength = np.broadcast_to(np.asarray(cell_sidelength, dtype=np.float64), (3,))
        snapshot_sidelength = np.broadcast_to(np.asarray(snapshot_sidelength, dtype=np.float64), (3,))

        expected_snapshot_sidelength = n_cells_per_axis * cell_sidelength

        # check cells sum to boxsize
        if not np.allclose(expected_snapshot_sidelength, snapshot_sidelength, rtol=1e-5):
            raise ValueError(f"n_cells_per_axis * cell_sidelength ({expected_snapshot_sidelength}) does not match snapshot_sidelength ({snapshot_sidelength})")

        # wrap around the box
        wrapped_cell_shift = np.mod(cell_shift, n_cells_per_axis)

        return wrapped_cell_shift.astype(np.float64) * cell_sidelength

    def _transform_snapshot_coordinates(self, coords, rot_angles,cell_data,
                    reflections=None, periodic_cell_shift=None,
                    #cell_data=None, snapshot_number=None,
                    order="xyz", degrees=False,
                    out=None, inplace=False):
        """
        Generate a new set of coordinataes making a new orientation
            of the snapshot through rotations, reflections and wrapping coordinates
            about the periodic boundaries.

        Params:
            coords : ndarray, shape (N, 3)
            rot_angles : array-like, shape (3,)
                (angle_x, angle_y, angle_z), radians unless degrees=True.
            reflections : array-like, shape (3,), optional
                Per-axis reflection signs (+1/-1). Pass None for no reflection.
            periodic_cell_shift : array-like of int, shape (3,)
                Number of cells to shift along x, y, z axes
            cell_data : dictionary of snapshot cell metadata.
            snapshot_number : int, snapshot number
            order : str
                Rotation composition order, a permutation of "x", "y", "z".
            degrees : bool
            out : ndarray, shape (N, 3), optional
                Preallocated output buffer (avoids reallocation on repeated calls).
            inplace : bool
                If True, overwrite `coords` in place instead of allocating new memory.

        Returns:
            Coordinates: (N,3) for the new snapshot orientation.

        """

        nr_cell_axis = cell_data["nr_cells_axis"]
        cell_sidelength=cell_data["cell_size"]
        snapshot_sidelength=cell_data["snap_boxsize"]

        shift_vector = self._cell_shift_to_vector(periodic_cell_shift, nr_cell_axis,  cell_sidelength, snapshot_sidelength)

        return box_structure.rotate_coords_cartesian(
                coords, rot_angles,
                reflections=reflections, periodic_shift=shift_vector, sidelengths=snapshot_sidelength,
                order=order, degrees=degrees,
                out=out, inplace=inplace
            )

    def _validate_ang_radius_deg(self, ang_radius_deg):
        """
        Ang_radius_deg does not needs validating unless an upper bound has been placed on it. 
        SnapshotBeam overrides this, SnapshotAllSky doesn't override it.
        """
        pass

    def _gather_files_for_ptype(self, current_ptype, shell_z, ang_radius_deg, use_snapshots=None):
        """
        Identify every (snapshot, tile, file) needed to populate a lightcone
        shell for a single particle type, without reading any particle data.

        Populates self.snap_file_offset_lengths,
        self.npart_per_file, self.last_snapshot_read, self._file_order_index
        and self.npart_kept_per_files. 

        :return: self.snap_file_offset_lengths, a list of FileReadSpec entries.
        """
        # new empty state
        self.__clear_shell_read_state()

        if use_snapshots is None:
            self.snapshots_to_read = nz.snapshot_number_in_range(
                    boxsize_resolution=self.box_res,
                    redshift_range=shell_z,
                    redshift_buffer=(0.025,0.025),
                    snapshot_buffer=(0,0),
                    bounds="strict", decimals=5
                )[::-1]
        else:
            self.snapshots_to_read = np.asarray(use_snapshots)

        self.__remove_snapshots_outside_shell(shell_z) # sanity check to catch edge cases and rounding errors that slip through

        # iterate through snapshot cell data to identify cells, filenames, offsets, lengths and particles per file to read.
        self.snap_file_offset_lengths=[] #snapshot number, file number, offsets (of cells in file), lengths (of cells in file)
        self.npart_per_file=[]

        for snap_nr in self.snapshots_to_read:
            # clear active snapshot state
            self.__clear_snapshot_read_state()

            # record each of this snapshot's tile pieces into self.last_snapshot_read & snap_file_offset_lengths
            # (a single snapshot can need more than one piece)
            self.__gather_cells_and_tiles(snapshot_number=snap_nr, shell_z=shell_z, current_ptype=current_ptype, ang_radius_deg=ang_radius_deg)

        self._file_order_index = {
            (snap_nr, tile, file_nr): idx
            for idx, (snap_nr, tile, file_nr) in enumerate(sorted((fd.snap_nr, fd.tile, fd.file_nr) for fd in self.snap_file_offset_lengths))
        }
        self.npart_kept_per_file = np.zeros(len(self.snap_file_offset_lengths), dtype=int)

        # temporary comm barrier for testing
        if self.comm is not None:
            self.comm.barrier()

        # check for no particles existing in identified cells
        if np.sum(self.npart_per_file)==0:
            message(self.comm_rank,f"\nNo particles found in selected cells\n")
            return self.snap_file_offset_lengths

        # wipe data as needed
        self.__clear_snapshot_read_state()

        # print of all files to read from:
        for i in range(len(self.snap_file_offset_lengths)):
            file_data=self.snap_file_offset_lengths[i]
            j = self.__last_snapshot_index(file_data.snap_nr, file_data.tile)
            update_cell_str=f"\tsnap: {file_data.snap_nr:<2},\ttile: {str(file_data.tile):<10},\tfile: {file_data.file_nr:<2},\tnumb_part: {self.npart_per_file[j][file_data.file_nr]} [{np.sum(file_data.lengths)}]"
            message(self.comm_rank,update_cell_str)

        return self.snap_file_offset_lengths

    def __populate_lightcone_with_ptype(self, particle_type, property_names, shell_z, ang_radius_deg, redistribute=False, use_snapshots=None):

        self._gather_files_for_ptype(particle_type, shell_z, ang_radius_deg, use_snapshots)

        # check for no particles existing in identified cells
        if np.sum(self.npart_per_file)==0:
            return {}

        # select method for reading files
        method = "parallel" if self.comm is not None else "serial"

        property_names=list(property_names)
        needed_properties = ["ParticleIDs","Coordinates","SmoothingLengths"] if particle_type!="PartType1" else ["ParticleIDs","Coordinates"]
        for prop_name in needed_properties:
            if prop_name not in property_names:
                property_names.append(prop_name)

        if self.snap_file_offset_lengths is None:
            raise ValueError("Have not collected filenames and cells to read")

        # serial mode of reading in particle data
        if method=="serial":

            self.particle_data = {name : None for name in property_names}
            self.particle_offset={}
            self.previous_offset={}


            # read particles from files and place into ligthcone beam
            self.__read_and_place_particles(self.snap_file_offset_lengths, particle_type, property_names, ang_radius_deg) #, total_ptype_to_read)

            self._attach_expansion_factors()

        elif method=="parallel":
            # use parallel reading of particle data
            if self.comm is None:
                raise ValueError("must supply an MPI communicator (comm) to read in parallel")
            self.__parallel_read_and_place_particles(particle_type, property_names, ang_radius_deg)
            if redistribute:
                # evenly distribute particles across all ranks
                self.__redistribute_particles_evenly(property_names)
        else:
            raise ValueError("method not recognised")

        return self.particle_data

    def __gather_cells_and_tiles(self, snapshot_number, shell_z, current_ptype, ang_radius_deg):

        # sanity check snapshot specific values
        if self.__snap_cell_data is not None:
            raise ValueError(f"snapshot specific values are not being cleared after iterations")
        elif self.__cell_data_snap_nr is not None:
            raise ValueError(f"snapshot specific values are not being cleared after iterations")
        elif self.__snap_redshift_range_to_populate is not None:
            raise ValueError(f"snapshot specific values are not being cleared after iterations")

        snapshot_z_range = nz.snapshot_redshift_range(snapshot_number, self.box_res)
        # show read out of snapshot redshift info
        snapshot_to_shell_str=(
            f"snapshot: {snapshot_number}"+
            f"\n\tcentre redshift {nz.snapshot_number_redshifts( snapshot_number, self.box_res)}"+
            f"\n\tredshift range: {snapshot_z_range[0]} - {snapshot_z_range[1]}"
            )
        message(self.comm_rank,snapshot_to_shell_str)
        snap_shell_z_range=tuple(
                    [snapshot_z_range[0] if snapshot_z_range[0] > shell_z[0] else shell_z[0],
                    snapshot_z_range[1] if snapshot_z_range[1] < shell_z[1] else shell_z[1]]
                )

        self.__snap_redshift_range_to_populate=snap_shell_z_range


        # write new snapshot specific values, reuse of data is stored in cache
        self.__reset_cell_read_state(snapshot_number)
        self.__snap_redshift_range_to_populate=snap_shell_z_range


        # give read out of max beam diameter for snapshots redshift range in shell
        #max_diameter = self._diameter_at_redshift(snap_shell_z_range[1], ang_radius_deg)

        r_beam = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")
        add_part_from_snap_str=(
            f"add particles from snapshot {snapshot_number} to lightcone shell"+
            f"\n\tredshift:\t{snap_shell_z_range[0]:.5f} - {snap_shell_z_range[1]:.5f}"+
            f"\n\tradii:\t{r_beam[0]:.5f} - {r_beam[1]:.5f} [Mpc]" #+f"\n\tmax diameter of shell:\t{max_diameter} [Mpc]"
            )

        message(self.comm_rank,add_part_from_snap_str)

        del r_beam #, max_diameter

        # set up dictionary of ptypes in snapshot files:
        ptypes_in_current_snap = self.__files_with_ptype(snapshot_number, self.__snap_cell_data["nr_files"])

        #iterate through tile pieces -- _snapshot_tile_idx adds
        # transverse tiles (alongside the usual line-of-sight ones) once
        # the beam's diameter exceeds one box sidelength, so a wide beam
        # no longer requires max_diameter < box sidelength
        for tile, z_sub_min, z_sub_max in self._snapshot_tile_idx(snap_shell_z_range, ang_radius_deg):
            # read in instructions for how to reconstruct the snapshot box
            snapshot_rotation_angles, snapshot_rotation_reflections, snapshot_periodic_shifts = self._snapshot_reorientation(tile)
            orientation_str=(
                f"Reorienting snapshot {snapshot_number} (tile {tile}, z={z_sub_min:.5f}-{z_sub_max:.5f}):"
                f"\n\tRotation:\t{snapshot_rotation_angles} [deg]"+
                f"\n\tReflection:\t{snapshot_rotation_reflections}"+
                f"\n\tPeriodic Shift:\t{snapshot_periodic_shifts}"
                )
            message(self.comm_rank,orientation_str)
            repositioned_cell_centres=np.empty(np.shape(self.__snap_cell_data["cell_centres"]))
            __ = self._transform_snapshot_coordinates(
                                            coords=self.__snap_cell_data["cell_centres"],
                                            rot_angles=snapshot_rotation_angles,
                                            reflections=snapshot_rotation_reflections,
                                            periodic_cell_shift=snapshot_periodic_shifts,
                                            cell_data=self.__snap_cell_data,
                                            order="xyz",
                                            degrees=True,
                                            out=repositioned_cell_centres,
                                            inplace=False
                                        )
            repositioned_cell_centres += self.get_snapshot_reposition_coords(tile, self.__snap_cell_data["snap_boxsize"])


            beam_idx, __ = self._in_shell(
                    coords=repositioned_cell_centres,
                    ang_radius_deg=ang_radius_deg,
                    z_min=snap_shell_z_range[0],
                    z_max=snap_shell_z_range[1],
                    buffer_length=self.__snap_cell_data["cell_size"][0],
                    buffer_shape="cube",
                    return_bool=False
                )

            message(self.comm_rank,f"\nnumber of cells in beam: {np.shape(beam_idx)[0]}")
            if len(beam_idx) > 0:
                message(self.comm_rank,"coordinates range of cell centres in beam")
                #if self.comm_rank is None or self.comm_rank==0:
                print_coord_ranges(repositioned_cell_centres[beam_idx,:], indent=2, rank=self.comm_rank)

            # unique files to read
            file_numbers = np.unique(self.__snap_cell_data["cells_in_file"][current_ptype][beam_idx])
            npart_per_file=np.zeros(self.__snap_cell_data["nr_files"], dtype=int)

            for ii, file_nr in enumerate(file_numbers):

                # check for no particles in file
                if file_nr not in np.asarray(ptypes_in_current_snap[current_ptype], dtype=int):
                    continue

                infile_cell_idx=beam_idx[self.__snap_cell_data["cells_in_file"][current_ptype][beam_idx]==file_nr]
                offsets=self.__snap_cell_data["offsets"][current_ptype][infile_cell_idx]
                lengths=self.__snap_cell_data["lengths"][current_ptype][infile_cell_idx]

                # number of particles to read in = sum of lengths
                npart_to_read = np.sum(self.__snap_cell_data["lengths"][current_ptype][infile_cell_idx])

                # check for no particles in cells
                if npart_to_read==0:
                    continue

                npart_per_file[file_nr]=npart_to_read

                self.snap_file_offset_lengths.append(
                    FileReadSpec(snapshot_number, tile, file_nr, offsets, lengths)
                )

            self.npart_per_file.append(npart_per_file)
            # store the snapshot's full shell range here too, so the
            # particle-level filter in __read_and_place_particles uses the
            # same correct (wider) radial bound as the cell-level one above
            self.__last_snapshot_read(snapshot_number, tile, snap_shell_z_range)

    def __read_and_place_particles(self, files_to_read, current_ptype, property_names, ang_radius_deg):
        """
        Read the given FileReadSpec entries with and re-orient each file's coordinates into the beam.
        Use for both the serial and parallel read methods.
        The total number of particles to read with the serial method the is global sum total and
        for the parallel method the total is the number of particles in the files on the rank.
        """

        snap_z_range = {(record.snap_nr, record.tile): record.z_updated for record in self.last_snapshot_read}
        kept_parts = {prop: [] for prop in property_names}
        kept_parts["SnapshotNumber"]=[] #store snapshot of each particle. 
        npart_read_total = 0
        npart_kept_total = 0

        self._local_kept_per_file_updates = []

        for file_data in files_to_read:

            snap_nr=file_data.snap_nr
            tile=file_data.tile
            file_nr=file_data.file_nr
            filename = self.snapshot_format.format(snap_nr=snap_nr, file_nr=file_nr)
            infile_offset, infile_lengths = merge_cells(file_data.offsets, file_data.lengths)
            jj=self.__last_snapshot_index(snap_nr, tile)
            npart_infile = self.npart_per_file[jj][file_nr]

            self.__reset_cell_read_state(snap_nr)

            # update the output particle data
            file_particle_data = self.__read_particles_from_cells(current_ptype, filename, infile_offset, infile_lengths, property_names, npart_infile)

            snapshot_rotation_angles, snapshot_rotation_reflections, snapshot_periodic_shifts = self._snapshot_reorientation(tile)

            new_coords=np.empty((npart_infile, 3))
            __ = self._transform_snapshot_coordinates(
                                    #self.particle_data["Coordinates"][last_offset:current_offset].to_value("Mpc"),
                                    file_particle_data["Coordinates"].to_value("Mpc"),
                                    snapshot_rotation_angles,
                                    reflections=snapshot_rotation_reflections,
                                    periodic_cell_shift=snapshot_periodic_shifts,
                                    cell_data=self.__snap_cell_data,
                                    order="xyz",
                                    degrees=True,
                                    out=new_coords,
                                    inplace=False
                                    )

            new_coords+=self.get_snapshot_reposition_coords(tile, self.__snap_cell_data["snap_boxsize"])

            # replace coordinates
            file_particle_data["Coordinates"] = (new_coords*unyt.Mpc).to(file_particle_data["Coordinates"].units)

            del new_coords

            message(self.comm_rank, "\ncoordinates range of particles read")
            print_coord_ranges(file_particle_data["Coordinates"], indent=2, rank=self.comm_rank)

            zmin, zmax = snap_z_range[(snap_nr, tile)]
            local_beam_idx, __ = self.__select_paticles_in_shell(
                file_particle_data["Coordinates"].to_value("Mpc"),
                zmin=zmin,
                zmax=zmax,
                ang_radius_deg=ang_radius_deg,
                #method="approx",
                method="exact",
                r_tol=0*unyt.Mpc,
                )

            npart_read_total += npart_infile
            npart_kept_total += len(local_beam_idx)

            message(self.comm_rank, f"Particles read:\t{npart_infile}\tkept in beam:\t{len(local_beam_idx)}")

            file_idx = self._file_order_index[(snap_nr, tile, file_nr)]
            self.npart_kept_per_file[file_idx] = len(local_beam_idx)
            self._local_kept_per_file_updates.append((file_idx, len(local_beam_idx)))

            for prop in property_names:
                kept_parts[prop].append(file_particle_data[prop][local_beam_idx])
            
            kept_parts["SnapshotNumber"].append(unyt.unyt_array(np.fill_like(local_beam_idx, fill_value=snap_nr, dtype=int), units=unyt.dimensionless))

        if self.comm is not None:
            npart_read_global = self.comm.allreduce(npart_read_total)
            npart_kept_global = self.comm.allreduce(npart_kept_total)
        else:
            npart_read_global = npart_read_total
            npart_kept_global = npart_kept_total

        if npart_read_global > 0:
            message(self.comm_rank,f"\nParticles read:\t{npart_read_global}\tkept in beam:\t{npart_kept_global} [{npart_kept_global/npart_read_global * 100:.3f}%]")
        else:
            message(self.comm_rank,f"\nNo particles found in selected cells")

        for prop in property_names:
            if len(kept_parts[prop]) > 0:
                self.particle_data[prop] = unyt.uconcatenate(kept_parts[prop])
        
        if len(kept_parts["SnapshotNumber"]) > 0:
            self.particle_data["SnapshotNumber"] = unyt.uconcatenate(kept_parts["SnapshotNumber"])

    def __fill_missing_particle_data(self, property_names, comm):
        """
        A rank assigned zero files. Use collective allgather so
        every rank learns them from a rank that did read
        something, and fill in a correct zero-length unyt_array.
        """
        all_unit_metadata = comm.allgather(self._unit_metadata)
        if self.unit_registry is None:
            #for metadata in comm.allgather(self._unit_metadata):
            for metadata in all_unit_metadata:
                if metadata is not None:
                    self._unit_metadata = metadata
                    self.unit_registry = sw_units.unit_registry_from_metadata(metadata)
                    break

        for prop in property_names:
            arr = self.particle_data[prop]
            info = None if arr is None else (arr.dtype, arr.shape[1:], str(arr.units))
            all_info = comm.allgather(info)

            if arr is None:
                for other in all_info:
                    if other is not None:
                        dtype, trailing_shape, units_str = other
                        self.particle_data[prop] = unyt.unyt_array(
                            np.zeros((0,)+tuple(trailing_shape), dtype=dtype), units_str, registry=self.unit_registry)
                        break
                else:
                    raise RuntimeError(f"no rank read any {prop} particles; cannot determine its dtype/units")

    def _attach_expansion_factors(self):
        """
        Filter particles down to the particles inside the
        beam compute scale factors (ExpansionFactors as lightcone property).
        """

        npart_in_beam = self.particle_data["Coordinates"].shape[0]
        message(self.comm_rank,f"Total particles in beam:\t{npart_in_beam}")

        if npart_in_beam > 0:
            message(self.comm_rank, "\ncoordinates range of particles in beam")
            print_coord_ranges(self.particle_data["Coordinates"][:], indent=2, rank=self.comm_rank)

        # compute distance from observer
        r_comoving = unyt.unyt_array(np.zeros(npart_in_beam), units = self.particle_data["Coordinates"].units)
        r_comoving[:]=np.sqrt(
                self.particle_data["Coordinates"][:, 0]**2 +
                self.particle_data["Coordinates"][:, 1]**2 +
                self.particle_data["Coordinates"][:, 2]**2
            )

        # determine scale factor of particles in the beam
        self.particle_data["ExpansionFactors"]=unyt.unyt_array(np.zeros(npart_in_beam), units = unyt.dimensionless)
        self.particle_data["ExpansionFactors"][:]=self.__comoving_distance_to_scalefactor(r_comoving)*unyt.dimensionless

    def __read_particles_from_cells(self, ptype, filename, infile_offset, infile_lengths, property_names, numb_part_infile):

        file_particle_data = {}

        with h5py.File(filename, "r") as infile:
            # set max number of particles that can be read from file
            numb_all_infile = np.sum(infile["Header"].attrs["NumPart_ThisFile"][:])
            if self.unit_registry is None:
                    self._unit_metadata = sw_units.snapshot_unit_metadata(infile)
                    self.unit_registry = sw_units.unit_registry_from_metadata(self._unit_metadata) #sw_units.unit_registry_from_snapshot(infile)
            for prop in property_names:
                # Find the dataset for this property
                if prop in infile[ptype]:
                    infile_dset = infile[ptype][prop] # read hdf5 type object
                else:
                    raise ValueError(f"{prop} not found")

                # Create output array, if we didn't already
                if self.particle_data[prop] is None:
                    #infile_shape = list(infile_dset.shape)
                    infile_shape = list(infile_dset.shape)
                    #infile_shape[0] = np.sum(numb_part_total)
                    infile_shape[0] = numb_part_infile
                    # set property dtype
                    if prop=="ParticleIDs":
                        prop_dtype=np.int64
                    else:
                        prop_dtype=infile_dset.dtype
                    #set property units
                    if unyt is not None:
                        units = sw_units.units_from_attributes(dict(infile_dset.attrs), self.unit_registry)

                        # set values = -1 unless assigned
                        #self.particle_data[prop] = unyt.unyt_array(-1*np.ones(infile_shape, dtype=prop_dtype), units) # data is array for all particles in the current file z1<z<z2
                        arr = unyt.unyt_array(-1*np.ones(infile_shape, dtype=prop_dtype), units) # data is array for all particles in the current file z1<z<z2
                        if "a*" in str(units) or "*a" in str(units):
                            arr = sw_units.drop_a_from_comoving_property(arr)
                    else:
                        #self.particle_data[prop] = -1*np.ones(infile_shape, dtype=prop_dtype)
                        arr = -1*np.ones(infile_shape, dtype=prop_dtype)

                # to track that total added to file matches input total to read
                num_added=0
                offset = 0

                for (clen, coff) in zip(infile_lengths, infile_offset):
                    i1 = max((coff, 0)) # the starting index for infile data, either at start of file (=0, especially if global offset value is -ve relative to file), or offset further within the file if there are multiple sets fo adjacent cells
                    i2 = min((coff+clen), int(numb_all_infile)) # final index for infile data, either for all particles within the file if all cells need to be read, or to end of where adjacent cells are within file
                    num = i2 - i1 # number of particles to be read from the file

                    if num > 0:
                        arr[offset:offset+num] = infile_dset[i1:i2,...]
                        offset += num
                        num_added+=num

                assert num_added == numb_part_infile
                file_particle_data[prop] = arr
            del num_added

        return file_particle_data

    def __parallel_read_and_place_particles(self, current_ptype, property_names, ang_radius_deg):
        """
        Parallel counterpart to the serial method.
        MPI ranks are assigned distinct entries with one (snapshot, file)
        per rank where possible. Every rank then reads only its own files
        and places them in the beam before filtering to the beam locally.

        Note: each rank contains its own local slice of the shell's particles.
        """

        #comm_rank = comm.Get_rank()
        #comm_size = comm.Get_size()

        # distribute the files to read across ranks, one file per rank where
        # there are at least as many files as ranks
        nr_files = len(self.snap_file_offset_lengths)
        files_on_rank = phdf5.assign_files(nr_files, self.comm_size)
        first_on_rank = np.cumsum(files_on_rank) - files_on_rank
        first = first_on_rank[self.comm_rank]
        num = files_on_rank[self.comm_rank]
        my_files = self.snap_file_offset_lengths[first:first+num]

        self.particle_data = {name : None for name in property_names}
        #self.particle_offset={}
        #self.previous_offset={}


        # local total for just the files this rank owns
        total_ptype_to_read = 0
        for file_data in my_files:
            jj = self.__last_snapshot_index(file_data.snap_nr, file_data.tile)
            total_ptype_to_read += self.npart_per_file[jj][file_data.file_nr]

        rank_message(self.comm_rank, f"\treading {len(my_files)}/{nr_files} files, {total_ptype_to_read} particles")
        #print(f"[rank {self.comm_rank}/{self.comm_size}] reading {len(my_files)}/{nr_files} files, {total_ptype_to_read} particles")


        self.__read_and_place_particles(my_files, current_ptype, property_names, ang_radius_deg)


        all_kept_per_file_updates = self.comm.allgather(self._local_kept_per_file_updates)
        for updates in all_kept_per_file_updates:
            for file_idx, count in updates:
                self.npart_kept_per_file[file_idx] = count

        # a rank assigned zero files never allocates self.particle_data --
        # every rank must return arrays of consistent dtype/units so that
        # downstream MPI reductions see the same set of properties on every rank
        self.__fill_missing_particle_data(property_names, self.comm)

        #self.__filter_and_finalize_beam_particles(shell_z, ang_radius_deg)
        self._attach_expansion_factors()

    def __select_paticles_in_shell(self, coords, zmin, zmax, ang_radius_deg, method="exact", r_tol=2*unyt.Mpc):

        if method=="approx":
            buffer_length=apply_expected_units(r_tol, unyt.Mpc).to_value("Mpc")
            buffer_shape="sphere"

        elif method=="exact":
            buffer_length=apply_expected_units(0., unyt.Mpc).to_value("Mpc")
            buffer_shape="sphere"

        elif method=="all":
            assert np.ndim(coords)==2
            assert np.shape(coords)[-1]==3
            return (np.arange(np.shape(coords)[0]), None)


        return self._in_shell(coords, ang_radius_deg=ang_radius_deg, z_min=zmin, z_max=zmax, buffer_length=buffer_length, buffer_shape=buffer_shape, return_bool=False)

    def __redistribute_particles_evenly(self, property_names):
        """
        Redistribute particles evenly across all MPI ranks.
        After this call every rank will hold roughly equal number of particles.
        """
        all_props = list(property_names)
        if "ExpansionFactors" not in all_props:
            all_props.append("ExpansionFactors")

        n_local = self.particle_data[all_props[0]].shape[0]
        nperproc = self.comm.allgather(n_local)
        ntot = sum(nperproc)

        ndesired = np.full(self.comm_size, ntot // self.comm_size, dtype=np.int64)
        ndesired[: ntot % self.comm_size] += 1

        for prop in all_props:
            arr = self.particle_data[prop]
            units = arr.units
            new_values = psort.repartition(arr.to_value(units), ndesired, comm=self.comm)
            self.particle_data[prop] = unyt.unyt_array(new_values, units, registry=self.unit_registry)

        message(self.comm_rank, f"Redistributed {ntot} particles evenly across {self.comm_size} ranks")

    def gather_files(self, current_ptype, shell_z, ang_radius_deg, use_snapshots=None):
        """
        Serial-only counterpart to the file-identification step inside
        place_snapshot_particles_in_shell. For a single particle type,
        determines every (snapshot, tile, file) needed to populate the
        given redshift shell, without reading any particle data.

        Follow up by calling place_file_in_shell once per index into the
        returned list to read and place each file individually, instead of
        the whole shell at once (as load_ptype_in_beam(method="serial")
        does via place_snapshot_particles_in_shell).

        :param current_ptype: particle type to gather, e.g. "PartType1"
        :param shell_z: (z_min, z_max) redshift range of the lightcone shell
        :param ang_radius_deg: angular radius of the beam, degrees
        :param use_snapshots: optional explicit list/array of snapshot
            numbers to use instead of the ones auto-selected from shell_z
        :return: (numb_files, self.snap_file_offset_lengths) -- the number
            of files found and the list of FileReadSpec entries itself;
            pass an index in range(numb_files) to place_file_in_shell.
        """
        if self.comm is not None:
            raise ValueError("gather_files only supports serial (non-MPI) use")

        self._validate_ang_radius_deg(ang_radius_deg)
        #if ang_radius_deg > MAX_BEAM_ANG_RADIUS_DEG:
        #    radius_error_str=f"angular radius ({ang_radius_deg} [deg]) exceeds the maximum radius of {MAX_BEAM_ANG_RADIUS_DEG} [deg]"
        #    raise ValueError(radius_error_str)

        files = self._gather_files_for_ptype(current_ptype, shell_z, ang_radius_deg, use_snapshots)
        return len(files), files

    def place_file_in_shell(self, file_number, current_ptype, property_names, ang_radius_deg):
        """
        Serial-only: read and place a single file -- selected by index into
        self.snap_file_offset_lengths, as returned by gather_files -- into
        the beam.

        Returns the same per-property particle-data dict as
        place_snapshot_particles_in_shell, but containing only the
        particles kept from this one file (an empty dict if none of this
        file's particles fell inside the beam).

        :param file_number: index into self.snap_file_offset_lengths
        :param current_ptype: particle type being placed, e.g. "PartType1"
            (must match the ptype passed to gather_files)
        :param property_names: particle properties to read, e.g.
            ["ParticleIDs", "Coordinates"]
        :param ang_radius_deg: angular radius of the beam, degrees (must
            match the value passed to gather_files)
        """
        if self.comm is not None:
            raise ValueError("place_file_in_shell only supports serial (non-MPI) use")
        if self.snap_file_offset_lengths is None:
            raise ValueError("must call gather_files before place_file_in_shell")

        property_names = list(property_names)
        needed_properties = ["ParticleIDs","Coordinates","SmoothingLengths"] if current_ptype!="PartType1" else ["ParticleIDs","Coordinates"]
        for prop_name in needed_properties:
            if prop_name not in property_names:
                property_names.append(prop_name)
        file_data = self.snap_file_offset_lengths[file_number]

        self.particle_data = {name: None for name in property_names}
        self.__read_and_place_particles([file_data], current_ptype, property_names, ang_radius_deg)

        if self.particle_data["Coordinates"] is None:
            message(self.comm_rank, f"\nNo particles kept from file {file_number}\n")
            return {}

        self._attach_expansion_factors()

        return self.particle_data

    def place_snapshot_particles_in_shell(self, lightcone_redshift_range, ang_radius_deg, property_names, particle_types=["PartType0", "PartType1", "PartType5"], use_snapshots=None, comm=None, redistibute_particles=False):
        """
        Read and place every particle of the given type(s) into a lightcone shell, if they fall within the lightcones footprint. 
        """
        self._validate_ang_radius_deg(ang_radius_deg)

        self._define_mpi_mode(comm, redistibute_particles)
        message(self.comm_rank, f"\nUsing MPI parallel reading method\n")

        combined_dset ={}
        if np.ndim(particle_types)>0:
            # iterate through particle types
            for ptype in particle_types:
                message(self.comm_rank, seperator_str()+f"\nPlaceing {ptype} in lightcone")
                combined_dset[ptype] = self.__populate_lightcone_with_ptype(ang_radius_deg=ang_radius_deg, shell_z=lightcone_redshift_range, property_names=property_names, particle_type=ptype,redistribute=redistibute_particles, use_snapshots=use_snapshots)

        elif np.ndim(particle_types)==0:
            # return for singular particle type
            ptype = particle_types
            message(self.comm_rank, seperator_str()+f"\nPlaceing {ptype} in lightcone")
            combined_dset[ptype]=self.__populate_lightcone_with_ptype(ang_radius_deg=ang_radius_deg, shell_z=lightcone_redshift_range, property_names=property_names, particle_type=ptype,redistribute=redistibute_particles, use_snapshots=use_snapshots)

        return combined_dset

    def inverse_transform_snapshot_coordinates(self, coords, rot_angles, cell_data,
                    reflections=None, periodic_cell_shift=None,
                    order="xyz", degrees=False,
                    out=None, inplace=False):
        """
        Given coordinates produced by transform_snapshot_coordinates,
        recover the original input coordinates.

        Params: identical to transform_snapshot_coordinates.

        Returns:
            Coordinates: (N,3) snapshot-frame coordinates that, if passed
            through transform_snapshot_coordinates with the same other
            arguments, would reproduce the input `coords`.
        """

        nr_cell_axis = cell_data["nr_cells_axis"]
        cell_sidelength=cell_data["cell_size"]
        snapshot_sidelength=cell_data["snap_boxsize"]

        shift_vector = self.__cell_shift_to_vector(periodic_cell_shift, nr_cell_axis,  cell_sidelength, snapshot_sidelength)

        return box_structure.rotate_coords_cartesian(
                coords, rot_angles,
                reflections=reflections, periodic_shift=shift_vector, sidelengths=snapshot_sidelength,
                order=order, degrees=degrees,
                out=out, inplace=inplace, invert=True
            )

    def __cell_data_for_snapshot(self, snap_nr):
        """
        Shared cache lookup used by Snapshot2Lightcone and Lightcone2Snapshot.
        Same cache as __reset_cell_read_state, therefore repeated calls reuse it.
        """
        if snap_nr not in self._cell_data_cache:
            self._cell_data_cache[snap_nr] = self.get_cell_data(snap_nr)
        return self._cell_data_cache[snap_nr]

    def Snapshot2Lightcone(self, snapshot_number, coords, tile=None):
        """
        Translate snapshot coordinates into lightcone coordinates, for 
        every tile this snapshot currently supplies to the lightcone.

        A single snapshot number can supply more than one periodic-replica
        tile to the same shell, with its own independent
        rotation/reflection/periodic-shift/reposition.

        This function only knows about tiles from the most recently gathered shell and  
        relies on self.last_snapshot_read being populated by gather_files,
        place_snapshot_particles_in_shell or the internal
        _gather_files_for_ptype they both call.

        Params
            snapshot_number : int
            coords : ndarray, shape (N, 3)
                Comoving snapshot-frame (box) coordinates, Mpc.
            tile : tuple, optional
                A specific periodic-replica tile -- (nx, ny, n_los) for
                SnapshotBeam, (nx, ny, nz) for SnapshotAllSky. If given,
                self.last_snapshot_read is not consulted at all, so this
                works standalone (no gather_files/place_snapshot_particles_in_shell
                call needed first). If None (default), every tile this
                snapshot supplies to the last-gathered shell is used.
        Returns
            dict {tile: lightcone_coords}, one entry per tile this
            snapshot supplies to the last-gathered shell. `tile` is
            whatever _tiles_intersecting_shell returns for this class
            -- (nx, ny, n_los) for SnapshotBeam, (nx, ny, nz) for
            SnapshotAllSky. lightcone_coords has shape (N, 3).
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be a vector with shape (N, 3)")

        if tile is not None:
            tiles = [tile]
        else:
            if not self.last_snapshot_read:
                raise ValueError(
                    "no shell-read state to look up tiles from -- call "
                    "gather_files or place_snapshot_particles_in_shell "
                    "first, or pass tile explicitly to skip this lookup"
                )

            tiles = [record.tile for record in self.last_snapshot_read
                     if record.snap_nr == snapshot_number]
            if not tiles:
                raise ValueError(
                    f"snapshot {snapshot_number} not found in the last "
                    "shell-read state -- pass tile explicitly to skip "
                    "this lookup"
                )

        cell_data = self.__cell_data_for_snapshot(snapshot_number)
        boxsize = cell_data["snap_boxsize"]

        lightcone_coords_by_tile = {}
        for tile in tiles:
            rot_angles, reflections, periodic_shift = self._snapshot_reorientaton(tile)
            oriented = self.transform_snapshot_coordinates(
                coords, rot_angles, cell_data,
                reflections=reflections, periodic_cell_shift=periodic_shift,
                order="xyz", degrees=True, inplace=False,
            )
            lightcone_coords_by_tile[tile] = oriented + self.get_snapshot_reposition_coords(tile, boxsize)

        return lightcone_coords_by_tile

    def Lightcone2Snapshot(self, coords, snapshot_number=None, tile=None):
        """
        Inverse of Snapshot2Lightcone, translate lightcone
        coordinates back into the snapshot numbers and snapshot coordinates.

        Which snapshot a lightcone-frame point belongs to is determined
        from its comoving distance alone. 
        Within a matched snapshot, exactly as in Snapshot2Lightcone, the
        snapshot alone doesn't say which of its tiles a given point's
        orientation should be inverted to. Therefore, this
        inverts every one of that snapshot's tiles and returns all of
        them. Only one is the 'real' originating tile for any
        given point 

        Relies on self.last_snapshot_read (like Snapshot2Lightcone)
        and therefore only knows about snapshots/tiles from the most recently
        gathered shell.

        Params
            coords : ndarray, shape (N, 3)
                Comoving lightcone-frame coordinates, Mpc.
            snapshot_number : int, optional
                Skip distance-based snapshot matching and assume every
                point belongs to this snapshot.
            tile : tuple, optional
                Skip tile lookup and invert only through this one tile.
                Requires snapshot_number to also be given.
        Returns
            dict {snapshot_number: {tile: (point_index, snapshot_coords)}}
            -- point_index is an integer array indexing into the input
            `coords` (which of the N input points matched this
            snapshot_number), and snapshot_coords (shape
            (len(point_index), 3)) is those points' snapshot-frame
            coordinates after inverting through `tile`'s orientation.
            Points whose comoving distance doesn't fall inside any
            recorded snapshot's range are dropped (with a message()
            warning naming how many).
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be a vector with shape (N, 3)")

        if tile is not None and snapshot_number is None:
            raise ValueError("tile requires snapshot_number to also be given")

        # point_idx_by_snap[snap_nr]: which input points belong to snap_nr.
        # tiles_by_snap[snap_nr]: which tiles to invert them through.
        if snapshot_number is not None:
            # every point assigned directly, no distance-based filtering
            point_idx_by_snap = {snapshot_number: np.arange(coords.shape[0])}
            if tile is not None:
                tiles_by_snap = {snapshot_number: [tile]}
            else:
                if not self.last_snapshot_read:
                    raise ValueError(
                        "no shell-read state to look up tiles from -- call "
                        "gather_files or place_snapshot_particles_in_shell "
                        "first, or pass tile explicitly to skip this lookup"
                    )
                found_tiles = [record.tile for record in self.last_snapshot_read
                               if record.snap_nr == snapshot_number]
                if not found_tiles:
                    raise ValueError(
                        f"snapshot {snapshot_number} not found in the last "
                        "shell-read state -- pass tile explicitly to skip "
                        "this lookup"
                    )
                tiles_by_snap = {snapshot_number: found_tiles}
        else:
            if not self.last_snapshot_read:
                raise ValueError(
                    "no shell-read state to look up snapshots from -- call "
                    "gather_files or place_snapshot_particles_in_shell "
                    "first, or pass snapshot_number explicitly to skip "
                    "this lookup"
                )
            # every tile recorded for each snapshot, and the (z_min, z_max)
            # range that snapshot (all its tiles alike) was assigned
            tiles_by_snap = {}
            snap_z_range = {}
            for record in self.last_snapshot_read:
                tiles_by_snap.setdefault(record.snap_nr, []).append(record.tile)
                snap_z_range.setdefault(record.snap_nr, record.z_updated)

            r = np.linalg.norm(coords, axis=1)
            point_idx_by_snap = {}
            for snap_nr, z_range in snap_z_range.items():
                r_min, r_max = unyt.unyt_array.from_astropy(
                    self.cosmo.comoving_distance(z_range)).to_value("Mpc")
                point_idx = np.flatnonzero((r >= r_min) & (r <= r_max))
                if point_idx.size > 0:
                    point_idx_by_snap[snap_nr] = point_idx

            n_unmatched = coords.shape[0] - sum(idx.size for idx in point_idx_by_snap.values())
            if n_unmatched > 0:
                message(self.comm_rank,
                        f"\nLightcone2Snapshot: {n_unmatched} point(s) did not fall "
                        "within any snapshot's comoving-distance range in the last "
                        "shell-read state\n")

        results = {}
        for snap_nr, point_idx in point_idx_by_snap.items():

            cell_data = self.__cell_data_for_snapshot(snap_nr)
            boxsize = cell_data["snap_boxsize"]

            tile_results = {}
            for t in tiles_by_snap[snap_nr]:
                rot_angles, reflections, periodic_shift = self._snapshot_reorientaton(t)
                repositioned = coords[point_idx] - self.get_snapshot_reposition_coords(t, boxsize)
                snapshot_coords = self.inverse_transform_snapshot_coordinates(
                    repositioned, rot_angles, cell_data,
                    reflections=reflections, periodic_cell_shift=periodic_shift,
                    order="xyz", degrees=True, inplace=False,
                )
                tile_results[t] = (point_idx, snapshot_coords)
            results[snap_nr] = tile_results


        return results

    def place_halos_in_shell(self, halo_format, lightcone_redshift_range, ang_radius_deg, use_snapshots=None):
        """
        Halo counterpart of place_snapshot_particles_in_shell. 
        Place haloes from SOAP catalogues into a lightcone 
        shell using the same
        methods as used for particles
        (_snapshot_tile_pieces, _particles_in_beam) and Snapshot2Lightcone.

        Reads halo positions via lightcone_io.halo_catalogue.SOAPCatalogue, 
        the same as used to build the real halo lightcones. 
        No per-cell/file/offset indexing to exploit so we read a snapshot's entire
        halo catalogue in one collective call.

        Track the minimum number of properties (snapshot number, catalogue index)
        required to look up any other properties in SOAP catalgues later:

          - Lightcone/HaloCentre: the halo's centre, transformed into the
            lightcone frame exactly like a particle's Coordinates.
          - Lightcone/SnapshotNumber: which snapshot the halo came from.
          - InputHalos/HaloCatalogueIndex: the halo's index in that
            snapshot's own SOAP catalogue (a native SOAP field -- distinct
            from SOAPCatalogue's own InputHalos/SOAPIndex, which this
            method never reads).
          - Lightcone/ExpansionFactor: the same per-point interpolated
            expansion factor _attach_expansion_factors computes for
            particles (via __lightcone_scalefactors), evaluated at each
            halo's own new lightcone comoving distance -- not the
            snapshot's single scale factor, exactly as a particle's own
            ExpansionFactors differs slightly from its parent snapshot's.

        :param halo_format: format string for SOAP catalogue filenames
            (using {snap_nr}), passed straight to
            lightcone_io.halo_catalogue.SOAPCatalogue.
        :param lightcone_redshift_range: (z_min, z_max) shell to populate.
        :param ang_radius_deg: passed through to _validate_ang_radius_deg,
            _snapshot_tile_pieces and _particles_in_beam exactly like
            place_snapshot_particles_in_shell (None for an all-sky
            observer -- see SnapshotAllSky's thin wrapper).
        :param use_snapshots: optional explicit snapshot number array/list,
            as in gather_files -- bypasses the usual redshift-range lookup.

        :return: dict of unyt arrays, one row per halo kept:
            Lightcone/HaloCentre (N, 3) Mpc, 
            Lightcone/SnapshotNumber (N,),
            InputHalos/HaloCatalogueIndex (N,), 
            Lightcone/ExpansionFactor (N,)

        """
        self._validate_ang_radius_deg(ang_radius_deg)

        if use_snapshots is None:
            snapshots_to_read = nz.snapshot_number_in_range(
                    boxsize_resolution=self.box_res,
                    redshift_range=lightcone_redshift_range,
                    redshift_buffer=(0.025, 0.025),
                    snapshot_buffer=(0, 0),
                    bounds="strict", decimals=5
                )[::-1]
        else:
            snapshots_to_read = np.asarray(use_snapshots)

        empty_result = {
            "Lightcone/HaloCentre": unyt.unyt_array(np.zeros((0, 3)), unyt.Mpc),
            "Lightcone/SnapshotNumber": unyt.unyt_array(np.zeros((0,), dtype=int), unyt.dimensionless),
            "InputHalos/HaloCatalogueIndex": unyt.unyt_array(np.zeros((0,), dtype=int), unyt.dimensionless),
            "Lightcone/ExpansionFactor": unyt.unyt_array(np.zeros((0,)), unyt.dimensionless),
        }

        if len(snapshots_to_read) == 0:
            message(self.comm_rank, "\nNo snapshots found in range for halo placement\n")
            return empty_result

        to_read = ["InputHalos/HaloCentre", "InputHalos/HaloCatalogueIndex"]
        halo_cat = hc.SOAPCatalogue(halo_format, int(np.min(snapshots_to_read)), int(np.max(snapshots_to_read)))

        kept_centre = []
        kept_snap_nr = []
        kept_index = []

        for snap_nr in snapshots_to_read:
            snap_nr = int(snap_nr)

            snapshot_z_range = nz.snapshot_redshift_range(snap_nr,self.box_res)
            snap_shell_z_range = (
                max(snapshot_z_range[0], lightcone_redshift_range[0]),
                min(snapshot_z_range[1], lightcone_redshift_range[1]),
            )
            if snap_shell_z_range[0] >= snap_shell_z_range[1]:
                # this snapshot's own redshift range doesn't actually
                # overlap the requested shell -- mirrors
                # __test_snapshot_against_shell_redshifts's filtering
                continue

            message(self.comm_rank, f"\nReading SOAP halos for snapshot {snap_nr}\n")
            halo_data = halo_cat.read(snap_nr, to_read)
            halo_index = halo_data["InputHalos/HaloCatalogueIndex"].value.astype(int)
            if halo_index.shape[0] == 0:
                continue
            halo_centre_box = sw_units.drop_a_from_comoving_property(halo_data["InputHalos/HaloCentre"]).to_value("Mpc")

            # populate self.__snap_cell_data for this snapshot -- needed
            # by _snapshot_tile_pieces below (Snapshot2Lightcone gets its
            # own cell data independently, via self._cell_data_cache)
            self.__reset_cell_read_state(snap_nr)

            for tile, z_sub_min, z_sub_max in self._snapshot_tile_pieces(snap_shell_z_range, ang_radius_deg):
                lightcone_coords = self.Snapshot2Lightcone(snap_nr, halo_centre_box, tile=tile)[tile]

                keep_idx, __ = self._particles_in_beam(
                        lightcone_coords, zmin=snap_shell_z_range[0], zmax=snap_shell_z_range[1],
                        ang_radius_deg=ang_radius_deg, method="exact", r_tol=0 * unyt.Mpc,
                    )
                if len(keep_idx) == 0:
                    continue

                kept_centre.append(lightcone_coords[keep_idx])
                kept_snap_nr.append(np.full(len(keep_idx), snap_nr, dtype=int))
                kept_index.append(halo_index[keep_idx])

            message(self.comm_rank, f"\tsnap: {snap_nr},\thalos kept from this snapshot: "
                                     f"{sum(len(a) for a in kept_snap_nr) if kept_snap_nr else 0}")

        if not kept_centre:
            return empty_result

        all_centre = np.concatenate(kept_centre, axis=0)
        all_snap_nr = np.concatenate(kept_snap_nr, axis=0)
        all_index = np.concatenate(kept_index, axis=0)

        r_comoving = np.linalg.norm(all_centre, axis=1)
        expansion_factor = self.__lightcone_scalefactors(r_comoving)

        message(self.comm_rank, f"\nTotal halos placed in lightcone shell: {all_snap_nr.shape[0]}\n")

        return {
            "Lightcone/HaloCentre": unyt.unyt_array(all_centre, unyt.Mpc),
            "Lightcone/SnapshotNumber": unyt.unyt_array(all_snap_nr, unyt.dimensionless),
            "InputHalos/HaloCatalogueIndex": unyt.unyt_array(all_index, unyt.dimensionless),
            "Lightcone/ExpansionFactor": unyt.unyt_array(expansion_factor, unyt.dimensionless),
        }


class SnapshotBeam(SnapshotLightcone):
    """

    """

    def __init__(self, boxsize_resolution, simulation_name, beam_vector, orientation_seed=0):
        """
        Defines the beam
        """
        super().__init__(boxsize_resolution, simulation_name, beam_vector, orientation_seed=orientation_seed)

        # orthonormal basis perpendicular to beam_vec, used to place tiles
        # transverse to the line of sight (see _snapshot_tile_idx) once
        # the beam's diameter exceeds one box sidelength. Computed here,
        # not in SnapshotLightcone.__init__, so a SnapshotAllSky instance
        # -- which has no notion of a single line of sight -- never
        # carries it at all.
        self.beam_transverse_e1, self.beam_transverse_e2 = self.__transverse_basis(self.beam_vec)

    def _validate_ang_radius_deg(self, ang_radius_deg):
        if ang_radius_deg > MAX_BEAM_ANG_RADIUS_DEG:
            radius_error_str=f"angular radius ({ang_radius_deg} [deg]) exceeds the maximum radius of {MAX_BEAM_ANG_RADIUS_DEG} [deg]"
            raise ValueError(radius_error_str)

    @staticmethod
    def __transverse_basis(beam_vector):
        """
        Orthonormal basis (e1, e2) perpendicular to beam_vector, used to
        place tiles transverse to the line of sight.
        """
        reference = np.array([1.0, 0.0, 0.0]) if abs(beam_vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(beam_vector, reference)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(beam_vector, e1)
        return e1, e2

    def _los_tile_idx(self, comoving_distance, boxsize):
        """
            How many times the periodic box has been replicated along a line of sight.
            If the comoving distance to the midplane redshit of the snapshots region
                being added to the lightcone is > L, then rotate for each box replication along the beam.
        """

        comoving_distance = apply_expected_units(comoving_distance, unyt.Mpc).to_value("Mpc")
        return int(np.floor(comoving_distance / boxsize + 0.5))

    def _tiles_intersecting_shell(self, r_min, r_max, boxsize, ang_radius_deg=None):
        """
            Split a snapshot's assigned redshift range within the 
            shell at every tile boundary it crosses.
            
            Returns a list of tuples: ((nx, ny, n_los), z_sub_min, z_sub_max)
            n_los is the replica index along beam_vec (as
            before), and (nx, ny) is the replica offset along the two
            transverse axes (0, 0) for a piece that doesn't need any.
        """
        half_angle = np.deg2rad(ang_radius_deg)
        
        #if self.comm_rank==0:
        #    message(self.comm_rank, f"redshift range:{snap_shell_z_range}")
        #self.comm.barrier()
        #self.comm.barrier()
        #quit()
        #r_min, r_max = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")

        along_min = r_min * np.cos(half_angle)
        tile_min = self._los_tile_idx(along_min, boxsize)
        tile_max = self._los_tile_idx(r_max, boxsize)

        #boxsize = self.__snap_cell_data["snap_boxsize"][0]
        half_L = 0.5 * boxsize
        
        w_max = r_max * np.tan(half_angle)

        tiles = []
        for n_los in range(tile_min, tile_max + 1):
            if w_max <= half_L:
                # one box replica already covers the beam's cross-section
                tiles.append((0, 0, n_los))
                continue

            # need transverse replicas too -- enumerate candidates out to
            # however many are needed, then keep only the ones whose
            # closest point to the beam axis is still within w_max
            n_transverse = int(np.ceil((w_max - half_L) / boxsize))
            for nx in range(-n_transverse, n_transverse + 1):
                tx_lo, tx_hi = nx * boxsize - half_L, nx * boxsize + half_L
                min_dx = 0.0 if tx_lo <= 0.0 <= tx_hi else min(abs(tx_lo), abs(tx_hi))
                for ny in range(-n_transverse, n_transverse + 1):
                    ty_lo, ty_hi = ny * boxsize - half_L, ny * boxsize + half_L
                    min_dy = 0.0 if ty_lo <= 0.0 <= ty_hi else min(abs(ty_lo), abs(ty_hi))
                    min_perp_dist = np.sqrt(min_dx**2 + min_dy**2)
                    if min_perp_dist > w_max:
                        continue
                    tiles.append((nx, ny, n_los))

        return tiles

    def _in_shell(self, coords, z_min, z_max, ang_radius_deg, buffer_length=0., buffer_shape="cube",  return_bool=False, cosmo=None):
        """
        Determine which points fall inside the lightcone shell defined by a minimum multipole ell_min and a
        redshift range [z_min, z_max].

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            Cartesian comoving lightcone coordinates in Mpc, columns (x, y, z_axis).
        ang_radius_deg : float
            Angular radius of the cone [deg]
        z_min, z_max : float
            Redshift bounds of the lightcone slice.

        Returns
        -------
        mask : ndarray of bool, shape (N,)
            True for points inside the lightcone wedge.
        """


        if (cosmo is None):
            cosmo=self.cosmo

        #if (beam_vector is None):
        #    beam_vector=self.beam_vec
        #if (ang_radius_deg is None):
        #    ang_radius_deg=self.ang_radius_deg


        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 3: # test for coords shape
            raise ValueError("coords must be a vector with shape (N, 3)")

        axis = self.beam_vec

        #theta_min = np.pi / ell_min # [radian]
        #half_angle = theta_min / 2.0
        half_angle=np.deg2rad(ang_radius_deg)
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
            _classify_wedge_numba(coords, axis, shell_inner, shell_outer,
                                   half_angle, buffer_radius, m)

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
                (r > shell_inner - buffer_radius) & (r <= shell_outer + buffer_radius) # dist from observer check
                & (along_axis > -buffer_radius) # direction check
                & (angle_from_axis <= half_angle + angle_buffer) # perpendicular distance check
            )

        if return_bool:
            return m, None
        else:
            return np.flatnonzero(m), np.flatnonzero(np.invert(m))

    def _diameter_at_redshift(self, z, ang_radius_deg, cosmo=None, use_comoving_dist=False):
        """
        Cone-diameter engine helper shared by __gather_cells_and_tiles's
        per-tile diagnostics (via _max_diameter_at_redshift below) and
        the public beam_diameter. Returns the diameter of a cone of
        angular radius ang_radius_deg at a given redshift.
            If use_comoving_dist == True, then z is passed as a comoving distance from the observer
        """
        # set cosmology
        if cosmo is not None:
            self.cosmo = cosmo

        half_angle = np.deg2rad(ang_radius_deg)
        if use_comoving_dist:
            h=z
        else:
            h = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(z)).to_value("Mpc")
        return h * np.tan(half_angle) * 2

    def beam_diameter(self, z, ang_radius_deg, cosmo=None, use_comoving_dist=False):
        """
        Returns diameter of the beam's cone (angular radius ang_radius_deg)
        at a given redshift.
            If use_comoving_dist == True, then z is passed as a comoving distance from the observer
        """
        return self._diameter_at_redshift(z, ang_radius_deg, cosmo=cosmo, use_comoving_dist=use_comoving_dist)

    def get_snapshot_reposition_coords(self, tile, snapshot_sidelengths):
        """
        Returns the coordinate transform required for snapshot to be centered in the beam at the snapshots redshift.

        tile is (nx, ny, n_los):
            n_los:      replicas along beam_vec (the line of sight)
            nx, ny:     replicas transverse to beam_vec, along e1 and e2,
                        once the beam's diameter exceeds one box sidelength
        """
        nx, ny, n_los = tile
        tile_centre_offset = (
            n_los * snapshot_sidelengths[0] * self.beam_vec
            + nx * snapshot_sidelengths[0] * self.beam_transverse_e1
            + ny * snapshot_sidelengths[0] * self.beam_transverse_e2
        )

        offset=np.array([
                -0.5*snapshot_sidelengths[0] + tile_centre_offset[0],
                -0.5*snapshot_sidelengths[1] + tile_centre_offset[1],
                -0.5*snapshot_sidelengths[2] + tile_centre_offset[2]
            ])

        return offset


class SnapshotAllSky(SnapshotLightcone):
    """
    
    """

    def __init__(self, boxsize_resolution, simulation_name, orientation_seed=0):
        """
        Pass to SnapshotLightcone
        """
        
        super().__init__(boxsize_resolution, simulation_name, beam_vector=None, orientation_seed=orientation_seed)

    def _tiles_intersecting_shell(self, r_min, r_max, boxsize, ang_radius_deg=None):
        """
        Every integer tile index (nx, ny, nz) whose periodic box replica
        -- centred at (nx, ny, nz)*boxsize, spanning
        [n_i*boxsize - boxsize/2, n_i*boxsize + boxsize/2) along each
        axis -- intersects the spherical annulus [r_min, r_max] around
        the observer at the origin.

        :param r_min, r_max: comoving distance bounds, Mpc
        :param boxsize: snapshot box side length, Mpc (box assumed cubic)
        :return: list of (nx, ny, nz) int tuples
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

    def _in_shell(self, coords, z_min, z_max, ang_radius_deg=None, buffer_length=0.,
                 buffer_shape="cube", return_bool=False, cosmo=None):
        """
        Radial-shell selection: every direction is "inside the beam" for an all-sky observer, 
        so only the comoving-distance bounds matter (ang_radius_deg is accepted for signature
        compatibility with the inherited pipeline but is unused).
        """
        if cosmo is None:
            cosmo = self.cosmo

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

    def get_snapshot_reposition_coords(self, tile, snapshot_sidelengths):
        """
        Coordinate offset placing 3D box tile `tile` = (nx, ny, nz) at
        global position (nx, ny, nz)*snapshot_sidelengths -- the periodic
        replica of the box the observer sees in that direction/distance.
        """
        tile_arr = np.asarray(tile, dtype=float)
        snapshot_sidelengths = np.asarray(snapshot_sidelengths, dtype=float)
        return -0.5 * snapshot_sidelengths + tile_arr * snapshot_sidelengths

    def _diameter_at_redshift(self, z, ang_radius_deg=None, cosmo=None, use_comoving_dist=False):
        if cosmo is not None:
            self.cosmo = cosmo

        if use_comoving_dist:
            h = z
        else:
            h = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(z)).to_value("Mpc")
        return h * 2
        
    def lightcone_diameter(self, z, cosmo=None, use_comoving_dist=False):
        """
        Returns diameter of the past lightcone
        """
        return self._diameter_at_redshift(z, cosmo=cosmo, use_comoving_dist=use_comoving_dist, ang_radius_deg=None)

    def gather_files(self, current_ptype, shell_z, use_snapshots=None):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.
        """
        return super().gather_files(current_ptype, shell_z, None, use_snapshots=use_snapshots)

    def place_file_in_shell(self, file_number, current_ptype, property_names):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.
        """

        return  super().place_file_in_shell(file_number=file_number, current_ptype=current_ptype, property_names=property_names, ang_radius_deg=None)

    def place_snapshot_particles_in_shell(self, lightcone_redshift_range, property_names,
                                           particle_types=["PartType0", "PartType1", "PartType5"],
                                           use_snapshots=None, comm=None):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.
        """

        return super().place_snapshot_particles_in_shell(
                lightcone_redshift_range=lightcone_redshift_range, 
                ang_radius_deg=None, 
                property_names=property_names,
                particle_types=particle_types, 
                use_snapshots=use_snapshots,
                comm=comm, redistibute_particles=False)

    def place_halos_in_shell(self, halo_format, lightcone_redshift_range, use_snapshots=None):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.
        """
        return super().place_halos_in_shell(halo_format, lightcone_redshift_range, None, use_snapshots=use_snapshots)

