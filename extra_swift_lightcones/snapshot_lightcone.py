#!/bin/env python
import numpy as np
import unyt
import h5py
import argparse
from collections import namedtuple
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from lightcone_io.particle_reader import merge_cells
import lightcone_io.halo_catalogue as hc
from . import snapshot_orientation as box_structure
from . import snapshot_units as sw_units
from . import  swift_snapshot_redshift_conversion as nz
import datetime as dt
import warnings

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

# define simple printout functions
def seperator_str(n=35,line_seperator="~"):
    """
    Make fancy line seperation, 
    
    Returns a newline followed by the separator line.
    :param  n:              number of characters in the line
    :type   n:              int
    :param  line_seperator: character used to draw the line
    :type   line_seperator: str
    """
    sep_str=line_seperator * n
    return "\n"+sep_str

def message(rank, m, time_date_update=False):
    """
    Print a new message if on the prime (zero) rank, or if not using MPI. 

    :param  rank:               MPI rank. If None, always print
    :type   rank:               int
    :param  m:                  message to print
    :type   m:                  str
    :param  time_date_update:   If True, prefix the message with the current time
    :type   time_date_update:   boolean
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
    Print a new message with a timestamp and the rank number on each rank.

    :param  rank:   MPI rank
    :type   rank:   int
    :param  m:      message to print
    :type   m:      str
    """
    current_time=dt.datetime.now()
    time_str=current_time.strftime("%H:%M:%S")
    print('\t[Rank {rank_nr:03d}] [@{print_time}]\t'.format(rank_nr=rank,print_time=time_str) + m)

def print_coord_ranges(x, indent=0, rank=None):
    """
    Print the minimum and maximum of the coordinates along each axis.

    :param  x:      coordinates, shape (N, 3)
    :type   x:      np.ndarray or unyt.unyt_array
    :param  indent: number of tabs to indent each line by
    :type   indent: int
    :param  rank:   MPI rank, only rank 0 prints. If None, always print
    :type   rank:   int
    """
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
    
    # setting parallel=False when using MPI parallel processing
    @njit(parallel=False, fastmath=True, cache=True)
    def _classify_wedge_numba(coords, axis, chi_inner, chi_outer, half_angle,
                               buffer_radius, mask):
        """
        Flag points inside a cone about the axis that lie between two comoving distances. 

        Returns the updated mask, True where point lies within the cone and comoving distances. 

        :param  coords:         comoving coordinates [Mpc] relative to the observer, shape (N, 3)
        :type   coords:         np.ndarray
        :param  axis:           unit vector along the axis of the cone
        :type   axis:           np.ndarray
        :param  chi_inner:      inner comoving distance [Mpc] of the shell
        :type   chi_inner:      float
        :param  chi_outer:      outer comoving distance [Mpc] of the shell
        :type   chi_outer:      float
        :param  half_angle:     angular radius [rad] of the cone
        :type   half_angle:     float
        :param  buffer_radius:  distance [Mpc] to extend the boundaries of the shell by
        :type   buffer_radius:  float
        :param  mask:           output boolean buffer, shape (N,)
        :type   mask:           np.ndarray
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
                (r > chi_inner - buffer_radius) and (r <= chi_outer + buffer_radius)
                and (along > -buffer_radius)
                and (angle <= half_angle + angle_buffer)
            )
        return mask

def define_redshift_at_comoving_distance_function(cosmo, zmin=0.0001, zmax=5.05, n_grid=3000, method="fast"):
    """
    Build an interpolation function for the redshift at a given comoving distance.

    Returns a CubicSpline function taking comoving distance [Mpc] and returning redshift.

    :param  cosmo:  cosmology of the simulation
    :type   cosmo:  astropy cosmology object
    :param  zmin:   minimum redshift of the grid
    :type   zmin:   float
    :param  zmax:   maximum redshift of the grid
    :type   zmax:   float
    :param  n_grid: number of points in the grid
    :type   n_grid: int
    :param  method: 'fast', a uniform grid in redshift. 'precise', a uniform grid in comoving distance
    :type   method: str
    """
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

# ways of locking the orientation of the box tiles into layers
ORIENTATION_LOCK_OPTIONS = (None, "none", "cube", "sphere")

class SnapshotLightcone():
    """
    Fundamental code used to constuct lightcones from snapshots with 
    SnapshotBeam and SnapshotAllSky sub classes. 

    This class tracks:
        1. snapshot cosmology 
        2. MPI ranks
        3. periodic replications of snapshots and their reorientation
            to avoid exact replication of structures. 
        4. reading and placing particles from snapshots into 
            the lightcone

    The exact methods of identifying particles within the lightcones footprint and how the 
    snapshot box replication occurs is handled in the SnapshotBeam and SnapshotAllSky sub classes. 
    """

    def __init__(self, boxsize_resolution, simulation_name, beam_vector, 
                    simulation_base_dir_format="/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}",  
                    orientation_seed=0, verbose=1, orientation_lock=None):
        """
        Define the lightcone's cosmology, units and vector (where applicable).

        :param  boxsize_resolution:         FLAMINGO box size and resolution label, e.g. "L1000N1800"
        :type   boxsize_resolution:         str
        :param  simulation_name:            name of the simulation, e.g. "HYDRO_FIDUCIAL"
        :type   simulation_name:            str
        :param  beam_vector:                direction vector of the line of sight as an array of 3 floats. None for all-sky
        :type   beam_vector:                numpy.ndarray
        :param  simulation_base_dir_format: formatted path to the simulation directory, with {box_res} and {sim_name} fields
        :type   simulation_base_dir_format: str
        :param  orientation_seed:           seed used to select the rotation, reflection and periodic shift of each box tile. 
        :type   orientation_seed:           int
        :param  orientation_lock:           "none":   every box tile has its own orientation. 
                                            "cube": every box tile in a cube shell of boxes around the observer's box shares one 
                                                orientation, so each shell is periodically continuous and the 
                                                orientation only changes on the faces between shells. 
                                            "sphere": 
                                                everything in a spherical shell, between (n-1/2) and (n+1/2) box 
                                                sidelengths from the observer, shares one orientation, so the 
                                                orientation only changes at those distances. A box crossing a 
                                                sphere is used once for each shell, and tiles are labelled 
                                                (nx, ny, nz, layer)
        :type   orientation_lock:           str
        """

        # simulation values
        self.box_res=boxsize_resolution
        self.sim_name=simulation_name
        self.simulation_dir=simulation_base_dir_format.format(box_res=self.box_res, sim_name=self.sim_name)
        self.snapshot_format = self.simulation_dir+'/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5'
        
        self.verbose = int(verbose)
        
        # define the cosmology from the snapshots
        self.__define_snapshot_cosmology(int(nz.snapshot_number_redshifts(snapshot_number=0, boxsize_resolution=self.box_res, inverse=True)), 0)

        #self.__clear_all_internal_values()

        # define the beam vector (direction of line of sight)
        if beam_vector is None:
            self.beam_vec = None
        else:
            self.beam_vec=self.__define_beam_vector(beam_vector)
        
        self.orientation_seed = int(orientation_seed)
        if orientation_lock not in ORIENTATION_LOCK_OPTIONS:
            raise ValueError(f"orientation_lock must be one of {ORIENTATION_LOCK_OPTIONS}, not {orientation_lock!r}")
        self.orientation_lock = orientation_lock

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
        """
        Reset the values specific to the snapshot currently being read.
        """
        # snapshot specific values
        # points to state of current snapshot and cell data
        self.__snap_cell_data=None
        self.__cell_data_snap_nr=None
        self.__snap_redshift_range_to_populate=None

    def __reset_cell_read_state(self, snap_nr):
        """
        Point the current snapshot cell data to the given snapshot, reading it into the cache if needed.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        """
        if snap_nr not in self._cell_data_cache:
            self._cell_data_cache[snap_nr] = self.__get_cell_data(snap_nr)
        self.__snap_cell_data = self._cell_data_cache[snap_nr] # current snapshot cell data
        self.__cell_data_snap_nr=snap_nr # current snapshot number of cell data

    def __last_snapshot_read(self, snap_nr, tile, z_updated):
        """
        Record a snapshot tile read into the current shell.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        :param  tile:       periodic replica tile index
        :type   tile:       tuple
        :param  z_updated:  redshift range of the shell populated by this snapshot
        :type   z_updated:  tuple
        """
        if self.last_snapshot_read is None:
            self.last_snapshot_read=[]
        self.last_snapshot_read.append(SnapshotReadRecord(snap_nr, tile, z_updated))

    def __last_snapshot_index(self, snap_nr, tile):
        """
        Find the index of a snapshot tile in the records of the snapshots read.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        :param  tile:       periodic replica tile index
        :type   tile:       tuple
        """
        for idx, record in enumerate(self.last_snapshot_read):
            if record.snap_nr == snap_nr and record.tile == tile:
                return idx
        raise ValueError(f"snapshot {snap_nr}, tile {tile} not found in last_snap records")

    @staticmethod
    def __define_beam_vector(beam_vector):
        """
        Check the beam vector and normalise it.

        Returns the beam vector as a unit vector.

        :param  beam_vector:    direction vector as an array of 3 floats
        :type   beam_vector:    numpy.ndarray
        """
        beam_vector = np.asarray(beam_vector, dtype=float)
        if beam_vector.shape != (3,): # test for vector shape
            raise ValueError("beam_vector must be a vector with shape (3,)")
        norm = np.linalg.norm(beam_vector)
        if norm == 0: # test for non-zero vector
            raise ValueError("beam_vector must be a nonzero vector")
        return beam_vector / norm # ensure unit vector

    def __get_cell_data(self, snap_nr):
        """
        Read the cell metadata of a snapshot.

        Returns a dict of the number of cells, cell size, cell centres, box size, number of files 
        and the file, offsets and lengths of each cell per particle type.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        """
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
        """
        Find which files of a snapshot contain each particle type.

        Returns a dict of file numbers per particle type.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        :param  numb_files: number of files in the snapshot
        :type   numb_files: int
        """
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
        """
        Define the cosmology from a snapshot file.

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  file_number:        file number of the snapshot
        :type   file_number:        int
        """
        try:
            self.cosmo = Snapshot_Cosmology_For_Lightcone(self.snapshot_format.format(snap_nr=snapshot_number, file_nr=file_number)).COSMO

        except Exception or (FileNotFoundError, OSError) as e:
            self.cosmo = Snapshot_Cosmology_For_Lightcone(self.simulation_dir+'/snapshots').COSMO
            

    def __comoving_distance_to_scalefactor(self, r, cosmo=None):
        """
        Compute the scale factor of particles based on comoving distance from observer.

        :param  r:      comoving distance from the observer [Mpc]
        :type   r:      float, np.ndarray or unyt.unyt_array (units of Mpc or equivalent)
        :param  cosmo:  cosmology. If None, use the snapshot cosmology
        :type   cosmo:  astropy cosmology object
        """
        if cosmo is None:
            cosmo=self.cosmo
        if self.interp_redshift_from_comoving_distance is None:
            raise ValueError("interpolation function not defined")

        r = sw_units.apply_expected_units(r, unyt.Mpc) # place distance in terms of Mpc
        a = 1./(self.interp_redshift_from_comoving_distance(r.to_value("Mpc"))+1.)

        return a

    def __redshift_from_comoving_distance(self, shell_z, n_grid, method="precise"):
        """
        Define the interpolation function for the redshift at a given comoving distance over the redshift range.

        :param  shell_z:    minimum and maximum redshift [z_min, z_max]
        :type   shell_z:    sequence of two floats
        :param  n_grid:     number of points in the interpolation grid
        :type   n_grid:     int
        :param  method:     'fast' or 'precise', see define_redshift_at_comoving_distance_function
        :type   method:     str
        """
        zmin=shell_z[0]-0.01 if shell_z[0]>0.01 else shell_z[0]
        zmax=shell_z[1]+0.01
        self.interp_redshift_from_comoving_distance = define_redshift_at_comoving_distance_function(self.cosmo, zmin, zmax, n_grid, method="precise")

    def __remove_snapshots_outside_shell(self, shell_z):
        """
        Remove snapshots to read that do not fit within the bounds of the shell.

        :param  shell_z:    minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:    sequence of two floats
        """
        m = np.ones(len(self.snapshots_to_read), dtype=bool)
        for ii, snap_nr in enumerate(self.snapshots_to_read):
            snapshot_z_range = nz.snapshot_redshift_range(snap_nr, self.box_res)
            if snapshot_z_range[0] > shell_z[1]:
                m[ii]=0
            elif  snapshot_z_range[1] < shell_z[0]:
                m[ii]=0
        self.snapshots_to_read=self.snapshots_to_read[m]

    def _log(self, m, level=1, time_date_update=False):
        """
        If self.verbose is equal to given level, print a message (prime (zero) rank only for MPI compatibility). 

        :param  m:                  message to print
        :type   m:                  str
        :param  level:              1 for a summary, 2 for details
        :type   level:              int
        :param  time_date_update:   If True, prefix the message with the current time
        :type   time_date_update:   boolean
        """
        if self.verbose >= level:
            message(self.comm_rank, m, time_date_update=time_date_update)

    def _rank_log(self, m, level=1):
        """
        Print a message on each rank, if self.verbose is equal to given level. 

        :param  m:      message to print
        :type   m:      str
        :param  level:  1 for a summary, 2 for details
        :type   level:  int
        """
        if self.verbose >= level:
            rank_message(self.comm_rank, m)

    def _log_coord_ranges(self, title, coords):
        """
        Print the minimum and maximum of the coordinates along each axis, if self.verbose == 2.

        :param  title:  line printed before the ranges
        :type   title:  str
        :param  coords: coordinates, shape (N, 3)
        :type   coords: np.ndarray or unyt.unyt_array
        """
        if self.verbose >= 2:
            message(self.comm_rank, title)
            print_coord_ranges(coords, indent=2, rank=self.comm_rank)

    def _define_mpi_mode(self, comm, redistibute_particles):
        """
        Enable MPI mode for this output. In MPI mode each rank reads a subset of the selected particles.

        :param  comm:                   MPI communicator. If None, use serial mode
        :type   comm:                   mpi4py.MPI.Comm
        :param  redistibute_particles:  If True, redistribute particles evenly across ranks after reading
        :type   redistibute_particles:  boolean
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
        Find every tile a snapshot's assigned redshift range touches + the snapshot's 
        full (unsplit) redshift range.

        Returns a list of tuples: (tile, z_min, z_max).

        :param  snap_shell_z_range: redshift range of the shell populated by the snapshot [z_min, z_max]
        :type   snap_shell_z_range: tuple
        :param  ang_radius_deg:     angular radius [deg] of the beam, passed to _tiles_intersecting_shell. 
                                        Unused by SnapshotAllSky
        :type   ang_radius_deg:     float
        """
        r_min, r_max = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")

        boxsize = self.__snap_cell_data["snap_boxsize"][0]

        tiles = self._tiles_intersecting_shell(r_min, r_max, boxsize, ang_radius_deg)
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

    def _snapshot_reorientation(self, tile):
        """
        Orientation (rotation, reflection, periodic shift) for snapshot box 'tile' (nx, ny, n_los). 
        Create a new permutation of the snapshot if the comoving distance to this redshift 
        is > box sidelength or in a set orientation lock layer. 
        With orientation_lock="cube" or "sphere" every tile in the same layer takes the orientation 
        of tile (0, 0, layer), so the boxes in a layer are periodically continuous.

        Returns a tuple, (rotation angles [deg], reflections, periodic cell shifts)
        
        :param  tile:   periodic replica tile index
        :type   tile:   tuple
        """
        n_ang = box_structure.SnapshotBeamAngles_quaters.shape[0]
        n_ref = box_structure.SnapshotBeamAngles_reflections.shape[0]
        n_shift = box_structure.SnapshotBeamAngles_shifts.shape[0]
        
        if self.orientation_lock is not None:
            nx, ny, nz = 0, 0, self._orientation_layer(tile)
        else:
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
        Convert cell shift to a comoving distance vector.

        Returns the shift vector as an np.ndarray, shape (3,).

        :param  cell_shift:             number of cells to shift along x, y, z. If None, no shift
        :type   cell_shift:             array-like of int, shape (3,)
        :param  n_cells_per_axis:       number of cells per sidelength
        :type   n_cells_per_axis:       int or array-like, shape (3,)
        :param  cell_sidelength:        length along each axis of a cell
        :type   cell_sidelength:        float or array-like, shape (3,)
        :param  snapshot_sidelength:    length along each axis of snapshot
        :type   snapshot_sidelength:    float or array-like, shape (3,)
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
        Generate a new set of coordinates making a new orientation of the snapshot 
        through rotations, reflections and wrapping coordinates about the periodic boundaries.

        Returns the coordinates for the new snapshot orientation.

        :param  coords:                 coordinates, shape (N, 3)
        :type   coords:                 np.ndarray
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
        :type   degrees:                boolean
        :param  out:                    preallocated output buffer, shape (N, 3) (avoids reallocation on repeated calls)
        :type   out:                    np.ndarray
        :param  inplace:                If True, overwrite coords in place instead of allocating new memory
        :type   inplace:                boolean
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
        ang_radius_deg does not need validating unless an upper bound has been placed on it. 
        SnapshotBeam overrides this, SnapshotAllSky doesn't override it.

        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        """
        pass

    def _gather_files_for_ptype(self, current_ptype, shell_z, ang_radius_deg, use_snapshots=None):
        """
        Identify every (snapshot, tile, file) needed to populate a lightcone
        shell for a single particle type, without reading any particle data.

        Populates self.snap_file_offset_lengths, self.npart_per_file, self.last_snapshot_read, 
        self._file_order_index and self.npart_kept_per_file. 

        Returns self.snap_file_offset_lengths, a list of FileReadSpec entries.

        :param  current_ptype:  particle type to gather, e.g. "PartType1"
        :type   current_ptype:  str
        :param  shell_z:        minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or np.ndarray
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
            self._log(f"\nNo {current_ptype} particles found in selected cells\n", level=1)
            return self.snap_file_offset_lengths

        # wipe data as needed
        self.__clear_snapshot_read_state()

        snapshots_used = sorted({int(file_data.snap_nr) for file_data in self.snap_file_offset_lengths})
        self._log(f"{current_ptype}: {len(self.snap_file_offset_lengths)} files to read from snapshots {snapshots_used}, {int(np.sum(self.npart_per_file))} particles in the selected cells", level=1)


        # print of all files to read from:
        for i in range(len(self.snap_file_offset_lengths)):
            file_data=self.snap_file_offset_lengths[i]
            j = self.__last_snapshot_index(file_data.snap_nr, file_data.tile)
            update_cell_str=f"\tsnap: {file_data.snap_nr:<2},\ttile: {str(file_data.tile):<10},\tfile: {file_data.file_nr:<2},\tnumb_part: {self.npart_per_file[j][file_data.file_nr]} [{np.sum(file_data.lengths)}]"
            self._log(update_cell_str, level=2)

        return self.snap_file_offset_lengths

    def __populate_lightcone_with_ptype(self, particle_type, property_names, shell_z, ang_radius_deg, redistribute=False, use_snapshots=None):
        """
        Read and place all particles of one type into the lightcone shell.

        Returns a dict of particle properties, empty if no particles are found.

        :param  particle_type:  particle type to place, e.g. "PartType1"
        :type   particle_type:  str
        :param  property_names: particle properties to read, required properties are added if missing
        :type   property_names: list
        :param  shell_z:        minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        :param  redistribute:   If True, redistribute particles evenly across MPI ranks
        :type   redistribute:   boolean
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or np.ndarray
        """
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
        """
        Find the cells and files of a snapshot, for every tile, that fall within the lightcone shell. 
        Updates self.snap_file_offset_lengths, self.npart_per_file and self.last_snapshot_read.

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  shell_z:            minimum and maximum redshift of the shell [z_min, z_max]
        :type   shell_z:            sequence of two floats
        :param  current_ptype:      particle type, e.g. "PartType1"
        :type   current_ptype:      str
        :param  ang_radius_deg:     angular radius [deg] of the beam
        :type   ang_radius_deg:     float
        """

        # sanity check snapshot specific values
        if self.__snap_cell_data is not None:
            raise ValueError(f"snapshot specific values are not being cleared after iterations")
        elif self.__cell_data_snap_nr is not None:
            raise ValueError(f"snapshot specific values are not being cleared after iterations")
        elif self.__snap_redshift_range_to_populate is not None:
            raise ValueError(f"snapshot specific values are not being cleared after iterations")

        snapshot_z_range = nz.snapshot_redshift_range(snapshot_number, self.box_res)
        # show read out of snapshot redshift info
        snapshot_to_shell_str=(f"snapshot: {snapshot_number}"+f"\n\tcentre redshift {nz.snapshot_number_redshifts( snapshot_number, self.box_res)}"+f"\n\tredshift range: {snapshot_z_range[0]} - {snapshot_z_range[1]}")
        self._log(snapshot_to_shell_str, level=2)
        snap_shell_z_range=tuple(
                    [snapshot_z_range[0] if snapshot_z_range[0] > shell_z[0] else shell_z[0],
                    snapshot_z_range[1] if snapshot_z_range[1] < shell_z[1] else shell_z[1]]
                )

        self.__snap_redshift_range_to_populate=snap_shell_z_range


        # write new snapshot specific values, reuse of data is stored in cache
        self.__reset_cell_read_state(snapshot_number)
        self.__snap_redshift_range_to_populate=snap_shell_z_range


        # give read out of max beam diameter for snapshots redshift range in shell
        r_beam = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(snap_shell_z_range)).to_value("Mpc")
        add_part_from_snap_str=(
            f"add particles from snapshot {snapshot_number} to lightcone shell" +
            f"\n\tredshift:\t{snap_shell_z_range[0]:.5f} - {snap_shell_z_range[1]:.5f}" +
            f"\n\tradii:\t{r_beam[0]:.5f} - {r_beam[1]:.5f} [Mpc]" 
            )

        self._log(add_part_from_snap_str, level=2)

        del r_beam #, max_diameter

        # set up dictionary of ptypes in snapshot files:
        ptypes_in_current_snap = self.__files_with_ptype(snapshot_number, self.__snap_cell_data["nr_files"])

        #iterate through tiles pieces
        # add transverse tiles when the beam's diameter exceeds one box sidelength. 
        for tile, z_sub_min, z_sub_max in self._snapshot_tile_idx(snap_shell_z_range, ang_radius_deg):
            # read in instructions for how to reconstruct the snapshot box
            snapshot_rotation_angles, snapshot_rotation_reflections, snapshot_periodic_shifts = self._snapshot_reorientation(tile)
            orientation_str=(f"Reorienting snapshot {snapshot_number} (tile {tile}, z={z_sub_min:.5f}-{z_sub_max:.5f}"+
                (f", orientation layer {self._orientation_layer(tile)}" if self.orientation_lock else "")+"):"
                f"\n\tRotation:\t{snapshot_rotation_angles} [deg]"+
                f"\n\tReflection:\t{snapshot_rotation_reflections}"+
                f"\n\tPeriodic Shift:\t{snapshot_periodic_shifts}")
            self._log(orientation_str, level=2)
            
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
                    #z_min=snap_shell_z_range[0],
                    #z_max=snap_shell_z_range[1],
                    z_min=z_sub_min,
                    z_max=z_sub_max,
                    buffer_length=self.__snap_cell_data["cell_size"][0],
                    buffer_shape="cube",
                    return_bool=False
                )

            self._log(f"\nnumber of cells in beam: {np.shape(beam_idx)[0]}", level=2)

            if len(beam_idx) > 0:
                self._log_coord_ranges("coordinates range of cell centres in beam", repositioned_cell_centres[beam_idx,:])

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
            # store the snapshot's full shell range here, so __read_and_place_particles uses the
            # same correct (wider) radial bound.
            #self.__last_snapshot_read(snapshot_number, tile, snap_shell_z_range)
            self.__last_snapshot_read(snapshot_number, tile, (z_sub_min, z_sub_max))

    def __read_and_place_particles(self, files_to_read, current_ptype, property_names, ang_radius_deg):
        """
        Read the given FileReadSpec entries and re-orient each file's coordinates into the beam.

        :param  files_to_read:  files to read
        :type   files_to_read:  list of FileReadSpec
        :param  current_ptype:  particle type, e.g. "PartType1"
        :type   current_ptype:  str
        :param  property_names: particle properties to read
        :type   property_names: list
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
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

            self._log_coord_ranges("\ncoordinates range of particles read", file_particle_data["Coordinates"])

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

            if len(files_to_read) > 1: # otherwise the same as the total below
                self._log(f"Particles read:\t{npart_infile}\tkept in beam:\t{len(local_beam_idx)}", level=2)

            file_idx = self._file_order_index[(snap_nr, tile, file_nr)]
            self.npart_kept_per_file[file_idx] = len(local_beam_idx)
            self._local_kept_per_file_updates.append((file_idx, len(local_beam_idx)))

            for prop in property_names:
                kept_parts[prop].append(file_particle_data[prop][local_beam_idx])
            
            kept_parts["SnapshotNumber"].append(unyt.unyt_array(np.full_like(local_beam_idx, fill_value=snap_nr, dtype=int), units=unyt.dimensionless))

        if self.comm is not None:
            npart_read_global = self.comm.allreduce(npart_read_total)
            npart_kept_global = self.comm.allreduce(npart_kept_total)
        else:
            npart_read_global = npart_read_total
            npart_kept_global = npart_kept_total

        if npart_read_global > 0:
            self._log(f"\nParticles read:\t{npart_read_global}\tkept in beam:\t{npart_kept_global} [{npart_kept_global/npart_read_global * 100:.3f}%]", level=1)
        else:
            self._log("\nNo particles found in selected cells", level=1)

        for prop in property_names:
            if len(kept_parts[prop]) > 0:
                self.particle_data[prop] = unyt.uconcatenate(kept_parts[prop])
        
        if len(kept_parts["SnapshotNumber"]) > 0:
            self.particle_data["SnapshotNumber"] = unyt.uconcatenate(kept_parts["SnapshotNumber"])

    def __fill_missing_particle_data(self, property_names, comm):
        """
        Fill in a correct zero-length unyt_array for a rank assigned zero files. 
        Use the unit registery and each property's dtype + units from a rank that did read something.

        :param  property_names: particle properties read
        :type   property_names: list
        :param  comm:           MPI communicator
        :type   comm:           mpi4py.MPI.Comm
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
        Compute the scale factors of the particles inside the beam 
        from their comoving distance, stored as the ExpansionFactors lightcone property.
        """
        npart_in_beam = self.particle_data["Coordinates"].shape[0]
        self._log(f"Total particles in beam:\t{npart_in_beam}", level=2)

        if npart_in_beam > 0:
            self._log_coord_ranges("\ncoordinates range of particles in beam", self.particle_data["Coordinates"][:])

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
        """
        Read the particles in the selected cells of a snapshot file.

        Returns a dict of unyt arrays per property.

        :param  ptype:              particle type, e.g. "PartType1"
        :type   ptype:              str
        :param  filename:           path to the snapshot file
        :type   filename:           str
        :param  infile_offset:      offset of each set of adjacent cells in the file
        :type   infile_offset:      np.ndarray
        :param  infile_lengths:     number of particles in each set of adjacent cells
        :type   infile_lengths:     np.ndarray
        :param  property_names:     particle properties to read
        :type   property_names:     list
        :param  numb_part_infile:   total number of particles to read from the file
        :type   numb_part_infile:   int
        """
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
        MPI ranks are assigned distinct entries with one (snapshot, file) per rank where possible. 
        Each rank reads only its own files and places them in the beam. 
        Filtering occurs locally. 

        Note: each rank contains its own local slice of the shell's particles.

        :param  current_ptype:  particle type, e.g. "PartType1"
        :type   current_ptype:  str
        :param  property_names: particle properties to read
        :type   property_names: list
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        """

        # distribute the files to read across ranks, one file per rank for when there are at least as many files as ranks
        nr_files = len(self.snap_file_offset_lengths)
        files_on_rank = phdf5.assign_files(nr_files, self.comm_size)
        first_on_rank = np.cumsum(files_on_rank) - files_on_rank
        first = first_on_rank[self.comm_rank]
        num = files_on_rank[self.comm_rank]
        my_files = self.snap_file_offset_lengths[first:first+num]

        self.particle_data = {name : None for name in property_names}

        # local total for just the files this rank owns
        total_ptype_to_read = 0
        for file_data in my_files:
            jj = self.__last_snapshot_index(file_data.snap_nr, file_data.tile)
            total_ptype_to_read += self.npart_per_file[jj][file_data.file_nr]

        self._rank_log(f"\treading {len(my_files)}/{nr_files} files, {total_ptype_to_read} particles", level=1)

        self.__read_and_place_particles(my_files, current_ptype, property_names, ang_radius_deg)

        all_kept_per_file_updates = self.comm.allgather(self._local_kept_per_file_updates)
        for updates in all_kept_per_file_updates:
            for file_idx, count in updates:
                self.npart_kept_per_file[file_idx] = count

        # Fill space for when rank has no files assigned. 
        self.__fill_missing_particle_data(property_names, self.comm)

        self._attach_expansion_factors()

    def __select_paticles_in_shell(self, coords, zmin, zmax, ang_radius_deg, method="exact", r_tol=2*unyt.Mpc):
        """
        Select the particles inside the lightcone shell.

        Returns a tuple of (indices inside the shell, indices outside the shell).

        :param  coords:         comoving lightcone coordinates [Mpc], shape (N, 3)
        :type   coords:         np.ndarray
        :param  zmin:           minimum redshift of the shell
        :type   zmin:           float
        :param  zmax:           maximum redshift of the shell
        :type   zmax:           float
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        :param  method:         'exact', no buffer. 'approx', extend the shell by r_tol. 'all', keep every particle
        :type   method:         str
        :param  r_tol:          buffer distance used with the 'approx' method
        :type   r_tol:          float or unyt.unyt_quantity (units of Mpc or equivalent)
        """
        if method=="approx":
            buffer_length=sw_units.apply_expected_units(r_tol, unyt.Mpc).to_value("Mpc")
            buffer_shape="sphere"

        elif method=="exact":
            buffer_length=sw_units.apply_expected_units(0., unyt.Mpc).to_value("Mpc")
            buffer_shape="sphere"

        elif method=="all":
            assert np.ndim(coords)==2
            assert np.shape(coords)[-1]==3
            return (np.arange(np.shape(coords)[0]), None)


        return self._in_shell(coords, ang_radius_deg=ang_radius_deg, z_min=zmin, z_max=zmax, buffer_length=buffer_length, buffer_shape=buffer_shape, return_bool=False)

    def __redistribute_particles_evenly(self, property_names):
        """
        Redistribute particles evenly across all ranks.
        Every rank should then hold roughly equal amounts of particles.

        :param  property_names: particle properties to redistribute, ExpansionFactors is always included
        :type   property_names: list
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

        self._log(f"Redistributed {ntot} particles evenly across {self.comm_size} ranks", level=1)

    def gather_files(self, current_ptype, shell_z, ang_radius_deg, use_snapshots=None):
        """
        
        For a single particle type, determine every (snapshot, tile, file) needed to populate the
        given redshift shell, without reading any particle data.
        
        Serial-only. 

        Returns (numb_files, self.snap_file_offset_lengths). 
            the number of files found and the list of FileReadSpec entries itself. 

        :param  current_ptype:  particle type to gather, e.g. "PartType1"
        :type   current_ptype:  str
        :param  shell_z:        minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or np.ndarray
        """
        if self.comm is not None:
            raise ValueError("gather_files only supports serial (non-MPI) use")

        self._validate_ang_radius_deg(ang_radius_deg)

        files = self._gather_files_for_ptype(current_ptype, shell_z, ang_radius_deg, use_snapshots)
        return len(files), files

    def place_file_in_shell(self, file_number, current_ptype, property_names, ang_radius_deg):
        """
        Read and place a single file, selected by index, into the beam.

        Serial-only. 

        Returns the same per property particle data as
        place_snapshot_particles_in_shell, but containing only the
        particles kept in the selected file. 
        Returns an empty dictionary if there are no particles from the file inside the beam. 

        :param  file_number:    index into self.snap_file_offset_lengths
        :type   file_number:    int
        :param  current_ptype:  particle type being placed, e.g. "PartType1" 
                                    (must match the ptype passed to gather_files)
        :type   current_ptype:  str
        :param  property_names: particle properties to read, e.g. ["ParticleIDs", "Coordinates"]
        :type   property_names: list
        :param  ang_radius_deg: angular radius [deg] of the beam (must match the value passed to gather_files)
        :type   ang_radius_deg: float
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
            self._log(f"\nNo particles kept from file {file_number}\n", level=2)
            return {}

        self._attach_expansion_factors()

        return self.particle_data

    def place_snapshot_particles_in_shell(self, lightcone_redshift_range, ang_radius_deg, property_names, particle_types=["PartType0", "PartType1", "PartType5"], use_snapshots=None, comm=None, redistibute_particles=False):
        """
        Read and place every particle of the given type(s) into a lightcone shell, if they fall within the lightcone's footprint. 

        Returns a dict of particle data per particle type.

        :param  lightcone_redshift_range:   minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   lightcone_redshift_range:   sequence of two floats
        :param  ang_radius_deg:             angular radius [deg] of the beam
        :type   ang_radius_deg:             float
        :param  property_names:             particle properties to read, e.g. ["ParticleIDs", "Coordinates"]
        :type   property_names:             list
        :param  particle_types:             particle type(s) to place, e.g. "PartType1"
        :type   particle_types:             str or list
        :param  use_snapshots:              snapshot numbers to use instead of the ones auto-selected from the redshift range
        :type   use_snapshots:              list or np.ndarray
        :param  comm:                       MPI communicator. If None, read in serial
        :type   comm:                       mpi4py.MPI.Comm
        :param  redistibute_particles:      If True, redistribute particles evenly across MPI ranks
        :type   redistibute_particles:      boolean
        """

        self._validate_ang_radius_deg(ang_radius_deg)

        self._define_mpi_mode(comm, redistibute_particles)

        if self.comm is not None:
            self._log(f"\nReading snapshot files in parallel on {self.comm_size} MPI ranks\n", level=1)
            

        combined_dset ={}
        if np.ndim(particle_types)>0:
            # iterate through particle types
            for ptype in particle_types:
                self._log(seperator_str()+f"\nPlacing {ptype} in lightcone", level=1)
                combined_dset[ptype] = self.__populate_lightcone_with_ptype(ang_radius_deg=ang_radius_deg, shell_z=lightcone_redshift_range, property_names=property_names, particle_type=ptype,redistribute=redistibute_particles, use_snapshots=use_snapshots)

        elif np.ndim(particle_types)==0:
            # return for singular particle type
            ptype = particle_types
            self._log(seperator_str()+f"\nPlacing {ptype} in lightcone", level=1)
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
        """
        Recover the original snapshot frame coordiantes from the transformed coordinates. 
        
        i.e., given coordinates produced by _transform_snapshot_coordinates return the inital input coordinates.

        Returns the snapshot frame coordinates passed through _transform_snapshot_coordinates. 

        :param  coords:                 transformed coordinates, shape (N, 3)
        :type   coords:                 np.ndarray
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
        :type   degrees:                boolean
        :param  out:                    preallocated output buffer, shape (N, 3)
        :type   out:                    np.ndarray
        :param  inplace:                If True, overwrite coords in place instead of allocating new memory
        :type   inplace:                boolean
        """
        nr_cell_axis = cell_data["nr_cells_axis"]
        cell_sidelength=cell_data["cell_size"]
        snapshot_sidelength=cell_data["snap_boxsize"]

        shift_vector = self._cell_shift_to_vector(periodic_cell_shift, nr_cell_axis,  cell_sidelength, snapshot_sidelength)

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

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        """
        if snap_nr not in self._cell_data_cache:
            self._cell_data_cache[snap_nr] = self.__get_cell_data(snap_nr)
        return self._cell_data_cache[snap_nr]

    def Snapshot2Lightcone(self, snapshot_number, coords, tile=None):
        """
        Translate snapshot coordinates into lightcone coordinates, for  every tile this snapshot currently supplies to the lightcone.

        This function only knows about tiles from the most recently gathered shell and  
        relies on self.last_snapshot_read being populated by either gather_files, place_snapshot_particles_in_shell or the internal
        _gather_files_for_ptype they both call.

        Note: A single snapshot number can supply more than one periodic replica tile to the same shell due to independant rotation, reflection and periodic shifts.

        Returns a dict {tile: lightcone_coords}, one entry per tile this snapshot supplies to the 
        last gathered shell. 

        :param  snapshot_number:    snapshot number
        :type   snapshot_number:    int
        :param  coords:             comoving snapshot-frame (box) coordinates [Mpc], shape (N, 3)
        :type   coords:             np.ndarray
        :param  tile:               a specific periodic-replica tile, (nx, ny, n_los) for SnapshotBeam, 
                                        (nx, ny, nz) for SnapshotAllSky. If given, self.last_snapshot_read 
                                        is not consulted at all, so this works standalone (no gather_files/
                                        place_snapshot_particles_in_shell call needed first). If None, every 
                                        tile this snapshot supplies to the last-gathered shell is used.
        :type   tile:               tuple
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
            rot_angles, reflections, periodic_shift = self._snapshot_reorientation(tile)
            oriented = self._transform_snapshot_coordinates(
                coords, rot_angles, cell_data,
                reflections=reflections, periodic_cell_shift=periodic_shift,
                order="xyz", degrees=True, inplace=False,
            )
            lightcone_coords_by_tile[tile] = oriented + self.get_snapshot_reposition_coords(tile, boxsize)

        return lightcone_coords_by_tile

    def Lightcone2Snapshot(self, coords, snapshot_number=None, tile=None):
        """
        Inverse of Snapshot2Lightcone, translate lightcone frame
        coordinates back into the snapshot numbers and snapshot frame coordinates.

        This function relies on self.last_snapshot_read and only knows about snapshots and tiles from the most recently gathered shell.
        
        Which snapshot a lightcone frame particle belongs to is determined from its comoving distance alone. 
        Within a matched snapshot the snapshot number alone doesn't say which of its tiles a given point's orientation should be inverted to. 
        Therefore, this inverts every one of that snapshot's tiles and returns all of them. 
        Only one is the 'real' originating tile for any given point . 

        Returns a dict {snapshot_number: {tile: (point_index, snapshot_coords)}}. 
            point_index:        an integer array indexing into the input coords, i.e., which of the input points matched the snapshot_number
            snapshot_coords:    the corresponding snapshot frame coordinates after inverting through tile's orientation
        
        Note: Points whose comoving distance doesn't fall inside any recorded snapshot's range are dropped

        :param  coords:             comoving lightcone-frame coordinates [Mpc], shape (N, 3)
        :type   coords:             np.ndarray
        :param  snapshot_number:    skip distance-based snapshot matching and assume every point belongs to this snapshot
        :type   snapshot_number:    int
        :param  tile:               skip tile lookup and invert only through this one tile. 
                                        Requires snapshot_number to also be given
        :type   tile:               tuple
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
                #snap_z_range.setdefault(record.snap_nr, record.z_updated)
                
                tile_z_range[(record.snap_nr, record.tile)] = record.z_updated
                z_lo, z_hi = snap_z_range.get(record.snap_nr, record.z_updated)
                snap_z_range[record.snap_nr] = (min(z_lo, record.z_updated[0]), max(z_hi, record.z_updated[1]))

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
                warnings.warn(f"Lightcone2Snapshot: {n_unmatched} point(s) not within any snapshots comoiving distance range in the lasy shell read state")

        results = {}
        #for snap_nr, point_idx in point_idx_by_snap.items():
        for snap_nr, snap_point_idx in point_idx_by_snap.items():
        

            cell_data = self.__cell_data_for_snapshot(snap_nr)
            boxsize = cell_data["snap_boxsize"]

            tile_results = {}
            for t in tiles_by_snap[snap_nr]:
                point_idx = snap_point_idx
                if snapshot_number is None and tile_z_range[(snap_nr, t)] != snap_z_range[snap_nr]:
                    # the tile supplies only part of the snapshot's range, i.e. orientation_lock="sphere", 
                    # keep the points in its own range
                    t_r_min, t_r_max = unyt.unyt_array.from_astropy(
                        self.cosmo.comoving_distance(tile_z_range[(snap_nr, t)])).to_value("Mpc")
                    point_idx = point_idx[(r[point_idx] >= t_r_min) & (r[point_idx] <= t_r_max)]
                    if point_idx.size == 0:
                        continue
                rot_angles, reflections, periodic_shift = self._snapshot_reorientation(t)
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
        Place haloes from SOAP catalogues into a lightcone shell using the same methods as used for particles and Snapshot2Lightcone.

        Read a snapshot's entire SOAP halo catalogue in one collective read. 

        Track the minimum number of properties (snapshot number, catalogue index)
        required to look up any other properties in SOAP catalgues later:

          - Lightcone/HaloCentre: the halo's centre, transformed into the
            lightcone frame exactly like a particle's Coordinates.
          - Lightcone/SnapshotNumber: the snapshot the halo came from.
          - InputHalos/HaloCatalogueIndex: the halo's index in that
            snapshot's own SOAP catalogue. 
          - Lightcone/ExpansionFactor: expansion factors are computed for all
            at the halo's new lightcone comoving distance.

        Returns a dict of halo properties, one row per halo kept: 
            {
                Lightcone/HaloCentre,
                Lightcone/SnapshotNumber, 
                InputHalos/HaloCatalogueIndex,
                Lightcone/ExpansionFactor
                }

        :param  halo_format:                format string for SOAP catalogue filenames (using {snap_nr}), 
                                                passed straight to lightcone_io.halo_catalogue.SOAPCatalogue
        :type   halo_format:                str
        :param  lightcone_redshift_range:   minimum and maximum redshift of the shell to populate [z_min, z_max]
        :type   lightcone_redshift_range:   sequence of two floats
        :param  ang_radius_deg:             angular radius [deg] of the beam, passed through exactly like 
                                                place_snapshot_particles_in_shell (None for an all-sky 
                                                observer, see SnapshotAllSky's thin wrapper)
        :type   ang_radius_deg:             float
        :param  use_snapshots:              snapshot numbers to use, as in gather_files, bypassing the usual 
                                                redshift-range lookup
        :type   use_snapshots:              list or np.ndarray
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
            self._log("\nNo snapshots found in range for halo placement\n", level=1)

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
                # this snapshot's own redshift range doesn't overlap the requested shell 
                continue

            self._log(f"\nReading SOAP halos for snapshot {snap_nr}\n", level=2)
            halo_data = halo_cat.read(snap_nr, to_read)
            halo_index = halo_data["InputHalos/HaloCatalogueIndex"].value.astype(int)
            if halo_index.shape[0] == 0:
                continue

            halo_centre_box = sw_units.drop_a_from_comoving_property(halo_data["InputHalos/HaloCentre"]).to_value("Mpc")

            self.__reset_cell_read_state(snap_nr)

            for tile, z_sub_min, z_sub_max in self._snapshot_tile_idx(snap_shell_z_range, ang_radius_deg):
                lightcone_coords = self.Snapshot2Lightcone(snap_nr, halo_centre_box, tile=tile)[tile]

                keep_idx, __ = self.__select_paticles_in_shell(
                        lightcone_coords, zmin=z_sub_min, zmax=z_sub_max,
                        #lightcone_coords, zmin=snap_shell_z_range[0], zmax=snap_shell_z_range[1],
                        ang_radius_deg=ang_radius_deg, method="exact", r_tol=0 * unyt.Mpc,
                    )
                if len(keep_idx) == 0:
                    continue

                kept_centre.append(lightcone_coords[keep_idx])
                kept_snap_nr.append(np.full(len(keep_idx), snap_nr, dtype=int))
                kept_index.append(halo_index[keep_idx])

            self._log(f"\tsnap: {snap_nr},\thalos kept from this snapshot:\t{sum(len(a) for a in kept_snap_nr) - nr_kept_before}", level=2)

        if not kept_centre:
            return empty_result

        all_centre = np.concatenate(kept_centre, axis=0)
        all_snap_nr = np.concatenate(kept_snap_nr, axis=0)
        all_index = np.concatenate(kept_index, axis=0)

        r_comoving = np.linalg.norm(all_centre, axis=1)
        expansion_factor = self.__comoving_distance_to_scalefactor(r_comoving)

        self._log(f"\nTotal halos placed in lightcone shell: {all_snap_nr.shape[0]}\n", level=1)

        return {
            "Lightcone/HaloCentre": unyt.unyt_array(all_centre, unyt.Mpc),
            "Lightcone/SnapshotNumber": unyt.unyt_array(all_snap_nr, unyt.dimensionless),
            "InputHalos/HaloCatalogueIndex": unyt.unyt_array(all_index, unyt.dimensionless),
            "Lightcone/ExpansionFactor": unyt.unyt_array(expansion_factor, unyt.dimensionless),
        }


class SnapshotBeam(SnapshotLightcone):
    """
    Class for constructing a pencil beam lightcone, about a single line of sight, from snapshots.
    """

    def __init__(self, boxsize_resolution, simulation_name, beam_vector, orientation_seed=0, verbose=1, orientation_lock=None):
        """
        Define the beam.

        :param  boxsize_resolution: FLAMINGO box size and resolution label, e.g. "L1000N1800"
        :type   boxsize_resolution: str
        :param  simulation_name:    name of the simulation, e.g. "HYDRO_FIDUCIAL"
        :type   simulation_name:    str
        :param  beam_vector:        direction vector of the line of sight as an array of 3 floats
        :type   beam_vector:        numpy.ndarray
        :param  orientation_seed:   seed used to select the rotation, reflection and periodic shift of each box tile
        :type   orientation_seed:   int
        :param  verbose:            amount printed. 0: nothing, only warnings. 1: a summary of each call. 
                                        2: details of every snapshot, box tile and file
        :type   verbose:            int
        :param  orientation_lock:           None: every box tile has its own orientation. "cube": every box tile 
                                            in a cube shell of boxes around the observer's box shares one 
                                            orientation, so each shell is periodically continuous and the 
                                            orientation only changes on the faces between shells
        :type   orientation_lock:           str
        """
        super().__init__(boxsize_resolution, simulation_name, beam_vector, orientation_seed=orientation_seed, verbose=verbose, orientation_lock=orientation_lock)

        # set up orthonormal basis perpendicular to beam_vec, 
        # used to place tiles transverse to the line of sight once
        # the beam's diameter exceeds one box sidelength. 
        self.beam_transverse_e1, self.beam_transverse_e2 = self.__transverse_basis(self.beam_vec)

        self.__init_tile_frame()

    def __init_tile_frame(self):
        """
        Sets self._tile_frame, a signed permutation matrix mapping lattice positions to tile labels.

        Tiles are placed on the lattice of whole box sidelengths along x, y and z, so they fill
        the beam exactly once for any beam direction. 
        A tiles are labelled as (nx, ny, n_los), its lattice
        position along the x, y or z axis closest to e1, e2 and beam_vec (with the same sign). 

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
        Raise ValueError if the angular radius exceeds MAX_BEAM_ANG_RADIUS_DEG.

        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        """
        if ang_radius_deg > MAX_BEAM_ANG_RADIUS_DEG:
            radius_error_str=f"angular radius ({ang_radius_deg} [deg]) exceeds the maximum radius of {MAX_BEAM_ANG_RADIUS_DEG} [deg]"
            raise ValueError(radius_error_str)

    @staticmethod
    def __transverse_basis(beam_vector):
        """
        Orthonormal basis (e1, e2) perpendicular to beam_vector, used to
        place tiles transverse to the line of sight.

        :param  beam_vector:    unit direction vector of the line of sight
        :type   beam_vector:    numpy.ndarray
        """
        reference = np.array([1.0, 0.0, 0.0]) if abs(beam_vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(beam_vector, reference)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(beam_vector, e1)
        return e1, e2

    def _los_tile_idx(self, comoving_distance, boxsize):
        """
        Returns the number of times the periodic box has been replicated along a line of sight.

        :param  comoving_distance:  comoving distance along the line of sight [Mpc]
        :type   comoving_distance:  float or unyt.unyt_quantity (units of Mpc or equivalent)
        :param  boxsize:            snapshot box side length [Mpc]
        :type   boxsize:            float
        """
        comoving_distance = sw_units.apply_expected_units(comoving_distance, unyt.Mpc).to_value("Mpc")
        return int(np.floor(comoving_distance / boxsize + 0.5))

    def _tiles_intersecting_shell(self, r_min, r_max, boxsize, ang_radius_deg=None):
        """
        Find every tile that the beam crosses between two comoving distances.
        
        The tiles are the periodic replicas of the box on the lattice of whole box sidelengths. 
        Neighbouring tiles share faces and fill the beam exactly once, for any beam direction.
        A tile may be kept that only just misses the beam, its cells are then all rejected when 
        the cells are selected.

        Returns a list of tuples (nx, ny, n_los), the tile's lattice position along the axes 
        closest to e1, e2 and beam_vec (see __init_tile_frame).
        Returns a list of tuples (nx, ny, n_los).
            n_los:      is the replica index along beam_vec (how many replications)
            (nx, ny):   is the replica offset along the two transverse axes.

        :param  r_min:          minimum comoving distance [Mpc]
        :type   r_min:          float
        :param  r_max:          maximum comoving distance [Mpc]
        :type   r_max:          float
        :param  boxsize:        snapshot box side length [Mpc] (box assumed cubic)
        :type   boxsize:        float
        :param  ang_radius_deg: angular radius [deg] of the beam
        :type   ang_radius_deg: float
        """
        half_angle = np.deg2rad(ang_radius_deg)
        half_L = 0.5 * boxsize

        along_min = r_min * np.cos(half_angle)
        #tile_min = self._los_tile_idx(along_min, boxsize)
        #tile_max = self._los_tile_idx(r_max, boxsize)
        
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
        
        #for n_los in range(tile_min, tile_max + 1):
        #    if w_max <= half_L:
        #        # one box replica already covers the beam's cross-section
        #        tiles.append((0, 0, n_los))
        #        continue
        #
        #    # add transverse replicas
        #    n_transverse = int(np.ceil((w_max - half_L) / boxsize))
        #    for nx in range(-n_transverse, n_transverse + 1):
        #        tx_lo, tx_hi = nx * boxsize - half_L, nx * boxsize + half_L
        #        min_dx = 0.0 if tx_lo <= 0.0 <= tx_hi else min(abs(tx_lo), abs(tx_hi))
        #        for ny in range(-n_transverse, n_transverse + 1):
        #            ty_lo, ty_hi = ny * boxsize - half_L, ny * boxsize + half_L
        #            min_dy = 0.0 if ty_lo <= 0.0 <= ty_hi else min(abs(ty_lo), abs(ty_hi))
        #            min_perp_dist = np.sqrt(min_dx**2 + min_dy**2)
        #            if min_perp_dist > w_max:
        #                continue
        #            tiles.append((nx, ny, n_los))
        #
        #return tiles

    def _in_shell(self, coords, z_min, z_max, ang_radius_deg, buffer_length=0., buffer_shape="cube",  return_bool=False, cosmo=None):
        """
        Determine which points fall inside the lightcone shell defined by an angular radius and the redshift range (z_min, z_max).

        Returns a tuple, (indices inside the shell, indices outside the shell). 
        If return_bool is True, returns (boolean mask, None), where it is True for points inside the lightcone wedge.

        :param  coords:         Cartesian comoving lightcone coordinates [Mpc], shape (N, 3)
        :type   coords:         np.ndarray
        :param  z_min:          minimum redshift of the lightcone slice
        :type   z_min:          float
        :param  z_max:          maximum redshift of the lightcone slice
        :type   z_max:          float
        :param  ang_radius_deg: angular radius [deg] of the cone
        :type   ang_radius_deg: float
        :param  buffer_length:  size [Mpc] of the buffer added to the boundaries of the shell
        :type   buffer_length:  float
        :param  buffer_shape:   'cube', buffer_length is a cube sidelength. 'sphere', buffer_length is a sphere radius
        :type   buffer_shape:   str
        :param  return_bool:    If True, return a boolean mask instead of indices
        :type   return_bool:    boolean
        :param  cosmo:          cosmology. If None, use the snapshot cosmology
        :type   cosmo:          astropy cosmology object
        """


        if (cosmo is None):
            cosmo=self.cosmo

        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 3: # test for coords shape
            raise ValueError("coords must be a vector with shape (N, 3)")

        axis = self.beam_vec
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
            # use faster identification method
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
        Returns the diameter [Mpc] of a cone of angular radius ang_radius_deg at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  ang_radius_deg:     angular radius [deg] of the cone
        :type   ang_radius_deg:     float
        :param  cosmo:              cosmology. If given, replaces the stored cosmology
        :type   cosmo:              astropy cosmology object
        :param  use_comoving_dist:  If True, then z is passed as a comoving distance from the observer
        :type   use_comoving_dist:  boolean
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
        Returns diameter [Mpc] of the beam's cone (angular radius ang_radius_deg) at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  ang_radius_deg:     angular radius [deg] of the beam
        :type   ang_radius_deg:     float
        :param  cosmo:              cosmology. If given, replaces the stored cosmology
        :type   cosmo:              astropy cosmology object
        :param  use_comoving_dist:  If True, then z is passed as a comoving distance from the observer
        :type   use_comoving_dist:  boolean
        """
        return self._diameter_at_redshift(z, ang_radius_deg, cosmo=cosmo, use_comoving_dist=use_comoving_dist)

    def get_snapshot_reposition_coords(self, tile, snapshot_sidelengths):
        """
        Returns the coordinate offset of a snapshots geometric centre at its position on the lattice of 
        whole box sidelengths. 

        :param  tile:                   (nx, ny, n_los), the tile's lattice position along the x, y or z axis 
                                            closest to e1, e2 and beam_vec (see __init_tile_frame)
        :param  tile:                   (nx, ny, n_los). n_los, replicas along beam_vec (the line of sight). 
                                            nx, ny, replicas transverse to beam_vec, along e1 and e2, 
                                            once the beam's diameter exceeds one box sidelength
        :type   tile:                   tuple
        :param  snapshot_sidelengths:   side lengths [Mpc] of the snapshot box
        :type   snapshot_sidelengths:   np.ndarray
        """
        #nx, ny, n_los = tile
        #tile_centre_offset = (
        #    n_los * snapshot_sidelengths[0] * self.beam_vec
        #    + nx * snapshot_sidelengths[0] * self.beam_transverse_e1
        #    + ny * snapshot_sidelengths[0] * self.beam_transverse_e2
        #)
        #
        #offset=np.array([
        #        -0.5*snapshot_sidelengths[0] + tile_centre_offset[0],
        #        -0.5*snapshot_sidelengths[1] + tile_centre_offset[1],
        #        -0.5*snapshot_sidelengths[2] + tile_centre_offset[2]
        #    ])
        #return offset

        # lattice position of the tile along x, y, z
        lattice_position = self._tile_frame.T @ np.asarray(tile[:3], dtype=np.int64)
        #lattice_position = self._tile_frame.T @ np.asarray(tile, dtype=np.int64)
        snapshot_sidelengths = np.asarray(snapshot_sidelengths, dtype=float)
        return -0.5 * snapshot_sidelengths + lattice_position * snapshot_sidelengths


class SnapshotAllSky(SnapshotLightcone):
    """
    Class for constructing an all-sky lightcone from snapshots.
    """

    def __init__(self, boxsize_resolution, simulation_name, orientation_seed=0, verbose=1, orientation_lock=None):
        """
        Define the all-sky lightcone, passed to SnapshotLightcone with no beam vector.

        :param  boxsize_resolution: FLAMINGO box size and resolution label, e.g. "L1000N1800"
        :type   boxsize_resolution: str
        :param  simulation_name:    name of the simulation, e.g. "HYDRO_FIDUCIAL"
        :type   simulation_name:    str
        :param  orientation_seed:   seed used to select the rotation, reflection and periodic shift of each box tile
        :type   orientation_seed:   int
        :param  verbose:            amount printed. 0: nothing, only warnings. 1: a summary of each call. 
                                        2: details of every snapshot, box tile and file
        :type   verbose:            int
        :param  orientation_lock:           None: every box tile has its own orientation. "cube": every box tile 
                                            in a cube shell of boxes around the observer's box shares one 
                                            orientation, so each shell is periodically continuous and the 
                                            orientation only changes on the faces between shells
        :type   orientation_lock:           str
        """
        
        super().__init__(boxsize_resolution, simulation_name, beam_vector=None, orientation_seed=orientation_seed, verbose=verbose, orientation_lock=orientation_lock)

    def _tiles_intersecting_shell(self, r_min, r_max, boxsize, ang_radius_deg=None):
        """

        Every integer tile index (nx, ny, nz) whose periodic box replica intersects the spherical annulus [r_min, r_max] around
        the observer at the origin.
        Each box is centred at (nx, ny, nz)*boxsize and has a length of [n_i*boxsize - boxsize/2, n_i*boxsize + boxsize/2) along each axis. 

        Returns a list of (nx, ny, nz) int tuples.

        :param  r_min:          minimum comoving distance [Mpc]
        :type   r_min:          float
        :param  r_max:          maximum comoving distance [Mpc]
        :type   r_max:          float
        :param  boxsize:        snapshot box side length [Mpc] (box assumed cubic)
        :type   boxsize:        float
        :param  ang_radius_deg: unused, accepted for signature compatibility with SnapshotBeam
        :type   ang_radius_deg: float
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
        Returns a tuple of (indices inside the shell, indices outside the shell). If return_bool is True, 
        returns (boolean mask, None).

        :param  coords:         Cartesian comoving lightcone coordinates [Mpc], shape (N, 3)
        :type   coords:         np.ndarray
        :param  z_min:          minimum redshift of the lightcone shell
        :type   z_min:          float
        :param  z_max:          maximum redshift of the lightcone shell
        :type   z_max:          float
        :param  ang_radius_deg: unused, accepted for signature compatibility with the inherited pipeline
        :type   ang_radius_deg: float
        :param  buffer_length:  size [Mpc] of the buffer added to the boundaries of the shell
        :type   buffer_length:  float
        :param  buffer_shape:   'cube', buffer_length is a cube sidelength. 'sphere', buffer_length is a sphere radius
        :type   buffer_shape:   str
        :param  return_bool:    If True, return a boolean mask instead of indices
        :type   return_bool:    boolean
        :param  cosmo:          cosmology. If None, use the snapshot cosmology
        :type   cosmo:          astropy cosmology object
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
        Returns the coordinate offset of a snapshot box from the observer. 

        :param  tile:                   (nx, ny, nz) tile index
        :type   tile:                   tuple
        :param  snapshot_sidelengths:   side lengths [Mpc] of the snapshot box
        :type   snapshot_sidelengths:   np.ndarray
        """
        tile_arr = np.asarray(tile[:3], dtype=float)
        snapshot_sidelengths = np.asarray(snapshot_sidelengths, dtype=float)
        return -0.5 * snapshot_sidelengths + tile_arr * snapshot_sidelengths

    def _diameter_at_redshift(self, z, ang_radius_deg=None, cosmo=None, use_comoving_dist=False):
        """
        Returns the diameter [Mpc] of the past lightcone at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  ang_radius_deg:     unused, accepted for signature compatibility with SnapshotBeam
        :type   ang_radius_deg:     float
        :param  cosmo:              cosmology. If given, replaces the stored cosmology
        :type   cosmo:              astropy cosmology object
        :param  use_comoving_dist:  If True, then z is passed as a comoving distance from the observer
        :type   use_comoving_dist:  boolean
        """
        if cosmo is not None:
            self.cosmo = cosmo

        if use_comoving_dist:
            h = z
        else:
            h = unyt.unyt_array.from_astropy(self.cosmo.comoving_distance(z)).to_value("Mpc")
        return h * 2
        
    def lightcone_diameter(self, z, cosmo=None, use_comoving_dist=False):
        """
        Returns diameter [Mpc] of the past lightcone at a given redshift.

        :param  z:                  redshift, or comoving distance [Mpc] if use_comoving_dist is True
        :type   z:                  float
        :param  cosmo:              cosmology. If given, replaces the stored cosmology
        :type   cosmo:              astropy cosmology object
        :param  use_comoving_dist:  If True, then z is passed as a comoving distance from the observer
        :type   use_comoving_dist:  boolean
        """
        return self._diameter_at_redshift(z, cosmo=cosmo, use_comoving_dist=use_comoving_dist, ang_radius_deg=None)

    def gather_files(self, current_ptype, shell_z, use_snapshots=None):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.

        :param  current_ptype:  particle type to gather, e.g. "PartType1"
        :type   current_ptype:  str
        :param  shell_z:        minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   shell_z:        sequence of two floats
        :param  use_snapshots:  snapshot numbers to use instead of the ones auto-selected from shell_z
        :type   use_snapshots:  list or np.ndarray
        """
        return super().gather_files(current_ptype, shell_z, None, use_snapshots=use_snapshots)

    def place_file_in_shell(self, file_number, current_ptype, property_names):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.

        :param  file_number:    index into self.snap_file_offset_lengths
        :type   file_number:    int
        :param  current_ptype:  particle type being placed, e.g. "PartType1" 
                                    (must match the ptype passed to gather_files)
        :type   current_ptype:  str
        :param  property_names: particle properties to read, e.g. ["ParticleIDs", "Coordinates"]
        :type   property_names: list
        """
        return  super().place_file_in_shell(file_number=file_number, current_ptype=current_ptype, property_names=property_names, ang_radius_deg=None)

    def place_snapshot_particles_in_shell(self, lightcone_redshift_range, property_names,
                                           particle_types=["PartType0", "PartType1", "PartType5"],
                                           use_snapshots=None, comm=None):
        """
        Wrapper over SnapshotLightcone implementation that sets ang_radius_deg=None.

        :param  lightcone_redshift_range:   minimum and maximum redshift of the lightcone shell [z_min, z_max]
        :type   lightcone_redshift_range:   sequence of two floats
        :param  property_names:             particle properties to read, e.g. ["ParticleIDs", "Coordinates"]
        :type   property_names:             list
        :param  particle_types:             particle type(s) to place, e.g. "PartType1"
        :type   particle_types:             str or list
        :param  use_snapshots:              snapshot numbers to use instead of the ones auto-selected from the redshift range
        :type   use_snapshots:              list or np.ndarray
        :param  comm:                       MPI communicator. If None, read in serial
        :type   comm:                       mpi4py.MPI.Comm
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

        :param  halo_format:                format string for SOAP catalogue filenames (using {snap_nr})
        :type   halo_format:                str
        :param  lightcone_redshift_range:   minimum and maximum redshift of the shell to populate [z_min, z_max]
        :type   lightcone_redshift_range:   sequence of two floats
        :param  use_snapshots:              snapshot numbers to use, bypassing the usual redshift-range lookup
        :type   use_snapshots:              list or np.ndarray
        """
        return super().place_halos_in_shell(halo_format, lightcone_redshift_range, None, use_snapshots=use_snapshots)

