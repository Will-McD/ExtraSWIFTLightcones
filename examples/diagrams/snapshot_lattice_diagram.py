#!/bin/env python
#
import argparse
import tempfile
from pathlib import Path
import h5py
import numpy as np
from swiftlet.simulation_suites import Simulation
from swiftlet.snapshot_cosmology import SnapshotCosmology
from swiftlet.snapshot_lightcone import SnapshotAllSky, SnapshotBeam
from swiftlet.snapshot_orientation import write_orientation_tables
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from swiftlet.example_plot_settings import plot_settings, cmap_MCG

"""
Build fake FLAMINGO snapshots (redshift 0 - 0.425, written to a temporary directory) using a box sidelength of 1Gpc and 
placeing all particles in a regular lattice write snapshots .

Repeat this for when we do and do-not re-orient the tiles:
    - not re-oriented:  every box tile is the snapshot box as it is (no rotation, reflection or shift). To do this 
                            we build a using an orientation table of one unchanged row, resulting in the lightcone 
                            being just the initial snapshot box repeated periodically. Therefore each particle is 
                            where its periodic copy of the box puts it.
    - re-oriented:      the default orientation tables, with each orientation lock: "none", "cube" and "sphere".


A correct lightcone will replicate the lattice structure exactly and places every point of the snapshot:
    - exactly once
    - with no gaps
    - with no repeats
    - for many box tiles
    - works the same for particles and haloes
    - if there is no rotation, reflection or shift we check the exact structure is perfectly replicated

"""

############################################################################################################
# DEFINE GLOBAL VARIABLES FOR THE FAKE SNAPSHOTS

# the fake simulation: a FLAMINGO 1 Gpc box label, so its snapshots have FLAMINGO's output redshifts
BOX_RES = "L1000N0900"
SIM_NAME = "LATTICE_EXAMPLE"
SNAPSHOTS = range(69, 78)  # z = 0 to 0.425, with dz=0.05 out to about 1.65 box lengths from the observer
BOXSIZE = 1000.  # [Mpc], L1
NCELLS_AXIS = 8 # cell number per axis in box
#CELL_SIZE = BOXSIZE / NCELLS_AXIS # size of fake cell structure. 
NFILES = 2 # split the cells over this many temp files
PTYPE = "PartType1" # use fake dark matter particles
LATTICE_SPACING = 25.  # [Mpc], distance between particles in the regular lattice

# use FLAMINGO like units for fake snapshot
# units of the fake snapshots: Mpc, 1e10 Msun and Mpc / (km/s), so H0 in internal units is H0 in km/s/Mpc
MPC_CM = 3.08567758e24
UNITS_CGS = {
    "Unit length in cgs (U_L)": MPC_CM,
    "Unit mass in cgs (U_M)": 1.98841e43,
    "Unit time in cgs (U_t)": 3.08567758e19,
    "Unit temperature in cgs (U_T)": 1.0,
    "Unit current in cgs (U_I)": 1.0,
}

# place holder cosmology params 
# make a simple flat cosmology with no radiation or neutrinos
COSMOLOGY = {
    "H0 [internal units]": 68., 
    "h": 0.68, 
    "Omega_m": 0.3, 
    "Omega_lambda": 0.7, 
    "Omega_b": 0.048,
    "w_0": -1., 
    "w_a": 0., 
    "T_CMB_0 [K]": 0., 
    "N_eff": 0.
    }

# DEFINE GLOBAL VARIABLES FOR THE LIGHTCONES MAKE WITH FAKE SNAPSHOTS

# the lightcones: name, beam vector (None for all-sky), angular radius [deg] and shell [comoving Mpc]
LIGHTCONES = [
    ("all-sky", None, None, (330., 1110.)),
    ("beam along z", (0, 0, 1), 25., (110., 1610.)),
    ("beam off axis (1, 2, 3)", (1, 2, 3), 25., (110., 1610.)),
]

# set order and type of lightcones to make. 
LOCKS = [None, "cube", "sphere"] # orientation locks 

# DEFINE GLOBAL VARIABLES FOR DIFFERENT THE DIFFERENT ORIENTATION TABLES

# the columns of the figures: (orientation, orientation lock). 
# Without re-orientation every tile is the same, so the lock makes no difference and only no lock is placed
NOT_REORIENTED = "not re-oriented"
REORIENTED = "re-oriented"
CASES = [(NOT_REORIENTED, None)] + [(REORIENTED, lock) for lock in LOCKS]

# one orientation that leaves the box unchanged: no quarter turns, no reflections and no cell shift
UNCHANGED_TABLES = {"QuarterTurns": [[0, 0, 0]], "Reflections": [[1, 1, 1]], "CellShifts": [[0, 0, 0]]}

############################################################################################################

def set_unit_attrs(dset, length_exponent, a_exponent, conversion=None):
    """
    Add SWIFT unit attributes to a dataset, and the CGS conversion factor SOAP adds.

    :param  dset:               the dataset
    :type   dset:               h5py.Dataset
    :param  length_exponent:    power of length in the dataset's units
    :type   length_exponent:    int
    :param  a_exponent:         power of the scale factor in the dataset's units (1 for comoving lengths)
    :type   a_exponent:         int
    :param  conversion:         factor to physical CGS units, as SOAP writes, or None for none
    :type   conversion:         float
    """
    for symbol in "ILMTt":
        dset.attrs[f"U_{symbol} exponent"] = [float(length_exponent if symbol == "L" else 0)]
    dset.attrs["a-scale exponent"] = [float(a_exponent)]
    dset.attrs["h-scale exponent"] = [0.]
    if conversion is not None:
        dset.attrs["Conversion factor to physical CGS (including cosmological corrections)"] = [conversion]


def write_metadata(f, redshift):
    """
    Write the physical constants, cosmology and units, as SWIFT snapshots and SOAP catalogues hold them.

    :param  f:          the open file
    :type   f:          h5py.File
    :param  redshift:   redshift of the snapshot
    :type   redshift:   float
    """
    constants = f.create_group("PhysicalConstants/CGS")
    constants.attrs["parsec"] = [3.08567758e18]
    constants.attrs["solar_mass"] = [1.98841e33]
    constants.attrs["newton_G"] = [6.674e-8]
    cosmology = f.create_group("Cosmology")
    cosmology.attrs["Redshift"] = [redshift]
    cosmology.attrs["Scale-factor"] = [1. / (1. + redshift)]
    for name, value in COSMOLOGY.items():
        cosmology.attrs[name] = [value]
    for group_name in ("Units", "InternalCodeUnits"):
        group = f.create_group(group_name)
        for name, value in UNITS_CGS.items():
            group.attrs[name] = [value]


def lattice_positions(boxsize=BOXSIZE, spacing=LATTICE_SPACING):
    """
    Positions [Mpc] of the uniform lattice filling the box, offset by half a spacing from the box 
    faces: the particle (and halo) with ID i is at row i.

    Returns positions of lattice points in a fake snapshot
    
    :param  boxsize:    side length [Mpc] of the box
    :type   boxsize:    float
    :param  spacing:    spacing [Mpc] of the lattice, which must divide the box
    :type   spacing:    float
    
    """
    
    n = int(round(boxsize / spacing))
    return (np.indices((n,) * 3).reshape(3, -1).T + 0.5) * spacing


def write_snapshot(snapshot_file_format, snap_nr, redshift, ids, pos, boxsize=BOXSIZE, ncells_axis=NCELLS_AXIS, nfiles=NFILES):
    """
    Write a fake snapshot of particles split over files by cell and include SWIFT style cell metadata.

    :param  snapshot_file_format:   path of a snapshot file, with {snap_nr} and {file_nr} fields
    :type   snapshot_file_format:   str
    :param  snap_nr:                snapshot number
    :type   snap_nr:                int
    :param  redshift:               redshift of the snapshot
    :type   redshift:               float
    :param  ids:                    particle IDs
    :type   ids:                    np.ndarray
    :param  pos:                    particle positions [Mpc], shape (N, 3)
    :type   pos:                    np.ndarray
    :param  boxsize:                side length [Mpc] of the box
    :type   boxsize:                float
    :param  ncells_axis:            number of SWIFT cells along each axis
    :type   ncells_axis:            int
    :param  nfiles:                 number of files of the snapshot
    :type   nfiles:                 int
    """
    
    # set up cell structure: 

    # get side length of cells
    cell_size = boxsize / ncells_axis
    # get cell centres in snapshot
    centres = (np.indices((ncells_axis,) * 3).reshape(3, -1).T + 0.5) * cell_size
    
    # total  numb cells in snapshot 
    ncells = len(centres)
    
    # file number cell is written too
    cell_file = np.arange(ncells) % nfiles
    
    # give each cell an ID
    cell_idx = np.floor(pos / cell_size).astype(int)
    
    cell = (cell_idx[:, 0] * ncells_axis + cell_idx[:, 1]) * ncells_axis + cell_idx[:, 2]

    counts = np.bincount(cell, minlength=ncells)

    # offset of each cell within its file; every file holds the offsets of all cells
    offsets = np.zeros(ncells, dtype=int)
    for file_nr in range(nfiles):
        file_cells = np.flatnonzero(cell_file == file_nr)
        offsets[file_cells] = np.cumsum(counts[file_cells]) - counts[file_cells]

    for file_nr in range(nfiles):
        file_cells = np.flatnonzero(cell_file == file_nr)
        in_file = np.concatenate([np.flatnonzero(cell == c) for c in file_cells])
        filename = Path(snapshot_file_format.format(snap_nr=snap_nr, file_nr=file_nr))
        filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filename, "w") as f:
            npart_file = np.zeros(7, dtype=int)
            npart_file[1] = len(in_file)
            npart_total = np.zeros(7, dtype=int)
            npart_total[1] = len(pos)
            header = f.create_group("Header")
            header.attrs["BoxSize"] = [boxsize] * 3
            header.attrs["NumFilesPerSnapshot"] = [nfiles]
            header.attrs["NumPart_ThisFile"] = npart_file
            header.attrs["NumPart_Total"] = npart_total

            cells = f.create_group("Cells/Meta-data")
            cells.attrs["nr_cells"] = [ncells]
            cells.attrs["dimension"] = [ncells_axis] * 3
            cells.attrs["size"] = [cell_size] * 3
            f["Cells/Centres"] = centres
            f[f"Cells/Files/{PTYPE}"] = cell_file
            f[f"Cells/OffsetsInFile/{PTYPE}"] = offsets
            f[f"Cells/Counts/{PTYPE}"] = counts

            write_metadata(f, redshift)

            dm = f.create_group(PTYPE)
            set_unit_attrs(dm.create_dataset("Coordinates", data=pos[in_file]), 1, 1)
            set_unit_attrs(dm.create_dataset("ParticleIDs", data=ids[in_file]), 0, 0)


def write_soap_catalogue(filename, redshift, halo_index, halo_centre):
    """
    Write a fake SOAP catalogue containing the coordinates of the fake 
    halo centres ("InputHalos/HaloCentre") [cMpc] and their catalogue indices ("InputHalos/HaloCatalogueIndex").

    :param  filename:       path of the catalogue
    :type   filename:       str
    :param  redshift:       redshift of the snapshot
    :type   redshift:       float
    :param  halo_index:     catalogue index of each halo
    :type   halo_index:     np.ndarray
    :param  halo_centre:    halo centres [comoving Mpc], shape (N, 3)
    :type   halo_centre:    np.ndarray
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as f:
        write_metadata(f, redshift)
        set_unit_attrs(f.create_dataset("InputHalos/HaloCentre", data=halo_centre), 1, 1,
                       conversion=MPC_CM / (1. + redshift))
        set_unit_attrs(f.create_dataset("InputHalos/HaloCatalogueIndex", data=halo_index), 0, 0, conversion=1.)


def write_lattice_simulation(run_dir, box_res=BOX_RES, sim_name=SIM_NAME, snapshots=SNAPSHOTS, boxsize=BOXSIZE,ncells_axis=NCELLS_AXIS, spacing=LATTICE_SPACING):
    """
    Write snapshots for the fake simulation. 
    Use the same regular lattice in every snapshot. Haloes and particles are at each lattice point whose catalogue index is the particle's ID.

    Returns the Simulation and the SOAP catalogue filename format, with a {snap_nr} field.


    :param  run_dir:        directory to write the simulation to, as <run_dir>/<box_res>/<sim_name>
    :type   run_dir:        str or pathlib.Path
    :param  box_res:        box size and resolution label, which sets the suite and its output redshifts, e.g.
                                FLAMINGO "L1000N0900" or COLIBRE "L0025N0376"
    :type   box_res:        str
    :param  sim_name:       name of the simulation
    :type   sim_name:       str
    :param  snapshots:      snapshot numbers to write
    :type   snapshots:      iterable of int
    :param  boxsize:        side length [Mpc] of the box
    :type   boxsize:        float
    :param  ncells_axis:    number of SWIFT cells along each axis
    :type   ncells_axis:    int
    :param  spacing:        spacing [Mpc] of the lattice, which must divide the box
    :type   spacing:        float

    """
    # a fake simulation stored in run_dir
    simulation = Simulation.from_label(box_res, sim_name, base_dir_format=f"{run_dir}/{{box_res}}/{{sim_name}}")
    halo_format = f"{simulation.directory}/SOAP/halo_properties_{{snap_nr:04d}}.hdf5"

    pos = lattice_positions(boxsize, spacing)
    ids = np.arange(len(pos))
    for snap_nr in snapshots:
        redshift = simulation.snapshot_redshift(snap_nr)
        write_snapshot(simulation.snapshot_format, snap_nr, redshift, ids, pos, boxsize, ncells_axis)
        write_soap_catalogue(halo_format.format(snap_nr=snap_nr), redshift, ids, pos)
    return simulation, halo_format


def write_unchanged_orientation_table(run_dir):
    """
    Write an orientation table of one row that leaves every box tile unchanged: no rotation, reflection or cell shift.

    Returns the path of the table file.

    :param  run_dir:    directory to write the table to
    :type   run_dir:    str or pathlib.Path
    """
    filename = Path(run_dir) / "unchanged_orientation_table.yaml"
    write_orientation_tables(filename, tables=UNCHANGED_TABLES)
    return str(filename)


def new_lightcone(simulation, beam_vector, orientation_lock, orientation_table=None):
    """
    Return the SnapshotLightcone object, 
        SnapshotBeam if beam_vector is given
        SnapshotAllSky if beam_vector is None

    :param  simulation:         the simulation
    :type   simulation:         Simulation
    :param  beam_vector:        direction of the beam, or None for all-sky
    :type   beam_vector:        tuple
    :param  orientation_lock:   orientation lock of the box tiles: None, "cube" or "sphere"
    :type   orientation_lock:   str
    :param  orientation_table:  file of orientation tables, or None for the default tables
    :type   orientation_table:  str
    """
    
    if beam_vector is None:
                return SnapshotAllSky(simulation_object=simulation, orientation_lock=orientation_lock, orientation_table=orientation_table, verbose=0)
    
    return SnapshotBeam(simulation_object=simulation, beam_vector=beam_vector, orientation_lock=orientation_lock,orientation_table=orientation_table, verbose=0)


def redshift_at_distance(cosmo, r):
    """
    Redshift at comoving distances [Mpc] from the observer, interpolated on a fine grid.

    :param  cosmo:  cosmology of the simulation
    :type   cosmo:  SnapshotCosmology
    :param  r:      comoving distances [Mpc]
    :type   r:      float or np.ndarray
    """
    z_grid = np.linspace(0, 0.5, 20001)
    r_grid = cosmo.COSMO.comoving_distance(z_grid).to_value("Mpc")
    return np.interp(r, r_grid, z_grid)


def place_lattice(lightcone, redshift_range, ang_radius_deg, kind, halo_format):
    """
    Place the input lattice in a lightcone shell.
    Use kind to select between reading particles from the fake snapshots or haloes from 
    the fake SOAP catalogue.  

    All outputs are sorted by snapshot number and ID

    Returns (snapshot number, ID, position [Mpc] relative to the observer),

    :param  lightcone:      the lightcone
    :type   lightcone:      SnapshotAllSky or SnapshotBeam
    :param  redshift_range: minimum and maximum redshift of the shell
    :type   redshift_range: tuple
    :param  ang_radius_deg: angular radius [deg] of a beam, None for all-sky
    :type   ang_radius_deg: float
    :param  kind:           "particles" or "haloes"
    :type   kind:           str
    :param  halo_format:    SOAP catalogue filename format, with a {snap_nr} field
    :type   halo_format:    str
    """
    
    beam_args = {} if ang_radius_deg is None else {"ang_radius_deg": ang_radius_deg}
    if kind == "particles":
        particles = lightcone.place_snapshot_particles_in_shell(redshift_range, property_names=["ParticleIDs", "Coordinates"],particle_types=PTYPE, **beam_args)[PTYPE]
        snap_nr, ids, coords = particles["SnapshotNumber"], particles["ParticleIDs"], particles["Coordinates"]
    else:
        haloes = lightcone.place_halos_in_shell(halo_format, redshift_range, **beam_args)
        snap_nr = haloes["Lightcone/SnapshotNumber"]
        ids = haloes["InputHalos/HaloCatalogueIndex"]
        coords = haloes["Lightcone/HaloCentre"]
    snap_nr = np.asarray(snap_nr, dtype=int)
    ids = np.asarray(ids, dtype=int)
    coords = np.asarray(coords.to_value("Mpc"), dtype=float).reshape(-1, 3)
    order = np.lexsort((ids, snap_nr))
    return snap_nr[order], ids[order], coords[order]


def lattice_in_shell(r_min, r_max, beam_vector, ang_radius_deg, boxsize=BOXSIZE, spacing=LATTICE_SPACING):
    """
    Identify the expected points of the lattice that fit within the lightcones shell. 
    Build a mask for all points in the lattice, that are between r_min and r_max [Mpc] from the 
    observer and in the beam: (k + 1/2) * spacing - L/2 along each axis, for every integer k. Where L 
    is the snapshot sidelength.

    The observer is always placed at the centre of the snapshot box, so this is the lattice the lightcone should hold, each point once.

    Returns coordinates 

    :param  r_min:          inner comoving distance [Mpc] of the shell
    :type   r_min:          float
    :param  r_max:          outer comoving distance [Mpc] of the shell
    :type   r_max:          float
    :param  beam_vector:    direction of the beam, or None for all-sky
    :type   beam_vector:    tuple
    :param  ang_radius_deg: angular radius [deg] of the beam
    :type   ang_radius_deg: float
    :param  boxsize:        side length [Mpc] of the box
    :type   boxsize:        float
    :param  spacing:        spacing [Mpc] of the lattice
    :type   spacing:        float
    """
    
    # get max num of replicas on each axis
    k_max = int(np.ceil((r_max + 0.5 * boxsize) / spacing)) + 1
    axis = (np.arange(-k_max, k_max + 1) + 0.5) * spacing - 0.5 * boxsize
    axis = axis[np.abs(axis) <= r_max + spacing]

    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    r = np.linalg.norm(points, axis=1)
    keep = (r >= r_min) & (r <= r_max)
    if beam_vector is not None:
        beam_vec = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
        keep[keep] = points[keep] @ beam_vec / r[keep] >= np.cos(np.deg2rad(ang_radius_deg))
    return points[keep]


def lattice_check(coords, expected, boxsize=BOXSIZE, spacing=LATTICE_SPACING):
    """
    Compare the points placed with the lattice points expected in the shell and then identify
    lattice nodes that are 1) missing from the lattice or 2) incorrectly placed. 
    
        - Missing:              particle or halo is expected to be in the lightcone but not found in the lightcone 
        - Incorrectly placed:   particles or haloes that are: 
                                    A) expected to be in the lightcone but not placed in 
                                        the lightcone 
                                    B) not in the correct lattice structure
                                    C) appears more times than allowed by the orientation lock. For 'none' and 'sphere', 
                                        each  point in the lattice (particle or halo for a single snapshot number) 
                                        from a single snapshot can only appear as many times as that snapshot is repeated. 
                                        For 'cube', each lattice point of a snapshot can be repeated for the number 
                                        of tiles in each layer. 
                                       
    
    Returns a tuple (missing, incorrectly placed)

    :param  coords:     positions [Mpc] placed
    :type   coords:     np.ndarray
    :param  expected:   expected lattice positions [Mpc] 
    :type   expected:   np.ndarray
    :param  boxsize:    side length [Mpc] of the box
    :type   boxsize:    float
    :param  spacing:    spacing [Mpc] of the lattice
    :type   spacing:    float
    """
    
    def lattice_keys(points):
        # integer index (k_x, k_y, k_z) of each point on the lattice, and whether it is on the lattice
        k = np.rint((points + 0.5 * boxsize) / spacing - 0.5).astype(int)
        on_lattice = np.all(np.abs(points - ((k + 0.5) * spacing - 0.5 * boxsize)) < 1e-6, axis=1)
        return [tuple(key) for key in k], on_lattice

    # get key for lattice and if points are placed on the lattice: 
    # for points placed into the lattice 
    placed_keys, placed_on_lattice = lattice_keys(coords) 
    
    # for the expected points 
    expected_keys, _ = lattice_keys(expected) 


    counts = {}
    for key in placed_keys:
        counts[key] = counts.get(key, 0) + 1
    expected_set = set(expected_keys)

    # find points missing from lightcone
    missing = []
    for point, key in zip(expected, expected_keys):
        if key not in counts:
            missing.append(point)
    
    # find extra points in the lightcone
    extra = {}
    for point, key, on_lattice in zip(coords, placed_keys, placed_on_lattice):
        if counts[key] > 1 or key not in expected_set or not on_lattice:
            extra[key] = point
    
    return np.array(missing).reshape(-1, 3), np.array(list(extra.values())).reshape(-1, 3)


def not_reoriented_check(ids, coords, boxsize=BOXSIZE, spacing=LATTICE_SPACING):
    """
    Check the points of a lightcone that is not re-oriented. 
    Every tile is the snapshot box as it is, so a point placed in the lightcone must be the particle (or halo) whose position in the box 
    is the point wrapped back into the box.

    Returns the number of points carrying the wrong ID.

    :param  ids:        IDs of the points placed
    :type   ids:        np.ndarray
    :param  coords:     positions [Mpc] placed, relative to the observer
    :type   coords:     np.ndarray
    :param  boxsize:    side length [Mpc] of the box
    :type   boxsize:    float
    :param  spacing:    spacing [Mpc] of the lattice
    :type   spacing:    float
    """

    box_positions = lattice_positions(boxsize, spacing)[ids]
    wrapped = np.mod(coords + 0.5 * boxsize, boxsize)
    return int(np.sum(np.any(np.abs(wrapped - box_positions) > 1e-6, axis=1)))


def snapshot_handovers(simulation, cosmo, r_range):
    """
    Comoving distances [Mpc] inside the shell where one snapshot's redshift range ends and the next begins.

    :param  simulation: the simulation
    :type   simulation: Simulation
    :param  cosmo:      cosmology of the simulation
    :type   cosmo:      SnapshotCosmology
    :param  r_range:    inner and outer comoving distance [Mpc] of the shell
    :type   r_range:    tuple
    """
    edges_z = set()
    for snap_nr in SNAPSHOTS:
        edges_z.update(float(z) for z in simulation.snapshot_redshift_range(snap_nr))
    distances = cosmo.COSMO.comoving_distance(sorted(edges_z)).to_value("Mpc")
    return [r for r in distances if r_range[0] < r < r_range[1]]


def place_cases(simulation, cosmo, halo_format, unchanged_table):
    
    """
    Run through all scenarios and place the lattice in every lightcone, 
    with every orientation lock, as both particles and haloes.

    Returns dict {(lightcone name, lock, kind): (snapshot number, ID, position [Mpc])}.

    :param  simulation:         the fake lattice simulation
    :type   simulation:         Simulation
    :param  cosmo:              its cosmology
    :type   cosmo:              SnapshotCosmology
    :param  halo_format:        its SOAP catalogue filename format
    :type   halo_format:        str
    :param  unchanged_table:    file of the orientation table that leaves the tiles unchanged
    :type   unchanged_table:    str
    """

    # store placed lattice points
    placed = {}
    for name, beam_vector, ang_radius_deg, r_range in LIGHTCONES:
        redshift_range = tuple(redshift_at_distance(cosmo, np.array(r_range)))
        
        # iterate through oreintation tables & locks 
        for case in CASES:

            orientation, lock = case
            table = unchanged_table if orientation == NOT_REORIENTED else None
            
            # repeat for particles and haloes
            for kind in ["particles", "haloes"]:
                print(f"placing the lattice as {kind}: {name}, {orientation}, orientation_lock={lock}", flush=True)
                lightcone = new_lightcone(simulation, beam_vector, lock, orientation_table=table)
                placed[name, case, kind] = place_lattice(lightcone, redshift_range, ang_radius_deg, kind, halo_format)

    return placed


def plot_lattice_layer(placed, kind, simulation, cosmo, filename):
    """
    Create a plot showing one layer of the lattice in every lightcone type (full-sky and beam; rows) and 
    each orientation lock (columns), for both particles or haloes.

    Colour each lattice point shown by its initial distance from the corner of the snapshotbox, highlighting
    how each cell & therefore lattice points have been moved from the intal snapshot structure. 

    Returns the number of lattice points missing or repeated, over every lightcone and lock.

    :param  placed:     the points placed, see place_cases
    :type   placed:     dict
    :param  kind:       "particles" or "haloes"
    :type   kind:       str
    :param  simulation: the fake lattice simulation
    :type   simulation: Simulation
    :param  cosmo:      its cosmology
    :type   cosmo:      SnapshotCosmology
    :param  filename:   path of the figure
    :type   filename:   str or pathlib.Path
    """

    # set number of columns and rows as numb of lightcone types and locks to use
    nrow = len(LIGHTCONES)
    ncol = len(CASES)
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.21*ncol, 3.5*nrow), sharey="row", squeeze=False)
    
    # get layer of lattice
    layer = 0.5 * LATTICE_SPACING
    theta = np.linspace(0, 2 * np.pi, 721)
    
    # distance of each particle of the box from the corner (0, 0, 0) of the box
    corner_distance = np.linalg.norm(lattice_positions(), axis=1)
    max_corner_distance = np.sqrt(3.) * BOXSIZE
    
    total_problems = 0

    for row, (name, beam_vector, ang_radius_deg, r_range) in enumerate(LIGHTCONES):
        
        expected_all = lattice_in_shell(*r_range, beam_vector, ang_radius_deg)
        # the layer of lattice points shown (the closest to the observer across the beam), and the two axes in it
        normal = 2 if beam_vector is None else int(np.argmin(np.abs(beam_vector)))
        u, v = [axis for axis in range(3) if axis != normal]

        # find points in the layers
        def in_layer(points):
            return np.abs(points[:, normal] - layer) < 1e-3
            

        expected = expected_all[in_layer(expected_all)]

        pad = 2 * LATTICE_SPACING
        limits = [(expected[:, axis].min() - pad, expected[:, axis].max() + pad) for axis in (u, v)]
        
        #get comoving distances for each snapshots redshift range
        handovers = snapshot_handovers(simulation, cosmo, r_range) 

        for col, case in enumerate(CASES):
            
            ax = axes[row, col]
            
            orientation, lock = case # get specific table and lock
            
            snap_nr, ids, coords = placed[name, case, kind] # recall the placed particles coords, snapshots and ids

            missing_all, extra_all = lattice_check(coords, expected_all)
            total_problems += len(missing_all) + len(extra_all)
            tiles = len(np.unique(np.floor(coords / BOXSIZE + 0.5).astype(int), axis=0))
            
            # find particles visible in this layer 
            shown = in_layer(coords) 
        
            # get flagged particles: missing, extra/incorrect
            missing, extra = missing_all[in_layer(missing_all)], extra_all[in_layer(extra_all)] 

            # box tile edges at (n + 1/2) box lengths
            for axis_limits, draw in ((limits[0], ax.axvline), (limits[1], ax.axhline)):
                first_edge = int(np.floor(axis_limits[0] / BOXSIZE - 0.5))
                last_edge = int(np.ceil(axis_limits[1] / BOXSIZE + 0.5))
                for n in range(first_edge, last_edge + 1):
                    draw((n + 0.5) * BOXSIZE, color="darkgray", lw=1.0, zorder=1)
            
            # overlay the shell (solid) and snapshot edges (dashed) for the layer
            circles = [(r_range[0], False), (r_range[1], False)]
            for r in handovers:
                circles.append((r, True))
            for r, dashed in circles:
                if r > layer:
                    radius = np.sqrt(r**2 - layer**2)
                    edge_ls = "--" if dashed else "-"
                    ls_col = "darkslategray" if dashed else "darkgray"
                    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=ls_col, ls=edge_ls, lw=0.8, zorder=2)

            # plot lattice points in the layer
            # colour each particle based on distance from initial snapshot (0,0,0) coords
            ax.scatter(
                coords[shown, u], coords[shown, v], 
                s=3.5 if beam_vector is None else 5.0, 
                c=corner_distance[ids[shown]], 
                cmap=cmap_MCG, 
                vmin=0., vmax=max_corner_distance, 
                edgecolors='black', 
                linewidths=0.175,
                zorder=3,
                rasterized=True
            )

            # plot errors if they are in the layer 
            if len(missing):
                ax.scatter(missing[:, u], missing[:, v], s=40, marker="x", color="indianred", linewidths=1.4, zorder=4)
            if len(extra):
                ax.scatter(extra[:, u], extra[:, v], s=40, facecolors="none", edgecolors="darksalmon", linewidths=1.4,
                           zorder=4)

            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_aspect("equal")
            ax.tick_params(colors="darkslategray", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("lightgray")
            if row == len(LIGHTCONES) - 1:
                ax.set_xlabel(f"{'xyz'[u]} [Mpc]", color="darkslategray", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{name}\n\n{'xyz'[v]} [Mpc]", color="k", fontsize=10)
            if row == 0:
                title = orientation if orientation == NOT_REORIENTED else f"{orientation}\norientation_lock = {lock}"
                ax.set_title(f"{title}\n", color="k", fontsize=11, fontweight="bold")
            
            # overlay summary showing number of points, tiles, snapshots used  
            txt_offset=0.025
            ax.text(
                1. - txt_offset, 1. - txt_offset, 
                f"points: {len(coords):,}\ntiles: {tiles}\nsnapshots: {len(np.unique(snap_nr))}" , 
                color="k", 
                fontsize=8, va="top", ha="right",
                bbox=dict(
                    facecolor="white",
                    edgecolor="k",
                    linewidth=0.8,alpha=0.7,
                    boxstyle="square,pad=0.3",
                    ),
                transform=ax.transAxes, 
                zorder=10,
                )

    handles = [
        #Line2D([], [], ls="", marker="o", color="k", markersize=3, label=f"lattice points (one layer)"),
        Line2D([], [], ls="", marker="x", color="indianred", markersize=7, markeredgewidth=1.4, label="missing lattice point"),
        Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor="darksalmon", markersize=7, markeredgewidth=1.4, label="incorrect lattice point"),
        Line2D([], [], color="darkgray", lw=1.0, label="tile edges"),
        Line2D([], [], color="darkslategray", lw=0.8, ls="--", label="snapshot edges"),
    ]
    
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.06))
    fig.suptitle(f"{kind} lattice", fontsize=13, color="k", y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.98))
    
    # colour bar of where the particles sit in the snapshot box
    colour_ax = fig.add_axes([0.3, 0.045, 0.4, 0.009])
    colour_bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0., max_corner_distance), cmap=cmap_MCG), cax=colour_ax, orientation="horizontal")
    colour_bar.set_label("distance of the particle from the (0, 0, 0) in the snapshot box [Mpc]", fontsize=9)
    colour_bar.ax.tick_params(labelsize=8)
    fig.savefig(filename, dpi=250)
    plt.close(fig)
    return total_problems


def haloes_match_particles(placed):
    """
    Check the haloes are placed exactly where the particles are, in every lightcone and lock, and print the result.

    Returns True if they all match.

    :param  placed: the points placed, see place_cases
    :type   placed: dict
    """
    all_match = True
    for name, _, _, _ in LIGHTCONES:
        for case in CASES:
            particles, haloes = placed[name, case, "particles"], placed[name, case, "haloes"]
            same = (len(particles[2]) == len(haloes[2]) and np.array_equal(particles[0], haloes[0])
                    and np.array_equal(particles[1], haloes[1]) and np.allclose(particles[2], haloes[2], atol=1e-6))
            all_match = all_match and same
            print(f"{name},  {case[0]}, orientation_lock={case[1]}: haloes {'match' if same else 'DO NOT match'} the particles")
    return all_match


def not_reoriented_ids_correct(placed):
    """
    Check every particle and halo of the lightcones that are not re-oriented carries the ID of the point of the
    box it was copied from.

    Returns the number of points carrying the wrong ID.

    :param  placed: the points placed, see place_cases
    :type   placed: dict
    """
    wrong = 0
    for name, _, _, _ in LIGHTCONES:
        for kind in ["particles", "haloes"]:
            _, ids, coords = placed[name, (NOT_REORIENTED, None), kind]
            nr_wrong = not_reoriented_check(ids, coords)
            wrong += nr_wrong
            print(f"{name}, {NOT_REORIENTED}, {kind}: {len(ids) - nr_wrong:,} of {len(ids):,} points carry the ID of their periodic copy of the box"+ ("" if nr_wrong == 0 else f", {nr_wrong} WRONG"))
    return wrong


def run(run_dir, output_dir):
    """
    Write the fake lattice simulation to run_dir, place it in every lightcone and lock, and plot the results.

    Returns the number of lattice points missing or repeated, over every case, particles and haloes.

    :param  run_dir:    directory to write the fake simulation to
    :type   run_dir:    str or pathlib.Path
    :param  output_dir: directory to save the figures to
    :type   output_dir: str or pathlib.Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"writing the fake lattice simulation to {run_dir}", flush=True)
    
    unchanged_table = write_unchanged_orientation_table(run_dir) # write 1 line table for no changes
    simulation, halo_format = write_lattice_simulation(run_dir) # write fake snapshots 
    cosmo = SnapshotCosmology.from_simulation(simulation) # get fak snapshot cosmology
    
    placed = place_cases(simulation, cosmo, halo_format, unchanged_table)
    problems = 0 if haloes_match_particles(placed) else 1
    problems += not_reoriented_ids_correct(placed)
    
    # use default orientation table
    for kind in ["particles", "haloes"]:
        filename = output_dir / f"lattice_{kind}.png"
        kind_problems = plot_lattice_layer(placed, kind, simulation, cosmo, filename)
        problems += kind_problems
        print(f"{kind}: {'no' if kind_problems == 0 else kind_problems} missing or repeated lattice points, figure saved as {filename}")

    # repeat with table that does not allow for reorientation of new tiles 

    return problems


if __name__ == "__main__":

    output_dir="example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_settings() 

    with tempfile.TemporaryDirectory() as run_dir:
        problems = run(run_dir, output_dir)

    print("every lattice point placed exactly once" if problems == 0 else f"{problems} problems found")