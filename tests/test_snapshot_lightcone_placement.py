#!/bin/env python
"""
Tests of where SnapshotLightcone places objects, using small fake snapshots and SOAP catalogues
written to a temporary directory. 

No simulation data is required.

- Haloes placed from a SOAP catalogue (place_halos_in_shell) end up exactly where a particle at the
  same position in the snapshot is placed (place_snapshot_particles_in_shell).
- Particles on a uniform lattice across the whole snapshot, placed in a shell reaching just under
  half a box length from the observer, end up on the same lattice centred on the observer, each once.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import unyt

h5py = pytest.importorskip("h5py")
astropy_cosmology = pytest.importorskip("astropy.cosmology")
snapshot_lightcone = pytest.importorskip("swiftlet.snapshot_lightcone")
from swiftlet import swift_snapshot_redshift_conversion as nz

BOX_RES = "L1000N0900"
SNAPSHOTS = range(69, 78)  # z = 0 to 0.425, out to about 1.65 box lengths from the observer
BOXSIZE = 1000.
NCELLS_AXIS = 10
CELL_SIZE = BOXSIZE / NCELLS_AXIS
NFILES = 2
PTYPE = "PartType1"
MPC_CM = 3.08567758e24
UNITS_CGS = {
    "Unit length in cgs (U_L)": MPC_CM,
    "Unit mass in cgs (U_M)": 1.98841e43,
    "Unit time in cgs (U_t)": 3.08567758e19,
    "Unit temperature in cgs (U_T)": 1.0,
    "Unit current in cgs (U_I)": 1.0,
}

# haloes, with a particle at the centre of each
HALO_SIM = "HALO_TEST"
NHALOS = 4000
NEDGE = 400  # the first haloes of each snapshot sit exactly on cell faces or at the box origin

# uniform lattice of particles, offset by half a spacing from the box faces
LATTICE_SIM = "LATTICE_TEST"
LATTICE_SPACING = 25.
R_MIN, R_MAX = 20., 0.5 * BOXSIZE - 1.  # shell just inside half a box length from the observer


# tools for building fake SWIFT datasets: 

def set_unit_attrs(dset, length_exponent, a_exponent, conversion=None):
    """
    Add SWIFT unit attributes to a dataset, and the CGS conversion factor SOAP adds.
    """
    for symbol in "ILMTt":
        dset.attrs[f"U_{symbol} exponent"] = [float(length_exponent if symbol == "L" else 0)]
    dset.attrs["a-scale exponent"] = [float(a_exponent)]
    dset.attrs["h-scale exponent"] = [0.]
    if conversion is not None:
        dset.attrs["Conversion factor to physical CGS (including cosmological corrections)"] = [conversion]


def write_metadata(f, redshift):
    """
    Physical constants, cosmology and units, as written to SWIFT snapshots and SOAP catalogues.
    """
    constants = f.create_group("PhysicalConstants/CGS")
    constants.attrs["parsec"] = [3.08567758e18]
    constants.attrs["solar_mass"] = [1.98841e33]
    constants.attrs["newton_G"] = [6.674e-8]
    cosmology = f.create_group("Cosmology")
    cosmology.attrs["Redshift"] = [redshift]
    cosmology.attrs["Scale-factor"] = [1. / (1. + redshift)]
    cosmology.attrs["h"] = [0.68]
    for group_name in ("Units", "InternalCodeUnits"):
        group = f.create_group(group_name)
        for name, value in UNITS_CGS.items():
            group.attrs[name] = [value]


def write_snapshot(snap_dir, snap_nr, ids, pos):
    """
    Write one fake snapshot of dark matter particles, split over files by cell.
    """
    redshift = nz.snapshot_number_redshifts(snap_nr, BOX_RES)
    centres = (np.indices((NCELLS_AXIS,)*3).reshape(3, -1).T + 0.5) * CELL_SIZE
    ncells = len(centres)
    cell_file = np.arange(ncells) % NFILES

    cell_idx = np.floor(pos / CELL_SIZE).astype(int)
    cell = (cell_idx[:, 0] * NCELLS_AXIS + cell_idx[:, 1]) * NCELLS_AXIS + cell_idx[:, 2]
    counts = np.bincount(cell, minlength=ncells)

    # offset of each cell within its file, every file holds the offsets of all cells
    offsets = np.zeros(ncells, dtype=int)
    for file_nr in range(NFILES):
        file_cells = np.flatnonzero(cell_file == file_nr)
        offsets[file_cells] = np.cumsum(counts[file_cells]) - counts[file_cells]

    snap_dir.mkdir(parents=True)
    for file_nr in range(NFILES):
        file_cells = np.flatnonzero(cell_file == file_nr)
        in_file = np.concatenate([np.flatnonzero(cell == c) for c in file_cells])

        with h5py.File(snap_dir / f"flamingo_{snap_nr:04d}.{file_nr}.hdf5", "w") as f:
            npart_file = np.zeros(7, dtype=int)
            npart_file[1] = len(in_file)
            npart_total = np.zeros(7, dtype=int)
            npart_total[1] = len(pos)
            header = f.create_group("Header")
            header.attrs["BoxSize"] = [BOXSIZE] * 3
            header.attrs["NumFilesPerSnapshot"] = [NFILES]
            header.attrs["NumPart_ThisFile"] = npart_file
            header.attrs["NumPart_Total"] = npart_total

            cells = f.create_group("Cells/Meta-data")
            cells.attrs["nr_cells"] = [ncells]
            cells.attrs["dimension"] = [NCELLS_AXIS] * 3
            cells.attrs["size"] = [CELL_SIZE] * 3
            f["Cells/Centres"] = centres
            f[f"Cells/Files/{PTYPE}"] = cell_file
            f[f"Cells/OffsetsInFile/{PTYPE}"] = offsets
            f[f"Cells/Counts/{PTYPE}"] = counts

            write_metadata(f, redshift)

            dm = f.create_group(PTYPE)
            set_unit_attrs(dm.create_dataset("Coordinates", data=pos[in_file]), 1, 1)
            set_unit_attrs(dm.create_dataset("ParticleIDs", data=ids[in_file]), 0, 0)


def write_soap_catalogue(filename, snap_nr, halo_index, halo_centre):
    """
    Write a fake SOAP catalogue holding the halo centres [comoving Mpc] and their catalogue indices,
    read with lightcone_io's SOAPCatalogue.
    """
    redshift = nz.snapshot_number_redshifts(snap_nr, BOX_RES)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as f:
        write_metadata(f, redshift)
        set_unit_attrs(f.create_dataset("InputHalos/HaloCentre", data=halo_centre), 1, 1,
                       conversion=MPC_CM / (1. + redshift))
        set_unit_attrs(f.create_dataset("InputHalos/HaloCatalogueIndex", data=halo_index), 0, 0, conversion=1.)


def halo_positions(rng):
    """
    Random halo centres [Mpc] in the box, with some exactly on cell faces and at the origin of the box,
    where a halo and a particle could be put in different cells.
    """
    pos = rng.uniform(0, BOXSIZE, (NHALOS, 3))
    pos[:NEDGE // 2] = rng.integers(0, NCELLS_AXIS, (NEDGE // 2, 3)) * CELL_SIZE
    pos[NEDGE // 2, :] = 0.
    pos[NEDGE // 2 + 1:NEDGE, 0] = rng.integers(0, NCELLS_AXIS, NEDGE - NEDGE // 2 - 1) * CELL_SIZE
    return pos


def redshift_at_distance(r):
    """
    Redshift at comoving distances r [Mpc] from the observer, interpolated on a fine grid.
    """
    z_grid = np.linspace(0, 0.5, 20001)
    r_grid = Cosmology.COSMO.comoving_distance(z_grid).to_value("Mpc")
    return np.interp(r, r_grid, z_grid)


def lattice_positions():
    """
    Positions [Mpc] of a uniform lattice filling the box, offset by half a spacing from the box faces.
    """
    n = int(round(BOXSIZE / LATTICE_SPACING))
    return (np.indices((n,)*3).reshape(3, -1).T + 0.5) * LATTICE_SPACING


class Cosmology:
    """
    Stand in for lightcone_io's Snapshot_Cosmology_For_Lightcone, the fake snapshots only have minimal metadata.
    """
    COSMO = astropy_cosmology.FlatLambdaCDM(H0=68, Om0=0.3)

    def __init__(self, filename):
        pass


def merge_cells(offsets, lengths):
    """
    Merge the index ranges of cells that are adjacent in a file, for when lightcone_io is not available.
    """
    order = np.argsort(offsets)
    merged_offsets, merged_lengths = [], []
    for offset, length in zip(np.asarray(offsets)[order], np.asarray(lengths)[order]):
        if length == 0:
            continue
        if merged_offsets and merged_offsets[-1] + merged_lengths[-1] == offset:
            merged_lengths[-1] += length
        else:
            merged_offsets.append(offset)
            merged_lengths.append(length)
    return np.array(merged_offsets, dtype=int), np.array(merged_lengths, dtype=int)


def write_fake_runs(run_dir):
    """
    Write the fake simulations to run_dir.

    Returns the SOAP catalogue filename format of the halo simulation.
    """
    # fake halo simulation: one particle at the centre of each halo, with the halo's catalogue index as its ID. 
    # As in SOAP the index is only unique within a snapshot: every snapshot's haloes are numbered from 0
    halo_dir = run_dir / BOX_RES / HALO_SIM
    for snap_nr in SNAPSHOTS:
        pos = halo_positions(np.random.default_rng(snap_nr))
        #index = np.arange(NHALOS) + snap_nr * 10**6
        index = np.arange(NHALOS) # make it more realistic by having repeats per snapshot. Now start indexing without offset or reliance on snapshot number. 
        write_snapshot(halo_dir / "snapshots" / f"flamingo_{snap_nr:04d}", snap_nr, index, pos)
        write_soap_catalogue(halo_dir / "SOAP" / f"halo_properties_{snap_nr:04d}.hdf5", snap_nr, index, pos)

    # lattice simulation: the same lattice in every snapshot
    lattice_dir = run_dir / BOX_RES / LATTICE_SIM
    pos = lattice_positions()
    for snap_nr in SNAPSHOTS:
        write_snapshot(lattice_dir / "snapshots" / f"flamingo_{snap_nr:04d}", snap_nr, np.arange(len(pos)), pos)
    return soap_format(run_dir)


def soap_format(run_dir):
    """
    SOAP catalogue filename format of the fake halo simulation in run_dir.
    """
    return str(run_dir / BOX_RES / HALO_SIM / "SOAP" / "halo_properties_{snap_nr:04d}.hdf5")


def patch_snapshot_lightcone(patch, run_dir):
    """
    Point SnapshotLightcone at the fake simulations in run_dir.

    :param  patch:      used to make the changes, undone with patch.undo()
    :type   patch:      pytest.MonkeyPatch
    :param  run_dir:    directory the fake simulations were written to
    :type   run_dir:    pathlib.Path
    """
    patch.setattr(snapshot_lightcone, "Snapshot_Cosmology_For_Lightcone", Cosmology)
    if not callable(snapshot_lightcone.merge_cells):
        patch.setattr(snapshot_lightcone, "merge_cells", merge_cells)
    init = snapshot_lightcone.SnapshotLightcone.__init__
    defaults = list(init.__defaults__)
    defaults[0] = str(run_dir) + "/{box_res}/{sim_name}"
    patch.setattr(init, "__defaults__", tuple(defaults))


# 'run' fake SWIFT simulations, i.e. place fake particles in snapshot and catalogue

@pytest.fixture(scope="module")
def fake_run_dir(tmp_path_factory):
    """
    Write the fake simulations and point SnapshotLightcone at them.

    Returns the directory of the fake simulations.
    """
    run_dir = tmp_path_factory.mktemp("runs")
    write_fake_runs(run_dir)
    patch = pytest.MonkeyPatch()
    patch_snapshot_lightcone(patch, run_dir)
    yield run_dir
    patch.undo()


@pytest.fixture(scope="module")
def fake_runs(fake_run_dir):
    """
    The fake simulations, see fake_run_dir.

    Returns the SOAP catalogue filename format of the halo simulation.
    """
    return soap_format(fake_run_dir)


def new_lightcone(sim_name, beam_vector, orientation_lock):
    """
    SnapshotBeam along beam_vector, or SnapshotAllSky if beam_vector is None, using a fake simulation.
    """
    if beam_vector is None:
        return snapshot_lightcone.SnapshotAllSky(BOX_RES, sim_name, verbose=0, orientation_lock=orientation_lock)
    return snapshot_lightcone.SnapshotBeam(BOX_RES, sim_name, beam_vector, verbose=0, orientation_lock=orientation_lock)


def shell_args(lightcone, ang_radius_deg):
    """
    Angular radius argument of the placing methods: only SnapshotBeam takes one.
    """
    return {} if isinstance(lightcone, snapshot_lightcone.SnapshotAllSky) else {"ang_radius_deg": ang_radius_deg}


def sorted_rows(snap_nr, ids, coords, a, edge):
    """
    Rows of (snapshot, ID, position, expansion factor), sorted so two sets of rows can be compared.
    If edge, only the rows of the haloes on cell faces, otherwise only the rest.
    """
    #keep = (ids % 10**6 < NEDGE) == edge
    keep = (ids < NEDGE) == edge # updated to match new halo id's without offsets
    snap_nr, ids, coords, a = snap_nr[keep], ids[keep], coords[keep], a[keep]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], ids, snap_nr))
    return snap_nr[order], ids[order], coords[order], a[order]


def place_haloes_and_particles(halo_format, beam_vector, ang_radius_deg, redshift_range, orientation_lock, edge):
    """
    Place the haloes and the particles at their centres in a lightcone shell.

    Returns the sorted rows of the haloes and of the particles, see sorted_rows.
    """
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock)
    halos = lightcone.place_halos_in_shell(halo_format, redshift_range, **shell_args(lightcone, ang_radius_deg))
    particles = lightcone.place_snapshot_particles_in_shell(redshift_range, property_names=["ParticleIDs", "Coordinates"],
                                                            particle_types=PTYPE, **shell_args(lightcone, ang_radius_deg))[PTYPE]
    halo_rows = sorted_rows(halos["Lightcone/SnapshotNumber"].value.astype(int),
                            halos["InputHalos/HaloCatalogueIndex"].value.astype(int),
                            halos["Lightcone/HaloCentre"].to_value("Mpc"),
                            halos["Lightcone/ExpansionFactor"].value, edge)
    particle_rows = sorted_rows(particles["SnapshotNumber"].value.astype(int),
                                particles["ParticleIDs"].value.astype(int),
                                particles["Coordinates"].to_value("Mpc"),
                                particles["ExpansionFactors"].value, edge)
    return halo_rows, particle_rows


def assert_same_rows(halo_rows, particle_rows):
    """
    The haloes and particles come from the same snapshots, with the same IDs, positions and expansion factors.
    """
    np.testing.assert_array_equal(halo_rows[0], particle_rows[0])
    np.testing.assert_array_equal(halo_rows[1], particle_rows[1])
    np.testing.assert_allclose(halo_rows[2], particle_rows[2], atol=1e-6)
    np.testing.assert_allclose(halo_rows[3], particle_rows[3], rtol=1e-10)


# params for lightcone beam tests

LIGHTCONES = [
    pytest.param((0, 0, 1), 20., (0.05, 0.25), id="beam_z_axis"),
    pytest.param((1, 2, 3), 15., (0.05, 0.25), id="beam_off_axis"),
    pytest.param(None, None, (0.1, 0.2), id="all_sky"),
]


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_haloes_placed_as_particles_at_their_centres(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """
    Haloes read from SOAP catalogues are placed in the lightcone exactly as particles at the same positions in the
    snapshots: from the same snapshots, at the same lightcone positions and expansion factors, each the same number of times.
    """
    halo_rows, particle_rows = place_haloes_and_particles(fake_runs, beam_vector, ang_radius_deg, redshift_range,
                                                          orientation_lock, edge=False)
    assert len(halo_rows[0]) > 0
    assert_same_rows(halo_rows, particle_rows)

    # every halo is in the shell, and in the beam
    r = np.linalg.norm(halo_rows[2], axis=1)
    r_min, r_max = Cosmology.COSMO.comoving_distance(redshift_range).to_value("Mpc")
    assert np.all((r >= r_min - 1e-6) & (r <= r_max + 1e-6))
    if beam_vector is not None:
        cos_angle = halo_rows[2] @ (np.asarray(beam_vector) / np.linalg.norm(beam_vector)) / r
        assert np.all(cos_angle >= np.cos(np.deg2rad(ang_radius_deg)) - 1e-9)


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_haloes_on_cell_faces_placed_as_particles(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """
    Place the centre of haloes on the edges and/or faces of the snapshot box tiles. 
    Ensure that both particles and haloes have:
        1) the same re-orientation for edge cases. 
        2) are not wrapped around the tile, but instead remain on the tiles face. 
    """
    halo_rows, particle_rows = place_haloes_and_particles(fake_runs, beam_vector, ang_radius_deg, redshift_range,
                                                          orientation_lock, edge=True)
    assert len(halo_rows[0]) > 0
    assert_same_rows(halo_rows, particle_rows)


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg", [
    pytest.param(None, None, id="all_sky"),
    pytest.param((0, 0, 1), 30., id="beam_z_axis"),
    pytest.param((1, 2, 3), 30., id="beam_off_axis"),
])
def test_lattice_within_half_a_box_length(fake_runs, beam_vector, ang_radius_deg, orientation_lock):
    """
    Place particles from a uniform lattive across the whole snapshot box into a lightcone redshift shell with a radius 
    approx L/2. Therefore only the observer's box tile, [-L/2, L/2) along each axis, is within this shell. 

    Test that regardless of the orientation, the particles placed in the lightcone must be the
    points of the lattice centred on the observer that lie in the shell (and the beam), each exactly once, taken
    from the snapshot whose redshift range holds their distance from the observer.
    """
    lightcone = new_lightcone(LATTICE_SIM, beam_vector, orientation_lock)
    z_min, z_max = redshift_at_distance([R_MIN, R_MAX])
    particles = lightcone.place_snapshot_particles_in_shell((z_min, z_max), property_names=["ParticleIDs", "Coordinates"],
                                                            particle_types=PTYPE, **shell_args(lightcone, ang_radius_deg))[PTYPE]
    coords = particles["Coordinates"].to_value("Mpc")
    snap_nr = particles["SnapshotNumber"].value.astype(int)

    # the lattice centred on the observer, in the shell and the beam
    expected = lattice_positions() - 0.5 * BOXSIZE
    r_expected = np.linalg.norm(expected, axis=1)
    assert np.min(np.abs(r_expected - R_MAX)) > 1e-3 and np.min(np.abs(r_expected - R_MIN)) > 1e-3
    in_shell = (r_expected >= R_MIN) & (r_expected <= R_MAX)
    if beam_vector is not None:
        beam_vec = np.asarray(beam_vector) / np.linalg.norm(beam_vector)
        cos_angle = expected @ beam_vec / r_expected
        assert np.min(np.abs(cos_angle - np.cos(np.deg2rad(ang_radius_deg)))) > 1e-9
        in_shell &= cos_angle >= np.cos(np.deg2rad(ang_radius_deg))
    expected = expected[in_shell]

    # every point is on the lattice, inside the observer's tile, and each lattice point is placed exactly once
    assert len(coords) == len(expected)
    assert np.all(np.abs(coords) < 0.5 * BOXSIZE)
    lattice_idx = np.rint(coords / LATTICE_SPACING - 0.5).astype(int)
    np.testing.assert_allclose(coords, (lattice_idx + 0.5) * LATTICE_SPACING, atol=1e-6)
    expected_idx = np.rint(expected / LATTICE_SPACING - 0.5).astype(int)
    np.testing.assert_array_equal(np.unique(lattice_idx, axis=0), np.unique(expected_idx, axis=0))
    assert len(np.unique(lattice_idx, axis=0)) == len(lattice_idx)

    # each particle of a snapshot is placed at most once, as only one tile is used
    for snap in np.unique(snap_nr):
        ids = particles["ParticleIDs"].value.astype(int)[snap_nr == snap]
        assert len(np.unique(ids)) == len(ids)

    # each point comes from the snapshot whose redshift range holds its distance from the observer
    z = redshift_at_distance(np.linalg.norm(coords, axis=1))
    for snap in np.unique(snap_nr):
        z_lo, z_hi = nz.snapshot_redshift_range(snap, BOX_RES)
        assert np.all((z[snap_nr == snap] >= z_lo - 1e-6) & (z[snap_nr == snap] <= z_hi + 1e-6))


@pytest.mark.parametrize("use_numba", [True, False])
def test_reorientation_of_points_on_cell_faces(use_numba, monkeypatch):
    """
    Quarter turns, reflections and whole cell shifts map the points on the cell faces of a box onto cell faces again,
    one to one, staying within [0, L] along each axis, with the numba and numpy versions giving the same result.
    """
    box_structure = snapshot_lightcone.box_structure
    if use_numba and not box_structure._HAVE_NUMBA:
        pytest.skip("numba is not installed")
    monkeypatch.setattr(box_structure, "_HAVE_NUMBA", use_numba)

    # every corner of every cell, and points on the cell faces
    corners = np.indices((NCELLS_AXIS,)*3).reshape(3, -1).T * CELL_SIZE
    on_faces = corners + np.array([0., 0.5, 0.25]) * CELL_SIZE
    points = np.concatenate([corners, on_faces])

    rng = np.random.default_rng(3)
    for _ in range(50):
        angles = rng.integers(-2, 3, 3) * 90
        reflections = rng.choice([-1, 1], 3)
        shift = rng.integers(-NCELLS_AXIS, NCELLS_AXIS, 3) * CELL_SIZE
        out = box_structure.rotate_coords_cartesian(points, angles, reflections=reflections, periodic_shift=shift,
                                                    sidelengths=BOXSIZE, degrees=True)
        assert np.all((out >= 0) & (out <= BOXSIZE))
        # still on the cell faces, exactly, and no two points mapped to the same place within the periodic box
        np.testing.assert_array_equal(out / CELL_SIZE * 4, np.round(out / CELL_SIZE * 4))
        assert len(np.unique(np.mod(out, BOXSIZE), axis=0)) == len(points)
        # the inverse recovers the original points
        back = box_structure.rotate_coords_cartesian(out, angles, reflections=reflections, periodic_shift=shift,
                                                     sidelengths=BOXSIZE, degrees=True, invert=True)
        np.testing.assert_allclose(np.mod(back, BOXSIZE), points, atol=1e-9)


# functions to test the consistency of the particles placed over many box tiles, snapshots and shells

def particle_rows(particles):
    """
    Rows of (snapshot, particle ID, position [Mpc]) of placed particles, sorted so two sets can be compared.
    """
    snap_nr = particles["SnapshotNumber"].value.astype(int)
    ids = particles["ParticleIDs"].value.astype(int)
    coords = particles["Coordinates"].to_value("Mpc")
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], ids, snap_nr))
    return snap_nr[order], ids[order], coords[order]


def assert_same_particles(rows, other_rows):
    """
    Two sets of placed particles are the same: same snapshots, IDs and positions.
    """
    assert len(rows[0]) == len(other_rows[0])
    np.testing.assert_array_equal(rows[0], other_rows[0])
    np.testing.assert_array_equal(rows[1], other_rows[1])
    np.testing.assert_allclose(rows[2], other_rows[2], atol=1e-6)


def place_particles(lightcone, redshift_range, ang_radius_deg):
    """
    Place the dark matter particles of a fake simulation in a shell of the lightcone.
    """
    return lightcone.place_snapshot_particles_in_shell(redshift_range, property_names=["ParticleIDs", "Coordinates"],
                                                       particle_types=PTYPE, **shell_args(lightcone, ang_radius_deg))[PTYPE]


def lattice_in_shell(r_min, r_max, beam_vector, ang_radius_deg):
    """
    Points [Mpc] of the lattice filling all space that the snapshot lattice makes when its box is tiled around the observer,
    (k + 1/2) * spacing - L/2 for every integer k along each axis, between r_min and r_max from the observer and in the beam.
    Checks no point is within 1e-3 Mpc of the edges of the shell or the beam.
    """
    k_max = int(np.ceil((r_max + 0.5 * BOXSIZE) / LATTICE_SPACING)) + 1
    axis = (np.arange(-k_max, k_max + 1) + 0.5) * LATTICE_SPACING - 0.5 * BOXSIZE
    axis = axis[np.abs(axis) <= r_max + LATTICE_SPACING]
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    r = np.linalg.norm(points, axis=1)
    assert np.min(np.abs(r - r_min)) > 1e-3 and np.min(np.abs(r - r_max)) > 1e-3
    keep = (r >= r_min) & (r <= r_max)
    if beam_vector is not None:
        beam_vec = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
        cos_angle = points[keep] @ beam_vec / r[keep]
        cos_edge = np.cos(np.deg2rad(ang_radius_deg))
        assert np.min(np.abs(cos_angle - cos_edge)) > 1e-9
        keep[keep] = cos_angle >= cos_edge
    return points[keep]


def lattice_index(coords):
    """
    Integer index (k_x, k_y, k_z) of points of the lattice of lattice_in_shell, checking they are on it.
    """
    k = np.rint((coords + 0.5 * BOXSIZE) / LATTICE_SPACING - 0.5).astype(int)
    np.testing.assert_allclose(coords, (k + 0.5) * LATTICE_SPACING - 0.5 * BOXSIZE, atol=1e-6)
    return k


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, r_range", [
    pytest.param(None, None, (440., 1110.), id="all_sky"),
    pytest.param((0, 0, 1), 25., (110., 1610.), id="beam_z_axis"),
    pytest.param((1, 2, 3), 25., (110., 1610.), id="beam_off_axis"),
    pytest.param((1, 1, 1), 40., (110., 1610.), id="beam_diagonal_wide"),
])
def test_lattice_over_many_tiles(fake_runs, beam_vector, ang_radius_deg, r_range, orientation_lock):
    """
    Place particles from a uniform lattive across the whole snapshot box into a lightcone redshift shell with a radius 
    larger than L/2. Therefore many tiles are needed within this shell. 

    Test that the particles placed in the lightcone retain the same lattice structure:
        1) There are no gaps in the lattice
        2) The spaces between particles has not decreased or increased.  

    """
    lightcone = new_lightcone(LATTICE_SIM, beam_vector, orientation_lock)
    particles = place_particles(lightcone, redshift_at_distance(r_range), ang_radius_deg)
    snap_nr, ids, coords = particle_rows(particles)

    expected = lattice_in_shell(*r_range, beam_vector, ang_radius_deg)
    placed_idx = lattice_index(coords)
    expected_idx = lattice_index(expected)
    unique_idx, counts = np.unique(placed_idx, axis=0, return_counts=True)
    assert np.all(counts == 1), f"{np.sum(counts > 1)} lattice points placed more than once"
    missing = len(expected_idx) - len(unique_idx)
    assert missing == 0, f"{missing} lattice points not placed"
    np.testing.assert_array_equal(unique_idx, np.unique(expected_idx, axis=0))

    # more than one snapshot fills the shell, and each point comes from the one holding its distance
    assert len(np.unique(snap_nr)) > 1
    z = redshift_at_distance(np.linalg.norm(coords, axis=1))
    for snap in np.unique(snap_nr):
        z_lo, z_hi = nz.snapshot_redshift_range(snap, BOX_RES)
        assert np.all((z[snap_nr == snap] >= z_lo - 1e-6) & (z[snap_nr == snap] <= z_hi + 1e-6))


@pytest.mark.parametrize("orientation_lock", [None, "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_ranges", [
    pytest.param(None, None, (0.1, 0.163, 0.2), id="all_sky"),
    pytest.param((1, 2, 3), 20., (0.05, 0.137, 0.3), id="beam_off_axis"),
])
def test_consecutive_shells(fake_runs, beam_vector, ang_radius_deg, redshift_ranges, orientation_lock):
    """
    Test that placing two consecutive shells ([z0, z1] and [z1, z2]), gives exactly the particles of expected from a larger shell [z0, z2]. 
    At the shared boundary check for:
        1) missing particles 
        2) duplicated particles
    """
    z0, z1, z2 = redshift_ranges
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock)
    inner = particle_rows(place_particles(lightcone, (z0, z1), ang_radius_deg))
    outer = particle_rows(place_particles(lightcone, (z1, z2), ang_radius_deg))
    whole = particle_rows(place_particles(lightcone, (z0, z2), ang_radius_deg))

    assert len(inner[0]) > 0 and len(outer[0]) > 0
    combined = tuple(np.concatenate([a, b]) for a, b in zip(inner, outer))
    order = np.lexsort((combined[2][:, 2], combined[2][:, 1], combined[2][:, 0], combined[1], combined[0]))
    assert_same_particles(tuple(a[order] for a in combined), whole)


@pytest.mark.parametrize("orientation_lock", ["cube", "sphere"])
@pytest.mark.parametrize("beam_vector", [(0, 0, 1), (1, 2, 3), (-1, 0.5, 0.2)])
def test_beam_is_part_of_all_sky(fake_runs, beam_vector, orientation_lock):
    """
    Test that for a given orientation lock (i.e. every tile's orientation is set by its layer) a beam is exactly the part of the
    all-sky lightcone inside its cone. 
    The overlapping footprints of the BEAM and ALLSKY methods should return the same particles from the same snapshots at the same positions.
    """
    redshift_range, ang_radius_deg = (0.1, 0.2), 20.
    all_sky = particle_rows(place_particles(new_lightcone(HALO_SIM, None, orientation_lock), redshift_range, None))
    beam = particle_rows(place_particles(new_lightcone(HALO_SIM, beam_vector, orientation_lock), redshift_range, ang_radius_deg))

    beam_vec = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
    cos_angle = all_sky[2] @ beam_vec / np.linalg.norm(all_sky[2], axis=1)
    in_cone = cos_angle >= np.cos(np.deg2rad(ang_radius_deg))
    assert len(beam[0]) > 0
    assert_same_particles(tuple(a[in_cone] for a in all_sky), beam)


@pytest.mark.parametrize("orientation_lock", [None, "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_file_by_file_placement(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """
    Test that placing the particles one file at a time gives exactly the particles of reading and 
    placing all files are once.
    """
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock)
    args = shell_args(lightcone, ang_radius_deg)
    whole = particle_rows(place_particles(lightcone, redshift_range, ang_radius_deg))

    numb_files, files = lightcone.gather_files(PTYPE, redshift_range, **args)
    assert numb_files > 0
    per_file = []
    for file_number in range(numb_files):
        particles = lightcone.place_file_in_shell(file_number, PTYPE, ["ParticleIDs", "Coordinates"], **args)
        if particles:
            particles["SnapshotNumber"] = unyt.unyt_array(np.full(len(particles["ParticleIDs"]), files[file_number].snap_nr),
                                                          "dimensionless")
            per_file.append({name: particles[name].copy() for name in ("SnapshotNumber", "ParticleIDs", "Coordinates")})
    combined = {name: unyt.uconcatenate([d[name] for d in per_file]) for name in per_file[0]}
    assert_same_particles(particle_rows(combined), whole)


def mpiexec_command():
    """
    mpiexec of the MPI library mpi4py uses, next to the Python executable if it is there (e.g. the mpich wheel),
    otherwise on the PATH. None if there is none.
    """
    local = Path(sys.executable).parent / "mpiexec"
    return str(local) if local.is_file() else shutil.which("mpiexec")


@pytest.mark.parametrize("nr_ranks", [2, 3])
@pytest.mark.parametrize("sim_name, beam_vector, ang_radius_deg, redshift_range, orientation_lock, redistribute", [
    pytest.param(HALO_SIM, None, None, (0.1, 0.2), None, False, id="all_sky"),
    pytest.param(HALO_SIM, None, None, (0.1, 0.2), "sphere", False, id="all_sky_sphere"),
    pytest.param(HALO_SIM, (1, 2, 3), 20., (0.05, 0.3), "cube", False, id="beam"),
    pytest.param(HALO_SIM, (1, 2, 3), 20., (0.05, 0.3), None, True, id="beam_redistributed"),
    pytest.param(LATTICE_SIM, (0, 0, 1), 25., (0.05, 0.3), None, False, id="lattice_beam"),
    # one snapshot and at most two files, so some ranks are given no files to read
    pytest.param(HALO_SIM, (0, 0, 1), 20., (0.06, 0.07), None, True, id="beam_redistributed_ranks_without_files"),
])
def test_mpi_matches_serial(fake_run_dir, tmp_path, nr_ranks, sim_name, beam_vector, ang_radius_deg, redshift_range,
                            orientation_lock, redistribute):
    """
    Test that the mpi and serial methods achieve the same results. 
    """
    pytest.importorskip("mpi4py")
    mpiexec = mpiexec_command()
    if mpiexec is None:
        pytest.skip("mpiexec is not available")

    args = dict(run_dir=str(fake_run_dir), sim_name=sim_name, beam_vector=beam_vector, ang_radius_deg=ang_radius_deg,
                redshift_range=redshift_range, orientation_lock=orientation_lock, redistribute=redistribute,
                output=str(tmp_path / "placed"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    run = subprocess.run([mpiexec, "-n", str(nr_ranks), sys.executable, str(Path(__file__).parent / "mpi_place_particles.py"),
                          json.dumps(args)], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    ranks = [np.load(tmp_path / f"placed.{rank}.npz") for rank in range(nr_ranks)]
    for rank in ranks:
        assert len(rank["snap_nr"]) == len(rank["ids"]) == len(rank["coords"])
    assert sum(len(rank["ids"]) > 0 for rank in ranks) > 1, "the particles were not split between the ranks"
    snap_nr, ids, coords = (np.concatenate([rank[name] for rank in ranks]) for name in ("snap_nr", "ids", "coords"))
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], ids, snap_nr))
    parallel = snap_nr[order], ids[order], coords[order]

    lightcone = new_lightcone(sim_name, beam_vector, orientation_lock)
    serial = particle_rows(place_particles(lightcone, redshift_range, ang_radius_deg))
    assert_same_particles(parallel, serial)


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("sim_name", [HALO_SIM, LATTICE_SIM])
def test_all_sky_within_half_a_box_length_is_the_snapshot(fake_runs, sim_name, orientation_lock):
    """
    Test that for when the tile is not rotated, reflected or has a periodic shift the particles are in the 
    same position about the observer as in the snapshot. 

    An all-sky lightcone with radius L/2 is filled only by tile, (0, 0, 0). 
    With orientation_seed=0 that tile is not rotated, reflected or shifted (with or without a lock), so every
    particle of the snapshots within half a box length of the centre of the box is placed exactly once, at its box
    position less half a box length, from the snapshot whose redshift range holds its distance from the observer.
    """
    lightcone = new_lightcone(sim_name, None, orientation_lock)
    observer_tile = (0, 0, 0, 0) if orientation_lock == "sphere" else (0, 0, 0)  # "sphere" labels tiles with their layer
    rot_angles, reflections, shift = lightcone._snapshot_reorientation(observer_tile)
    assert np.all(rot_angles == 0) and np.all(reflections == 1) and np.all(shift == 0)

    r_max = 0.5 * BOXSIZE
    redshift_range = (0., float(redshift_at_distance(r_max)))
    placed = particle_rows(place_particles(lightcone, redshift_range, None))

    # every particle of each snapshot, at its box position less half a box length, in the part of the shell
    # the snapshot fills
    snap_length_in_mpc = (MPC_CM * unyt.cm).to_value("Mpc")
    expected = [], [], []
    near_boundary = 0
    for snap_nr in SNAPSHOTS:
        z_lo, z_hi = nz.snapshot_redshift_range(snap_nr, BOX_RES)
        r_lo, r_hi = Cosmology.COSMO.comoving_distance([max(z_lo, 0.), min(z_hi, redshift_range[1])]).to_value("Mpc")
        if z_lo >= redshift_range[1]:
            continue
        if sim_name == HALO_SIM:
            #ids, pos = np.arange(NHALOS) + snap_nr * 10**6, halo_positions(np.random.default_rng(snap_nr))
            ids, pos = np.arange(NHALOS), halo_positions(np.random.default_rng(snap_nr)) # updated to match new halo ids without offset. 
        else:
            pos = lattice_positions()
            ids = np.arange(len(pos))
        # in Mpc, converted from the snapshot length unit as the placed coordinates are
        lightcone_pos = (pos - 0.5 * BOXSIZE) * snap_length_in_mpc
        r = np.linalg.norm(lightcone_pos, axis=1)
        # leave out the particles within 1e-6 Mpc of the edges of the shell or a snapshot's range, which may go either way
        edge = (np.abs(r - r_lo) < 1e-6) | (np.abs(r - r_hi) < 1e-6)
        near_boundary += np.count_nonzero(edge & (r >= r_lo - 1e-6) & (r <= r_hi + 1e-6))
        keep = (r >= r_lo) & (r <= r_hi) & ~edge
        expected[0].append(np.full(np.count_nonzero(keep), snap_nr))
        expected[1].append(ids[keep])
        expected[2].append(lightcone_pos[keep])
    expected = tuple(np.concatenate(a) for a in expected)
    order = np.lexsort((expected[2][:, 2], expected[2][:, 1], expected[2][:, 0], expected[1], expected[0]))
    expected = tuple(a[order] for a in expected)

    # leave out the same particles from those placed
    r_placed = np.linalg.norm(placed[2], axis=1)
    snapshot_edges = Cosmology.COSMO.comoving_distance(
        [z for snap_nr in SNAPSHOTS for z in nz.snapshot_redshift_range(snap_nr, BOX_RES)] + [redshift_range[1]]).to_value("Mpc")
    on_edge = np.any(np.abs(r_placed[:, None] - snapshot_edges[None, :]) < 1e-6, axis=1)
    placed = tuple(a[~on_edge] for a in placed)

    assert near_boundary <= 10
    assert len(placed[0]) > 0
    assert_same_particles(placed, expected)
    np.testing.assert_allclose(placed[2], expected[2], rtol=0, atol=1e-9)  # the same positions, to float rounding


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", [
    pytest.param(None, None, (0.1, 0.27), id="all_sky"),
    pytest.param((1, 2, 3), 20., (0.05, 0.4), id="beam_off_axis"),
    pytest.param((0, 0, 1), 40., (0.05, 0.4), id="beam_z_axis_wide"),
])
def test_number_density(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """
    Test that the number of particles placed matches the number density of the snapshots times the volume they fill, within
    3 sigma. Repeat for the whole redshift shell, the shell filled by each individual snapshot and for each octant on the sky.
    Excludes particles/ haloes on snapshot box faces.
    """
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock)
    snap_nr, ids, coords = particle_rows(place_particles(lightcone, redshift_range, ang_radius_deg))
    #uniform = ids % 10**6 >= NEDGE
    uniform = ids >= NEDGE # updated to match 0 offset snapshot IDs
    snap_nr, coords = snap_nr[uniform], coords[uniform]

    density = (NHALOS - NEDGE) / BOXSIZE**3  # [Mpc^-3]
    solid_angle = 4 * np.pi if beam_vector is None else 2 * np.pi * (1 - np.cos(np.deg2rad(ang_radius_deg)))

    def expected_count(z_lo, z_hi, fraction_of_sky=1.):
        r_lo, r_hi = Cosmology.COSMO.comoving_distance([z_lo, z_hi]).to_value("Mpc")
        return density * fraction_of_sky * solid_angle / 3 * (r_hi**3 - r_lo**3)

    def assert_poisson(count, expected, what):
        assert abs(count - expected) < 3 * np.sqrt(expected), \
            f"{what}: {count} particles, {expected:.0f} expected ({(count - expected) / np.sqrt(expected):+.1f} sigma)"

    # whole shell
    assert_poisson(len(snap_nr), expected_count(*redshift_range), "whole shell")

    # the part of the shell each snapshot fills
    snapshots_used = 0
    for snap in SNAPSHOTS:
        z_lo, z_hi = nz.snapshot_redshift_range(snap, BOX_RES)
        z_lo, z_hi = max(z_lo, redshift_range[0]), min(z_hi, redshift_range[1])
        if z_lo >= z_hi:
            assert np.count_nonzero(snap_nr == snap) == 0
            continue
        snapshots_used += 1
        assert_poisson(np.count_nonzero(snap_nr == snap), expected_count(z_lo, z_hi), f"snapshot {snap}")
    assert snapshots_used > 1

    # each octant of the sky
    if beam_vector is None:
        octant = (coords[:, 0] > 0) * 4 + (coords[:, 1] > 0) * 2 + (coords[:, 2] > 0)
        for i in range(8):
            assert_poisson(np.count_nonzero(octant == i), expected_count(*redshift_range, fraction_of_sky=1/8), f"octant {i}")


# objects repeated over the box tiles

def placed_particles_by_tile(lightcone, redshift_range, ang_radius_deg):
    """
    Place dark matter particles in a shell one file at a time and label each particle with the tile it was placed from.

    Returns arrays of the snapshot number, particle ID and tile label of each particle placed.
    """
    args = shell_args(lightcone, ang_radius_deg)
    numb_files, files = lightcone.gather_files(PTYPE, redshift_range, **args)
    snap_nr, ids, tiles = [], [], []
    for file_number in range(numb_files):
        particles = lightcone.place_file_in_shell(file_number, PTYPE, ["ParticleIDs", "Coordinates"], **args)
        if not particles:
            continue
        n = len(particles["ParticleIDs"])
        snap_nr.append(np.full(n, files[file_number].snap_nr))
        ids.append(particles["ParticleIDs"].value.astype(int))
        tiles.append(np.tile(np.asarray(files[file_number].tile, dtype=int), (n, 1)))
    return np.concatenate(snap_nr), np.concatenate(ids), np.concatenate(tiles)


def placed_haloes_by_tile(lightcone, halo_format, redshift_range, ang_radius_deg, monkeypatch):
    """
    Place the haloes of the SOAP catalogues in a shell and label each halo with the tile it was placed from.
    Use place_halos_in_shell to move haloes into one tile at a time then keep those in the shell, 
    so the tile of each halo kept is the one of the last call to Snapshot2Lightcone.

    Returns arrays of the snapshot number, halo ID (SOAP catalogue index) and tile label of each halo placed.
    """
    current = {}
    kept_tiles = []
    to_lightcone = lightcone.Snapshot2Lightcone
    select_in_shell = lightcone._SnapshotLightcone__select_paticles_in_shell

    def snapshot2lightcone(snapshot_number, coords, tile=None):
        current["tile"] = tile
        return to_lightcone(snapshot_number, coords, tile=tile)

    def select_and_record_tile(*args, **kwargs):
        keep_idx, other = select_in_shell(*args, **kwargs)
        kept_tiles.extend([current["tile"]] * len(keep_idx))
        return keep_idx, other

    monkeypatch.setattr(lightcone, "Snapshot2Lightcone", snapshot2lightcone)
    monkeypatch.setattr(lightcone, "_SnapshotLightcone__select_paticles_in_shell", select_and_record_tile)
    halos = lightcone.place_halos_in_shell(halo_format, redshift_range, **shell_args(lightcone, ang_radius_deg))
    monkeypatch.undo()

    snap_nr = halos["Lightcone/SnapshotNumber"].value.astype(int)
    assert len(kept_tiles) == len(snap_nr), "could not match every halo placed to a tile"
    return snap_nr, halos["InputHalos/HaloCatalogueIndex"].value.astype(int), np.array(kept_tiles, dtype=int).reshape(-1, 3)


def assert_unique_ids_in_fake_run(run_dir):
    """
    A repeated ID in the lightcone can only be the same object placed again.
    Every particle of the fake simulation data has its own ParticleID, and every halo of a SOAP catalogue
    has its own HaloCatalogueIndex and snapshot number.
    """
    halo_dir = run_dir / BOX_RES / HALO_SIM
    halo_ids_of_snapshot = []
    for snap_nr in SNAPSHOTS:
        snapshot_files = sorted((halo_dir / "snapshots" / f"flamingo_{snap_nr:04d}").glob("*.hdf5"))
        particle_ids = []
        for name in snapshot_files:
            with h5py.File(name, "r") as f:
                particle_ids.append(f[f"{PTYPE}/ParticleIDs"][...])
        particle_ids = np.concatenate(particle_ids)
        with h5py.File(halo_dir / "SOAP" / f"halo_properties_{snap_nr:04d}.hdf5", "r") as f:
            halo_ids = f["InputHalos/HaloCatalogueIndex"][...]
        assert len(np.unique(particle_ids)) == len(particle_ids) == NHALOS
        assert len(np.unique(halo_ids)) == len(halo_ids) == NHALOS
        halo_ids_of_snapshot.append(np.sort(halo_ids))
    # the same IDs in every snapshot
    assert all(np.array_equal(ids, halo_ids_of_snapshot[0]) for ids in halo_ids_of_snapshot)

def assert_no_repeats_in_a_tile(snap_nr, ids, tiles, what):
    """
    Test that a particle is not repeated more than the number of tiles within the lightcone. 
    Each tile holds one copy of the snapshot box, so an object (snapshot, ID) is placed at most once in a tile. Across the
    lightcone it is placed at most once per tile its snapshot fills, so never more often than the number of those tiles.

    Returns the most times any one object was placed.
    """
    rows = np.column_stack([snap_nr, ids, tiles])
    unique_rows, counts = np.unique(rows, axis=0, return_counts=True)
    repeated = unique_rows[counts > 1]
    assert len(repeated) == 0, f"{len(repeated)} {what} placed more than once in a tile, e.g. (snapshot, ID, tile) = {repeated[:3].tolist()}"

    objects, placements = np.unique(np.column_stack([snap_nr, ids]), axis=0, return_counts=True)
    for snap in np.unique(snap_nr):
        n_tiles = len(np.unique(tiles[snap_nr == snap], axis=0))
        most = placements[objects[:, 0] == snap].max()
        assert most <= n_tiles, f"a {what[:-1]} of snapshot {snap} placed {most} times, its snapshot fills {n_tiles} tiles"
    assert np.all(placements <= len(np.unique(tiles, axis=0)))
    return placements.max()


# example params for duplication tests. 
@pytest.mark.parametrize("orientation_lock", [None, "cube"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", [
    pytest.param(None, None, (0.1, 0.27), id="all_sky"),
    pytest.param((1, 2, 3), 20., (0.05, 0.4), id="beam_off_axis"),
    pytest.param((0, 0, 1), 40., (0.05, 0.4), id="beam_z_axis_wide"),
])
def test_no_repeats_within_a_tile(fake_run_dir, fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock,
                                  monkeypatch):
    """
    Test the number of times a halo and particle are placed in the same tiles.
    Excluding orientation_lock=='sphere', each tile is one re-oriented copy of the snapshot box, so no particle 
    or halo should be placed twice in the same tile. For the whole lightcone each particle or halo is can only placed 
    as many times as there are new tiles filled by the corresponding snapshot box. 
    """
    assert_unique_ids_in_fake_run(fake_run_dir)

    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock)
    particles = placed_particles_by_tile(lightcone, redshift_range, ang_radius_deg)
    haloes = placed_haloes_by_tile(lightcone, fake_runs, redshift_range, ang_radius_deg, monkeypatch)
    assert len(particles[0]) > 0 and len(haloes[0]) > 0

    most_particle = assert_no_repeats_in_a_tile(*particles, "particles")
    most_halo = assert_no_repeats_in_a_tile(*haloes, "haloes")
    if beam_vector is None:
        # the shell reaches past the observer's box, so objects are placed in more than one tile
        assert most_particle > 1 and most_halo > 1

    # the same objects in the same tiles
    def sorted_rows_by_tile(snap_nr, ids, tiles):
        rows = np.column_stack([snap_nr, ids, tiles])
        return rows[np.lexsort(rows.T[::-1])]
    np.testing.assert_array_equal(sorted_rows_by_tile(*haloes), sorted_rows_by_tile(*particles))
