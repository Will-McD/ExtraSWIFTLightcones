#!/bin/env python
"""
Tests of where SnapshotLightcone places objects, using small fake snapshots and SOAP catalogues
written to a temporary directory. No simulation data is required.

    - Haloes placed from a SOAP catalogue (place_halos_in_shell) end up exactly where a particle at the
      same position in the snapshot is placed (place_snapshot_particles_in_shell).
    - Particles on a uniform lattice across the whole snapshot, placed in a shell reaching just under
      half a box length from the observer, end up on the same lattice centred on the observer, each once.
"""
import numpy as np
import pytest
import unyt

h5py = pytest.importorskip("h5py")
astropy_cosmology = pytest.importorskip("astropy.cosmology")
snapshot_lightcone = pytest.importorskip("extra_swift_lightcones.snapshot_lightcone")
from extra_swift_lightcones import snapshot_units as sw_units
from extra_swift_lightcones import swift_snapshot_redshift_conversion as nz

BOX_RES = "L1000N0900"
SNAPSHOTS = range(72, 78)
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
    Write a fake SOAP catalogue holding the halo centres [comoving Mpc] and their catalogue indices.
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


class SOAPCatalogue:
    """
    Stand in for lightcone_io's SOAPCatalogue, reading the fake SOAP catalogues with the package's own unit
    registry. The lightcone_io reader calls float() on the one-element metadata arrays SOAP writes, which
    numpy >= 2.5 refuses.
    """
    def __init__(self, halo_format, first_snap, last_snap):
        self.halo_format = halo_format

    def read(self, snap_nr, to_read):
        with h5py.File(self.halo_format.format(snap_nr=snap_nr), "r") as f:
            registry = sw_units.unit_registry_from_metadata(sw_units.snapshot_unit_metadata(f))
            units = {"InputHalos/HaloCentre": "a*snap_length", "InputHalos/HaloCatalogueIndex": "dimensionless"}
            return {name: unyt.unyt_array(f[name][...], units[name], registry=registry) for name in to_read}


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


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory):
    """
    Write the fake simulations and point SnapshotLightcone at them.

    Returns the SOAP catalogue filename format of the halo simulation.
    """
    run_dir = tmp_path_factory.mktemp("runs")

    # halo simulation: one particle at the centre of each halo, with the halo's catalogue index as its ID
    halo_dir = run_dir / BOX_RES / HALO_SIM
    halo_format = str(halo_dir / "SOAP" / "halo_properties_{snap_nr:04d}.hdf5")
    for snap_nr in SNAPSHOTS:
        pos = halo_positions(np.random.default_rng(snap_nr))
        index = np.arange(NHALOS) + snap_nr * 10**6
        write_snapshot(halo_dir / "snapshots" / f"flamingo_{snap_nr:04d}", snap_nr, index, pos)
        write_soap_catalogue(halo_dir / "SOAP" / f"halo_properties_{snap_nr:04d}.hdf5", snap_nr, index, pos)

    # lattice simulation: the same lattice in every snapshot
    lattice_dir = run_dir / BOX_RES / LATTICE_SIM
    pos = lattice_positions()
    for snap_nr in SNAPSHOTS:
        write_snapshot(lattice_dir / "snapshots" / f"flamingo_{snap_nr:04d}", snap_nr, np.arange(len(pos)), pos)

    patch = pytest.MonkeyPatch()
    patch.setattr(snapshot_lightcone, "Snapshot_Cosmology_For_Lightcone", Cosmology)
    patch.setattr(snapshot_lightcone.hc, "SOAPCatalogue", SOAPCatalogue, raising=False)
    if not callable(snapshot_lightcone.merge_cells):
        patch.setattr(snapshot_lightcone, "merge_cells", merge_cells)
    init = snapshot_lightcone.SnapshotLightcone.__init__
    defaults = list(init.__defaults__)
    defaults[0] = str(run_dir) + "/{box_res}/{sim_name}"
    patch.setattr(init, "__defaults__", tuple(defaults))
    yield halo_format
    patch.undo()


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
    keep = (ids % 10**6 < NEDGE) == edge
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


@pytest.mark.xfail(strict=True, reason=(
    "coordinates are converted to unyt's Mpc (3.0856775809623e24 cm) but the box, cell sizes and periodic shifts "
    "stay in the snapshot length unit (3.08567758e24 cm), so a point on a cell face that the periodic shift wraps to "
    "the box face moves just below it and is thrown a box length across the tile. The halo path keeps it there, "
    "the particle path drops it as its cell is not read for that tile"))
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES[1:])
def test_haloes_on_cell_faces_placed_as_particles(fake_runs, beam_vector, ang_radius_deg, redshift_range):
    """
    As test_haloes_placed_as_particles_at_their_centres, for the haloes exactly on cell faces or at the box origin.
    """
    halo_rows, particle_rows = place_haloes_and_particles(fake_runs, beam_vector, ang_radius_deg, redshift_range,
                                                          orientation_lock=None, edge=True)
    assert_same_rows(halo_rows, particle_rows)


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg", [
    pytest.param(None, None, id="all_sky"),
    pytest.param((0, 0, 1), 30., id="beam_z_axis"),
    pytest.param((1, 2, 3), 30., id="beam_off_axis"),
])
def test_lattice_within_half_a_box_length(fake_runs, beam_vector, ang_radius_deg, orientation_lock):
    """
    Particles on a uniform lattice across the whole snapshot, placed in a shell reaching just under half a box length from
    the observer. Only the observer's box tile, [-L/2, L/2) along each axis, reaches into this shell. Its periodic
    shift is a whole number of cells, and its reflections and 90 degree rotations are about the centre of the box, so
    each maps the lattice onto itself. Whatever the orientation, the particles placed in the lightcone must be the
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