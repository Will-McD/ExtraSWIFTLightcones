import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("lightcone_io")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/diagrams"))
import snapshot_lattice_diagram as lattice  # noqa: E402
from swiftlet.repeat_check import find_close_cell_repeats, CloseRepeatWarning  # noqa: E402
from swiftlet.snapshot_cosmology import SnapshotCosmology  # noqa: E402
from swiftlet.snapshot_lightcone import SnapshotAllSky  # noqa: E402
from swiftlet.snapshot_orientation import write_orientation_tables  # noqa: E402

"""
Test the cell-level close repeat checker on fake snapshots built with a regular lattice 
of particles that is the same across all snapshots: 
    - without re-orientation (an orientation table of one unchanged row) nothing is reported
    - with orientation_lock="cube" no two copies of a cell in the same layer are reported
    - every close repeat of a particle (two copies within 1 deg on the sky and closer than a box length) is in a
        reported cell pair
    - a table that reflects some tiles and not others makes copies of the cells on the faces between them touch
    - the touching cells can also found from the tables alone (snapshot_orientation.check_neighbouring_tiles)
    - the warning and the HDF5 file
"""

ALL_SKY_SHELL = (440., 1110.)  # [Mpc], from snapshots 72 to 75, through 27 tiles


@pytest.fixture(scope="module")
def lattice_run(tmp_path_factory):
    """
    Write the fake lattice simulation and an unchanged orientation table.

    Returns (Simulation, cosmology, directory).
    """
    run_dir = tmp_path_factory.mktemp("repeat_check_runs")
    simulation, _ = lattice.write_lattice_simulation(run_dir)
    lattice.write_unchanged_orientation_table(run_dir)
    return simulation, SnapshotCosmology.from_simulation(simulation), run_dir


def shell(cosmo, r_range=ALL_SKY_SHELL):
    """
    Redshift range of a shell of comoving distances [Mpc].
    """
    return tuple(lattice.redshift_at_distance(cosmo, np.array(r_range)))


def cube_layer(tiles):
    """
    Cube layer max(|n|) of tile labels, shape (N, 4) or (N, 3).
    """
    return np.max(np.abs(np.asarray(tiles)[:, :3]), axis=1)


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, r_range", [
    pytest.param(None, None, ALL_SKY_SHELL, id="all_sky"),
    pytest.param((1, 2, 3), 25., (110., 1610.), id="beam_off_axis"),
])
def test_nothing_reported_without_reorientation(lattice_run, beam_vector, ang_radius_deg, r_range, orientation_lock):
    """
    The box repeated periodically has no close repeats: copies of a cell are whole box lengths apart.
    """
    simulation, cosmo, run_dir = lattice_run
    lightcone = lattice.new_lightcone(simulation, beam_vector, orientation_lock, orientation_table=str(run_dir / "unchanged_orientation_table.yaml"))
    
    with warnings.catch_warnings():
        warnings.simplefilter("error", CloseRepeatWarning)
        report = find_close_cell_repeats(lightcone, shell(cosmo, r_range), ang_radius_deg=ang_radius_deg)

    assert report.nr_cell_copies > 0 and report.nr_cells_repeated > 0
    
    # the counts of the shell: every pair of copies of a cell, from the copies of each repeated cell
    assert report.nr_tiles > 1 and report.nr_cells_repeated <= report.nr_cells <= report.nr_cell_copies
    copies = report.cells["nr_copies"]

    assert report.nr_cell_pairs == np.sum(copies * (copies - 1) // 2)
    assert report.nr_touching == 0 and report.nr_line_of_sight == 0 and len(report.pairs["cell"]) == 0
    assert report.closest_separation >= report.boxsize - 1e-6


def test_cube_lock_reports_nothing_within_a_layer(lattice_run):
    """
    The tiles of a cube layer share one orientation, so copies of a cell in one layer are the box repeated: only
    copies in different layers can be close repeats, and none share a face.
    """
    simulation, cosmo, _ = lattice_run
    report = find_close_cell_repeats(lattice.new_lightcone(simulation, None, "cube"), shell(cosmo, (110., 1610.)), warn=False)
    
    assert report.nr_touching == 0
    assert report.nr_line_of_sight > 0
    assert np.all(cube_layer(report.pairs["tile_a"]) != cube_layer(report.pairs["tile_b"]))


def particle_close_repeats(snap_nr, ids, coords, boxsize, angle_deg=1.):
    """
    Pairs of copies of the same particle (snapshot, ID) within angle_deg of each other on the sky and closer than a
    box length.

    Returns (snapshot, ID, position a, position b) of each pair.
    """
    order = np.lexsort((ids, snap_nr))
    snap_nr, ids, coords = snap_nr[order], ids[order], coords[order]
    direction = coords / np.linalg.norm(coords, axis=1)[:, None]
    same = (snap_nr[1:] == snap_nr[:-1]) & (ids[1:] == ids[:-1])
    a, b = [], []
    # every pair of copies: the copies of a particle are next to each other once sorted
    step = 1
    
    while np.any(same):
        i = np.flatnonzero(same)
        a.append(i)
        b.append(i + step)
        step += 1
        same = same[:-1] & (snap_nr[step:] == snap_nr[:-step]) & (ids[step:] == ids[:-step]) if step < len(ids) \
            else np.zeros(0, bool)
    
    a, b = np.concatenate(a), np.concatenate(b)
    close = ((np.sum(direction[a] * direction[b], axis=1) > np.cos(np.deg2rad(angle_deg))) & (np.linalg.norm(coords[a] - coords[b], axis=1) < boxsize))
    
    return snap_nr[a[close]], ids[a[close]], coords[a[close]], coords[b[close]]


@pytest.mark.parametrize("orientation_lock", [None, "cube"])
def test_particle_close_repeats_are_in_reported_cell_pairs(lattice_run, orientation_lock):
    """
    Place the lattice as particles and find every close repeat of a particle (two copies within 1 deg and closer
    than a box length). Each must be in a pair of copies of its cell that the checker reports as a line of sight
    pair: the same snapshot and cell, in the same two tiles.
    """
    simulation, cosmo, _ = lattice_run
    redshift_range = shell(cosmo, (110., 1610.))
    lightcone = lattice.new_lightcone(simulation, None, orientation_lock)
    report = find_close_cell_repeats(lightcone, redshift_range, warn=False)
    snap_nr, ids, coords = lattice.place_lattice(lattice.new_lightcone(simulation, None, orientation_lock), redshift_range, None, "particles", None)
    pair_snap, pair_ids, coords_a, coords_b = particle_close_repeats(snap_nr, ids, coords, lattice.BOXSIZE)
    assert len(pair_snap) > 0

    # the cell of each particle in the box, numbered as in the snapshot's cell metadata
    cell_index = np.floor(lattice.lattice_positions()[pair_ids] / (lattice.BOXSIZE / lattice.NCELLS_AXIS)).astype(int)
    pair_cell = (cell_index[:, 0] * lattice.NCELLS_AXIS + cell_index[:, 1]) * lattice.NCELLS_AXIS + cell_index[:, 2]

    def tile(points):
        return np.floor(points / lattice.BOXSIZE + 0.5).astype(int)

    pairs = report.pairs
    reported = set()
    
    for s, c, ta, tb in zip(pairs["snapshot"][pairs["line_of_sight"]], pairs["cell"][pairs["line_of_sight"]], pairs["tile_a"][pairs["line_of_sight"]], pairs["tile_b"][pairs["line_of_sight"]]):
        reported.add((int(s), int(c), frozenset([tuple(ta[:3]), tuple(tb[:3])])))
    
    for s, c, ta, tb in zip(pair_snap, pair_cell, tile(coords_a), tile(coords_b)):
        assert (int(s), int(c), frozenset([tuple(ta), tuple(tb)])) in reported, \
            f"close repeat of a particle of cell {c} of snapshot {s} in tiles {tuple(ta)}, {tuple(tb)} not reported"


def test_reflected_tiles_make_touching_cells(lattice_run, tmp_path):
    """
    With a table of two rows, unchanged and reflected along x, some neighbouring tiles differ only by the reflection.
    Across their face along x, each cell on the face touches its own mirror image: the checker reports these as
    touching pairs, one cell apart along x in tiles next to each other along x.
    """

    simulation, cosmo, _ = lattice_run
    table = tmp_path / "reflect_x.yaml"
    write_orientation_tables(table, tables={
        "QuarterTurns": [[0, 0, 0], [0, 0, 0]], 
        "Reflections": [[1, 1, 1], [-1, 1, 1]], 
        "CellShifts": [[0, 0, 0], [0, 0, 0]]
        })
    
    lightcone = lattice.new_lightcone(simulation, None, None, orientation_table=str(table))
    report = find_close_cell_repeats(lightcone, shell(cosmo), warn=False)
    touching = report.pairs["touching"]
    
    # each touching pair is counted once, though a pair straddling a snapshot hand-over has a row for each snapshot
    different = {(int(c), frozenset([tuple(a), tuple(b)]))
                 for c, a, b in zip(report.pairs["cell"][touching], report.pairs["tile_a"][touching],
                                    report.pairs["tile_b"][touching])}
    
    assert report.nr_touching > 0 and report.nr_touching == len(different) <= np.sum(touching)
    
    offset = report.pairs["centre_b"][touching] - report.pairs["centre_a"][touching]
    np.testing.assert_allclose(np.abs(offset), [[report.cell_size, 0., 0.]] * len(offset), atol=1e-6)
    
    tile_step = report.pairs["tile_b"][touching][:, :3] - report.pairs["tile_a"][touching][:, :3]
    np.testing.assert_array_equal(np.abs(tile_step), [[1, 0, 0]] * len(tile_step))


@pytest.mark.parametrize("orientation_seed", [0, 2])
def test_table_check_finds_the_touching_cells_of_the_lightcone(lattice_run, orientation_seed):
    """
    The touching cells of the lightcone shell (find_close_cell_repeats) are found from the tables alone
    (snapshot_orientation.check_neighbouring_tiles), in the same two tiles: the two move the cells alike. Without the
    shell, the tables find more (whole tiles); given the shell, exactly the same cells, as many as the touching pairs
    the shell check counts.
    """
    
    from swiftlet.snapshot_orientation import check_neighbouring_tiles, default_orientation_tables
    
    simulation, cosmo, _ = lattice_run
    r_range = (110., 1610.)
    
    # the first default tables, which make some cells touch (the default tables make none)
    lightcone = SnapshotAllSky(simulation_object=simulation, orientation_seed=orientation_seed, orientation_table="default_v1", verbose=0)
    report = find_close_cell_repeats(lightcone, shell(cosmo, r_range), warn=False)
    
    assert report.nr_touching > 0
    touching = report.pairs["touching"]
    in_shell = set()
    
    for ta, tb, c in zip(report.pairs["tile_a"][touching], report.pairs["tile_b"][touching], report.pairs["cell"][touching]):
        in_shell.add((frozenset([tuple(ta[:3]), tuple(tb[:3])]), int(c)))

    def touching_cells(tables):
        return {(frozenset([tuple(ta), tuple(tb)]), int(c))
                for ta, tb, c in zip(tables.touching["tile_a"], tables.touching["tile_b"], tables.touching["cell"])}

    whole_tiles = check_neighbouring_tiles(default_orientation_tables("default_v1"), lattice.NCELLS_AXIS, None,
                                           orientation_seed,
                                           extent=3, warn=False)
    assert in_shell < touching_cells(whole_tiles)

    shell_r_range = tuple(cosmo.COSMO.comoving_distance(list(shell(cosmo, r_range))).to_value("Mpc"))
    
    in_tables_shell = check_neighbouring_tiles(default_orientation_tables("default_v1"), lattice.NCELLS_AXIS, None,
                                               orientation_seed, extent=3, warn=False, boxsize=lattice.BOXSIZE,
                                               shell=shell_r_range)
    
    assert touching_cells(in_tables_shell) == in_shell
    assert in_tables_shell.nr_touching_cells == report.nr_touching


def test_close_repeat_check_example(lattice_run, tmp_path):
    """
    examples/snapshot_close_repeat_check.py checks every lock, writes a report for each and plots them.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import snapshot_close_repeat_check as example
    
    simulation, cosmo, _ = lattice_run
    example.run(simulation, shell(cosmo), [0], None, None, 1., tmp_path)
    
    for lock in example.LOCKS:
        with h5py.File(tmp_path / f"close_repeats.lock_{lock}.seed_0.hdf5", "r") as f:
            assert f.attrs["nr_cell_copies"] > 0 and f.attrs["nr_tiles"] > 0 and f.attrs["nr_cell_pairs"] > 0
    
    assert (tmp_path / "close_repeats.png").is_file()


def test_warning_and_hdf5_file(lattice_run, tmp_path):
    """
    Close repeats give a CloseRepeatWarning, and the HDF5 file holds the pairs, the repeated cells, the counts and
    the lightcone's settings.
    """
    simulation, cosmo, _ = lattice_run
    filename = tmp_path / "close_repeats.hdf5"
    
    # the first default tables, which make some cells touch (the default tables make none)
    lightcone = lattice.new_lightcone(simulation, None, None, orientation_table="default_v1")
    
    with pytest.warns(CloseRepeatWarning, match="replicas sharing a face"):
        report = find_close_cell_repeats(lightcone, shell(cosmo, (110., 1610.)), output_file=str(filename))
    
    assert report.nr_touching > 0
    
    with h5py.File(filename, "r") as f:
        assert f.attrs["nr_touching"] == report.nr_touching
        assert f.attrs["nr_line_of_sight"] == report.nr_line_of_sight
        assert len(f["pairs/cell"]) == len(report.pairs["cell"])
        assert f["pairs/tile_a"].shape == (len(report.pairs["cell"]), 4)
    
        np.testing.assert_allclose(f["cells/closest_separation"][...], report.cells["closest_separation"])
        assert f["lightcone"].attrs["orientation_lock"] == "none"


