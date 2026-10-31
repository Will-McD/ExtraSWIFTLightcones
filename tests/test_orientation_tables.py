#!/bin/env python
import numpy as np
import pytest
pytest.importorskip("yaml")
from swiftlet import snapshot_orientation as so

"""
Tests of the orientation tables of the snapshot tiles:
    - the default tables
    - generating tables with every distinct orientation
    - checking tables
    - reading + writing .yaml or .yml files

"""

def orientation_keys(tables):
    """
    Each row of the tables as a hashable key: the matrix of its quarter turns and reflections, and its cell shift.
    """
    keys = []
    for turns, reflections, shift in zip(tables["QuarterTurns"], tables["Reflections"], tables["CellShifts"]):
        matrix = so._build_rotation_matrix(turns * 90., degrees=True) * reflections[np.newaxis, :]
        keys.append(tuple(np.rint(matrix).astype(int).ravel()) + tuple(int(n) for n in shift))
    return keys


# The fingerprints of the installed tables. 
# To keep reproducibility these should not change. 
DEFAULT_V1_FINGERPRINT = "dd43da3e96d8add4"
DEFAULT_V2_FINGERPRINT = "7006e905808e256f"


def test_installed_tables():
    """
    Test that the default tables are installed and accessible: 
        - default_v1:   a small 79 row table used for tests and has intentional 
                            clashes of cells
        - default_v2:   a no-clash table to used to build lightcones  
    """

    assert so.DEFAULT_ORIENTATION_TABLE == "default_v2"
    v1, v2 = so.default_orientation_tables("default_v1"), so.default_orientation_tables()
    assert so.orientation_tables_fingerprint(v1) == DEFAULT_V1_FINGERPRINT
    assert len(v1["QuarterTurns"]) == 79 and set(np.unique(v1["CellShifts"]).tolist()) == {0, 1, -1, 2, -2, 4}
    assert so.orientation_tables_fingerprint(v2) == DEFAULT_V2_FINGERPRINT
    assert len(v2["QuarterTurns"]) == 5184
    for name in ("QuarterTurns", "CellShifts"):
        np.testing.assert_array_equal(v2[name][0], [0, 0, 0])
    np.testing.assert_array_equal(v2["Reflections"][0], [1, 1, 1])
    keys = orientation_keys(v2)
    assert len(set(keys)) == len(keys), "orientations repeated in default_v2"
    turns_and_reflections = [key[:9] for key in keys]
    assert len(set(turns_and_reflections)) == 48
    assert set(np.unique(turns_and_reflections, axis=0, return_counts=True)[1]) == {108}
    assert np.all(np.isin(v2["CellShifts"], so.EIGHTH_SHIFT_VALUES))
    # read by name, as a lightcone does
    for name, fingerprint in (("default_v1", DEFAULT_V1_FINGERPRINT), ("default_v2", DEFAULT_V2_FINGERPRINT),
                              ("default", DEFAULT_V2_FINGERPRINT)):
        assert so.orientation_tables_fingerprint(so.read_orientation_tables(name)) == fingerprint
    with pytest.raises(ValueError, match="installed"):
        so.default_orientation_tables("default_v3")
    # the installed files are YAML, which any YAML reader reads as the same tables
    yaml = pytest.importorskip("yaml")
    for name, tables in (("default_v1", v1), ("default_v2", v2)):
        with open(so.installed_orientation_table_file(name)) as f:
            contents = yaml.safe_load(f)
        assert contents["name"] == name and contents["fingerprint"] == so.orientation_tables_fingerprint(tables)
        assert contents["columns"] == list(so.ORIENTATION_TABLE_COLUMNS)
        rows = np.column_stack([tables[n] for n in so.ORIENTATION_TABLE_NAMES])
        np.testing.assert_array_equal(np.array(contents["rows"]), rows)
    # each call gives its own copy, so changing it doesn't change the installed tables
    v2["CellShifts"][0] = 1
    assert so.orientation_tables_fingerprint(so.default_orientation_tables()) == DEFAULT_V2_FINGERPRINT


def test_installed_v1_tables_are_those_of_the_tool():
    """
    The installed default_v1 tables are the rows copied into tools/make_installed_orientation_tables.py.
    """
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tools" / "make_installed_orientation_tables.py"
    spec = importlib.util.spec_from_file_location("make_installed_orientation_tables", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert so.orientation_tables_fingerprint(tool.default_v1_tables()) == DEFAULT_V1_FINGERPRINT
    assert tool.FINGERPRINTS == {"default_v1": DEFAULT_V1_FINGERPRINT, "default_v2": DEFAULT_V2_FINGERPRINT}


@pytest.mark.parametrize("nr_cells_axis", [16, 32])
def test_default_v2_is_clash_free(nr_cells_axis):
    """
    No two rows of default_v2 make a cell touch its own copy across the face between neighbouring tiles (64 cells per
    axis is checked by tools/make_default_orientation_tables.py --check 64, as it takes minutes).
    """
    assert so.tables_are_clash_free(so.default_orientation_tables(), nr_cells_axis) == 0
    assert so.tables_are_clash_free(so.default_orientation_tables("default_v1"), nr_cells_axis) > 0


def test_generator_makes_the_installed_default_v2():
    """
    generate_clash_free_tables, as tools/make_default_orientation_tables.py calls it, makes exactly the installed
    default_v2 tables.
    """
    tables = so.generate_clash_free_tables(seed=21)
    assert so.orientation_tables_fingerprint(tables) == DEFAULT_V2_FINGERPRINT


def test_cell_shifts_in_cells():
    """
    Shifts in eighths of the box are whole cells for multiples of 8 cells per axis; anything else is refused.
    """
    np.testing.assert_array_equal(so.cell_shifts_in_cells([1, -3, 4], 32), [4, -12, 16])
    np.testing.assert_array_equal(so.cell_shifts_in_cells([1, -3, 4], [16, 64, 8]), [2, -24, 4])
    
    for nr_cells_axis in (10, 4, 0):
        with pytest.raises(ValueError, match="multiple of 8"):
            so.cell_shifts_in_cells([1, 0, 0], nr_cells_axis)

    with pytest.raises(ValueError, match="CellShifts must be a multiple of 1/8"):
        so.check_orientation_tables({"QuarterTurns": [[0, 0, 0]], "Reflections": [[1, 1, 1]], "CellShifts": [[5, 0, 0]]})
    
    with pytest.raises(ValueError, match="CellShifts must be a multiple of 1/8"):
        so.check_orientation_tables({"QuarterTurns": [[0, 0, 0]], "Reflections": [[1, 1, 1]], "CellShifts": [[-4, 0, 0]]})


def test_tables_in_whole_cells_are_refused(tmp_path):
    """
    Test that tables or files with units of cell shifts not equal to 1/8 are refused.
    """
    tables = {"QuarterTurns": [[0, 0, 0], [1, 0, 0]], "Reflections": [[1, 1, 1], [-1, 1, 1]], "CellShifts": [[0, 0, 0], [2, 0, -1]]}
    
    with pytest.raises(ValueError, match="incorrect units"):
        so.check_orientation_tables(dict(tables, cell_shift_unit="cells"))
    
    assert so.orientation_tables_fingerprint(dict(tables, cell_shift_unit="eighths")) == \
        so.orientation_tables_fingerprint(tables)
    
    filename = tmp_path / "tables.yaml"
    fingerprint = so.write_orientation_tables(filename, tables=tables)
    
    assert so.orientation_tables_fingerprint(so.read_orientation_tables(filename)) == fingerprint
    
    filename.write_text(filename.read_text().replace("cell_shift_unit: eighths", "cell_shift_unit: cells"))

    with pytest.raises(ValueError, match="incorrect units"):
        so.read_orientation_tables(filename)

def test_generated_tables_hold_every_distinct_orientation():
    tables = so.generate_orientation_tables()
    keys = orientation_keys(tables)
    
    # test keys are consistent for allowed ranges
    assert len(keys) == 48 * 8**3 == 24576
    assert len(set(keys)) == len(keys), "orientations repeated in the generated tables"
    assert len({key[:9] for key in keys}) == 48  # every signed permutation of the axes
    assert set(np.unique(tables["CellShifts"]).tolist()) == set(so.EIGHTH_SHIFT_VALUES)

    # the first row leaves the box unchanged
    np.testing.assert_array_equal(tables["QuarterTurns"][0], [0, 0, 0])
    np.testing.assert_array_equal(tables["Reflections"][0], [1, 1, 1])
    np.testing.assert_array_equal(tables["CellShifts"][0], [0, 0, 0])


def test_generated_tables_are_reproducible():
    same = [so.generate_orientation_tables(seed=3) for _ in range(2)]
    other = so.generate_orientation_tables(seed=4)
    assert so.orientation_tables_fingerprint(same[0]) == so.orientation_tables_fingerprint(same[1])
    assert so.orientation_tables_fingerprint(other) != so.orientation_tables_fingerprint(same[0])
    assert set(orientation_keys(other)) == set(orientation_keys(same[0]))  # the same rows, in another order


def test_generated_tables_options():
    
    tables = so.generate_orientation_tables(n_rows=100)
    assert len(tables["QuarterTurns"]) == 100
    
    np.testing.assert_array_equal(tables["CellShifts"][0], [0, 0, 0])
    np.testing.assert_array_equal(so.generate_orientation_tables(n_rows=100)["CellShifts"], so.generate_orientation_tables()["CellShifts"][:100])
    
    tables = so.generate_orientation_tables(cell_shift_values=(0, 2, -2, 4))
    
    assert len(tables["QuarterTurns"]) == 48 * 4**3 and set(np.unique(tables["CellShifts"]).tolist()) == {0, 2, -2, 4}
    
    with pytest.raises(ValueError):
        so.generate_orientation_tables(n_rows=0)
    
    with pytest.raises(ValueError):
        so.generate_orientation_tables(n_rows=24577)
    
    with pytest.raises(ValueError):
        so.generate_orientation_tables(cell_shift_values=(1, 2))  # no 0
    
    with pytest.raises(ValueError, match="CellShifts must be a multiple of 1/8"):
        so.generate_orientation_tables(cell_shift_values=(0, 8))  # not in eighths


@pytest.mark.parametrize("change, message", [
    (lambda t: t.pop("CellShifts"), "missing"),
    (lambda t: t.update(CellShifts=t["CellShifts"][:-1]), "same number of rows"),
    (lambda t: t.update(Reflections=np.where(t["Reflections"] > 0, 2, -1)), "Reflections"),
    (lambda t: t.update(QuarterTurns=t["QuarterTurns"] + 0.5), "whole numbers"),
    (lambda t: t.update(QuarterTurns=t["QuarterTurns"][:, :2]), "shape"),
])
def test_invalid_tables_are_refused(change, message):
    tables = dict(so.default_orientation_tables())
    change(tables)
    with pytest.raises(ValueError, match=message):
        so.check_orientation_tables(tables)


def test_write_and_read_tables(tmp_path):
    """
    Test that Generated tables do not change between reading and writing when read back. 
    """
    filename = tmp_path / "orientations.yaml"
    
    fingerprint = so.write_orientation_tables(filename, seed=5)
    
    tables = so.read_orientation_tables(filename)
    
    expected = so.generate_orientation_tables(seed=5)
    
    for name in so.ORIENTATION_TABLE_NAMES:
        np.testing.assert_array_equal(tables[name], expected[name])

    assert so.orientation_tables_fingerprint(tables) == fingerprint

    so.write_orientation_tables(filename, tables=so.default_orientation_tables())

    assert so.orientation_tables_fingerprint(so.read_orientation_tables(filename)) == \
        so.orientation_tables_fingerprint(so.default_orientation_tables())


def test_write_and_read_yaml_tables(tmp_path):
    """
    Tables written to a .yaml file read back exactly, also with yaml.safe_load, with what describes them.
    """
    yaml = pytest.importorskip("yaml")
    filename = tmp_path / "orientations.yaml"
    fingerprint = so.write_orientation_tables(filename, seed=5, attributes={"made_by": "me", "checked_for": [16, 32]})
    tables = so.read_orientation_tables(filename)
    expected = so.generate_orientation_tables(seed=5)
    
    for name in so.ORIENTATION_TABLE_NAMES:
        np.testing.assert_array_equal(tables[name], expected[name])
    
    assert so.orientation_tables_fingerprint(tables) == fingerprint
    
    with open(filename) as f:
        contents = yaml.safe_load(f)
    
    assert contents["fingerprint"] == fingerprint and contents["number_of_rows"] == 24576
    assert contents["seed"] == 5 and contents["generated"] is True and contents["cell_shift_unit"] == "eighths"
    assert contents["made_by"] == "me" and contents["checked_for"] == [16, 32]
    
    np.testing.assert_array_equal(np.array(contents["rows"])[:, 6:], expected["CellShifts"])


def test_yaml_tables_written_by_hand(tmp_path):
    """
    A YAML file laid out differently (rows in block style, no fingerprint) reads as the same tables; a file whose rows
    no longer match its fingerprint, or with the wrong columns or rows, is refused.
    """
    columns = "[" + ", ".join(so.ORIENTATION_TABLE_COLUMNS) + "]"
    by_hand = tmp_path / "by_hand.yml"
    by_hand.write_text(f"# my tables\ncell_shift_unit: eighths\ncolumns: {columns}\nrows:\n"
                       "  - [0, 0, 0, 1, 1, 1, 0, 0, 0]\n  -\n    - 1\n    - 0\n    - 0\n    - -1\n    - 1\n"
                       "    - 1\n    - 2\n    - 0\n    - -1\n")
    tables = so.read_orientation_tables(by_hand)
    np.testing.assert_array_equal(tables["CellShifts"], [[0, 0, 0], [2, 0, -1]])

    filename = tmp_path / "edited.yaml"
    so.write_orientation_tables(filename, tables=REFLECT_X_TABLES)
    filename.write_text(filename.read_text().replace("-1, 1, 1, 0, 0, 0]", "-1, 1, 1, 1, 0, 0]"))
    with pytest.raises(ValueError, match="changed since it was written"):
        so.read_orientation_tables(filename)
    for name, text, message in (
            ("columns.yaml", "columns: [a, b]\nrows:\n  - [1, 2]\n", "columns must be"),
            ("short.yaml", f"columns: {columns}\nrows:\n  - [0, 0, 0, 1, 1, 1, 0, 0]\n", "list of 9 values"),
            ("words.yaml", f"columns: {columns}\nrows:\n  - [0, 0, 0, 1, 1, 1, 0, 0, x]\n", "only numbers"),
            ("no_rows.yaml", "name: nothing\n", "no 'rows'")):
        (tmp_path / name).write_text(text)
        with pytest.raises(ValueError, match=message):
            so.read_orientation_tables(tmp_path / name)


def test_reading_a_file_without_tables_fails(tmp_path):
    """
    Test that a .yaml file that is not orientation table is refused with a clear error.
    """
    for name, contents in (("binary.yaml", b"\x89HDF\r\n\x1a\n\x00\x00"), ("not_yaml.yaml", b"rows: [1, 2\n  - ]: :"),
                           ("no_rows.yaml", b"name: nothing\n")):
        (tmp_path / name).write_bytes(contents)
        with pytest.raises(ValueError, match="not an orientation table file"):
            so.read_orientation_tables(tmp_path / name)


def test_only_yaml_files(tmp_path):
    """
    Orientation tables are written and read only as YAML files: any other file name is refused.
    """
    for name in ("tables.hdf5", "tables.txt", "tables"):
        with pytest.raises(ValueError, match="must end in .yaml"):
            so.write_orientation_tables(tmp_path / name, tables=REFLECT_X_TABLES)
        (tmp_path / name).write_text("")
        with pytest.raises(ValueError, match="must end in .yaml"):
            so.read_orientation_tables(tmp_path / name)
    so.write_orientation_tables(tmp_path / "tables.yml", tables=REFLECT_X_TABLES)
    assert so.orientation_tables_fingerprint(so.read_orientation_tables(tmp_path / "tables.yml")) == \
        so.orientation_tables_fingerprint(REFLECT_X_TABLES)

# cells touching their own copy in a neighbouring tile

UNCHANGED_TABLES = {"QuarterTurns": [[0, 0, 0]], "Reflections": [[1, 1, 1]], "CellShifts": [[0, 0, 0]]}
REFLECT_X_TABLES = {"QuarterTurns": [[0, 0, 0], [0, 0, 0]], "Reflections": [[1, 1, 1], [-1, 1, 1]],
                    "CellShifts": [[0, 0, 0], [0, 0, 0]]}


@pytest.mark.parametrize("orientation_lock", [None, "cube"])
def test_unchanged_tiles_never_touch(orientation_lock):
    """
    Tiles that are all the box as it is are the box repeated periodically: no cell touches its own copy.
    """
    report = so.check_neighbouring_tiles(UNCHANGED_TABLES, 16, orientation_lock, extent=2)
    assert report.nr_neighbour_pairs == 3 * 4 * 5 * 5 and report.nr_reoriented_pairs == 0
    assert report.nr_touching_cells == 0


def test_reflected_neighbours_touch_on_their_face():
    """
    A tile reflected along x next to (along x) one that is not reflects the cells on their shared face onto
    themselves: all n^2 cells of the face touch their own copy. Neighbours along y or z, or reflected alike, do not
    touch.
    """
    from swiftlet.snapshot_lightcone import TILE_HASH_PRIMES
    n, extent = 8, 2
    with pytest.warns(Warning, match="cells touching their own replica"):
        report = so.check_neighbouring_tiles(REFLECT_X_TABLES, n, None, extent=extent)

    def reflected(tile):
        return (tile[0] * TILE_HASH_PRIMES[0] + tile[1] * TILE_HASH_PRIMES[1] + tile[2] * TILE_HASH_PRIMES[2]) % 2

    tiles = np.indices((2 * extent + 1,) * 3).reshape(3, -1).T - extent
    differing = [tuple(t) for t in tiles if t[0] < extent and reflected(t) != reflected(t + [1, 0, 0])]
    assert len(differing) > 0
    assert report.nr_touching_pairs == len(differing) and report.nr_touching_cells == n * n * len(differing)
    np.testing.assert_array_equal(report.touching["tile_b"] - report.touching["tile_a"], [[1, 0, 0]] * report.nr_touching_cells)
    # the cells on the x faces: i = 0 or i = n - 1
    i = report.touching["cell"] // (n * n)
    assert np.all((i == 0) | (i == n - 1))



@pytest.mark.parametrize("nr_cells_axis", [8, 32])
def test_cube_lock_touches_only_between_layers(nr_cells_axis):
    """
    With orientation_lock="cube" the tiles of a layer share one orientation, so only pairs on the faces between
    layers are oriented differently, and only they can touch.
    """
    report = so.check_neighbouring_tiles(so.generate_orientation_tables(seed=4), nr_cells_axis, "cube", extent=3,
                                         warn=False)
    assert 0 < report.nr_reoriented_pairs < report.nr_neighbour_pairs
    layer_a = np.max(np.abs(report.touching["tile_a"]), axis=1)
    layer_b = np.max(np.abs(report.touching["tile_b"]), axis=1)
    assert np.all(layer_a != layer_b)


def test_check_neighbouring_tiles_reads_a_table_file(tmp_path):
    filename = tmp_path / "tables.yaml"
    so.write_orientation_tables(filename, tables=REFLECT_X_TABLES)
    from_file = so.check_neighbouring_tiles(str(filename), 8, extent=2, warn=False)
    from_dict = so.check_neighbouring_tiles(REFLECT_X_TABLES, 8, extent=2, warn=False)
    assert from_file.nr_touching_cells == from_dict.nr_touching_cells > 0

def test_check_neighbouring_tiles_in_a_shell():
    """
    Given the box size and a shell, only touching copies whose cells overlap the shell are kept: a shell holding every
    tile keeps them all, a shell inside half a box length (the observer's tile, which touches nothing) or beyond every
    tile keeps none, and a shell in between keeps some.
    """
    n, boxsize, extent = 8, 800., 2
    everything = so.check_neighbouring_tiles(REFLECT_X_TABLES, n, None, extent=extent, warn=False)
    assert everything.nr_touching_cells > 0 and everything.shell is None

    def in_shell(shell):
        return so.check_neighbouring_tiles(REFLECT_X_TABLES, n, None, extent=extent, warn=False, boxsize=boxsize,
                                           shell=shell).nr_touching_cells

    assert in_shell((0., 10. * boxsize)) == everything.nr_touching_cells
    assert in_shell((0., 0.2 * boxsize)) == 0
    assert in_shell((10. * boxsize, 11. * boxsize)) == 0
    assert 0 < in_shell((0.5 * boxsize, 1.5 * boxsize)) < everything.nr_touching_cells
    
    with pytest.raises(ValueError, match="boxsize"):
        so.check_neighbouring_tiles(REFLECT_X_TABLES, n, None, extent=extent, shell=(0., 1.))


def test_check_neighbouring_tiles_rejects_sphere():
    with pytest.raises(ValueError, match="check_neighbouring_tiles takes orientation_lock None or 'cube'"):
        so.check_neighbouring_tiles(UNCHANGED_TABLES, 16, "sphere")


