#!/bin/env python
"""
Tests of the orientation tables of the snapshot tiles:
    - the default tables
    - generating tables with every distinct orientation
    - checking tables
    - reading + writing .hdf5 files

"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from swiftlet import snapshot_orientation as so


def orientation_keys(tables):
    """
    Each row of the tables as a hashable key: the matrix of its quarter turns and reflections, and its cell shift.
    """
    keys = []
    for turns, reflections, shift in zip(tables["QuarterTurns"], tables["Reflections"], tables["CellShifts"]):
        matrix = so._build_rotation_matrix(turns * 90., degrees=True) * reflections[np.newaxis, :]
        keys.append(tuple(np.rint(matrix).astype(int).ravel()) + tuple(int(n) for n in shift))
    return keys


def test_default_tables_are_the_built_in_tables():
    tables = so.default_orientation_tables()
    assert len(tables["QuarterTurns"]) == 79
    np.testing.assert_array_equal(tables["QuarterTurns"], so.SnapshotBeamAngles_quaters)
    np.testing.assert_array_equal(tables["Reflections"], so.SnapshotBeamAngles_reflections)
    np.testing.assert_array_equal(tables["CellShifts"], so.SnapshotBeamAngles_shifts)


def test_generated_tables_hold_every_distinct_orientation():
    tables = so.generate_orientation_tables()
    keys = orientation_keys(tables)
    assert len(keys) == 48 * 7**3 == 16464
    assert len(set(keys)) == len(keys), "orientations repeated in the generated tables"
    assert len({key[:9] for key in keys}) == 48  # every signed permutation of the axes
    assert set(np.unique(tables["CellShifts"]).tolist()) == {0, 4, -4, 8, -8, 16, -16}
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
    np.testing.assert_array_equal(so.generate_orientation_tables(n_rows=100)["CellShifts"],
                                  so.generate_orientation_tables()["CellShifts"][:100])
    tables = so.generate_orientation_tables(cell_shift_values=(0, 8, -8, 16, -16, 32, -32))
    assert len(tables["QuarterTurns"]) == 16464 and np.abs(tables["CellShifts"]).max() == 32
    with pytest.raises(ValueError):
        so.generate_orientation_tables(n_rows=0)
    with pytest.raises(ValueError):
        so.generate_orientation_tables(n_rows=16465)
    with pytest.raises(ValueError):
        so.generate_orientation_tables(cell_shift_values=(4, 8))  # no 0


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
    filename = tmp_path / "orientations.hdf5"
    fingerprint = so.write_orientation_tables(filename, seed=5)
    tables = so.read_orientation_tables(filename)
    expected = so.generate_orientation_tables(seed=5)
    for name in so.ORIENTATION_TABLE_NAMES:
        np.testing.assert_array_equal(tables[name], expected[name])
    assert so.orientation_tables_fingerprint(tables) == fingerprint
    with h5py.File(filename, "r") as f:
        assert f.attrs["fingerprint"] == fingerprint
        assert f.attrs["number_of_rows"] == 16464 and f.attrs["seed"] == 5 and f.attrs["generated"]

    # given tables, e.g. the default ones
    so.write_orientation_tables(filename, tables=so.default_orientation_tables())
    assert so.orientation_tables_fingerprint(so.read_orientation_tables(filename)) == \
        so.orientation_tables_fingerprint(so.default_orientation_tables())


def test_reading_a_file_without_tables_fails(tmp_path):
    filename = tmp_path / "other.hdf5"
    with h5py.File(filename, "w") as f:
        f["QuarterTurns"] = np.zeros((3, 3), dtype=int)
    with pytest.raises(ValueError, match="not an orientation table file"):
        so.read_orientation_tables(filename)

