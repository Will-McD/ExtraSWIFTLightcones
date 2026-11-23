"""
Test the read plan: each snapshot file a lightcone needs must have a plan on where its particles get placed in the
lightcone and the plan must describe exactly the reads of the lightcones (snapshot, tile, file) pieces.
"""

import numpy as np
import pytest
from lightcone_helpers import tile_file_pieces
from test_snapshot_lightcone_placement import (  # noqa: F401 the fixtures, used by name
    HALO_SIM,
    PTYPE,
    fake_run_dir,
    fake_runs,
    new_lightcone,
)

from swiftlet.snapshot_lightcone import FilePlan, TileUse, _make_file_plan

# tests for one files plan, built by hand

# a snapshot of 6 cells; their offsets in the file are not in cell order
OFFSETS = np.array([40, 0, 25, 10, 70, 55])
LENGTHS = np.array([15, 10, 15, 15, 5, 15])


def test_make_file_plan_by_hand():
    """Test that shared cells are in the plan, in file order and then ordered by position."""
    uses = [
        ((0, 0, 0), (0.1, 0.15), np.array([2, 0, 3])),
        ((1, 0, 0), (0.1, 0.15), np.array([3, 5])),
        ((0, 1, 0, 1), (0.15, 0.2), np.array([5, 2])),
    ]
    plan = _make_file_plan(77, 3, uses, OFFSETS, LENGTHS)

    assert (plan.snap_nr, plan.file_nr) == (77, 3)

    np.testing.assert_array_equal(plan.cells, [3, 2, 0, 5])  # offsets 10, 25, 40, 55
    np.testing.assert_array_equal(plan.offsets, [10, 25, 40, 55])
    np.testing.assert_array_equal(plan.lengths, [15, 15, 15, 15])

    assert [use.tile for use in plan.uses] == [(0, 0, 0), (1, 0, 0), (0, 1, 0, 1)]
    assert plan.uses[2].z_range == (0.15, 0.2)

    for use, (_, _, cells) in zip(plan.uses, uses):
        np.testing.assert_array_equal(plan.cells[use.cells], cells)

    assert plan.nr_particles == 60  # 4 cells, each once
    assert plan.nr_placed == 45 + 30 + 30  # each cell once per use

    # each use's cells, in its own order
    np.testing.assert_array_equal(plan.offsets[plan.uses[0].cells], [25, 40, 10])
    np.testing.assert_array_equal(plan.lengths[plan.uses[0].cells], [15, 15, 15])
    np.testing.assert_array_equal(plan.offsets[plan.uses[2].cells], [55, 25])


def test_make_file_plan_with_copies_of_a_cell():
    """Repeat last test but with a cell needed by several tiles instead of just 2."""
    copies = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)]  # five copies of cell 4
    uses = [(tile, (0.1, 0.2), np.array([4])) for tile in copies]
    uses.insert(2, ((0, 0, 0), (0.1, 0.2), np.array([1, 4])))  # one more tile, with another cell too
    plan = _make_file_plan(70, 0, uses, OFFSETS, LENGTHS)

    np.testing.assert_array_equal(plan.cells, [1, 4])  # each once, in file order (offsets 0, 70)
    np.testing.assert_array_equal(plan.offsets, [0, 70])

    assert [use.tile for use in plan.uses] == [use[0] for use in uses]
    position_of_4 = 1

    assert sum(position_of_4 in use.cells for use in plan.uses) == 6  # six copies of the one planned cell

    assert plan.nr_particles == 10 + 5  # read once
    assert plan.nr_placed == 6 * 5 + 10  # placed once per copy

    assert len(plan.uses) == 6
    for use, (tile, _, cells) in zip(plan.uses, uses):
        assert use.tile == tile
        np.testing.assert_array_equal(plan.offsets[use.cells], OFFSETS[cells])
        np.testing.assert_array_equal(plan.lengths[use.cells], LENGTHS[cells])

    # every tile needs exactly the same cells: placed n times what is read
    same = [(tile, (0.1, 0.2), np.array([5, 2, 0])) for tile in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]]
    plan = _make_file_plan(70, 1, same, OFFSETS, LENGTHS)
    np.testing.assert_array_equal(plan.cells, [2, 0, 5])  # offsets 25, 40, 55
    assert plan.nr_particles == 45
    assert plan.nr_placed == 4 * plan.nr_particles

    for use in plan.uses:
        np.testing.assert_array_equal(plan.cells[use.cells], [5, 2, 0])  # each tile keeps its own order

    # a shared cell holding no particles is planned like any other, and adds nothing to read or place
    lengths = LENGTHS.copy()
    lengths[3] = 0
    plan = _make_file_plan(70, 2, [((0, 0, 0), (0.1, 0.2), [3, 2]), ((0, 0, 1), (0.1, 0.2), [3])], OFFSETS, lengths)

    np.testing.assert_array_equal(plan.cells, [3, 2])

    assert plan.nr_particles == 15 and plan.nr_placed == 15


def test_file_plan_types():
    """The plan and its uses are NamedTuples of the documented fields."""
    plan = _make_file_plan(1, 0, [((0, 0, 0), (0.0, 0.1), [1])], OFFSETS, LENGTHS)

    assert isinstance(plan, FilePlan) and isinstance(plan.uses, tuple) and isinstance(plan.uses[0], TileUse)
    assert plan._fields == ("snap_nr", "file_nr", "cells", "offsets", "lengths", "uses")
    assert plan.uses[0]._fields == ("tile", "z_range", "cells")
    assert plan.nr_particles == plan.nr_placed == 10


# test the plans of real shells, against the gather step's specs

SHELLS = [
    pytest.param((1, 2, 3), 5.0, (0.05, 0.25), id="beam_5deg"),
    pytest.param((0, 0, 1), 20.0, (0.05, 0.25), id="beam_20deg"),
    pytest.param(None, None, (0.1, 0.2), id="all_sky"),
]


def piece_key(snap_nr, tile, file_nr, offsets, lengths):
    """Returns a hashable key of a (snapshot, tile, file) piece, with its arrays as tuples."""
    return (
        int(snap_nr),
        tuple(int(i) for i in tile),
        int(file_nr),
        tuple(int(o) for o in offsets),
        tuple(int(n) for n in lengths),
    )


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", SHELLS)
def test_plans_match_the_tile_pieces(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """
    Check that gather_files returns the plans for each file.

    The plans must be exactly the lightcones (snapshot, tile, file) pieces and each (snapshot, file) is planned once.
    A plan's cells must be distinct, in file order, with the cell metadata's offsets and lengths, and be exactly the
    cells it needs. The particles to read and to place add up.
    """
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg)

    nr_files, plans = lightcone.gather_files(PTYPE, redshift_range)

    shell = lightcone.last_shell

    pieces = tile_file_pieces(shell)
    assert plans is shell.plans and nr_files == len(plans) > 0 and pieces

    # the same reads
    from_plans = [
        piece_key(plan.snap_nr, use.tile, plan.file_nr, plan.offsets[use.cells], plan.lengths[use.cells])
        for plan in plans
        for use in plan.uses
    ]
    assert sorted(from_plans) == sorted(piece_key(*piece) for piece in pieces)

    # each file once, in snapshot then file order
    keys = [(plan.snap_nr, plan.file_nr) for plan in plans]
    assert len(set(keys)) == len(keys) == len({(p.snap_nr, p.file_nr) for p in pieces})
    snapshot_order = {int(n): i for i, n in enumerate(shell.snapshots)}
    assert keys == sorted(keys, key=lambda key: (snapshot_order[key[0]], key[1]))

    z_ranges = {(p.snap_nr, p.tile): tuple(p.z_updated) for p in shell.pieces}

    for plan in plans:
        cell_data = shell.cell_data[plan.snap_nr]
        assert len(np.unique(plan.cells)) == len(plan.cells)

        # in file order, never overlapping (a cell with no particles can share its offset with the next)
        assert np.all(plan.offsets[1:] >= plan.offsets[:-1] + plan.lengths[:-1])

        np.testing.assert_array_equal(plan.offsets, cell_data["offsets"][PTYPE][plan.cells])
        np.testing.assert_array_equal(plan.lengths, cell_data["lengths"][PTYPE][plan.cells])
        np.testing.assert_array_equal(cell_data["cells_in_file"][PTYPE][plan.cells], plan.file_nr)

        used = np.unique(np.concatenate([use.cells for use in plan.uses]))
        np.testing.assert_array_equal(used, np.arange(len(plan.cells)))

        # each use is a piece of the shell, with its redshift range
        for use in plan.uses:
            assert z_ranges[(plan.snap_nr, use.tile)] == use.z_range

            if orientation_lock == "sphere":
                assert len(use.tile) == 4

        assert plan.nr_particles == int(np.sum(plan.lengths))
        assert plan.nr_placed == sum(int(np.sum(plan.lengths[use.cells])) for use in plan.uses)
        assert plan.nr_placed >= plan.nr_particles

    # every particle is planned
    assert sum(plan.nr_placed for plan in plans) == sum(int(np.sum(piece.lengths)) for piece in pieces)


def test_all_sky_plans_share_cells(fake_runs):
    """Test that the all-sky lightcones tiles share cells as some files have more particles to place than to read."""
    lightcone = new_lightcone(HALO_SIM, None, "cube")
    lightcone.gather_files(PTYPE, (0.1, 0.2))
    plans = lightcone.last_shell.plans

    assert any(plan.nr_placed > plan.nr_particles for plan in plans)
    assert sum(plan.nr_particles for plan in plans) < sum(plan.nr_placed for plan in plans)


def test_placing_keeps_the_plans(fake_runs):
    """Test that placing particles gathers the plans too, and places the same particles as before."""
    lightcone = new_lightcone(HALO_SIM, (0, 0, 1), "cube", ang_radius_deg=20.0)
    particles = lightcone.place_snapshot_particles_in_shell((0.05, 0.25), ["ParticleIDs", "Coordinates"], PTYPE)[PTYPE]
    shell = lightcone.last_shell

    assert shell.plans
    assert sum(len(plan.uses) for plan in shell.plans) == len(tile_file_pieces(shell))
    assert len(particles["ParticleIDs"]) > 0
