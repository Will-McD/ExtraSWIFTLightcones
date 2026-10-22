#!/bin/env python
import numpy as np
import pytest

from swiftlet import snapshot_redshifts as sr
from swiftlet.simulation_suites import Simulation, read_output_redshifts

"""
Tests of swiftlet.snapshot_redshifts, the redshift ranges snapshots cover.
"""

FLAMINGO_L1_REDSHIFTS = read_output_redshifts("flamingo_L1.txt")
FLAMINGO_L2P8_REDSHIFTS = read_output_redshifts("flamingo_L2p8.txt")
COLIBRE_REDSHIFTS = read_output_redshifts("colibre.txt")

TABLES = {"FLAMINGO L1": FLAMINGO_L1_REDSHIFTS, "FLAMINGO L2p8": FLAMINGO_L2P8_REDSHIFTS, "COLIBRE": COLIBRE_REDSHIFTS}

@pytest.mark.parametrize("redshifts", TABLES.values(), ids=TABLES.keys())
def test_edges(redshifts):
    """
    Test that edges are consistent between snapshots
    """
    redshifts = np.asarray(redshifts)
    edges = sr.redshift_edges(redshifts)
    assert len(edges) == len(redshifts) + 1
    # the first and last snapshots end at their own redshifts, the others at the midpoints between snapshots
    assert edges[0] == redshifts[0] and edges[-1] == redshifts[-1]
    assert np.allclose(edges[1:-1], 0.5 * (redshifts[:-1] + redshifts[1:]))
    # the ranges tile the redshifts without gaps or overlaps, each holding its snapshot's redshift
    ranges = np.array([sr.snapshot_redshift_range(redshifts, n) for n in range(len(redshifts))])
    assert np.array_equal(ranges[:-1, 0], ranges[1:, 1])
    assert np.all((ranges[:, 0] <= redshifts) & (redshifts <= ranges[:, 1]))
    # negative snapshot numbers count back from the last snapshot
    assert sr.snapshot_redshift_range(redshifts, -1) == tuple(ranges[-1])


@pytest.mark.parametrize("redshifts", TABLES.values(), ids=TABLES.keys())
def test_snapshots_in_range_is_every_overlapping_snapshot(redshifts):
    rng = np.random.default_rng(0)
    nr_snapshots = len(redshifts)
    shells = [tuple(sorted(rng.uniform(0., 3., 2))) for _ in range(200)]
    # shells on the edges, tiny shells and shells at z = 0
    edges = sr.redshift_edges(redshifts)
    shells += [(edges[n + 1], edges[n]) for n in range(nr_snapshots)] + [(0., 1e-6), (0., 0.2), (1., 1.0001)]
    for z_min, z_max in shells:
        brute_force = [n for n in range(nr_snapshots)
                       if sr.snapshot_redshift_range(redshifts, n)[0] <= z_max
                       and sr.snapshot_redshift_range(redshifts, n)[1] >= z_min]
        assert sr.snapshots_in_range(redshifts, (z_min, z_max), range_tol=0.).tolist() == brute_force


def test_snapshots_in_range_examples():
    # a shell exactly one snapshot wide also takes the two snapshots it touches
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.175, 0.225)).tolist() == [72, 73, 74]
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1, 0.2)).tolist() == [73, 74, 75]
    assert sr.snapshots_in_range(COLIBRE_REDSHIFTS, (0.1, 0.2)).tolist() == list(range(114, 120))
    # a thin shell inside one snapshot (the pre-selection before snapshot_redshifts found none for this one)
    assert sr.snapshots_in_range(COLIBRE_REDSHIFTS, (1., 1.0001)).tolist() == [92]
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0., 0.01)).tolist() == [77]


@pytest.mark.parametrize("redshifts", TABLES.values(), ids=TABLES.keys())
def test_strict_bounds_leave_out_touching_snapshots(redshifts):
    rng = np.random.default_rng(1)
    nr_snapshots = len(redshifts)
    edges = sr.redshift_edges(redshifts)
    shells = [tuple(sorted(rng.uniform(0., 3., 2))) for _ in range(200)]
    shells += [(edges[n + 1], edges[n]) for n in range(nr_snapshots)] + [(1., 1.)]
    for z_min, z_max in shells:
        brute_force = [n for n in range(nr_snapshots)
                       if sr.snapshot_redshift_range(redshifts, n)[0] < z_max
                       and sr.snapshot_redshift_range(redshifts, n)[1] > z_min]
        assert sr.snapshots_in_range(redshifts, (z_min, z_max), range_tol=0., strict=True).tolist() == brute_force


#def test_bounds_examples():
#    # a shell exactly one snapshot wide: "equal" also takes the two snapshots it touches, "strict" only the one
#    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.175, 0.225), bounds="equal").tolist() == [72, 73, 74]
#    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.175, 0.225), bounds="strict").tolist() == [73]
#    assert Simulation.from_label("L1000N1800", "X").snapshots_in_range((0.175, 0.225), bounds="strict").tolist() == [73]
#    # away from the edges they agree
#    assert sr.snapshots_in_range(COLIBRE_REDSHIFTS, (0.1, 0.2), bounds="strict").tolist() == list(range(114, 120))
#    with pytest.raises(ValueError, match="bounds must be 'equal' or 'strict'"):
#        sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1, 0.2), bounds="loose")


def test_redshift_buffer():
    sim = Simulation.from_label("L1000N1800", "X")
    # 0.1 < z < 0.2 is snapshots 73 to 75; widened by 0.025 either side it touches 72 and 76 too
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1, 0.2)).tolist() == [73, 74, 75]
    redshift_buffer=0.025
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1-redshift_buffer, 0.2+redshift_buffer)).tolist() == [72, 73, 74, 75, 76]
    redshift_buffer=0.025
    assert sim.snapshots_in_range((0.1-redshift_buffer, 0.2+redshift_buffer)).tolist() == [72, 73, 74, 75, 76]
    # strict leaves out the ones the widened range only touches
    redshift_buffer=0.025
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1-redshift_buffer, 0.2+redshift_buffer), strict=True).tolist() == [73, 74, 75]
    # different buffers below and above, and the same as widening the range by hand
    redshift_buffer=(0.06, 0.)
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1-redshift_buffer[0], 0.2+redshift_buffer[1])).tolist() == [73, 74, 75, 76]
    
    redshift_buffer=(0.01, 0.07)
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1-redshift_buffer[0], 0.2+redshift_buffer[1])).tolist() == \
        sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.09, 0.27)).tolist()
    # below z = 0 there are no more snapshots
    redshift_buffer=0.5
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0., 0.01+redshift_buffer)).tolist() == list(range(67, 78))  # up to z = 0.51
    with pytest.raises(ValueError, match=f"redshifts must not be negative"):
        redshift_buffer=-0.3
        sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1-redshift_buffer, 0.2+redshift_buffer) )


def test_tolerance():
    # 0.275 + 0.05 = 0.325, but the edge between snapshots 70 and 71 is stored as 0.32499999999999996: snapshot 70
    # only touches the widened range, so "strict" should leave it out, which it does within the tolerance
    edge = sr.redshift_edges(FLAMINGO_L1_REDSHIFTS)[71]
    assert edge < 0.275 + 0.05 and np.isclose(edge, 0.325, rtol=0., atol=1e-12)
    redshift_buffer=(0., 0.05)
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.25-redshift_buffer[0], 0.275+redshift_buffer[1]),strict=True).tolist() == [71, 72]
    redshift_buffer=(0., 0.05)
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.25-redshift_buffer[0], 0.275+redshift_buffer[1]), range_tol=0.).tolist() == [70, 71, 72]
    # "equal" takes it either way, as it touches
    
    redshift_buffer=(0., 0.05)
    assert sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.25-redshift_buffer[0], 0.275+redshift_buffer[1])).tolist() == [70, 71, 72]
    
    # a range ending a hair before an edge touches it within the tolerance, not without
    assert 70 in sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.25, edge - 1e-12)).tolist()
    assert 70 not in sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.25, edge - 1e-12), range_tol=0.).tolist()
    with pytest.raises(ValueError, match="range_tol must not be negative"):
        sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.1, 0.2), range_tol=-1.)


def test_snapshot_at_redshift():
    assert sr.snapshot_at_redshift(FLAMINGO_L1_REDSHIFTS, 0.) == 77
    assert sr.snapshot_at_redshift(FLAMINGO_L2P8_REDSHIFTS, 0.) == 78
    assert sr.snapshot_at_redshift(COLIBRE_REDSHIFTS, 1.) == 92
    with pytest.raises(ValueError, match="no snapshot at redshift 0.01"):
        sr.snapshot_at_redshift(FLAMINGO_L1_REDSHIFTS, 0.01)


def test_input_checks():
    with pytest.raises(ValueError, match="decrease with snapshot number"):
        sr.redshift_edges([0., 0.5, 1.])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        sr.redshift_edges([0.])
    with pytest.raises(ValueError, match="must be \\[z_min, z_max\\]"):
        sr.snapshots_in_range(FLAMINGO_L1_REDSHIFTS, (0.2, 0.1))
    with pytest.raises(IndexError, match="out of range for 78 snapshots"):
        sr.snapshot_redshift_range(FLAMINGO_L1_REDSHIFTS, 78)


def test_simulation_uses_its_tables():
    assert np.array_equal(Simulation.from_label("L1000N3600", "X").output_redshifts, FLAMINGO_L2P8_REDSHIFTS)
    assert np.array_equal(Simulation.from_label("L1000N0900", "X").output_redshifts, FLAMINGO_L1_REDSHIFTS)
    assert np.array_equal(Simulation.from_label("L200m6", "X").output_redshifts, COLIBRE_REDSHIFTS)