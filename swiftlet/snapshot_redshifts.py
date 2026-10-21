#!/bin/env python
import numpy as np

"""
The redshifts a simulation's snapshots are spread over in a lightcone.

Nothing here depends on the simulation suite, instead the redshifts come from (FLAMINGO, COLIBRE, ...) simulations.Simulation.

Each snapshot covers the redshifts closer to its own than to its neighbours'. 
The edges are the midpoints between each snapshots own redshift and the redshifts of the snapshots before and after it. 
The first and final snapshots end at their own redshifts, so nothing is placed earlier than the first output or later than z = 0.

"""

def _as_redshifts(redshifts):
    """
    Check for one redshift per snapshot and that they are decreasing with snapshot number.

    Returns the redshifts as an np.ndarray of float.

    :param  redshifts:  redshift of each snapshot, in order of snapshot number
    :type   redshifts:  array-like
    """
    redshifts = np.asarray(redshifts, dtype=float)
    if redshifts.ndim != 1 or len(redshifts) < 2:
        raise ValueError("the output redshifts must be a 1D array of at least 2 snapshots")
    if not np.all(np.diff(redshifts) < 0):
        raise ValueError("the output redshifts must decrease with snapshot number")
    return redshifts


def redshift_edges(redshifts):
    """
    The edges of each snapshots redshift range.

    Returns an np.ndarray of len(redshifts) + 1 edges, decreasing: snapshot n covers edges[n + 1] < z <= edges[n].

    :param  redshifts:  redshift of each snapshot, in order of snapshot number
    :type   redshifts:  array-like
    """
    redshifts = _as_redshifts(redshifts)
    midpoints = 0.5 * (redshifts[:-1] + redshifts[1:])
    return np.concatenate([[redshifts[0]], midpoints, [redshifts[-1]]])


def snapshot_redshift_range(redshifts, snapshot_number):
    """
    Redshift range covered by a snapshot, with edges at the midpoints between neighbouring snapshots.
    
    Returns a tuple, (minimum redshift, maximum redshift)

    :param  redshifts:          redshift of each snapshot, in order of snapshot number
    :type   redshifts:          array-like
    :param  snapshot_number:    snapshot number (negative numbers count back from the last snapshot)
    :type   snapshot_number:    int
    """
    edges = redshift_edges(redshifts)
    nr_snapshots = len(edges) - 1
    if not -nr_snapshots <= snapshot_number < nr_snapshots:
        raise IndexError(f"snapshot number {snapshot_number} out of range for {nr_snapshots} snapshots")
    snapshot_number = int(snapshot_number) % nr_snapshots
    return edges[snapshot_number + 1], edges[snapshot_number]


def snapshots_in_range(redshifts, redshift_range, range_tol=1e-9):
    """
    Returns an array of snapshot numbers whose redshift ranges overlap with a given redshift range.

    :param  redshifts:      redshift of each snapshot, in order of snapshot number
    :type   redshifts:      array-like
    :param  redshift_range: minimum and maximum redshift [z_min, z_max]
    :type   redshift_range: sequence of two floats
    """
    if range_tol < 0:
        raise ValueError(f"atol must not be negative, not {atol}")
    z_min, z_max = (float(z) for z in redshift_range)
    if z_max < z_min:
        raise ValueError(f"redshift_range must be [z_min, z_max], not {redshift_range}")
    edges = redshift_edges(redshifts)
    overlaps = (edges[1:] <= z_max+range_tol) & (edges[:-1] >= z_min-range_tol)
    return np.flatnonzero(overlaps)


def snapshot_at_redshift(redshifts, redshift, atol=1e-6):
    """
    Number of the snapshot whose output redshift is a given redshift.

    Raises ValueError if no snapshot is at the redshift.

    :param  redshifts:  redshift of each snapshot, in order of snapshot number
    :type   redshifts:  array-like
    :param  redshift:   redshift of the snapshot
    :type   redshift:   float
    :param  atol:       largest difference between the redshift and the snapshot's redshift
    :type   atol:       float
    """
    redshifts = _as_redshifts(redshifts)
    nearest = int(np.argmin(np.abs(redshifts - redshift)))
    if not np.isclose(redshifts[nearest], redshift, rtol=0., atol=atol):
        raise ValueError(f"no snapshot at redshift {redshift} (nearest is snapshot {nearest} at {redshifts[nearest]})")
    return nearest
