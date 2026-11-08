#!/bin/env python
"""
The redshifts each snapshot of a simulation fills in a lightcone.

Each snapshot covers the redshifts closer to its own than to its neighbours'. The edges are the midpoints between each
snapshot's own redshift and the redshifts of the snapshots before and after it. The first and final snapshots end at
their own redshifts, so nothing is placed earlier than the first output or later than z = 0.

When snapshots are missing, attempt to fill the gap with the available snapshots without exceeding the maximum allowed
redshift spacing between adjacent snapshots (``MAX_SNAPSHOT_SPACING = 0.1``).
"""

import os
import warnings
import numpy as np
from typing import NamedTuple


# the advised largest redshift spacing between the snapshots filling a lightcone shell.
MAX_SNAPSHOT_SPACING = 0.1

# the environment variable that sets another limit, see max_snapshot_spacing
MAX_SNAPSHOT_SPACING_ENV = "SWIFTLET_MAX_SNAPSHOT_SPACING"


def max_snapshot_spacing():
    """
    The redshift spacing between the snapshots filling a lightcone shell at which to fail.

    MAX_SNAPSHOT_SPACING, unless the environment variable SWIFTLET_MAX_SNAPSHOT_SPACING is set to another
    positive number. It is read each time, so changing the variable while running takes effect from the next shell.
    Provide a clear warning if above above MAX_SNAPSHOT_SPACING.

    Returns a float.
    """
    severe_warning_limit = 0.25
    value = os.environ.get(MAX_SNAPSHOT_SPACING_ENV, "").strip()
    if value == "":
        return MAX_SNAPSHOT_SPACING
    try:
        spacing = float(value)
    except ValueError:
        raise ValueError(f"{MAX_SNAPSHOT_SPACING_ENV} must be a positive number, not {value!r}") from None
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"{MAX_SNAPSHOT_SPACING_ENV} must be a positive number, not {value!r}")
    if spacing > severe_warning_limit:
        warnings.warn(
            f"!!! {MAX_SNAPSHOT_SPACING_ENV}={spacing:g} lets snapshots more than {severe_warning_limit:g} apart in "
            "redshift populate a lightcone. The spacing is too large and will cause large discontinuities between "
            "neighbouring snapshots !!!",
            stacklevel=3,
        )
    elif spacing > MAX_SNAPSHOT_SPACING:
        warnings.warn(
            f"{MAX_SNAPSHOT_SPACING_ENV}={spacing:g} is above the default {MAX_SNAPSHOT_SPACING:g}. Snapshots"
            " further apart in redshift result in larger discontinuities in the lightcone",
            stacklevel=3,
        )
    return spacing


# the snapshots filling a redshift range, see snapshot_coverage
class SnapshotCoverage(NamedTuple):
    """
    The snapshots that fill a redshift range, returned by :func:`snapshot_coverage`.

    :ivar snapshots:        the snapshot numbers filling the range, from high to low redshift
    :vartype snapshots:     list of int
    :ivar ranges:           the minimum and maximum redshift each snapshot fills, {snapshot number: (z_min, z_max)}
    :vartype ranges:        dict
    :ivar missing:          the snapshot numbers that would have filled the range but are not available, from high to
                            low redshift
    :vartype missing:       list of int
    :ivar filled_by:        the snapshots whose ranges cover the redshifts of each missing snapshot, {missing snapshot
                            number: list of snapshot numbers}
    :vartype filled_by:     dict
    """

    snapshots: list
    ranges: dict
    missing: list
    filled_by: dict


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
    Returns the edges of each snapshots redshift range, in decreasing order, snapshot
    n covers edges[n + 1] < z <= edge[n].

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


def snapshots_in_range(redshifts, redshift_range, range_tol=1e-9, strict=False):
    """
    Returns an array of the snapshot numbers whose redshift ranges overlap a given redshift range.

    :param  redshifts:      redshift of each snapshot, in order of snapshot number
    :type   redshifts:      array-like
    :param  redshift_range: minimum and maximum redshift [z_min, z_max]
    :type   redshift_range: sequence of two floats
    :param  range_tol:      a snapshot edge within range_tol of the range counts as touching it
    :type   range_tol:      float
    :param  strict:         If True, leave out the snapshots whose ranges only touch the range (overlap it by no more
                            than range_tol)
    :type   strict:         bool
    """
    if range_tol < 0:
        raise ValueError(f"range_tol must not be negative, not {range_tol}")

    z_min, z_max = (float(z) for z in redshift_range)
    if z_min < 0 or z_max < 0:
        raise ValueError("redshifts must not be negative")
    elif z_max < z_min:
        raise ValueError(f"redshift_range must be [z_min, z_max], not {redshift_range}")

    edges = redshift_edges(redshifts)

    if strict:
        overlaps = (edges[1:] < z_max - range_tol) & (edges[:-1] > z_min + range_tol)
    else:
        overlaps = (edges[1:] <= z_max + range_tol) & (edges[:-1] >= z_min - range_tol)
    return np.flatnonzero(overlaps)


def snapshot_at_redshift(redshifts, redshift, range_tol=1e-6):
    """
    Number of the snapshot whose output redshift is a given redshift.

    Raises ValueError if no snapshot is at the redshift.

    :param  redshifts:  redshift of each snapshot, in order of snapshot number
    :type   redshifts:  array-like
    :param  redshift:   redshift of the snapshot
    :type   redshift:   float
    :param  range_tol:       largest difference between the redshift and the snapshot's redshift
    :type   range_tol:       float
    """
    redshifts = _as_redshifts(redshifts)
    nearest = int(np.argmin(np.abs(redshifts - redshift)))
    if not np.isclose(redshifts[nearest], redshift, rtol=0.0, atol=range_tol):
        raise ValueError(f"no snapshot at redshift {redshift} (nearest is snapshot {nearest} at {redshifts[nearest]})")
    return nearest


def available_redshift_edges(redshifts, available):
    """
    Edges of the available snapshots redshift ranges when some snapshots are missing.

    Returns a tuple, (the available snapshot numbers in increasing order, the redshift edges of each snapshot)

    :param  redshifts:  redshift of each snapshot, in order of snapshot number
    :type   redshifts:  array-like
    :param  available:  numbers of the snapshots that are available
    :type   available:  array-like of int
    """
    redshifts = _as_redshifts(redshifts)
    snapshots = np.unique(np.asarray(available, dtype=int))
    if len(snapshots) == 0:
        raise ValueError("no snapshots are available")
    if snapshots[0] < 0 or snapshots[-1] >= len(redshifts):
        raise IndexError(f"available snapshots out of range for {len(redshifts)} snapshots: {snapshots.tolist()}")
    z = redshifts[snapshots]
    edges = np.concatenate([[redshifts[0]], 0.5 * (z[:-1] + z[1:]), [redshifts[-1]]])
    return snapshots, edges


def snapshot_coverage(redshifts, redshift_range, available=None, max_spacing=None, range_tol=0.0):
    """
    The snapshots that fill a redshift range when some snapshots are missing.

    Without missing snapshots, each snapshot fills its own range.
    If any snapshots are missing, the available snapshots fill the range instead, each covering the redshifts closer to
    its own than to its available neighbours.

    Returns :class:`SnapshotCoverage`

    Raise a ValueError if the redshift spacing between neighbouring snapshots filling the range is greater than
    max_spacing.

    :param  redshifts:      redshift of each snapshot, in order of snapshot number
    :type   redshifts:      array-like
    :param  redshift_range: minimum and maximum redshift [z_min, z_max]
    :type   redshift_range: sequence of two floats
    :param  available:      numbers of the snapshots that are available. If None, all are
    :type   available:      array-like of int
    :param  max_spacing:    the redshift spacing between snapshots at which to fail. If None, use
                            MAX_SNAPSHOT_SPACING (0.1), or the environment variable SWIFTLET_MAX_SNAPSHOT_SPACING if it
                            is set
    :type   max_spacing:    float
    :param  range_tol:      a snapshot edge within range_tol of the range counts as touching it, see snapshots_in_range
    :type   range_tol:      float
    """
    # get redshift range of snapshots, available snapshots and missing snapshots
    redshifts = _as_redshifts(redshifts)

    # check upper limit from advised hard coded limit vs environmental variable
    if max_spacing is None:
        max_spacing = max_snapshot_spacing()

    z_min, z_max = (float(z) for z in redshift_range)
    needed = snapshots_in_range(redshifts, (z_min, z_max), range_tol=range_tol)
    available = np.arange(len(redshifts)) if available is None else np.unique(np.asarray(available, dtype=int))
    missing = np.setdiff1d(needed, available)

    if len(missing) == 0:
        snapshots = needed
        ranges = {int(n): tuple(float(z) for z in snapshot_redshift_range(redshifts, n)) for n in snapshots}
        spacings = [(int(a), int(b), redshifts[a] - redshifts[b]) for a, b in zip(snapshots[:-1], snapshots[1:])]
    else:
        if not np.any(np.isin(available, np.arange(len(redshifts)))):
            raise ValueError(
                (
                    f"none of the snapshots are available, the redshift range {z_min} - {z_max} "
                    f"needs snapshots {needed.tolist()}"
                )
            )

        # get edges of available snapshots and find required overlaps
        avail, edges = available_redshift_edges(redshifts, available)
        overlaps = (edges[1:] <= z_max + range_tol) & (edges[:-1] >= z_min - range_tol)
        snapshots = avail[overlaps]
        ranges = {int(n): (float(edges[i + 1]), float(edges[i])) for i, n in zip(np.flatnonzero(overlaps), snapshots)}
        spacings = [(int(a), int(b), redshifts[a] - redshifts[b]) for a, b in zip(snapshots[:-1], snapshots[1:])]

        # missing snapshots beyond the first or last available one: it reaches out to the first or last output
        if snapshots[-1] == avail[-1] and avail[-1] < len(redshifts) - 1 and z_min < redshifts[avail[-1]]:
            spacings.append((int(avail[-1]), len(redshifts) - 1, redshifts[avail[-1]] - redshifts[-1]))
        if snapshots[0] == avail[0] and avail[0] > 0 and z_max > redshifts[avail[0]]:
            spacings.append((0, int(avail[0]), redshifts[0] - redshifts[avail[0]]))

    # the spacings are compared to max_spacing with a margin, so a spacing that is max_spacing in the table (e.g.
    # 0.15 - 0.05) is not let through by its floating point representation
    too_wide = [(a, b, dz) for a, b, dz in spacings if dz >= max_spacing - 1e-9]
    if too_wide:
        gaps = ", ".join(
            f"{dz:.4g} between snapshots {a} (z = {redshifts[a]:.4g}) and {b} (z = {redshifts[b]:.4g})"
            for a, b, dz in too_wide
        )
        cause = (
            f"snapshots {missing.tolist()} are missing"
            if len(missing)
            else "the output redshifts are spaced this widely"
        )
        raise ValueError(
            f"the redshift spacing between snapshots is too large to fill {z_min} < z < {z_max}: {gaps} as {cause}"
            f"(the limit {max_spacing:g}, can be set by {MAX_SNAPSHOT_SPACING_ENV})"
        )
    # fill in gaps and return the SnapshotCoverage named tuple
    filled_by = {}
    for n in missing:
        lo, hi = snapshot_redshift_range(redshifts, n)
        filled_by[int(n)] = [int(m) for m in snapshots if ranges[int(m)][0] < hi and ranges[int(m)][1] > lo]
    return SnapshotCoverage([int(n) for n in snapshots], ranges, [int(n) for n in missing], filled_by)
