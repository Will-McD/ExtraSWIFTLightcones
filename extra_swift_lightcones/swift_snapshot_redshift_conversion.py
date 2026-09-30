#!/bin/env python
import os
import numpy as np
import re


def flamingo_snapshot_redshift(boxsize_resolution):    
    if (boxsize_resolution=="L1000N1800") or (boxsize_resolution=="L1000N0900"):
        snapshot_numbers=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77])
        redshift=np.array([15, 10.38, 9.51, 8.7, 7.95, 7.26, 6.63, 6.04, 5.5, 5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.95, 2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.45, 2.4, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.])
    elif (boxsize_resolution=="L1000N3600") or (boxsize_resolution=="L2800N5040"):
        snapshot_numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 32, 33, 34, 35, 36, 37, 38, 39,40, 41, 42, 43, 44, 45, 46, 47, 48, 49,50, 51, 52, 53, 54, 55, 56, 57, 58, 59,60, 61, 62, 63, 64, 65, 66, 67, 68, 69,70, 71, 72, 73, 74, 75, 76, 77, 78])
        redshift = np.array([15, 12.26, 10.38, 9.51, 8.7, 7.95, 7.26, 6.63, 6.04, 5.5,5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.95,2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.45,2.4, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2, 1.95,1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45,1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 0.95,0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.])
    else:
        raise ValueError("boxsize_resolution not recognised")
    return snapshot_numbers, redshift

def colibre_snapshot_redshift():

    snapshot_numbers = np.array(list(range(128)))

    redshifts = np.array([
        30.0000, 25.0000, 22.5000, 20.0000, 19.0000, 18.0000, 17.0000, 16.0000,
        15.0000, 14.5000, 14.0000, 13.5000, 13.0000, 12.5000, 12.0000, 11.5000,
        11.0000, 10.5000, 10.0000,  9.7500,  9.5000,  9.2500,  9.0000,  8.7500,
         8.5000,  8.2500,  8.0000,  7.7500,  7.5000,  7.3750,  7.2500,  7.1250,
         7.0000,  6.8750,  6.7500,  6.6250,  6.5000,  6.3750,  6.2500,  6.1250,
         6.0000,  5.8750,  5.7500,  5.6250,  5.5000,  5.3750,  5.2500,  5.1250,
         5.0000,  4.8750,  4.7500,  4.6250,  4.5000,  4.3750,  4.2500,  4.1250,
         4.0000,  3.8750,  3.7500,  3.6250,  3.5000,  3.3750,  3.2500,  3.1250,
         3.0000,  2.8750,  2.7500,  2.6250,  2.5000,  2.4375,  2.3750,  2.3125,
         2.2500,  2.1875,  2.1250,  2.0625,  2.0000,  1.9375,  1.8750,  1.8125,
         1.7500,  1.6875,  1.6250,  1.5625,  1.5000,  1.4375,  1.3750,  1.3125,
         1.2500,  1.1875,  1.1250,  1.0625,  1.0000,  0.9500,  0.9000,  0.8500,
         0.8000,  0.7500,  0.7000,  0.6500,  0.6000,  0.5500,  0.5000,  0.4750,
         0.4500,  0.4250,  0.4000,  0.3750,  0.3500,  0.3250,  0.3000,  0.2750,
         0.2500,  0.2250,  0.2000,  0.1800,  0.1600,  0.1400,  0.1200,  0.1000,
         0.0800,  0.0600,  0.0500,  0.0400,  0.0300,  0.0200,  0.0100,  0.0000,
    ])
    return snapshot_numbers, redshift

def snapshot_number_in_range(
                redshift_range, boxsize_resolution=None,
                redshift_buffer=(0.025,0.025), snapshot_buffer=(0,0),
                bounds="equal", decimals=5, use_colibre=False):
    """
        redshift_range: tuple, max, min redshifts 
        boxsize_resolution: flamingo simulations boxsize and cuberoot of numb particles. Not required for colibre
        redshift_buffer: tuple, additional redshift to search over below and above the min and max redshifts given
        snapshot_buffer: tuple, include additional snapshots, if possible, above or below the found range. 
        bounds: str, 'equal', use <= and >= as the limits. 'strict' use > and < as the limits
        decimals:  int, rounding precision of the redshift values. 
    """
    
    if use_colibre:
        snapshot_numbers, redshift = colibre_snapshot_redshift()
    else:
        if boxsize_resolution is None:
            raise ValueError("No FLAMINGO boxsize_resolution passed")
    
        snapshot_numbers, redshift = flamingo_snapshot_redshift(boxsize_resolution)

    # assign redshift range to each snapshot
    snapshot_edges = np.zeros((len(snapshot_numbers), 2))
    snapshot_edges[:-2,0]  = 0.5*(redshift[1:-1]+redshift[:-2])
    snapshot_edges[ 1:-1,1]  = 0.5*(redshift[1:-1]+redshift[:-2])
    snapshot_edges[ 0, 1]  = redshift[0]+0.5*(redshift[0]-redshift[1])
    
    if use_colibre:
        dz=0.005
    else:
        dz=0.025

    snapshot_edges[-1, 1]  = dz
    snapshot_edges[-2, :]  = [dz,  3*dz]

    zmin=redshift_range[0]
    zmax=redshift_range[1]

    if decimals is None:
        lc_min = zmin-redshift_buffer[0]
        lc_max = zmax+redshift_buffer[1]
    else:
        lc_min = np.round(zmin-redshift_buffer[0], decimals)
        lc_max = np.round(zmax+redshift_buffer[1], decimals)

    #snapshot lower bound mask, if snapshot min is >= lc min and <= lc max then that snapshots redshift range falls into lightcone range
    # The inverse is true for the upper bound
    # if any of the snapshot redshift range overlaps with the lightcone redshift range, inlcude the snapshot. 
    if bounds=="equal":
        lower_bound_in_lightcone_range =(snapshot_edges[:, 0]>=(lc_min)) & (snapshot_edges[:, 0]<=(lc_max))
        upper_bound_in_lightcone_range =(snapshot_edges[:, 1]>=(lc_min)) & (snapshot_edges[:, 1]<=(lc_max))
        # capture edge cases where lightcone range is exactly within the snapshot range
        lightcone_range_in_snapshot = (snapshot_edges[:, 0]<=(lc_min)) & (snapshot_edges[:, 1]>=(lc_min)) 
    elif bounds=="strict":
        lower_bound_in_lightcone_range =(snapshot_edges[:, 0]>(lc_min)) & (snapshot_edges[:, 0]<(lc_max))
        upper_bound_in_lightcone_range =(snapshot_edges[:, 1]>(lc_min)) & (snapshot_edges[:, 1]<(lc_max))
        # capture edge cases where lightcone range is exactly within the snapshot range
        lightcone_range_in_snapshot = (snapshot_edges[:, 0]<(lc_min)) & (snapshot_edges[:, 1]>(lc_min)) 
    else:
        raise ValueError("Bounds not recognised")

    snapshots_in_range=snapshot_numbers[lower_bound_in_lightcone_range | upper_bound_in_lightcone_range | lightcone_range_in_snapshot]

    # adjust min snapshot 
    if snapshot_buffer[0]!=0 and np.min(snapshots_in_range)>snapshot_numbers[0]:
        min_bound = np.min(snapshots_in_range)-snapshot_buffer[0]
        snapshots_in_range=np.concatenate([[min_bound], snapshots_in_range])
    
    # adjust max snapshot 
    if snapshot_buffer[1]!=0 and np.max(snapshots_in_range)<snapshot_numbers[-1]:
        max_bound = np.max(snapshots_in_range)+snapshot_buffer[1]
        snapshots_in_range=np.concatenate([snapshots_in_range,[max_bound]])

    return snapshots_in_range.astype(int)

def snapshot_number_redshifts(snapshot_number, boxsize_resolution=None, inverse=False, use_colibre=False):
    """
    Returns the redshift of snapshot number. 
        If inverse = True, returns snapshot number for redshift passed as snapshot_number param
    """

    if use_colibre:
        snapshot_numbers, redshift = colibre_snapshot_redshift()
    else:
        if boxsize_resolution is None:
            raise ValueError("No FLAMINGO boxsize_resolution passed")
        snapshot_numbers, redshift = flamingo_snapshot_redshift(boxsize_resolution)
    
    if inverse:
        idx = np.argmin(np.abs(redshift - snapshot_number))
        if not np.isclose(redshift[idx], snapshot_number, atol=1e-6):
            raise ValueError(f"redshift {snapshot_number} not found for {boxsize_resolution} (nearest available: {redshift[idx]})")
        return int(snapshot_numbers[idx])
    
    return redshift[snapshot_number]

def snapshot_redshift_range(snapshot_number, boxsize_resolution=None, use_colibre=False):

    if use_colibre:
        snapshot_numbers, redshift = colibre_snapshot_redshift()
    else:
        if boxsize_resolution is None:
            raise ValueError("No FLAMINGO boxsize_resolution passed")
        snapshot_numbers, redshift = flamingo_snapshot_redshift(boxsize_resolution)
    
    internal_edges = 0.5 * (redshift[:-1] + redshift[1:])
    first_edge = redshift[0] - (internal_edges[0] - redshift[0])
    last_edge = redshift[-1] + (redshift[-1] - internal_edges[-1])

    if last_edge < redshift[-1]:
        last_edge= redshift[-1]
    if first_edge > redshift[0]:
        first_edge= redshift[0]
    
    edges = np.concatenate([[first_edge], internal_edges, [last_edge]])

    if not (-len(redshift) <= snapshot_number < len(redshift)):
        raise IndexError(f"snapshot number {snapshot_number} out of range for {len(redshift)} snapshots")

    return edges[snapshot_number + 1], edges[snapshot_number]


# access the flamingo shell redshifts if downloaded and placed in virtual environment from ./venv_scripts/shell_redshifts.sh
_REDSHIFT_FILES = {
    "L1":   ("L1_REDSHIFTS_FILENAME",   "L1_shell_redshifts_z3.txt"),
    "L2p8": ("L2P8_REDSHIFTS_FILENAME", "L2p8_shell_redshifts_z5.txt"),
}

def flamingo_shell_redshift_file(box):
    """
    Return the path to the FLAMINGO shell redshifts .txt file for the 1000 Mpc ("L1") or 2800 Mpc ("L2p8") box sidelength simulations .
    Checks, in order:
      1. the L1_REDSHIFTS_FILENAME / L2P8_REDSHIFTS_FILENAME environment variable
      2. <repo>/data/redshifts/<file> (only when called from inside the package,
         e.g. an editable install)
    Raises FileNotFoundError if neither exists.
    """
    if box not in _REDSHIFT_FILES:
        raise ValueError(f"box must be one of {list(_REDSHIFT_FILES)}, got {box!r}")
    env_name, filename = _REDSHIFT_FILES[box]
    #
    candidates = [os.environ.get(env_name)]
    #
    # interactive session work around, _file__ only exists when this code lives in a .py
    module_file = globals().get("__file__")
    if module_file is not None:
        package_dir = os.path.dirname(os.path.abspath(module_file))
        candidates.append(os.path.join(package_dir, "..", "data", "redshifts", filename))
    #
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    #
    raise FileNotFoundError(
        f"FLAMINGO shell redshifts for {box} not found (looked in: "
        f"{[p for p in candidates if p]}). Set {env_name} or run "
        "additional_scripts/flamingo_shell_redshifts.sh"
    )


_BOX_RES_PATTERN = re.compile(r"(?:^|/)(L(\d+)N(\d+))(?=/|$)")

def flamingo_box_resolution(path):
    """
    Returns the box size / resolution of a FLAMINGO simultion from a path to the simuations data.

    e.g. "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/data/products.." -> "L1000N1800"

    Raises ValueError if none (or more than one different one) is found.
    """
    matches = {m.group(1) for m in _BOX_RES_PATTERN.finditer(str(path))}
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one FLAMINGO box/resolution label (e.g. L1000N1800) in {path!r}, found {sorted(matches) or 'none'}")
    return matches.pop()