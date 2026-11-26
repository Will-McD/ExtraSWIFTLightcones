#!/bin/env python
"""
Compare a halo in a FLAMINGO on-the-fly lightcones with the same halo in a lightcone that SWIFTLET builds from the
snapshots.

1.  Find a halo in FLAMINGO's halo lightcone. Find a halo in a shell of a lightcone, with M200c closest to a target
    mass, within the circular beam drawn on the sky.

2.  Find the position of FLAMINGO's observer in the snapshot box, given by --observer_position, otherwise it is either
    read from pre-defined tables of observer positions.

3.  Build a SnapshotBeam pointing at the selected halo and the observer is positioned at FLAMINGO's observer position.
    SWIFTLET puts its observer at the nearest cell corner, so its positions are moved by (corner - observer position)
    into FLAMINGO's frame. Place the particles and haloes of the same shell, and find the same halo again, by its
    snapshot and HaloCatalogueIndex.

4.  Plot the two versions of the beam side by side, and then draw r200c around the haloes above the given minimum mass
    in the redshift shell of each beam:
    - top row: FLAMINGO's HEALPix maps of the shell (SmoothedGasMass, DarkMatterMass), zoomed on the halo
    - bottom row: the same maps made from the particles SWIFTLET placed from the snapshots

What differs between the two beams:
    -   FLAMINGO's on-the-fly lightcones store each particle when it crossed the lightcone, SWIFTLET stores where
        particles are in the snapshot corresponding to the lightcones redshift, so structures differ by how far they
        moved between the two.
    -   SWIFTLET's observer is at the nearest cell corner. Moving its positions into FLAMINGO's frame makes the
        positions match, but the shell's edges are at distances from the corner, so they can differ by up to half a
        cell diagonal.

Notes:
    1.  The FLAMINGO HEALPix maps, halo lightcones and SOAP catalogues are read from the remote cosma server with
        hdfstream, or from local directories if the --local argument is given.
    2.  SnapshotBeam reads the snapshots and SOAP catalogues from the local FLAMINGO directories
    3.  The halo and map readers come from map_zoom_on_halo.py, and the smoothing and zoom helpers from
        snapshot_beam_zoom_on_haloes.py, in this directory.
"""

import argparse
import os
import sys
from pathlib import Path
import h5py
import healpy as hp
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import unyt
import cmasher as cmr  # noqa: F401 register "cmr.*" colour maps used
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_beam_zoom_on_haloes import (
    bin_particles,
    comoving_distance_to_redshift,
    draw_scale_bar,
    gnomonic_zoom,
    smooth_weighted_particles,
)
from swiftlet.example_plot_settings import plot_settings
from swiftlet.flamingo_lightcones import flamingo_shell_redshift_file
from swiftlet.simulation_suites import Simulation
from swiftlet.snapshot_cosmology import SnapshotCosmology
from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet.snapshot_orientation import observer_corner_and_shift
from swiftlet.snapshot_units import drop_a_from_comoving_property
from swiftlet.soap_reader import comoving, read_soap_selected_rows

# the maps compared: the FLAMINGO map name, the particle type and how SWIFTLET's particles are added to the map
MAPS = {
    "SmoothedGasMass": {"ptype": "PartType0", "smoothed": True, "cmap": "cmr.lilac", "label": "gas"},
    "DarkMatterMass": {"ptype": "PartType1", "smoothed": False, "cmap": "cmr.cosmic", "label": "dark matter"},
}
SURFACE_DENSITY_UNITS = r"$\Sigma$ [M$_\odot$ Mpc$^{-2}$]"


def FLAMINGO_observer_positions(lightcone_nr, boxsize_str):
    """
    Returns the position of the observer for a flamingo lightcone.

    :param  lightcone_nr: lightcone of observer number
    :type   lightcone_nr: int
    :param  boxsize_str: "L2p8" or "L1" for the 2.8 Gpc and 1000 Mpc boxsize
    :type   boxsize_str: str
    """
    # 1 Gpc boxes (L1000)
    obvs_dict_L1 = {
        0: (750.0, 750.0, 750.0),
        1: (250.0, 250.0, 250.0),
    }
    # 2.8 Gpc boxes (L2800)
    obvs_dict_L2p8 = {
        0: (700.0, 700.0, 700.0),
        1: (700.0, 700.0, 2100.0),
        2: (700.0, 2100.0, 700.0),
        3: (700.0, 2100.0, 2100.0),
        4: (2100.0, 700.0, 700.0),
        5: (2100.0, 700.0, 2100.0),
        6: (2100.0, 2100.0, 700.0),
        7: (2100.0, 2100.0, 2100.0),
    }
    if (boxsize_str == "L1") or ("L1000N" in boxsize_str):
        return obvs_dict_L1[lightcone_nr]
    elif (boxsize_str == "L2p8") or ("L2800" in boxsize_str):
        return obvs_dict_L2p8[lightcone_nr]
    else:
        raise ValueError("Boxsize not recognised, must be short hand: 'L1' or 'L2p8'")


def comoving_mpc(values):
    """
    Returns comoving values [Mpc] as a numpy array, without the scale factor of comoving units if they have one.

    :param  values: comoving lengths, e.g. halo centres
    :type   values: unyt.unyt_array
    """
    try:
        values = drop_a_from_comoving_property(values)
    except (AttributeError, KeyError, ValueError, TypeError):
        pass  # no scale factor in the units
    return np.asarray(values.to_value("Mpc"))


def flamingo_haloes_in_shell(halo_format, soap_format, shell_z, snapshots, vector, radius, remote_dir, min_mass):
    """
    Read the haloes of FLAMINGO's halo lightcone in a shell and within an angular radius of a direction.

    The halo lightcone has one file per snapshot. Returns a dict of arrays, one entry per halo:

    - "LightconeCentre": the centre [cMpc] in the lightcone, relative to FLAMINGO's observer
    - "SnapshotCentre": the centre [cMpc] in its snapshot box, from its SOAP catalogue
    - "Redshift", "M200c" [Msun], "R200c" [cMpc], "SnapshotNumber", "HaloCatalogueIndex"

    :param  halo_format:    format string of the halo lightcone files (using {snap_nr})
    :type   halo_format:    str
    :param  soap_format:    format string of the SOAP catalogues (using {snap_nr})
    :type   soap_format:    str
    :param  shell_z:        minimum and maximum redshift of the shell
    :type   shell_z:        sequence of two floats
    :param  snapshots:      snapshot numbers whose halo lightcone files can hold haloes of the shell
    :type   snapshots:      iterable of int
    :param  vector:         direction to search around
    :type   vector:         numpy.ndarray, shape (3,)
    :param  radius:         angular radius [rad] to search
    :type   radius:         float
    :param  remote_dir:     remote directory (hdfstream) of the files, or None for local files
    :type   remote_dir:     hdfstream.RemoteDirectory
    :param  min_mass:       smallest M200c [Msun] of the haloes kept
    :type   min_mass:       float
    """
    import lightcone_io.halo_reader as hr

    vector = np.asarray(vector, dtype=float) / np.linalg.norm(vector)
    names = [
        "Lightcone/HaloCentre",
        "Lightcone/Redshift",
        "InputHalos/HaloCatalogueIndex",
        "InputHalos/HaloCentre",  # from the SOAP catalogue: the centre in the snapshot box
        "SO/200_crit/TotalMass",
        "SO/200_crit/SORadius",
    ]
    found = []
    for snap_nr in snapshots:
        if remote_dir is None and not os.path.exists(halo_format.format(snap_nr=snap_nr)):
            print(f"\tsnapshot {snap_nr}: no halo lightcone file")
            continue
        try:
            halo_file = hr.HaloLightconeFile(
                filename=halo_format.format(snap_nr=snap_nr),
                soap_filename=soap_format.format(snap_nr=snap_nr),
                remote_dir=remote_dir,
            )
        except Exception as error:  # e.g. no halo lightcone file for this snapshot
            print(f"\tsnapshot {snap_nr}: no halo lightcone file ({error})")
            continue
        haloes = halo_file.read_halos_in_radius(vector, radius, names)
        centre = comoving_mpc(haloes["Lightcone/HaloCentre"])
        redshift = np.asarray(haloes["Lightcone/Redshift"].value)
        mass = haloes["SO/200_crit/TotalMass"].to_value("Msun")
        # read_halos_in_radius reads whole index pixels, so also cut on the angle
        cos_offset = centre @ vector / np.linalg.norm(centre, axis=1)
        keep = (redshift > shell_z[0]) & (redshift <= shell_z[1]) & (mass >= min_mass) & (cos_offset >= np.cos(radius))

        found.append(
            {
                "LightconeCentre": centre[keep],
                "SnapshotCentre": comoving_mpc(haloes["InputHalos/HaloCentre"])[keep],
                "Redshift": redshift[keep],
                "M200c": mass[keep],
                "R200c": comoving_mpc(haloes["SO/200_crit/SORadius"])[keep],
                "SnapshotNumber": np.full(np.count_nonzero(keep), int(snap_nr)),
                "HaloCatalogueIndex": np.asarray(haloes["InputHalos/HaloCatalogueIndex"].value, dtype=np.int64)[keep],
            }
        )
        print(f"\tsnapshot {snap_nr}: {np.count_nonzero(keep)} haloes in the shell and search radius")

    found = [haloes for haloes in found if len(haloes["Redshift"])]
    if not found:
        raise FileNotFoundError("no haloes found in the shell: try a larger search radius or a smaller minimum mass")

    return {name: np.concatenate([haloes[name] for haloes in found]) for name in found[0]}


def snapshot_box_and_cell_size(simulation):
    """
    Returns the side lengths [cMpc] of the snapshot box and of its cells, from its redshift 0 snapshot.

    :param  simulation: the simulation
    :type   simulation: Simulation
    """
    with h5py.File(simulation.snapshot_file(simulation.z0_snapshot, 0), "r") as snapshot:
        boxsize = np.ravel(snapshot["Header"].attrs["BoxSize"])[0]
        cell_size = np.max(np.ravel(snapshot["Cells/Meta-data"].attrs["size"]))
        length_cgs = float(np.ravel(snapshot["Units"].attrs["Unit length in cgs (U_L)"])[0])
    to_mpc = unyt.unyt_quantity(length_cgs, "cm").to_value("Mpc")
    return float(boxsize) * to_mpc, float(cell_size) * to_mpc


def cone_around_zoom(vector, zoom_angle, distance_range, offset, nr_rim_points=180):
    """
    The smallest cone seen from a point offset from the observer that holds the beams between two comoving distances
    from the observer.

    Returns a tuple (direction of the cone, angular radius [deg]).

    :param  vector:         direction of the centre of the zoom, from the observer
    :type   vector:         numpy.ndarray, shape (3,)
    :param  zoom_angle:     angular radius [rad] of the zoom
    :type   zoom_angle:     float
    :param  distance_range: smallest and largest comoving distance [Mpc] from the observer
    :type   distance_range: sequence of two floats
    :param  offset:         position [Mpc] of the observer as seen from the point (observer - point)
    :type   offset:         numpy.ndarray, shape (3,)
    :param  nr_rim_points:  number of points around each rim
    :type   nr_rim_points:  int
    """
    vector = np.asarray(vector, dtype=float) / np.linalg.norm(vector)

    across = np.cross(vector, [0.0, 0.0, 1.0])
    if np.linalg.norm(across) < 1e-8:
        across = np.cross(vector, [0.0, 1.0, 0.0])
    across /= np.linalg.norm(across)
    across_2 = np.cross(vector, across)

    phi = np.linspace(0.0, 2.0 * np.pi, nr_rim_points, endpoint=False)
    rim = np.cos(zoom_angle) * vector + np.sin(zoom_angle) * (
        np.cos(phi)[:, None] * across + np.sin(phi)[:, None] * across_2
    )

    directions = np.vstack([vector, rim])

    points = np.vstack([directions * distance for distance in distance_range]) + np.asarray(offset)
    unit = points / np.linalg.norm(points, axis=1)[:, None]

    axis = unit.mean(axis=0)
    axis /= np.linalg.norm(axis)

    radius = np.degrees(np.max(np.arccos(np.clip(unit @ axis, -1.0, 1.0))))

    return axis, 1.02 * radius + np.degrees(zoom_angle) * 0.02


def cut_to_shell(data, coordinates_name, distance_range):
    """
    Keep the rows of placed particles or haloes whose comoving distance is inside a range.

    :param  data:               placed particles or haloes, {name: array}
    :type   data:               dict
    :param  coordinates_name:   name of their positions, e.g. "Coordinates"
    :type   coordinates_name:   str
    :param  distance_range:     smallest and largest comoving distance [Mpc]
    :type   distance_range:     sequence of two floats
    """
    if coordinates_name not in data:
        return
    distance = np.linalg.norm(data[coordinates_name].to_value("Mpc"), axis=1)
    keep = (distance >= distance_range[0]) & (distance <= distance_range[1])
    for name in list(data):
        if hasattr(data[name], "__len__") and len(data[name]) == len(keep):
            data[name] = data[name][keep]


def swiftlet_shell(
    simulation,
    cosmo,
    halo_vector,
    zoom_angle,
    shell_z,
    observer_position,
    boxsize,
    cell_size,
    soap_format,
    verbose=1,
):
    """
    Place the particles and haloes of FLAMINGO's zoom on a shell, from the snapshots, with no re-orientation and the
    observer at FLAMINGO's observer position.

    SWIFTLET's observer is at the nearest cell corner, so the beam is the cone around the corner holding FLAMINGO's
    zoom, and the shell is widened by the distance to the corner. The positions are moved into FLAMINGO's frame
    (corner - observer position) and cut to FLAMINGO's shell, so both hold exactly the same volume.

    Returns a tuple (particles {ptype: data}, haloes, beam, shift [Mpc] into FLAMINGO's frame).

    :param  simulation:         the simulation
    :type   simulation:         Simulation
    :param  cosmo:              cosmology of the simulation
    :type   cosmo:              astropy.cosmology.FLRW
    :param  halo_vector:        direction of the centre of the zoom, from FLAMINGO's observer
    :type   halo_vector:        numpy.ndarray
    :param  zoom_angle:         angular radius [rad] of the zoom
    :type   zoom_angle:         float
    :param  shell_z:            minimum and maximum redshift of FLAMINGO's shell
    :type   shell_z:            sequence of two floats
    :param  observer_position:  FLAMINGO's observer position [Mpc] in the snapshot box
    :type   observer_position:  numpy.ndarray
    :param  boxsize:            side length [Mpc] of the snapshot box
    :type   boxsize:            float
    :param  cell_size:          side length [Mpc] of the snapshot's cells
    :type   cell_size:          float
    :param  soap_format:        format string of the local SOAP catalogues (using {snap_nr})
    :type   soap_format:        str
    :param  verbose:            amount SnapshotBeam prints
    :type   verbose:            int
    """
    observer_position = np.asarray(observer_position, dtype=float)

    corner, _ = observer_corner_and_shift(observer_position, boxsize, int(round(boxsize / cell_size)))
    offset = observer_position - corner  # FLAMINGO's observer as seen from the corner

    distance_range = cosmo.comoving_distance(list(shell_z)).to_value("Mpc")

    beam_vector, ang_radius_deg = cone_around_zoom(halo_vector, zoom_angle, distance_range, offset)

    if ang_radius_deg > 60.0:
        raise ValueError(
            f"the zoom needs a beam of {ang_radius_deg:.1f} deg, which is greater than the allowed 60 degree max radius"
            " , choose a smaller halo or a halo that is further away form the observer."
        )

    wide_z = comoving_distance_to_redshift(
        cosmo,
        np.array([max(distance_range[0] - np.linalg.norm(offset), 1.0), distance_range[1] + np.linalg.norm(offset)]),
    )

    beam = SnapshotBeam(
        simulation,
        beam_vector=beam_vector,
        ang_radius_deg=ang_radius_deg,
        orientation_table="no_reorientation",
        observer_position=observer_position,
        verbose=verbose,
    )

    particles = beam.place_snapshot_particles_in_shell(
        tuple(wide_z),
        property_names=["Masses"],
        particle_types=sorted({settings["ptype"] for settings in MAPS.values()}),
    )

    haloes = beam.place_halos_in_shell(tuple(wide_z), halo_format=soap_format)

    shift = -offset  # corner - observer position

    for data in particles.values():
        if "Coordinates" in data:
            data["Coordinates"] = data["Coordinates"] + unyt.unyt_array(shift, "Mpc")
        cut_to_shell(data, "Coordinates", distance_range)

    haloes["Lightcone/HaloCentre"] = haloes["Lightcone/HaloCentre"] + unyt.unyt_array(shift, "Mpc")

    cut_to_shell(haloes, "Lightcone/HaloCentre", distance_range)

    return particles, haloes, beam, shift


def swiftlet_halo_properties(haloes, soap_format):
    """
    Look up M200c and R200c of the SWIFTLET haloes in the SOAP catalogues of their snapshots.

    Returns a tuple (M200c [Msun], R200c [cMpc]), one value per halo.
    """
    snap_nr = haloes["Lightcone/SnapshotNumber"].value.astype(int)
    rows = haloes["InputHalos/SOAPIndex"].value.astype(np.int64)
    mass, radius = np.zeros(len(rows)), np.zeros(len(rows))
    for snapshot in np.unique(snap_nr):
        of_snapshot = np.flatnonzero(snap_nr == snapshot)

        # a halo can be placed more than once (in more than one tile): read each row once
        unique_rows, row_of_halo = np.unique(rows[of_snapshot], return_inverse=True)

        # read rows from catalogue
        soap = read_soap_selected_rows(
            soap_format.format(snap_nr=int(snapshot)), ["SO/200_crit/TotalMass", "SO/200_crit/SORadius"], unique_rows
        )
        mass[of_snapshot] = soap["SO/200_crit/TotalMass"].to_value("Msun")[row_of_halo]
        radius[of_snapshot] = comoving(soap["SO/200_crit/SORadius"]).to_value("Mpc")[row_of_halo]

    return mass, radius


def find_same_halo(haloes, snap_nr, catalogue_index, lightcone_centre):
    """
    Find a halo in FLAMINGO and SWIFTLET halo lightcones.

    The same haloes in swiftlet come from the same snapshot with the same HaloCatalogueIndex,
    nearest to where FLAMINGO's lightcone has it (a halo can be placed more than once).

    Returns its row in the haloes, or None if no halo of that snapshot and index was placed.

    :param  haloes:             haloes placed by SnapshotLightcone.place_halos_in_shell
    :type   haloes:             LightconeOutput
    :param  snap_nr:            the halo's snapshot
    :type   snap_nr:            int
    :param  catalogue_index:    the halo's HaloCatalogueIndex
    :type   catalogue_index:    int
    :param  lightcone_centre:   the halo's centre [Mpc] in FLAMINGO's lightcone
    :type   lightcone_centre:   numpy.ndarray
    """
    same = np.flatnonzero(
        (haloes["Lightcone/SnapshotNumber"].value.astype(int) == int(snap_nr))
        & (haloes["InputHalos/HaloCatalogueIndex"].value.astype(np.int64) == int(catalogue_index))
    )
    if len(same) == 0:
        return None
    centres = haloes["Lightcone/HaloCentre"].to_value("Mpc")[same]
    return int(same[np.argmin(np.linalg.norm(centres - lightcone_centre, axis=1))])


def swiftlet_maps(particles, nside, pixels, distance):
    """
    Make the maps of MAPS from the particles SWIFTLET placed, on the selected pixels: gas smoothed with its SPH kernel,
    dark matter added to the pixel each particle is in, as surface densities at the halo's distance.

    Returns {map name: value of each selected pixel [Msun / Mpc^2]}.

    :param  particles:  particles placed, {ptype: data}, in FLAMINGO's frame
    :type   particles:  dict
    :param  nside:      HEALPix resolution of the maps
    :type   nside:      int
    :param  pixels:     sorted indices of the selected pixels (ring order)
    :type   pixels:     numpy.ndarray
    :param  distance:   comoving distance [Mpc] of the halo
    :type   distance:   float
    """
    pixel_area = hp.nside2pixarea(nside) * distance**2
    maps = {}
    for map_name, settings in MAPS.items():
        data = particles.get(settings["ptype"], {})
        if "Coordinates" not in data or len(data["Coordinates"]) == 0:
            maps[map_name] = np.zeros(len(pixels))
            continue
        position = data["Coordinates"].to("Mpc")
        mass = data["Masses"].to_value("Msun")
        if settings["smoothed"] and "SmoothingLengths" in data:
            mass_map, _ = smooth_weighted_particles(
                position, data["SmoothingLengths"].to("Mpc"), mass, np.ones((len(mass), 1)), nside, pixels
            )
        else:
            mass_map = bin_particles(position.value, mass, nside, pixels)
        maps[map_name] = mass_map / pixel_area
    return maps


def flamingo_maps(shell, nside, pixels, distance):
    r"""
    Read the HEALPix maps from FLAMINGO's shell (selected pixels only) as surface densities.

    Returns {map name: value of each selected pixel [Msun / Mpc^2]}.

    :param  shell:      FLAMINGO's HEALPix maps of the shell
    :type   shell:      lightcone_io.healpix_maps.Shell
    :param  nside:      HEALPix resolution of the maps
    :type   nside:      int
    :param  pixels:     sorted indices of the selected pixels (ring order)
    :type   pixels:     numpy.ndarray
    :param  distance:   comoving distance [Mpc] of the halo
    :type   distance:   float
    """
    from map_zoom_on_halo import read_selected_pixels

    pixel_area = hp.nside2pixarea(nside) * distance**2
    maps = {}
    for map_name in MAPS:
        values = unyt.unyt_array(read_selected_pixels(shell[map_name], pixels), shell[map_name].units)
        maps[map_name] = values.to_value("Msun") / pixel_area
    return maps


def plot_comparison(zooms, haloes, halo, output_filename, percentiles=(1.0, 99.9)):
    """
    Plot FLAMINGO's zooms (top row) and SWIFTLET's (bottom row), a column for each map of MAPS sharing a colour scale.

    Every panel shows the haloes of both lightcones as circles of R200c. Show FLAMINGO as solid pink and SWIFTLET as
    dashed white rings, so the same halo has the same marker in both rows and their offsets can be seen in either.

    :param  zooms:              {"FLAMINGO": {map name: (image, extent)}, "SWIFTLET": {...}}
    :type   zooms:              dict
    :param  haloes:             {"FLAMINGO": (centres [Mpc], R200c [Mpc]), "SWIFTLET": (...)}, the haloes to draw
    :type   haloes:             dict
    :param  halo:               the halo zoomed on: "LightconeCentre" [Mpc], "R200c" [Mpc], "label"
    :type   halo:               dict
    :param  output_filename:    path to write the plot to
    :type   output_filename:    str
    :param  percentiles:        percentiles of the pixels of each column used as the colour scale limits
    :type   percentiles:        tuple

    """
    rows = list(zooms)
    fig, axs = plt.subplots(len(rows), len(MAPS), figsize=(3.0 * len(MAPS), 3.0 * len(rows) + 0.6), squeeze=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)

    halo_style = {"FLAMINGO": {"color": "deeppink", "ls": "-"}, "SWIFTLET": {"color": "white", "ls": "--"}}

    # the image plane: x along the zoom's east, y along its north, both perpendicular to the halo's direction
    centre = np.asarray(halo["LightconeCentre"])
    distance = np.linalg.norm(centre)
    los = centre / distance
    north = np.array([0.0, 0.0, 1.0]) - los[2] * los
    north = north / np.linalg.norm(north) if np.linalg.norm(north) > 1e-12 else np.array([1.0, 0.0, 0.0])
    east = np.cross(north, los)

    def on_image(centres):
        # gnomonic positions [Mpc at the halo's distance] of directions in the zoom
        centres = np.atleast_2d(centres)
        depth = centres @ los
        return np.stack([-(centres @ east) / depth, (centres @ north) / depth], axis=1) * distance

    for col, (map_name, settings) in enumerate(MAPS.items()):
        values = np.concatenate([zooms[row][map_name][0][np.isfinite(zooms[row][map_name][0])] for row in rows])
        positive = values[values > 0]
        vmin, vmax = np.percentile(positive, percentiles) if len(positive) else (1.0, 10.0)

        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = mpl.colormaps.get_cmap(settings["cmap"])
        cmap = cmap.with_extremes(bad="grey", under=cmap(0.0))

        for r, row in enumerate(rows):
            image, extent = zooms[row][map_name]
            image = np.where(image <= 0, 1e-3 * norm.vmin, image)
            ax = axs[r, col]
            img = ax.imshow(
                np.ma.masked_invalid(image), origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="none"
            )

            # the haloes of both lightcones, R200c
            for lightcone, (centres, radii) in haloes.items():
                style = halo_style.get(lightcone, {"color": "white", "ls": ":"})

                for i, ((x, y), r200c) in enumerate(zip(on_image(centres) if len(centres) else [], radii)):
                    ax.plot(
                        x + r200c * np.cos(theta),
                        y + r200c * np.sin(theta),
                        lw=0.7,
                        alpha=0.9,
                        label=f"{lightcone} haloes" if i == 0 and r == 0 and col == 0 else None,
                        **style,
                    )

            # set axis limits and square aspect ratio
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")

            # remove ticks on axis
            ax.set_xticks([])
            ax.set_yticks([])

            draw_scale_bar(ax, extent, distance)  # add scale bar
            # draw titles as needed
            if r == 0:
                ax.set_title(settings["label"], fontsize=10)
            if col == 0:
                ax.set_ylabel(row, fontsize=10)

        # add colour bars
        cbar = fig.colorbar(
            img, ax=axs[:, col], location="bottom", shrink=0.9, pad=0.02, label=SURFACE_DENSITY_UNITS, extend="min"
        )

        cbar.ax.xaxis.set_minor_formatter(NullFormatter())

    fig.suptitle(halo["label"], fontsize=10)

    # one legend for the halo circles, under the title, not over the maps
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        # on a dark background, as the circles are drawn for the dark maps
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(handles),
            fontsize=8,
            facecolor="0.25",
            edgecolor="none",
            framealpha=1.0,
            labelcolor="white",
        )
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a halo in a FLAMINGO lightcone with the same halo in a SWIFTLET SnapshotBeam"
    )
    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label, e.g. L1000N1800")
    parser.add_argument("--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation")
    parser.add_argument(
        "--flamingo_dir",
        default="FLAMINGO/L1_m9/L1_m9",
        help="directory of FLAMINGO's lightcone products (healpix_maps, halo_lightcone, SOAP-HBT): on the remote "
        "cosma server, or a local directory with --local",
    )
    parser.add_argument("--local", action="store_true", help="read FLAMINGO's products from local files")
    parser.add_argument("--lightcone_nr", type=int, default=0, help="FLAMINGO lightcone (observer) number")
    parser.add_argument("--shell_nr", type=int, default=1, help="shell of FLAMINGO's HEALPix maps")
    parser.add_argument("--nside", type=int, default=16384, help="HEALPix resolution of the maps")
    parser.add_argument("--vector", type=float, nargs=3, default=(1.0, 2.0, 3.0), help="direction to search around")
    parser.add_argument("--search_radius_deg", type=float, default=20.0, help="angular radius [deg] to search")
    parser.add_argument("--log_mass", type=float, default=14.0, help="target log10(M200c / Msun) of the halo")
    parser.add_argument(
        "--log_mass_min",
        type=float,
        default=13.0,
        help="minimum log10(M200c / Msun) of the highlighted haloes around the target halo",
    )
    parser.add_argument("--zoom_npix", type=int, default=128, help="radius of the zoom, in pixels")
    parser.add_argument(
        "--observer_position",
        type=float,
        nargs=3,
        default=None,
        help="FLAMINGO's observer position [Mpc] in the snapshot box. If not given, read from the halo lightcone file "
        "or estimated from the haloes",
    )
    parser.add_argument(
        "--output_dir", default="./example_outputs/compare_flamingo_and_snapshot_halo", help="output directory"
    )
    args = parser.parse_args()
    plot_settings()

    simulation = Simulation.from_label(args.box_res, args.sim_name)
    boxsize, cell_size = snapshot_box_and_cell_size(simulation)
    cosmo = SnapshotCosmology.from_simulation(simulation).COSMO
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FLAMINGO's lightcone products, remote (hdfstream) or local
    basedir = args.flamingo_dir

    if args.local:
        remote_dir = None
        halo_format = (
            f"{basedir}/sorted_hbt_lightcone_halos/lightcone{args.lightcone_nr}/lightcone_halos_{{snap_nr:04d}}.hdf5"
        )
        lightcone_dir = (
            f"{basedir}/neutrino_corrected_maps_downsampled_4096"
            if args.nside == 4096
            else f"{basedir}/neutrino_corrected_maps"
        )

    else:
        import hdfstream

        remote_dir = hdfstream.open("cosma", "/")
        halo_format = f"{basedir}/halo_lightcone/lightcone{args.lightcone_nr}/lightcone_halos_{{snap_nr:04d}}.hdf5"
        lightcone_dir = f"{basedir}/healpix_maps/nside_{args.nside}"

    flamingo_soap_format = f"{basedir}/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"

    for formatted_path in [remote_dir, halo_format, lightcone_dir, flamingo_soap_format]:
        print(formatted_path)

    # 1. the shell, and a halo in it from FLAMINGO's halo lightcone
    shell_redshifts = np.loadtxt(
        flamingo_shell_redshift_file("L2p8" if "L2800" in args.box_res else "L1"), delimiter=","
    )

    shell_z = tuple(float(z) for z in shell_redshifts[args.shell_nr])
    snapshots = [int(n) for n in simulation.snapshots_in_range((max(shell_z[0] - 0.05, 0.0), shell_z[1] + 0.05))]

    print(f"\nshell {args.shell_nr}: {shell_z[0]:.4f} < z <= {shell_z[1]:.4f}, halo lightcone files of {snapshots}")

    candidates = flamingo_haloes_in_shell(
        halo_format,
        flamingo_soap_format,
        shell_z,
        snapshots,
        args.vector,
        np.radians(args.search_radius_deg),
        remote_dir,
        min_mass=10 ** (args.log_mass - 1.0) if args.log_mass_min is None else 10**args.log_mass_min,
    )

    # a halo in the middle of the shell, closest to the target mass
    distance = np.linalg.norm(candidates["LightconeCentre"], axis=1)
    r_shell = cosmo.comoving_distance(list(shell_z)).to_value("Mpc")
    middle = (distance > r_shell[0] + 0.25 * np.diff(r_shell)[0]) & (distance < r_shell[1] - 0.25 * np.diff(r_shell)[0])
    if not np.any(middle):
        middle[:] = True

    choice = np.flatnonzero(middle)[np.argmin(np.abs(np.log10(candidates["M200c"][middle]) - args.log_mass))]
    halo = {name: values[choice] for name, values in candidates.items()}
    halo_distance = float(np.linalg.norm(halo["LightconeCentre"]))

    print(
        f"\nFLAMINGO halo: snapshot {halo['SnapshotNumber']}, HaloCatalogueIndex {halo['HaloCatalogueIndex']}, "
        f"M200c {halo['M200c']:.3e} Msun, R200c {halo['R200c']:.3f} cMpc, z {halo['Redshift']:.4f}\n"
        f"\tlightcone centre {np.round(halo['LightconeCentre'], 3)} Mpc ({halo_distance:.2f} Mpc from the observer)\n"
        f"\tsnapshot centre {np.round(halo['SnapshotCentre'], 3)} Mpc"
    )

    # 2. FLAMINGO's observer position in the snapshot box

    if args.observer_position is not None:
        observer_position, source = np.asarray(args.observer_position, dtype=float), "given with --observer_position"
    else:
        observer_position, source = (
            FLAMINGO_observer_positions(args.lightcone_nr, simulation.box_res),
            f"estimated pre-defined tables",
        )

    print(f"\nFLAMINGO observer position: {np.round(observer_position, 3)} Mpc ({source})")

    # 3. the same shell from the snapshots, with FLAMINGO's geometry, in a beam around the halo
    zoom_angle = args.zoom_npix * hp.nside2resol(args.nside)  # [rad]
    halo_vector = halo["LightconeCentre"] / halo_distance

    particles, swiftlet_haloes, beam, shift = swiftlet_shell(
        simulation,
        cosmo,
        halo_vector,
        zoom_angle,
        shell_z,
        observer_position,
        boxsize,
        cell_size,
        simulation.soap_format,
    )

    position, corner = beam.observer_snapshot_coords
    print(
        f"\nSWIFTLET observer at the cell corner {np.round(corner, 3)} Mpc: its positions are moved by "
        f"{np.round(shift, 3)} Mpc into FLAMINGO's frame"
    )

    row = find_same_halo(swiftlet_haloes, halo["SnapshotNumber"], halo["HaloCatalogueIndex"], halo["LightconeCentre"])

    if row is None:
        print(
            "\nthe halo isn't in SWIFTLET's lightcone from the same snapshot: the snapshots fill the shell's "
            "redshifts differently near their edges"
        )

    else:
        swiftlet_centre = swiftlet_haloes["Lightcone/HaloCentre"].to_value("Mpc")[row]
        separation = np.linalg.norm(swiftlet_centre - halo["LightconeCentre"])
        print(
            f"\nSWIFTLET halo (same snapshot and HaloCatalogueIndex): centre {np.round(swiftlet_centre, 3)} Mpc, "
            f"{separation:.3f} Mpc from FLAMINGO's ({separation / halo['R200c']:.3f} R200c)"
        )

    # 4. the maps of both, on the same pixels and projection

    # select the pixels with an additional buffer of *1.05 for the angular radius of the zoom.
    pixels = np.sort(hp.query_disc(args.nside, np.asarray(halo_vector, dtype=float), zoom_angle * 1.05, inclusive=True))

    import lightcone_io.healpix_maps as hm

    flamingo_shell = hm.Shell(
        lightcone_dir,
        f"lightcone{args.lightcone_nr}",
        shell_nr=args.shell_nr,
        remote_dir=remote_dir,
    )

    pixel_maps = {
        "FLAMINGO": flamingo_maps(flamingo_shell, args.nside, pixels, halo_distance),
        "SWIFTLET": swiftlet_maps(particles, args.nside, pixels, halo_distance),
    }

    zooms = {
        row_name: {
            map_name: gnomonic_zoom(values, pixels, args.nside, halo_vector, zoom_angle, halo_distance)
            for map_name, values in maps.items()
        }
        for row_name, maps in pixel_maps.items()
    }

    def in_zoom(centres):
        # the haloes whose direction is inside the zoom
        if len(centres) == 0:
            return np.zeros(0, dtype=bool)
        return (centres / np.linalg.norm(centres, axis=1)[:, None]) @ halo_vector >= np.cos(zoom_angle)

    # The haloes of both lightcones in the zoom, above the same M200c, with their R200c.
    # FLAMINGO's are read around the zoom itself, as it can reach past the search radius
    # SWIFTLET's masses and radii are looked up in its SOAP catalogues

    min_mass = 10 ** (args.log_mass - 1.0)  # allow for identification of haloes within 1 dex

    near = flamingo_haloes_in_shell(
        halo_format,
        flamingo_soap_format,
        shell_z,
        snapshots,
        halo_vector,
        zoom_angle,
        remote_dir,
        min_mass=min_mass,
    )

    inside = in_zoom(near["LightconeCentre"])
    flamingo_in_zoom = (near["LightconeCentre"][inside], near["R200c"][inside])
    swiftlet_centres = swiftlet_haloes["Lightcone/HaloCentre"].to_value("Mpc")

    swiftlet_m200c, swiftlet_r200c = swiftlet_halo_properties(swiftlet_haloes, simulation.soap_format)
    keep = in_zoom(swiftlet_centres) & (swiftlet_m200c >= min_mass)
    swiftlet_in_zoom = (swiftlet_centres[keep], swiftlet_r200c[keep])

    print(
        f"\nhaloes in the zoom with M200c >= {min_mass:.2e} Msun: {np.count_nonzero(inside)} in FLAMINGO's lightcone, "
        f"{np.count_nonzero(keep)} in SWIFTLET's"
    )

    output_filename = output_dir / (
        f"{args.box_res}_{args.sim_name}_lightcone{args.lightcone_nr}_shell{args.shell_nr}_"
        f"snap{int(halo['SnapshotNumber'])}_halo{int(halo['HaloCatalogueIndex'])}.png"
    )

    plot_comparison(
        zooms,
        {"FLAMINGO": flamingo_in_zoom, "SWIFTLET": swiftlet_in_zoom},
        {
            "LightconeCentre": halo["LightconeCentre"],
            "R200c": halo["R200c"],
            "label": (
                f"{args.box_res}/{args.sim_name} lightcone{args.lightcone_nr}, shell {args.shell_nr}: "
                rf"M$_{{200c}}$ = {halo['M200c']:.2e} M$_\odot$, z = {halo['Redshift']:.3f}"
            ),
        },
        output_filename,
    )
    print(f"\nplot written to {output_filename}")
