#!/bin/env python
import argparse
import os

import h5py
import numpy as np
import unyt
import healpy as hp
import lightcone_io.smoothed_map as smoothed_map
import lightcone_io.kernel as kernel
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
import cmasher as cmr  # noqa: F401 register "cmr.*" colour maps used
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullFormatter
from pathlib import Path

from swiftlet.snapshot_cosmology import SnapshotCosmology
from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet.soap_reader import select_soap_rows
from swiftlet.simulation_suites import Simulation, output_redshifts
from swiftlet.snapshot_redshifts import snapshot_redshift_range
from swiftlet.example_plot_settings import plot_settings


"""
The same as map_zoom_on_halo.py but for haloes from the snapshots using lightcones built with SnapshotBeam. 
For the fiducial L1_m9 FLAMINGO simulation select haloes using the following criteria:

    1. Find haloes with M200c close to 10^14 Msun in the SOAP catalogues of the snapshots covering 0.1 < z < 0.2.
    2. Keep the haloes if the zoomed region around them is inside the zero'th snapshot box, at distances SnapshotBeam fills from
        the halo's own snapshot. Then select n_haloes from the remaining candidates
    3. For each halo, build a beam pointing at it, through a slab around the halo, and make small HEALPix maps of
        its particles around the halo:
            - SmoothedGasMass:  gas mass smoothed with the SPH kernel, as the lightcone's maps.
            - GasTemperature:   mass weighted temperature of the gas, smoothed with the SPH kernel.
    4. Plot a row of the zooms for each halo. The masses are shown as the projected mass per comoving area at the
        distance of the halo.

Notes 
    1) SnapshotLightcone re-orients the coordinates of the box tiles but not the velocities, so the velocities are
only along the right directions in the box tile (0, 0, 0), which is not re-oriented. The haloes are chosen so that
every particle used is in that tile, and this is checked.

    2) SnapshotBeam reads the snapshots and SOAP catalogues from the local FLAMINGO directories, so this needs to run
on COSMA.
"""

# maps of each halo: the particle type, properties and how each particle is added to the map, the colour map and
# the colour bar label
MAPS = {
    "SmoothedGasMass": {"ptype": "PartType0", "kind": "smoothed_mass", "cmap": "cmr.lilac",
                        "label": "gas", "units": r"$\Sigma_{\rm gas}$ [M$_\odot$ Mpc$^{-2}$]"},
    "GasTemperature": {"ptype": "PartType0", "kind": "mass_weighted", "cmap": "cmr.ember",
                       "label": "gas temperature", "units": r"$T_{\rm gas}$ [K]"},
}

# particle properties to read for each particle type
PROPERTIES = {"PartType0": ["Masses", "Temperatures"]}


def comoving_distance_to_redshift(cosmo, distance, z_max=3.0, n_grid=20000):
    """
    Redshift at the given comoving distances, interpolated from a grid of redshifts.

    Returns the redshifts, the same shape as distance.

    :param  cosmo:      astropy cosmology
    :type   cosmo:      astropy.cosmology.FLRW
    :param  distance:   comoving distances [Mpc]
    :type   distance:   float or np.ndarray
    :param  z_max:      largest redshift of the grid
    :type   z_max:      float
    :param  n_grid:     number of redshifts in the grid
    :type   n_grid:     int
    """

    z_grid = np.linspace(0., z_max, n_grid)
    r_grid = cosmo.comoving_distance(z_grid).to_value("Mpc")
    return np.interp(distance, r_grid, z_grid)

def snapshot_boxsize(simulation):
    """
    Sidelength of the snapshot box of a simulation, from the header of its redshift 0 snapshot.
    Returns the box sidelength [comoving Mpc].
    :param  simulation: the simulation
    :type   simulation: Simulation
    """
    with h5py.File(simulation.snapshot_file(simulation.z0_snapshot, 0), "r") as snapshot:
        boxsize = np.ravel(snapshot["Header"].attrs["BoxSize"])[0]
        length_cgs = float(np.ravel(snapshot["Units"].attrs["Unit length in cgs (U_L)"])[0])
    return unyt.unyt_quantity(boxsize * length_cgs, "cm").to_value("Mpc")

def bin_particles(part_pos, part_val, nside, selected_pix):
    """
    Add the particles to the pixels they are in. 
    Assume a HEALPix map that is ring ordered. 

    Returns the value of each selected pixel.

    :param  part_pos:       comoving positions of the particles relative to the observer, shape (N, 3)
    :type   part_pos:       np.ndarray
    :param  part_val:       value of each particle to add to the map, e.g. masses
    :type   part_val:       np.ndarray
    :param  nside:          HEALPix resolution of the map
    :type   nside:          int
    :param  selected_pix:   sorted indices of the selected pixels
    :type   selected_pix:   np.ndarray
    """

    map_values = np.zeros(len(selected_pix))
    pix = hp.vec2pix(nside, part_pos[:, 0], part_pos[:, 1], part_pos[:, 2])
    idx = np.clip(np.searchsorted(selected_pix, pix), 0, len(selected_pix) - 1)
    keep = selected_pix[idx] == pix
    np.add.at(map_values, idx[keep], part_val[keep])
    return map_values


def find_candidate_haloes(soap_format, simulation, z_max, log_mass_range, z_min=0.):
    """
    Find the haloes with M200c in a range in the SOAP catalogues of the snapshots covering z <= z_max.

    Returns a dict of arrays, one entry per halo:
        {
            "SnapshotNumber",
            "HaloCatalogueIndex",
            "SOAPCentre" [cMpc],
            "M200c" [Msun],
            "R200c" [cMpc],
        }

    :param  soap_format:    format string of the SOAP catalogue filenames (using {snap_nr})
    :type   soap_format:    str
    :param  simulation:     the simulation, for its snapshots' redshifts
    :type   simulation:     Simulation
    :param  z_max:          largest redshift to search
    :type   z_max:          float
    :param  z_min:          lowest redshift to search
    :type   z_min:          float
    :param  log_mass_range: minimum and maximum log10(M200c / Msun)
    :type   log_mass_range: tuple
    """

    found = []
    for snap_nr in simulation.snapshots_in_range((0., z_max)):
        z_lo, z_hi = simulation.snapshot_redshift_range(int(snap_nr))
        # only the snapshots overlapping 0 < z < z_max, not those that only touch it
        if z_lo >= z_max or z_hi <= 0. or z_lo<z_min:
            continue
        if not os.path.exists(soap_format.format(snap_nr=int(snap_nr))):
            print("\tsnapshot {snap_nr}: no SOAP catalogue, skipped")
            continue
        units = {"SO/200_crit/TotalMass": "Msun", "InputHalos/HaloCatalogueIndex": "dimensionless",
                 "InputHalos/HaloCentre": "Mpc", "SO/200_crit/SORadius": "Mpc"}
        soap = select_soap_rows(
            soap_format.format(snap_nr=int(snap_nr)), "SO/200_crit/TotalMass",
            lambda mass: (mass.to_value("Msun") >= 10**log_mass_range[0]) & (mass.to_value("Msun") <= 10**log_mass_range[1]),
            list(units), units=units)
        found.append({
            "SnapshotNumber": np.full(len(soap["row"]), int(snap_nr)),
            "HaloCatalogueIndex": soap["InputHalos/HaloCatalogueIndex"].astype(int),
            "SOAPCentre": soap["InputHalos/HaloCentre"],
            "M200c": soap["SO/200_crit/TotalMass"],
            "R200c": soap["SO/200_crit/SORadius"],
        })
        print(f"\tsnapshot {snap_nr}: {len(soap['row'])} haloes in the mass range")
    if not found:
        raise FileNotFoundError(f"no SOAP catalogues found ({soap_format}) of the snapshots at z < {z_max}")
    return {name: np.concatenate([haloes[name] for haloes in found]) for name in found[0]}


def select_haloes(candidates, box_centre, boxsize, cosmo, simulation, z_max, margin, n_haloes, target_log_mass, z_min=0.):
    """
    Choose the haloes to zoom on. 
    Keep the haloes whose surrounding zoomed in region are
        - inside the snapshot box, without crossing its faces. e.g. SnapshotBeam shows them being in tile (0, 0, 0)
        - A distance from the box centre, below z_max and above z_min
    
    Choose the n_haloes from the candidates, that have M200c closest to the target. 

    Returns the indices of the chosen haloes in candidates.

    :param  candidates:         haloes from find_candidate_haloes
    :type   candidates:         dict
    :param  box_centre:         centre of the snapshot box [Mpc]
    :type   box_centre:         np.ndarray
    :param  boxsize:            snapshot box sidelength [Mpc]
    :type   boxsize:            float
    :param  cosmo:              cosmology of the snapshots
    :type   cosmo:              astropy.cosmology.FLRW
    :param  simulation:         the simulation, for its snapshots' redshifts
    :type   simulation:         Simulation
    :param  z_max:              largest redshift of the haloes in the SnapshotBeam
    :type   z_max:              float
    :param  margin:             comoving distance [Mpc] around each halo that must also fit
    :type   margin:             np.ndarray
    :param  n_haloes:           number of haloes to choose
    :type   n_haloes:           int
    :param  target_log_mass:    log10(M200c / Msun) to choose the haloes closest to
    :type   target_log_mass:    float
    """

    # haloes offset from the centre of the box
    offset = candidates["SOAPCentre"] - box_centre
    distance = np.linalg.norm(offset, axis=1) # distance to the haloes
    #in_box = np.all(np.abs(offset) + margin[:, None] < 0.5 * boxsize, axis=1) # check if in tile (0,0,0)

    snap_z_range = np.array([simulation.snapshot_redshift_range(int(s)) for s in candidates["SnapshotNumber"]])
    z_near = comoving_distance_to_redshift(cosmo, np.maximum(distance - margin, 0.))
    z_far = comoving_distance_to_redshift(cosmo, distance + margin)
    same_snapshot = (z_near > snap_z_range[:, 0]) & (z_far <= snap_z_range[:, 1]) & (z_far <= z_max) & (z_near >= z_min)

    usable = np.flatnonzero(same_snapshot)
    print(f"\n{len(distance)} candidate haloes filled from their own snapshot, {len(usable)} with both")
    if len(usable) < n_haloes:
        raise ValueError(f"only {len(usable)} haloes can be used, try a wider mass range")

    # closest in mass first, taking one halo per snapshot before taking a second from any snapshot
    usable = usable[np.argsort(np.abs(np.log10(candidates["M200c"][usable]) - target_log_mass))]
    chosen, used_snapshots = [], set()
    for idx in usable:
        if candidates["SnapshotNumber"][idx] not in used_snapshots:
            chosen.append(idx)
            used_snapshots.add(candidates["SnapshotNumber"][idx])
        if len(chosen) == n_haloes:
            break
    for idx in usable:
        if len(chosen) == n_haloes:
            break
        if idx not in chosen:
            chosen.append(idx)
    return np.array(chosen)



def smooth_weighted_particles(part_pos, part_hsml, part_weight, part_values, nside, selected_pix):
    """
    Smooth the particles onto the selected pixels of a HEALPix map (ring order) with their SPH kernel, as
    lightcone_io.smoothed_map.make_sky_map does, for the weight and the weighted values of the particles at once.

    Returns a tuple of (smoothed weight of each pixel, smoothed weight * value of each pixel, shape (npix, nvalues)).

    :param  part_pos:       comoving positions of the particles relative to the observer
    :type   part_pos:       unyt.unyt_array, shape (N, 3)
    :param  part_hsml:      comoving smoothing lengths of the particles
    :type   part_hsml:      unyt.unyt_array
    :param  part_weight:    weight of each particle, e.g. masses
    :type   part_weight:    np.ndarray
    :param  part_values:    values of each particle to weight, shape (N, nvalues)
    :type   part_values:    np.ndarray
    :param  nside:          HEALPix resolution of the map
    :type   nside:          int
    :param  selected_pix:   sorted indices of the selected pixels
    :type   selected_pix:   np.ndarray
    """
    part_weighted = part_weight[:, None] * part_values
    max_pixrad = hp.pixelfunc.max_pixrad(nside)
    ang_hsml = np.asarray(smoothed_map.find_angular_smoothing_length(part_pos, part_hsml))
    pos = np.asarray(part_pos.value)

    # particles smaller than a pixel update only the pixel they are in
    single_pixel = ang_hsml * kernel.kernel_gamma < max_pixrad
    weight_map = bin_particles(pos[single_pixel], part_weight[single_pixel], nside, selected_pix)
    weighted_map = np.stack([bin_particles(pos[single_pixel], part_weighted[single_pixel, i], nside, selected_pix)
                             for i in range(part_values.shape[1])], axis=1)

    # particles larger than a pixel are smoothed over the pixels within their kernel
    for part_nr in np.flatnonzero(~single_pixel):
        pix, kernel_weight = smoothed_map.explode_particle(nside, pos[part_nr], 1., ang_hsml[part_nr])
        idx = np.clip(np.searchsorted(selected_pix, pix), 0, len(selected_pix) - 1)
        keep = selected_pix[idx] == pix
        weight_map[idx[keep]] += part_weight[part_nr] * kernel_weight[keep]
        weighted_map[idx[keep]] += part_weighted[part_nr] * kernel_weight[keep, None]
    return weight_map, weighted_map


def gnomonic_zoom(pix_values, selected_pix, nside, centre_vector, zoom_angle, distance):
    """
    Gnomonic projection of the selected pixels in a HEALPix map. 

    Returns a tuple of (image, extent of the image [Mpc] at the given comoving distance). 
    Pixels outside the selected pixels are NaN.

    :param  pix_values:     value of each selected pixel
    :type   pix_values:     np.ndarray
    :param  selected_pix:   sorted indices of the selected pixels
    :type   selected_pix:   np.ndarray
    :param  nside:          HEALPix resolution of the map
    :type   nside:          int
    :param  centre_vector:  direction to centre the image on
    :type   centre_vector:  np.ndarray, shape (3,)
    :param  zoom_angle:     angular radius [rad] of the image
    :type   zoom_angle:     float
    :param  distance:       comoving distance [Mpc] used to convert the image extent from angles to lengths
    :type   distance:       float
    """
    lon, lat = hp.vec2ang(np.asarray(centre_vector, dtype=float), lonlat=True)
    resolution = hp.nside2resol(nside)  # [rad]
    r_npix = int(np.ceil(zoom_angle / resolution))

    def vec2index(x, y, z):
        # position of each pixel in pix_values, or the last entry (NaN) if it isn't a selected pixel
        pix = hp.vec2pix(nside, x, y, z)
        idx = np.clip(np.searchsorted(selected_pix, pix), 0, len(selected_pix) - 1)
        return np.where(selected_pix[idx] == pix, idx, len(selected_pix))

    gnom = hp.projector.GnomonicProj(rot=[float(lon[0]), float(lat[0])], xsize=2 * r_npix + 1, ysize=2 * r_npix + 1,
                                     reso=np.degrees(resolution) * 60)
    image = gnom.projmap(np.append(np.asarray(pix_values, dtype=np.float64), np.nan), vec2index)
    return image, np.asarray(gnom.get_extent()) * distance

def rescale_temperature_to_lightcone_redshift(Temp_snap, z_snap, z_lc):
    """
    Approximate the gas temperature at the lightcone redshift (z_lc)
    given the particle's actual temperature at its snapshot redshift (z_snap).

    Assume purely adiabatic evolution over the gap between the
    snapshot and lightcone redshift. 
    """
    return Temp_snap * ((1.0 + z_lc) / (1.0 + z_snap)) ** 2

def halo_maps(beam_particles, nside, beam_pix, distance):
    """
    Make the maps of MAPS from the particles of a beam, for the selected pixels around a halo.

    Returns {map name: value of each selected pixel}.

    :param  beam_particles: particles placed in the beam, per particle type
    :type   beam_particles: dict
    :param  nside:          HEALPix resolution of the maps
    :type   nside:          int
    :param  beam_pix:       sorted indices of the selected pixels
    :type   beam_pix:       np.ndarray
    :param  distance:       comoving distance [Mpc] of the halo, for the surface densities
    :type   distance:       float
    """
    gas = beam_particles["PartType0"]
    gas_pos = gas["Coordinates"].to("Mpc")
    gas_mass = gas["Masses"].to_value("Msun")
    gas_temperature = gas["Temperatures"].to_value("K")

    gas_weight, gas_weighted = smooth_weighted_particles(gas_pos, gas["SmoothingLengths"].to("Mpc"), gas_mass, np.stack([gas_temperature], axis=1), nside, beam_pix)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_values = gas_weighted / gas_weight[:, None]

    gas_surface_density=gas_weight / (hp.nside2pixarea(nside) * distance**2)
    return {
            "SmoothedGasMass": gas_surface_density,
            "GasTemperature": np.where(gas_weight > 0, mean_values[:, 0], np.nan),
        }


def draw_scale_bar(ax, extent, distance, fraction=0.25, colour="white"):
    """
    Draw an angular scale bar in the bottom left of a zoom, with a round length in arcmin of 
    the given fraction of the image width.

    :param  ax:         axes of the zoom
    :type   ax:         matplotlib.axes._axes.Axes
    :param  extent:     (x_min, x_max, y_min, y_max) of the image [cMpc]
    :type   extent:     tuple
    :param  distance:   comoving distance [cMpc] of the halo, used to convert the image extent to angles
    :type   distance:   float
    :param  fraction:   length of the bar, as a fraction of the image width, before rounding down
    :type   fraction:   float
    :param  colour:     colour of the bar and its label
    :type   colour:     str
    """
    x_min, x_max, y_min, y_max = extent
    width = abs(x_max - x_min)
    width_arcmin = np.degrees(width / distance) * 60.
    lengths_arcmin = np.array([0.5, 1., 2., 5., 10., 15., 20., 30., 60.])
    length_arcmin = lengths_arcmin[lengths_arcmin <= fraction * width_arcmin].max(initial=lengths_arcmin[0])
    length = np.radians(length_arcmin / 60.) * distance  # [cMpc] in the image
    x0 = min(x_min, x_max) + 0.08 * width
    y0 = min(y_min, y_max) + 0.08 * abs(y_max - y_min)
    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    ax.plot([x0, x0 + length], [y0, y0], color=colour, lw=1.5, solid_capstyle="butt", path_effects=outline)
    ax.text(x0 + 0.5 * length, y0 + 0.03 * width, 
    f"{length_arcmin:g} arcmin", ha="center", va="bottom", color=colour,
            fontsize=10, path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])


def plot_halo_zooms(zooms, haloes, output_filename, percentiles=(1., 99.9), r200c_col=None):
    """
    Plot a row of zooms for each halo, a column for each map of MAPS sharing a colour scale.

    :param  zooms:              list, for each halo, of {map name: (image, extent)}
    :type   zooms:              list
    :param  haloes:             list, for each halo, of {"R200c" [cMpc], "label"}
    :type   haloes:             list
    :param  output_filename:    path to write the plot to
    :type   output_filename:    str
    :param  percentiles:        percentiles of the pixels of each column used as the colour scale limits
    :type   percentiles:        tuple
    :param  r200c_col:          A string indicating the colour of line indicating R200c overlaid on the halo. 
                                    If None, do not overlay R200c on the halo. 
    :type   r200c_col:          str, None
    """
    nrows, ncols = len(zooms), len(MAPS)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.55 * nrows + 0.6), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 200)

    for col, (map_name, settings) in enumerate(MAPS.items()):
        values = np.concatenate([zoom[map_name][0][np.isfinite(zoom[map_name][0])] for zoom in zooms])
        cmap = mpl.colormaps.get_cmap(settings["cmap"]).copy()
        cmap.set_bad("grey")
        if settings["kind"] == "mass_weighted" and map_name == "GasLOSVelocity":
            # symmetric about zero
            vmax = np.percentile(np.abs(values), percentiles[1])
            norm, extend = Normalize(vmin=-vmax, vmax=vmax), "both"
        else:
            positive = values[values > 0]
            vmin, vmax = np.percentile(positive, percentiles)
            norm, extend = LogNorm(vmin=vmin, vmax=vmax), "min"
            cmap.set_under(cmap(0.))

        for row, (zoom, halo) in enumerate(zip(zooms, haloes)):
            image, extent = zoom[map_name]
            if isinstance(norm, LogNorm):
                # empty pixels at the bottom of the colour scale, pixels outside the zoom (NaN) in grey
                image = np.where(image <= 0, 1e-3 * norm.vmin, image)
            ax = axs[row, col]
            img = ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=extent, cmap=cmap, norm=norm,
                            interpolation="none")
            r200c = halo["R200c"]
            ax.plot(r200c * np.cos(theta), r200c * np.sin(theta), color=r200c_col, lw=0.8, ls="--",
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="black")])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            draw_scale_bar(ax, extent, halo["distance"])
            if row == 0:
                ax.set_title(settings["label"], fontsize=10)
            if col == 0:
                ax.set_ylabel(halo["label"], fontsize=10)

        cbar = fig.colorbar(img, ax=axs[:, col], location="bottom", shrink=0.9, pad=0.02, label=settings["units"],
                            extend=extend)
        cbar.ax.xaxis.set_minor_formatter(NullFormatter()) 

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Zoom on haloes in SnapshotBeam lightcones of a FLAMINGO or COLIBRE simulation")
    parser.add_argument("--box_res", default="L1000N1800",
                        help="box size and resolution label, e.g. L1000N1800 (FLAMINGO L1_m9) or L0400N3008 (COLIBRE L400_m7)")
    parser.add_argument("--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation, e.g. HYDRO_FIDUCIAL or Thermal")
    parser.add_argument("--soap_dir", default=None, help="directory of the SOAP catalogues, by default <simulation>/SOAP-HBT")
    parser.add_argument("--n_haloes", type=int, default=3, help="number of haloes to zoom on")
    parser.add_argument("--log_mass", type=float, default=14., help="target log10(M200c / Msun) of the haloes")
    parser.add_argument("--log_mass_width", type=float, default=0.1, help="half width [dex] of the mass range")
    parser.add_argument("--z_max", type=float, default=0.2, help="largest redshift of the haloes")
    parser.add_argument("--pixel_size", type=float, default=0.02,
                        help="size [cMpc] of the HEALPix pixels at the halo, which sets the nside of each map")
    parser.add_argument("--output_dir", default="./example_outputs/snapshot_beam_zoom_on_haloes",
                        help="output directory")
    args = parser.parse_args()
    plot_settings()

    # the simulation: its paths, output redshifts, cosmology and box, read locally on COSMA
    simulation = Simulation.from_label(args.box_res, args.sim_name)
    soap_dir = args.soap_dir or f"{simulation.directory}/SOAP-HBT"
    soap_format = f"{soap_dir}/halo_properties_{{snap_nr:04d}}.hdf5"
    cosmo = SnapshotCosmology.from_simulation(simulation).COSMO
    boxsize = snapshot_boxsize(simulation)  # [Mpc]
    box_centre = np.full(3, 0.5 * boxsize)
    print(f"{simulation.box_res}/{simulation.name} ({simulation.suite.name}): box {boxsize:g} cMpc, {len(simulation.output_redshifts)} snapshots, SOAP catalogues {soap_format}")

    # output dir of image, update as needed
    output_dir = "./example_outputs/snapshot_beam_zoom_on_haloes"
    Path(output_dir).mkdir(parents=True, exist_ok=True) # ensure output dir exists

    # haloes with M200c within 0.1 dex of 10^14 Msun at z < 0.2
    target_log_M200c = args.log_mass
    log_mass_width = args.log_mass_width
    z_box_corner = float(comoving_distance_to_redshift(cosmo, 0.5 * np.sqrt(3.) * boxsize))
    z_max = min(args.z_max, z_box_corner)
    n_haloes = args.n_haloes

    # half width of the zooms and half depth of the beam around each halo, in units of R200c
    zoom_r200c = 4.
    depth_r200c = 4.

    # resolution of the maps: nside of the pixels about pixel_size across at the halo
    pixel_size = 0.01  # [cMpc]

    # Find haloes near 10^14 Msun in the snapshots
    # Get ID, centre, mass and radii of candidate haloes
    print(f"\nHaloes with M200c within {log_mass_width} dex of 10^{target_log_M200c:g} Msun, snapshots at z < {z_max}:")
    candidates = find_candidate_haloes(soap_format, simulation, z_max, (target_log_M200c - log_mass_width, target_log_M200c + log_mass_width))
    
    # Find the haloes to zoom on. 
    # The region around each halo must stay in the box tile (0, 0, 0).
    # i.e. The beam around the halo + the zoom with the wider beam around it
    zoom_radius = zoom_r200c * candidates["R200c"]
    margin = np.sqrt((1.2 * zoom_radius)**2 + (depth_r200c * candidates["R200c"])**2)
    chosen = select_haloes(candidates, box_centre, boxsize, cosmo, simulation, z_max, margin, n_haloes, target_log_M200c)

    # Make a beam that points at each halo and then place beam onto a map
    zooms, haloes = [], []
    for idx in chosen:
        halo = {name: values[idx] for name, values in candidates.items()} # store ID, centre, radii and masses 
        snap_nr = int(halo["SnapshotNumber"]) # halo snapshot number 
        offset = halo["SOAPCentre"] - box_centre # get offset of halo from the centre of the snapshot box
        distance = np.linalg.norm(offset) # dist from centre of snapshot to halo
        redshift = float(comoving_distance_to_redshift(cosmo, distance)) # redshift of the halo in the lightcone (from comoving distance to observer)
        beam_vector = offset / distance # beams direction
        zoom_angle = zoom_r200c * halo["R200c"] / distance # angular size of zoom region
        depth = depth_r200c * halo["R200c"] # depth of zoom region 
        beam_redshift_range = tuple(comoving_distance_to_redshift(cosmo, np.array([distance - depth, distance + depth])))
        nside = int(2**np.round(np.log2(hp.nside2resol(1) * distance / pixel_size))) # get 'nside' of zoomed region from pixel resolution
        beam_pix = np.sort(hp.query_disc(nside, beam_vector, zoom_angle, inclusive=True)) # find pixels within the zoomed region. 

        # give update on each selected candidate:
        # show catalogue info
        print(f"\nhalo {halo['HaloCatalogueIndex']} of snapshot {snap_nr}:\tM200c {halo['M200c']:.4e} Msun, R200c {halo['R200c']:.3f} cMpc")
        # SOAP info 
        print(f"\n\tSOAP centre ({', '.join(f'{x:.2f}' for x in halo['SOAPCentre'])}) cMpc, relative to the box centre ({', '.join(f'{x:.2f}' for x in offset)}) cMpc")
        # Beam info
        beam_str =f"\n\tbeam vector ({', '.join(f'{x:.4f}' for x in beam_vector)})" # vector
        beam_str+=f"\n\tbeam {beam_redshift_range[0]:.4f} < z <= {beam_redshift_range[1]:.4f}" # redshift range 
        beam_str+=f"{np.degrees(1.2 * zoom_angle):.3f} deg, nside {nside}" # angular size
        print(beam_str)

        # create a beam for selected halo
        SB = SnapshotBeam(simulation_object=simulation, beam_vector=beam_vector, orientation_seed=0, orientation_lock=None, verbose=1)
        
        beam_particles = {}
        for ptype, property_names in PROPERTIES.items():
            beam_particles[ptype] = SB.place_snapshot_particles_in_shell(
                lightcone_redshift_range=beam_redshift_range,
                ang_radius_deg=np.degrees(1.2 * zoom_angle),  # increase angular size to allow smoothing of particles
                property_names=property_names,
                particle_types=[ptype],
                )[ptype]


        # check the halo is where it is expected in the SnapshotBeam,  i.e., in tile (0, 0, 0) and not re-oriented
        beam_centre = SB.Snapshot2Lightcone(snap_nr, halo["SOAPCentre"][None, :], tile=(0, 0, 0))[(0, 0, 0)][0]
        if not np.allclose(beam_centre, offset, atol=1e-4):
            raise RuntimeError(f"halo placed at {beam_centre} in the SnapshotBeam, expected {offset}")

        # create the maps for each halo property 
        maps = halo_maps(beam_particles, nside, beam_pix, distance)
        del beam_particles # no longer need particle data

        # store the maps to plotting
        zooms.append({name: gnomonic_zoom(values, beam_pix, nside, beam_vector, zoom_angle, distance) for name, values in maps.items()})
        
        # store halo info for plotting
        haloes.append({
            "R200c": halo["R200c"],
            "distance": distance,
            "label": rf"$M_{{200c}}$ = {halo['M200c'] / 1e14:.2f}$\times 10^{{14}}$ M$_\odot$"+"\n"+rf"$z$ = {redshift:.3f}"
        })

    # plot a row of zooms for each halo
    output_filename = (f"{output_dir}/snapshot_beam_zoom_on_{len(chosen)}_haloes.{simulation.box_res}.{simulation.name}.png")
    plot_halo_zooms(zooms, haloes, output_filename)
    print(f"\nSaved as: {output_filename}")