#!/bin/env python
import sys
import numpy as np
import healpy as hp
import unyt
import lightcone_io.halo_catalogue as hc
import lightcone_io.smoothed_map as smoothed_map
import lightcone_io.kernel as kernel
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
import cmasher as cmr  # noqa: F401 register "cmr.*" colour maps used
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullFormatter
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from pathlib import Path


from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet import snapshot_units as sw_units
from swiftlet import swift_snapshot_redshift_conversion as nz

from map_zoom_on_halo import plot_settings

"""
The same as map_zoom_on_halo.py but for haloes from the snapshots and lightcones built from the snapshots with SnapshotBeam. 
For the fiducial L1_m9 FLAMINGO simulation:

    1. Find haloes with M200c close to 10^14 Msun in the SOAP catalogues of the snapshots covering 0.1 < z < 0.2.
        SnapshotBeam puts the observer at the centre of the box, and with orientation_seed=0 and no
        orientation lock that tile is not re-oriented, so a halo's position relative to the box centre
        (500, 500, 500) Mpc is its position in a SnapshotBeam lightcone.
    2. Keep the haloes, if the zoomed region around them is inside the zero'th snapshot box, at distances SnapshotBeam fills from
        the halo's own snapshot. Then select n_haloes from the remaining candidates
    3. For each halo, build a beam pointing at it, through a slab around the halo, and make small HEALPix maps of
        its particles around the halo:
            - SmoothedGasMass:  gas mass smoothed with the SPH kernel, as the lightcone's maps.
            - GasTemperature:   mass weighted temperature of the gas, smoothed with the SPH kernel.
            - GasLOSVelocity:   mass weighted peculiar velocity of the gas along the line of sight (away from the
                                    observer is positive), smoothed with the SPH kernel.
    4. Plot a row of the zooms for each halo. The masses are shown as the projected mass per comoving area at the
        distance of the halo.

Note 1) SnapshotLightcone re-orients the coordinates of the box tiles but not the velocities, so the velocities are
only along the right directions in the box tile (0, 0, 0), which is not re-oriented. The haloes are chosen so that
every particle used is in that tile, and this is checked.

Note 2) SnapshotBeam reads the snapshots and SOAP catalogues from the local FLAMINGO directories, so this needs to run
on COSMA.
"""

# maps of each halo: the particle type, properties and how each particle is added to the map, the colour map and
# the colour bar label
MAPS = {
    "SmoothedGasMass": {"ptype": "PartType0", "kind": "smoothed_mass", "cmap": "cmr.lilac",
                        "label": "gas", "units": r"$\Sigma_{\rm gas}$ [M$_\odot$ Mpc$^{-2}$]"},
    "GasTemperature": {"ptype": "PartType0", "kind": "mass_weighted", "cmap": "cmr.ember",
                       "label": "gas temperature", "units": r"$T_{\rm gas}$ [K]"},
    "GasLOSVelocity": {"ptype": "PartType0", "kind": "mass_weighted", "cmap": "RdBu_r",
                       "label": "gas l.o.s. velocity", "units": r"$v_{\rm los}$ [km s$^{-1}$]"},
}

# particle properties to read for each particle type
PROPERTIES = {"PartType0": ["Masses", "Temperatures", "Velocities"]}


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


def find_candidate_haloes(soap_format, boxsize_resolution, z_max, log_mass_range, z_min=0.1):
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

    :param  soap_format:        format string of the SOAP catalogue filenames (using {snap_nr})
    :type   soap_format:        str
    :param  boxsize_resolution: FLAMINGO box size and resolution label, e.g. "L1000N1800"
    :type   boxsize_resolution: str
    :param  z_max:              largest redshift to search
    :type   z_max:              float
    :param  log_mass_range:     minimum and maximum log10(M200c / Msun)
    :type   log_mass_range:     tuple
    """
    snapshot_numbers, _ = nz.flamingo_snapshot_redshift(boxsize_resolution)
    found = []
    for snap_nr in snapshot_numbers:
        z_lo, z_hi = nz.snapshot_redshift_range(int(snap_nr), boxsize_resolution)
        if z_lo >= z_max or z_hi <= 0. or z_lo<z_min:
            continue
        halo_cat = hc.SOAPCatalogue(soap_format, int(snap_nr), int(snap_nr))
        soap = halo_cat.read(int(snap_nr), ["InputHalos/HaloCatalogueIndex", "InputHalos/HaloCentre",
                                            "SO/200_crit/TotalMass", "SO/200_crit/SORadius"])
        mass = soap["SO/200_crit/TotalMass"].to_value("Msun")
        with np.errstate(divide="ignore"):
            keep = (np.log10(mass) >= log_mass_range[0]) & (np.log10(mass) <= log_mass_range[1])
        found.append({
            "SnapshotNumber": np.full(np.count_nonzero(keep), int(snap_nr)),
            "HaloCatalogueIndex": soap["InputHalos/HaloCatalogueIndex"].value.astype(int)[keep],
            "SOAPCentre": sw_units.drop_a_from_comoving_property(soap["InputHalos/HaloCentre"][keep]).to_value("Mpc"),
            "M200c": mass[keep],
            "R200c": sw_units.drop_a_from_comoving_property(soap["SO/200_crit/SORadius"][keep]).to_value("Mpc"),
        })
        print(f"\tsnapshot {snap_nr}: {np.count_nonzero(keep)} haloes in the mass range")
    return {name: np.concatenate([haloes[name] for haloes in found]) for name in found[0]}


def select_haloes(candidates, box_centre, boxsize, cosmo, boxsize_resolution, z_max, margin, n_haloes, target_log_mass):
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
    :param  boxsize_resolution: FLAMINGO box size and resolution label, e.g. "L1000N1800"
    :type   boxsize_resolution: str
    :param  z_max:              largest redshift of the haloes in the SnapshotBeam
    :type   z_max:              float
    :param  margin:             comoving distance [Mpc] around each halo that must also fit
    :type   margin:             np.ndarray
    :param  n_haloes:           number of haloes to choose
    :type   n_haloes:           int
    :param  target_log_mass:    log10(M200c / Msun) to choose the haloes closest to
    :type   target_log_mass:    float
    """
    offset = candidates["SOAPCentre"] - box_centre
    distance = np.linalg.norm(offset, axis=1)
    in_box = np.all(np.abs(offset) + margin[:, None] < 0.5 * boxsize, axis=1)

    snap_z_range = np.array([nz.snapshot_redshift_range(int(s), boxsize_resolution) for s in candidates["SnapshotNumber"]])
    z_near = comoving_distance_to_redshift(cosmo, np.maximum(distance - margin, 0.))
    z_far = comoving_distance_to_redshift(cosmo, distance + margin)
    same_snapshot = (z_near > snap_z_range[:, 0]) & (z_far <= snap_z_range[:, 1]) & (z_far <= z_max)

    usable = np.flatnonzero(in_box & same_snapshot)
    print(f"\n{len(distance)} candidate haloes: {np.count_nonzero(in_box)} away from the box faces, "
          f"{np.count_nonzero(same_snapshot)} filled from their own snapshot, {len(usable)} with both")
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
    # peculiar velocity along the line of sight to each particle, away from the observer is positive
    line_of_sight = gas_pos.value / np.linalg.norm(gas_pos.value, axis=1)[:, None]
    gas_vlos = np.sum(gas["Velocities"].to_value("km/s") * line_of_sight, axis=1)
    gas_temperature = gas["Temperatures"].to_value("K")

    gas_weight, gas_weighted = smooth_weighted_particles(gas_pos, gas["SmoothingLengths"].to("Mpc"), gas_mass,
                                                         np.stack([gas_temperature, gas_vlos], axis=1), nside, beam_pix)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_values = gas_weighted / gas_weight[:, None]

    gas_surface_density=gas_weight / (hp.nside2pixarea(nside) * distance**2)
    return {
        "SmoothedGasMass": gas_surface_density,
        "GasTemperature": np.where(gas_weight > 0, mean_values[:, 0], np.nan),
        "GasLOSVelocity": np.where(gas_weight > 0, mean_values[:, 1], np.nan),
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
    
    plot_settings()

    # fiducial L1_m9 simulation

    boxsize_resolution = "L1000N1800"
    sim_name = "HYDRO_FIDUCIAL"
    boxsize = 1000.  # [Mpc]
    box_centre = np.full(3, 0.5 * boxsize)  # (500, 500, 500) [Mpc]

    # update to local or remote path as needed...
    # currently to local path on COSMA
    sim_dir = f"/cosma8/data/dp004/flamingo/Runs/{boxsize_resolution}/{sim_name}"
    soap_format = f"{sim_dir}/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"
    
    # output dir of image, update as needed
    output_dir = "./example_outputs/snapshot_beam_zoom_on_haloes"
    Path(output_dir).mkdir(parents=True, exist_ok=True) # ensure output dir exists

    cosmo = Snapshot_Cosmology_For_Lightcone(f"{sim_dir}/snapshots").COSMO

    # haloes with M200c within 0.1 dex of 10^14 Msun at z < 0.2
    n_haloes = 2
    target_log_M200c = 14
    log_mass_width = 0.1
    z_max = 0.2

    # half width of the zooms and half depth of the beam around each halo, in units of R200c
    zoom_r200c = 4.
    depth_r200c = 4.

    # resolution of the maps: nside of the pixels about pixel_size across at the halo
    pixel_size = 0.01  # [cMpc]


    # 1. haloes near 10^14 Msun in the snapshots covering z < 0.2
    # get ID, centre, mass and radii of candidate haloes
    print(f"\nHaloes with M200c within {log_mass_width} dex of 10^{target_log_M200c:g} Msun, snapshots at z < {z_max}:")
    candidates = find_candidate_haloes(soap_format, boxsize_resolution, z_max, (target_log_M200c - log_mass_width, target_log_M200c + log_mass_width))

    # 2. find the haloes to zoom on. 
    # The region around each halo must stay in the box tile (0, 0, 0).
    # i.e. The beam around the halo + the zoom with the wider beam around it
    zoom_radius = zoom_r200c * candidates["R200c"]
    margin = np.sqrt((1.2 * zoom_radius)**2 + (depth_r200c * candidates["R200c"])**2)
    chosen = select_haloes(candidates, box_centre, boxsize, cosmo, boxsize_resolution, z_max, margin, n_haloes, target_log_M200c)

    # 3. make a beam that points at each halo and then place beam onto a map
    zooms, haloes = [], []
    for idx in chosen:
        halo = {name: values[idx] for name, values in candidates.items()} # store ID, centre, radii and masses 
        snap_nr = int(halo["SnapshotNumber"]) # halo snapshot number 
        offset = halo["SOAPCentre"] - box_centre # get offset of halo from the centre of the snapshot box
        distance = np.linalg.norm(offset) # dist from centre of snapshot to halo
        redshift = float(comoving_distance_to_redshift(cosmo, distance)) # redshift of the halo in the lightcone (from comoving distance to observer)
        beam_vector = offset / distance # 
        zoom_angle = zoom_r200c * halo["R200c"] / distance
        depth = depth_r200c * halo["R200c"]
        beam_redshift_range = tuple(comoving_distance_to_redshift(cosmo, np.array([distance - depth, distance + depth])))
        nside = int(2**np.round(np.log2(hp.nside2resol(1) * distance / pixel_size)))
        beam_pix = np.sort(hp.query_disc(nside, beam_vector, zoom_angle, inclusive=True))

        # give massive update on each selected candidate
        # show catalogue info
        print(f"\nhalo {halo['HaloCatalogueIndex']} of snapshot {snap_nr}:\tM200c {halo['M200c']:.4e} Msun, R200c {halo['R200c']:.3f} cMpc")
        # SOAP info 
        print(f"\n\tSOAP centre ({', '.join(f'{x:.2f}' for x in halo['SOAPCentre'])}) cMpc, relative to the box centre ({', '.join(f'{x:.2f}' for x in offset)}) cMpc")
        # Beam info
        beam_str =f"\n\tbeam vector ({', '.join(f'{x:.4f}' for x in beam_vector)})" # vector
        beam_str+=f"\n\tbeam {beam_redshift_range[0]:.4f} < z <= {beam_redshift_range[1]:.4f}" # redshift range 
        beam_str+=f"{np.degrees(1.2 * zoom_angle):.3f} deg, nside {nside}" # angular size
        print(beam_str)

        # create specific beam for selected halo
        SB = SnapshotBeam(boxsize_resolution=boxsize_resolution, simulation_name=sim_name, beam_vector=beam_vector,
                          orientation_seed=0, orientation_lock=None, verbose=1)
        
        beam_particles = {}
        for ptype, property_names in PROPERTIES.items():
            beam_particles[ptype] = SB.place_snapshot_particles_in_shell(
                lightcone_redshift_range=beam_redshift_range,
                ang_radius_deg=np.degrees(1.2 * zoom_angle),  # increase angular size to allow smoothing of particles
                property_names=property_names,
                particle_types=[ptype],
                )[ptype]

            # every particle must come from the box tile (0, 0, 0), where the velocities are not re-oriented
            coords = beam_particles[ptype]["Coordinates"].to_value("Mpc")
            if np.any(np.abs(coords) >= 0.5 * boxsize):
                raise RuntimeError(f"{ptype} particles placed outside the box tile (0, 0, 0)")

        # check the halo is where we expect in the SnapshotBeam: in tile (0, 0, 0) and not re-oriented
        beam_centre = SB.Snapshot2Lightcone(snap_nr, halo["SOAPCentre"][None, :], tile=(0, 0, 0))[(0, 0, 0)][0]
        if not np.allclose(beam_centre, offset, atol=1e-4):
            raise RuntimeError(f"halo placed at {beam_centre} in the SnapshotBeam, expected {offset}")

        # create the maps for each halo property 
        maps = halo_maps(beam_particles, nside, beam_pix, distance)
        del beam_particles # no longer need particle data

        # store the maps to plot later
        zooms.append({name: gnomonic_zoom(values, beam_pix, nside, beam_vector, zoom_angle, distance)
                      for name, values in maps.items()})
        
        # store halo info for plotting later
        haloes.append({
            "R200c": halo["R200c"],
            "distance": distance,
            "label": rf"$M_{{200c}}$ = {halo['M200c'] / 1e14:.2f}$\times 10^{{14}}$ M$_\odot$"+"\n"+rf"$z$ = {redshift:.3f}"
        })

    # 4. a row of zooms for each halo
    output_filename = f"{output_dir}/snapshot_beam_zoom_on_{n_haloes}_haloes.png"
    plot_halo_zooms(zooms, haloes, output_filename)
    print(f"\nZooms saved as: {output_filename}")