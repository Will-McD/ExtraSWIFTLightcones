#!/bin/env python
"""
The square beam counterpart of ``snapshot_beam_zoom_on_haloes``: zoom on haloes with SnapshotSquareBeam lightcones.
"""

import argparse
import os
import numpy as np
import healpy as hp
from pathlib import Path
from swiftlet.snapshot_cosmology import SnapshotCosmology
from swiftlet.snapshot_lightcone import SnapshotSquareBeam
from swiftlet.simulation_suites import Simulation
from swiftlet.example_plot_settings import plot_settings

# the halo search, the maps and the plot of the circular beam's example, in the same directory
from snapshot_beam_zoom_on_haloes import (
    PROPERTIES,
    SMOOTHING_MARGIN,
    comoving_distance_to_redshift,
    find_candidate_haloes,
    halo_maps,
    plot_halo_zooms,
    select_haloes,
    snapshot_boxsize,
    zoom_margin,
)


def aligned_gnomonic_projector(centre_vector, side_axes, r_npix, resolution):
    """
    A gnomonic projector centred on centre_vector, with its image x axis along side_axes[0].

    :param  centre_vector:  unit vector to centre the image on (the square beam's axis)
    :type   centre_vector:  np.ndarray, shape (3,)
    :param  side_axes:      the square beam's side_axes, two unit vectors perpendicular to centre_vector
    :type   side_axes:      tuple
    :param  r_npix:         number of image pixels from the centre to each edge
    :type   r_npix:         int
    :param  resolution:     size [rad] of an image pixel
    :type   resolution:     float
    """
    lon, lat = hp.vec2ang(np.asarray(centre_vector, dtype=float), lonlat=True)
    u, v = (np.asarray(a, dtype=float) for a in side_axes)
    step = 1e-3

    def projector(psi):
        return hp.projector.GnomonicProj(
            rot=[float(lon[0]), float(lat[0]), psi],
            xsize=2 * r_npix + 1,
            ysize=2 * r_npix + 1,
            reso=np.degrees(resolution) * 60,
        )

    # the angle of u in the image without a rotation about the centre, then the rotation that puts it along +x
    x, y = projector(0.0).vec2xy(centre_vector + step * u)
    angle = float(np.degrees(np.arctan2(y, x)))
    for psi in (angle, -angle, angle + 180.0, -angle + 180.0):
        gnom = projector(psi)
        x, y = gnom.vec2xy(centre_vector + step * u)
        if x > 0 and abs(y) < 1e-6 * abs(x):
            _, y_v = gnom.vec2xy(centre_vector + step * v)
            if y_v >= 0:
                raise RuntimeError("the projection would show the sky mirrored")
            return gnom
    raise RuntimeError("could not align the gnomonic projection with the square beam's sides")


def north_up_rotation(beam_vector):
    """
    Returns the square_rotation_deg [deg] that turns a square beam so that north is up and east is left, matching
    healpy.

    The direction is taken from healpy's own gnomonic projection, as snapshot_beam_zoom_on_haloes uses it, so the
    zooms of a square beam are drawn exactly as those of a circular one, also at the poles, where north is undefined.

    :param  beam_vector:    direction of the beam
    :type   beam_vector:    np.ndarray, shape (3,)
    """
    centre = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
    lon, lat = hp.vec2ang(centre, lonlat=True)
    gnom = hp.projector.GnomonicProj(rot=[float(lon[0]), float(lat[0])], xsize=3, ysize=3, reso=1.0)
    right = np.asarray(gnom.xy2vec(1e-3, 0.0), dtype=float).ravel() - centre
    return SnapshotSquareBeam.rotation_to(centre, right)


def square_zoom_pixels(nside, centre_vector, side_axes, half_width):
    """
    The HEALPix pixels (ring order, sorted) of a square on the sky, centred on centre_vector.

    :param  nside:          HEALPix resolution of the map
    :type   nside:          int
    :param  centre_vector:  unit vector of the centre of the square
    :type   centre_vector:  np.ndarray, shape (3,)
    :param  side_axes:      the two unit vectors along the sides of the square
    :type   side_axes:      tuple
    :param  half_width:     half the width of the square in the tangent plane, tan(half its angular side length)
    :type   half_width:     float
    """
    u, v = (np.asarray(a, dtype=float) for a in side_axes)
    corners = [centre_vector + half_width * (su * u + sv * v) for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    return np.sort(hp.query_polygon(nside, np.array(corners), inclusive=True))


def square_gnomonic_zoom(pix_values, selected_pix, nside, centre_vector, side_axes, r_npix, distance):
    """
    Gnomonic projection of the selected pixels in a HEALPix map, with the image axes along the square beam's sides.

    Returns a tuple of (image, extent of the image [Mpc] at the given comoving distance): x is along side_axes[0] and
    y along -side_axes[1], the sky as the observer sees it. Pixels outside the selected pixels are NaN.

    :param  pix_values:     value of each selected pixel
    :type   pix_values:     np.ndarray
    :param  selected_pix:   sorted indices of the selected pixels
    :type   selected_pix:   np.ndarray
    :param  nside:          HEALPix resolution of the map
    :type   nside:          int
    :param  centre_vector:  direction to centre the image on (the square beam's axis)
    :type   centre_vector:  np.ndarray, shape (3,)
    :param  side_axes:      the square beam's side_axes
    :type   side_axes:      tuple
    :param  r_npix:         number of image pixels from the centre to each edge
    :type   r_npix:         int
    :param  distance:       comoving distance [Mpc] used to convert the image extent from angles to lengths
    :type   distance:       float
    """

    def vec2index(x, y, z):
        # position of each pixel in pix_values, or the last entry (NaN) if it isn't a selected pixel
        pix = hp.vec2pix(nside, x, y, z)
        idx = np.clip(np.searchsorted(selected_pix, pix), 0, len(selected_pix) - 1)
        return np.where(selected_pix[idx] == pix, idx, len(selected_pix))

    gnom = aligned_gnomonic_projector(centre_vector, side_axes, r_npix, hp.nside2resol(nside))
    image = gnom.projmap(np.append(np.asarray(pix_values, dtype=np.float64), np.nan), vec2index)
    return image, np.asarray(gnom.get_extent()) * distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zoom on haloes in SnapshotSquareBeam lightcones of a FLAMINGO or COLIBRE simulation"
    )
    parser.add_argument(
        "--box_res",
        default="L1000N1800",
        help="box size and resolution label, e.g. L1000N1800 (FLAMINGO L1_m9) or L0400N3008 (COLIBRE L400_m7)",
    )
    parser.add_argument(
        "--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation, e.g. HYDRO_FIDUCIAL or Thermal"
    )
    parser.add_argument(
        "--soap_dir", default=None, help="directory of the SOAP catalogues, by default <simulation>/SOAP-HBT"
    )
    parser.add_argument("--n_haloes", type=int, default=3, help="number of haloes to zoom on")
    parser.add_argument("--log_mass", type=float, default=14.0, help="target log10(M200c / Msun) of the haloes")
    parser.add_argument("--log_mass_width", type=float, default=0.1, help="half width [dex] of the mass range")
    parser.add_argument("--z_max", type=float, default=0.2, help="largest redshift of the haloes")
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=0.01,
        help="size [cMpc] of the HEALPix pixels at the halo, which sets the nside of each map",
    )
    parser.add_argument(
        "--output_dir", default="./example_outputs/snapshot_square_beam_zoom_on_haloes", help="output directory"
    )
    args = parser.parse_args()
    plot_settings()

    # the simulation: its paths, output redshifts, cosmology and box, read locally on COSMA
    simulation = Simulation.from_label(args.box_res, args.sim_name)
    soap_format = simulation.soap_format
    if args.soap_dir is not None:
        soap_format = os.path.join(args.soap_dir, os.path.basename(soap_format))
    cosmo = SnapshotCosmology.from_simulation(simulation).COSMO
    boxsize = snapshot_boxsize(simulation)  # [Mpc]
    box_centre = np.full(3, 0.5 * boxsize)
    print(
        f"{simulation.box_res}/{simulation.name} ({simulation.suite.name}): box {boxsize:g} cMpc, "
        f"{len(simulation.output_redshifts)} snapshots, SOAP catalogues {soap_format}"
    )

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # haloes with M200c within log_mass_width dex of 10^log_mass Msun, out to z_max but inside the box
    z_box_corner = float(comoving_distance_to_redshift(cosmo, 0.5 * np.sqrt(3.0) * boxsize))
    z_max = min(args.z_max, z_box_corner)

    # half width of the zooms and half depth of the beam around each halo, in units of R200c
    zoom_r200c = 4.0
    depth_r200c = 4.0

    print(
        f"\nHaloes with M200c within {args.log_mass_width} dex of 10^{args.log_mass:g} Msun, snapshots at z < {z_max}:"
    )
    candidates = find_candidate_haloes(
        soap_format, simulation, z_max, (args.log_mass - args.log_mass_width, args.log_mass + args.log_mass_width)
    )

    # the region the square beam reaches around each halo must stay in the box tile (0, 0, 0): out to its corners,
    # sqrt(2) times the half width of the square, and the depth of the slab
    zoom_radius = zoom_r200c * candidates["R200c"]
    margin = zoom_margin(candidates["R200c"], zoom_r200c, depth_r200c)
    chosen = select_haloes(candidates, box_centre, cosmo, simulation, z_max, margin, args.n_haloes, args.log_mass)

    # a square beam pointing at each halo, and its maps
    zooms, haloes = [], []
    for idx in chosen:
        halo = {name: values[idx] for name, values in candidates.items()}
        snap_nr = int(halo["SnapshotNumber"])
        offset = halo["SOAPCentre"] - box_centre  # the halo relative to the observer, at the centre of the box
        distance = np.linalg.norm(offset)
        redshift = float(comoving_distance_to_redshift(cosmo, distance))
        beam_vector = offset / distance
        depth = depth_r200c * halo["R200c"]
        beam_redshift_range = tuple(
            comoving_distance_to_redshift(cosmo, np.array([distance - depth, distance + depth]))
        )

        # the maps' pixels, about pixel_size across at the halo, and the image: r_npix pixels from its centre to each
        # edge, zoom_r200c R200c across at the halo
        nside = int(2 ** np.round(np.log2(hp.nside2resol(1) * distance / args.pixel_size)))
        resolution = hp.nside2resol(nside)  # [rad]
        r_npix = int(np.ceil(zoom_r200c * halo["R200c"] / distance / resolution))
        half_width = (r_npix + 0.5) * resolution  # [rad] in the tangent plane, to the edge of the image

        # the square beam: the image's square and the smoothing margin, its sides the image's axes
        ang_side_length_deg = float(2.0 * np.degrees(np.arctan(SMOOTHING_MARGIN * half_width)))
        square_beam = SnapshotSquareBeam(
            simulation=simulation,
            beam_vector=beam_vector,
            ang_side_length_deg=ang_side_length_deg,
            square_rotation_deg=north_up_rotation(beam_vector),
            orientation_seed=0,
            orientation_lock=None,
            verbose=1,
        )
        side_axes = square_beam.side_axes
        beam_pix = square_zoom_pixels(nside, square_beam.beam_vec, side_axes, half_width)

        print(
            f"\nhalo {halo['HaloCatalogueIndex']} of snapshot {snap_nr}: "
            f"M200c {halo['M200c']:.4e} Msun, R200c {halo['R200c']:.3f} cMpc"
            f"\n\tSOAP centre ({', '.join(f'{x:.2f}' for x in halo['SOAPCentre'])}) cMpc, "
            f"relative to the box centre ({', '.join(f'{x:.2f}' for x in offset)}) cMpc"
            f"\n\tsquare beam along ({', '.join(f'{x:.4f}' for x in beam_vector)}), "
            f"{ang_side_length_deg:.3f} deg side, turned {square_beam.square_rotation_deg:.1f} deg, "
            f"{beam_redshift_range[0]:.4f} < z <= {beam_redshift_range[1]:.4f}, "
            f"nside {nside}"
        )

        beam_particles = {}
        for ptype, property_names in PROPERTIES.items():
            beam_particles[ptype] = square_beam.place_snapshot_particles_in_shell(
                lightcone_redshift_range=beam_redshift_range,
                property_names=property_names,
                particle_types=[ptype],
            )[ptype]

        # check the halo is where it is expected in the beam, i.e. in tile (0, 0, 0), which is not re-oriented
        beam_centre = square_beam.Snapshot2Lightcone(snap_nr, halo["SOAPCentre"][None, :], tile=(0, 0, 0))[0]
        if not np.allclose(beam_centre, offset, atol=1e-4):
            raise RuntimeError(f"halo placed at {beam_centre} in the square beam, expected {offset}")

        maps = halo_maps(beam_particles, nside, beam_pix, distance)
        del beam_particles

        zooms.append(
            {
                name: square_gnomonic_zoom(values, beam_pix, nside, square_beam.beam_vec, side_axes, r_npix, distance)
                for name, values in maps.items()
            }
        )
        haloes.append(
            {
                "R200c": halo["R200c"],
                "distance": distance,
                "label": rf"$M_{{200c}}$ = {halo['M200c'] / 1e14:.2f}$\times 10^{{14}}$ M$_\odot$"
                + "\n"
                + rf"$z$ = {redshift:.3f}",
            }
        )

    # a row of zooms for each halo
    output_filename = (
        f"{output_dir}/snapshot_square_beam_zoom_on_{len(chosen)}_haloes.{simulation.box_res}.{simulation.name}.png"
    )
    plot_halo_zooms(zooms, haloes, output_filename)
    print(f"\nSaved as: {output_filename}")
