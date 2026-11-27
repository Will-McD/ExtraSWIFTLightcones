#!/bin/env python
"""
Example script to show the placement of haloes from the SOAP catalogues into a SnapshotSquareBeam lightcone shell.

The square beam counterpart of snapshot_beam_halo_placement.py: the same haloes, simulation and options, in a beam with
a square footprint on the sky.
"""

import os
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import hdfstream
from swiftlet.snapshot_lightcone import SnapshotSquareBeam
from swiftlet.simulation_suites import Simulation
from swiftlet.example_plot_settings import plot_settings, cmap_MCG


# the SOAP lookup, the plot settings and the server are those of the circular beam's example, in the same directory
from snapshot_beam_halo_placement import (
    MARKER_SIZE_POWER,
    MARKER_SIZE_RANGE,
    MASS_LABEL,
    MASS_PROPERTY,
    REMOTE_DIR,
    REMOTE_SERVER,
    lookup_soap_property,
)


def plot_halos_in_square_beam_3panel(
    ang_side_length_deg,
    side_axes,
    z_min,
    z_max,
    cosmo,
    halo_coords,
    halo_mass,
    figsize=(7, 7),
    show=False,
):
    """
    Scatter plot of halo centres placed into a square beam along z, projected onto the x-y, x-z and y-z planes,
    coloured and sized by the log of their mass.

    The grey areas are the shell's part of the square pyramid, projected onto each plane: in x-y the square seen along
    the beam (its sides curve in toward the corners, which the far sphere cuts off), in x-z and y-z a wedge with
    straight sides (the pyramid's faces) between the near and far distances.

    Haloes without a mass are not shown, e.g. satellites, which have no SO masses in SOAP-HBT.

    Returns the figure.

    :param  ang_side_length_deg:    angular side length [deg] of the square beam
    :type   ang_side_length_deg:    float
    :param  side_axes:              the square beam's side_axes: two unit vectors along its sides, in the x-y plane for
                                    a beam along z
    :type   side_axes:              tuple
    :param  z_min:                  minimum redshift of the shell passed to place_halos_in_shell
    :type   z_min:                  float
    :param  z_max:                  maximum redshift of the shell passed to place_halos_in_shell
    :type   z_max:                  float
    :param  cosmo:                  cosmology, with a comoving_distance(z) method
    :type   cosmo:                  astropy.cosmology.FLRW
    :param  halo_coords:            Lightcone/HaloCentre.to_value("Mpc") from place_halos_in_shell, shape (N, 3)
    :type   halo_coords:            np.ndarray
    :param  halo_mass:              mass of each halo [Msun], shape (N,)
    :type   halo_mass:              np.ndarray
    :param  figsize:                size of the figure
    :type   figsize:                tuple
    :param  show:                   If True, show the figure
    :type   show:                   boolean
    """
    # tangent of half the side length, inner & outer radius
    t = np.tan(0.5 * np.deg2rad(ang_side_length_deg))
    chi_near = cosmo.comoving_distance(z_min).to_value("Mpc")
    chi_far = cosmo.comoving_distance(z_max).to_value("Mpc")

    # only the haloes with a mass, drawn from low to high mass so the most massive are on top
    has_mass = halo_mass > 0
    log_mass = np.log10(halo_mass[has_mass])
    order = np.argsort(log_mass)
    halo_coords, log_mass = halo_coords[has_mass][order], log_mass[order]
    mass_min, mass_max = log_mass.min(), log_mass.max()
    size_scale = (log_mass - mass_min) / max(mass_max - mass_min, 1e-10)
    marker_size = MARKER_SIZE_RANGE[0] + size_scale**MARKER_SIZE_POWER * (MARKER_SIZE_RANGE[1] - MARKER_SIZE_RANGE[0])
    scatter_style = dict(
        c=log_mass,
        s=marker_size,
        cmap=cmap_MCG,
        vmin=mass_min,
        vmax=mass_max,
        edgecolors="white",
        linewidths=0.3,
        zorder=10,
    )

    # make fig
    fig = plt.figure(figsize=figsize, layout="constrained")
    ax_xy = fig.add_subplot(3, 1, 1)
    ax_xz = fig.add_subplot(3, 1, 2)
    ax_yz = fig.add_subplot(3, 1, 3, sharex=ax_xz)

    # space between the panels, and for the colour bar below them
    fig.get_layout_engine().set(h_pad=0.06)

    def corner_label(ax, text):
        ax.text(
            0.02,
            0.96,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            zorder=20,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
        )

    def draw_wedge(ax, horiz, vert, vert_label, panel_label, show_xlabel):
        # a point inside has |x| <= t z and |y| <= t z: seen side on, its largest distance along the axis at a height
        # h across it is on the far sphere, sqrt(chi_far^2 - h^2), and its smallest on the near sphere at the pyramid's
        # edge, sqrt((chi_near^2 - h^2) / (1 + t^2))
        h_far = t * chi_far / np.sqrt(1.0 + t**2)
        h_near = t * chi_near / np.sqrt(1.0 + 2.0 * t**2)
        far_v = np.linspace(-h_far, h_far, 200)
        near_v = np.linspace(-h_near, h_near, 200)
        far_h = np.sqrt(chi_far**2 - far_v**2)
        near_h = np.sqrt((chi_near**2 - near_v**2) / (1.0 + t**2))
        wedge_h = np.concatenate([far_h, near_h[::-1]])
        wedge_v = np.concatenate([far_v, near_v[::-1]])
        ax.fill(wedge_h, wedge_v, color="#ececea", zorder=1)
        ax.plot(far_h, far_v, color="black", zorder=2)
        ax.plot(near_h, near_v, color="black", lw=1.2, linestyle="--", zorder=2)
        for sign in (-1.0, 1.0):  # the pyramid's faces, from the near to the far sphere
            ax.plot([near_h[-1], far_h[-1]], [sign * h_near, sign * h_far], color="black", lw=0.8, zorder=2)
        ax.scatter(horiz, vert, **scatter_style)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.set_ylabel(vert_label)
        if show_xlabel:
            ax.set_xlabel("z [cMpc]")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        corner_label(ax, panel_label)
        ax.set_aspect("auto")

    def draw_square(ax, halo_x, halo_y):
        # the farthest point across the beam in each direction phi (in the frame of the sides): on the far sphere where
        # it meets the pyramid, rho^2 (1 + m^2 / t^2) = chi_far^2 with m = max(|cos phi|, |sin phi|)
        u, v = (np.asarray(a, dtype=float)[:2] for a in side_axes)
        phi = np.linspace(0.0, 2.0 * np.pi, 721)
        m = np.maximum(np.abs(np.cos(phi)), np.abs(np.sin(phi)))
        rho = chi_far / np.sqrt(1.0 + m**2 / t**2)
        edge = rho[:, None] * (np.cos(phi)[:, None] * u[None, :] + np.sin(phi)[:, None] * v[None, :])
        ax.fill(edge[:, 0], edge[:, 1], color="#ececea", zorder=1)
        ax.plot(edge[:, 0], edge[:, 1], color="black", lw=0.8, zorder=2)
        points = ax.scatter(halo_x, halo_y, **scatter_style)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.axvline(0, color="gray", lw=0.5, zorder=0)
        ax.set_xlabel("x [cMpc]")
        ax.set_ylabel("y [cMpc]")
        corner_label(ax, "x-y")
        ax.set_aspect("equal")
        return points

    points = draw_square(ax_xy, halo_coords[:, 0], halo_coords[:, 1])
    draw_wedge(ax_xz, halo_coords[:, 2], halo_coords[:, 0], "x [cMpc]", "x-z", show_xlabel=False)
    draw_wedge(ax_yz, halo_coords[:, 2], halo_coords[:, 1], "y [cMpc]", "y-z", show_xlabel=True)

    # side length and redshift range of the shell as the title of the figure, and the halo mass colour bar below
    fig.suptitle(rf"square beam (0, 0, 1), ${ang_side_length_deg:g}^\circ$ side,  ${z_min:.3f} < z \leq {z_max:.3f}$")
    fig.colorbar(points, ax=[ax_xy, ax_xz, ax_yz], location="bottom", shrink=0.6, aspect=30, label=MASS_LABEL)

    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place SOAP haloes in a SnapshotSquareBeam and plot them.")
    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label")
    parser.add_argument("--sim_name", default="L1_m9", help="name of the simulation (HYDRO_FIDUCIAL if run on COSMA)")
    parser.add_argument(
        "--side_deg",
        type=float,
        default=15.0,
        help="angular side length [deg] of the square (default 15: the square around the 7.5 deg circular beam)",
    )
    parser.add_argument("--output_dir", default="example_outputs/snapshot_lightcone", help="output directory")
    args = parser.parse_args()

    plot_settings()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # define the beam: redshift, side length and vector. The plot assumes a beam along z
    redshift_range = (0.05, 0.15)
    print(f"\nredshift: {redshift_range[0]} - {redshift_range[1]}")

    ang_side_length_deg = args.side_deg
    print(f"angular side length: {ang_side_length_deg:.3f} [deg]", flush=True)

    beam_vector = (0, 0, 1)
    print(f"beam unit vector: ({beam_vector[0]}, {beam_vector[1]}, {beam_vector[2]})\n", flush=True)

    # access the public FLAMINGO SOAP catalogues
    remote_dir = hdfstream.open(REMOTE_SERVER, REMOTE_DIR)
    simulation = Simulation.from_label(args.box_res, args.sim_name, remote=remote_dir)

    print(f"SOAP catalogues: {simulation.soap_format}")

    square_beam = SnapshotSquareBeam(simulation, beam_vector=beam_vector, ang_side_length_deg=ang_side_length_deg)
    print(
        f"corners {square_beam.ang_radius_deg:.3f} [deg] from the axis, "
        f"solid angle {np.degrees(np.degrees(square_beam.solid_angle)):.1f} [deg^2]"
    )

    # the simulation's own SOAP catalogues (simulation.soap_format), on disk or on the server
    halos = square_beam.place_halos_in_shell(lightcone_redshift_range=redshift_range)

    n_halos = halos["Lightcone/HaloCentre"].shape[0]
    print(f"\nPlaced {n_halos} halos in the square beam")

    if n_halos > 0:
        snap_numbers = halos["Lightcone/SnapshotNumber"].value.astype(int)
        print(f"Spanning snapshots: {sorted(np.unique(snap_numbers))}")

        # look up the mass of each halo in the SOAP catalogue of its snapshot, from its row there
        soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)
        halo_mass = lookup_soap_property(simulation, snap_numbers, soap_index, MASS_PROPERTY).to_value("Msun")

        print(f"\n{np.count_nonzero(halo_mass > 0)} of the halos have a {MASS_PROPERTY}")

        fig = plot_halos_in_square_beam_3panel(
            ang_side_length_deg,
            square_beam.side_axes,
            redshift_range[0],
            redshift_range[1],
            square_beam.cosmo,  # the lightcone's own cosmology, from the snapshots
            halo_coords=halos["Lightcone/HaloCentre"].to_value("Mpc"),
            halo_mass=halo_mass,
            figsize=(7, 7),
            show=False,
        )
        fig_name = output_dir + "/halos_in_square_beam.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot: {fig_name}")
    else:
        print("No halos found in this shell/beam")
