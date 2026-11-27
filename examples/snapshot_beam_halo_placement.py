#!/bin/env python
"""
Example script to show the placement of haloes from the SOAP catalogues into a SnapshotBeam lightcone shell.
"""

import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import hdfstream
from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet.simulation_suites import Simulation
from swiftlet.soap_reader import read_soap_selected_rows, comoving
from swiftlet.example_plot_settings import plot_settings, cmap_MCG


# environmental variables for the plots
# halo mass shown, and the colour map of its log10: one hue, from light (low mass) to dark (high mass)
MASS_PROPERTY = "SO/200_crit/TotalMass"
MASS_LABEL = r"$\log_{10}(M_{200\mathrm{c}}\,/\,\mathrm{M}_\odot)$"
MARKER_SIZE_RANGE = (1.0, 150.0)  # marker area [pt^2] of the least and most massive halo shown
MARKER_SIZE_POWER = 2.0  #

# remote server and FLAMINGO's directory for use when the simulation is not on local disk.
REMOTE_SERVER = "cosma"
REMOTE_DIR = "/FLAMINGO"


def lookup_soap_property(simulation, snap_numbers, soap_index, property_name):
    """
    For haloes in the lightcone, look up a properties in the SOAP catalogue by matching
    their InputHalos/SOAPIndex.

    Returns the property of each halo, in the units given by SOAP.

    :param  simulation:     the simulation the haloes were placed from, on disk or remote
    :type   simulation:     swiftlet.simulation_suites.Simulation
    :param  snap_numbers:   Lightcone/SnapshotNumber of each halo from place_halos_in_shell
    :type   snap_numbers:   np.ndarray
    :param  soap_index:     InputHalos/SOAPIndex of each halo from place_halos_in_shell
    :type   soap_index:     np.ndarray
    :param  property_name:  SOAP property, e.g. "SO/200_crit/TotalMass"
    :type   property_name:  str
    """
    values = None
    for snap_nr in np.unique(snap_numbers):
        in_snap = np.flatnonzero(snap_numbers == snap_nr)
        rows, where = np.unique(soap_index[in_snap], return_inverse=True)
        column = comoving(
            read_soap_selected_rows(simulation.soap_file(snap_nr), [property_name], rows, remote=simulation.remote)[
                property_name
            ]
        )

        if values is None:
            values = np.zeros(len(snap_numbers)) * column.units
        values[in_snap] = column[where]
    return values


def plot_halos_in_beam_3panel(
    angular_radius_deg,
    z_min,
    z_max,
    cosmo,
    halo_coords,
    halo_mass,
    beam_vector=(0.0, 0.0, 1.0),
    figsize=(7, 7),
    show=False,
):
    """
    Scatter plot of halo centres placed into the beam, projected onto the x-y, x-z, y-z planes, coloured and sized by
    the log of their mass.

    Haloes without a mass are not shown, e.g. satellites, which have no SO masses in SOAP-HBT.

    Returns the figure.

    :param  angular_radius_deg: angular radius [deg] of the cone passed to place_halos_in_shell
    :type   angular_radius_deg: float
    :param  z_min:              minimum redshift of the shell passed to place_halos_in_shell
    :type   z_min:              float
    :param  z_max:              maximum redshift of the shell passed to place_halos_in_shell
    :type   z_max:              float
    :param  cosmo:              object with a .z2r(z) method returning a comoving distance
    :type   cosmo:              object
    :param  halo_coords:        Lightcone/HaloCentre.to_value("Mpc") from place_halos_in_shell, shape (N, 3)
    :type   halo_coords:        np.ndarray
    :param  halo_mass:          mass of each halo [Msun], shape (N,)
    :type   halo_mass:          np.ndarray
    :param  beam_vector:        assumed (0, 0, 1) for the drawn wedge geometry to line up with the (z, x)/(z, y) panels
    :type   beam_vector:        array-like, shape (3,)
    :param  figsize:            size of the figure
    :type   figsize:            tuple
    :param  show:               If True, show the figure
    :type   show:               boolean
    """
    # compute angle, inner & outer radius
    half_angle = np.deg2rad(angular_radius_deg)
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
        theta = np.linspace(-half_angle, half_angle, 200)
        near_h, near_v = chi_near * np.cos(theta), chi_near * np.sin(theta)
        far_h, far_v = chi_far * np.cos(theta), chi_far * np.sin(theta)
        wedge_h = np.concatenate([far_h, near_h[::-1]])
        wedge_v = np.concatenate([far_v, near_v[::-1]])
        ax.fill(wedge_h, wedge_v, color="#ececea", zorder=1)
        ax.plot(far_h, far_v, color="black", zorder=2)
        ax.plot(near_h, near_v, color="black", lw=1.2, linestyle="--", zorder=2)
        ax.scatter(horiz, vert, **scatter_style)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.set_ylabel(vert_label)
        if show_xlabel:
            ax.set_xlabel("z [cMpc]")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        corner_label(ax, panel_label)
        ax.set_aspect("auto")

    def draw_disk(ax, halo_x, halo_y):
        r_far = chi_far * np.sin(half_angle)
        theta_full = np.linspace(0, 2 * np.pi, 200)
        ax.fill(r_far * np.cos(theta_full), r_far * np.sin(theta_full), color="#ececea", zorder=1)
        points = ax.scatter(halo_x, halo_y, **scatter_style)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.axvline(0, color="gray", lw=0.5, zorder=0)
        ax.set_xlabel("x [cMpc]")
        ax.set_ylabel("y [cMpc]")
        corner_label(ax, "x-y")
        ax.set_aspect("equal")
        return points

    points = draw_disk(ax_xy, halo_coords[:, 0], halo_coords[:, 1])
    draw_wedge(ax_xz, halo_coords[:, 2], halo_coords[:, 0], "x [cMpc]", "x-z", show_xlabel=False)
    draw_wedge(ax_yz, halo_coords[:, 2], halo_coords[:, 1], "y [cMpc]", "y-z", show_xlabel=True)

    # redshift range of the shell as the title of the figure, and the halo mass colour bar below the panels
    beam_str = ", ".join(f"{v:g}" for v in beam_vector)
    fig.suptitle(rf"beam ({beam_str}),  ${z_min:.3f} < z \leq {z_max:.3f}$")
    fig.colorbar(points, ax=[ax_xy, ax_xz, ax_yz], location="bottom", shrink=0.6, aspect=30, label=MASS_LABEL)

    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place SOAP haloes in a SnapshotBeam and plot them.")
    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label")
    parser.add_argument("--sim_name", default="L1_m9", help="name of the simulation")
    parser.add_argument("--output_dir", default="example_outputs/snapshot_lightcone", help="output directory")
    args = parser.parse_args()

    plot_settings()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # define the beam: redshift, radius and vector.
    redshift_range = (0.05, 0.15)
    print(f"\nredshift: {redshift_range[0]} - {redshift_range[1]}")

    ang_radius_deg = np.rad2deg(np.pi / 24)
    print(f"angular radius: {ang_radius_deg:.3f} [deg]", flush=True)

    beam_vector = (0, 0, 1)
    print(f"beam unit vector: ({beam_vector[0]}, {beam_vector[1]}, {beam_vector[2]})\n", flush=True)

    # use remote data
    remote_dir = hdfstream.open(REMOTE_SERVER, REMOTE_DIR)
    simulation = Simulation.from_label(args.box_res, args.sim_name, remote=remote_dir)
    print(f"SOAP catalogues: {simulation.soap_format}")

    snap_beam = SnapshotBeam(simulation, beam_vector=beam_vector, ang_radius_deg=ang_radius_deg)

    # use the simulations SOAP catalogues
    halos = snap_beam.place_halos_in_shell(lightcone_redshift_range=redshift_range)

    n_halos = halos["Lightcone/HaloCentre"].shape[0]

    print(f"\nPlaced {n_halos} halos in the lightcone beam")

    if n_halos > 0:
        snap_numbers = halos["Lightcone/SnapshotNumber"].value.astype(int)
        halo_index = halos["InputHalos/HaloCatalogueIndex"].value.astype(int)
        expansion_factor = halos["Lightcone/ExpansionFactor"].value

        print(f"Spanning snapshots: {sorted(np.unique(snap_numbers))}")

        # look up the mass of each halo in the SOAP catalogue of its snapshot, from its row there
        soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)
        halo_mass = lookup_soap_property(simulation, snap_numbers, soap_index, MASS_PROPERTY).to_value("Msun")

        print(f"\n{np.count_nonzero(halo_mass > 0)} of the halos have a {MASS_PROPERTY}")

        # the lightcone's own cosmology, from the snapshots
        plot_cosmo = snap_beam.snapshot_cosmology

        fig = plot_halos_in_beam_3panel(
            ang_radius_deg,
            redshift_range[0],
            redshift_range[1],
            plot_cosmo,
            halo_coords=halos["Lightcone/HaloCentre"].to_value("Mpc"),
            halo_mass=halo_mass,
            beam_vector=beam_vector,
            figsize=(7, 7),
            show=False,
        )
        fig_name = output_dir + "/halos_in_beam.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot: {fig_name}")
    else:
        print("No halos found in this shell/beam")
