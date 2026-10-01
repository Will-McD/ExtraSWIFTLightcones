#!/bin/env python
#

import sys
import numpy as np
from extra_swift_lightcones.snapshot_lightcone import SnapshotBeam
from  extra_swift_lightcones.swift_snapshot_redshift_conversion import flamingo_shell_redshift_file
from pathlib import Path
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone

import matplotlib.pyplot as plt

"""
    Place haloes from the SOAP catalogues into a SnapshotBeam lightcone shell. 
"""

def plot_settings():
    """
    matplotlib settings for this example. 
    """
    import matplotlib as mpl
    
    # Line Properties
    plt.rcParams["lines.linewidth"]=1.5 
    # Font options
    plt.rcParams["font.size"]=8
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["legend.labelspacing"]=0.25
    plt.rcParams["legend.columnspacing"]=0.75
    plt.rcParams["legend.borderpad"]=0.3
    plt.rcParams["legend.borderaxespad"]=0.7
    plt.rcParams["legend.fontsize"]=10
    # Figure layour settings
    plt.rcParams["figure.constrained_layout.use"] =True
    plt.rcParams["figure.constrained_layout.h_pad"] =0.005
    plt.rcParams["figure.constrained_layout.w_pad"] =0.005
    plt.rcParams["figure.constrained_layout.hspace"]=0.005
    plt.rcParams["figure.constrained_layout.wspace"]=0.005
    # axes font settings
    mpl.rcParams['axes.labelsize']=10
    mpl.rcParams['figure.labelsize']=10
    mpl.rcParams['xtick.labelsize']=10
    mpl.rcParams['ytick.labelsize']=10
    # axes tick settings
    plt.rcParams["xtick.direction"] ='in'
    plt.rcParams["ytick.direction"] ='in'
    plt.rcParams["xtick.top"]=True
    plt.rcParams["ytick.right"]= True


def plot_halos_in_beam_3panel(angular_radius_deg, z_min, z_max, cosmo,
                               halo_coords, beam_vector=(0.0, 0.0, 1.0),
                               figsize=(7, 7), show=False):
    """
    Scatter plot of halo centres placed into the beam, projected onto the 
    x-y, x-z, y-z planes. 

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
    :param  beam_vector:        assumed (0, 0, 1) for the drawn wedge geometry to line up with the (z, x)/(z, y) panels
    :type   beam_vector:        array-like, shape (3,)
    :param  figsize:            size of the figure
    :type   figsize:            tuple
    :param  show:               If True, show the figure
    :type   show:               boolean
    """
    half_angle = np.deg2rad(angular_radius_deg)
    chi_near = cosmo.z2r(z_min).to_value("Mpc")
    chi_far = cosmo.z2r(z_max).to_value("Mpc")

    fig = plt.figure(figsize=figsize)
    ax_xy = fig.add_subplot(3, 1, 1)
    ax_xz = fig.add_subplot(3, 1, 2)
    ax_yz = fig.add_subplot(3, 1, 3, sharex=ax_xz)
    fig.subplots_adjust(hspace=0.08)

    def draw_wedge(ax, horiz, vert, vert_label, show_xlabel):
        theta = np.linspace(-half_angle, half_angle, 200)
        near_h, near_v = chi_near * np.cos(theta), chi_near * np.sin(theta)
        far_h, far_v = chi_far * np.cos(theta), chi_far * np.sin(theta)
        wedge_h = np.concatenate([far_h, near_h[::-1]])
        wedge_v = np.concatenate([far_v, near_v[::-1]])
        ax.fill(wedge_h, wedge_v, color="darkgrey", alpha=0.5, zorder=1,
                 label=rf"shell (${z_min:.3f}<z\leq{z_max:.3f}$)")
        ax.plot(far_h, far_v, color="black", zorder=2)
        ax.plot(near_h, near_v, color="black", lw=1.2, linestyle="--", zorder=2)
        ax.scatter(horiz, vert, color="darkcyan", marker=".", s=12, alpha=0.6,
                    zorder=10, label="halos")
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.set_ylabel(vert_label)
        if show_xlabel:
            ax.set_xlabel("z [cMpc]")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_aspect("auto")

    def draw_disk(ax, halo_x, halo_y):
        r_far = chi_far * np.sin(half_angle)
        theta_full = np.linspace(0, 2 * np.pi, 200)
        ax.fill(r_far * np.cos(theta_full), r_far * np.sin(theta_full),
                 color="darkgrey", alpha=0.5, zorder=1)
        ax.scatter(halo_x, halo_y, color="darkcyan", marker=".", s=12, alpha=0.6, zorder=10)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.axvline(0, color="gray", lw=0.5, zorder=0)
        ax.set_xlabel("x [cMpc]")
        ax.set_ylabel("y [cMpc]")
        ax.set_aspect("equal")

    draw_disk(ax_xy, halo_coords[:, 0], halo_coords[:, 1])
    draw_wedge(ax_xz, halo_coords[:, 2], halo_coords[:, 0], "x [cMpc]", show_xlabel=False)
    draw_wedge(ax_yz, halo_coords[:, 2], halo_coords[:, 1], "y [cMpc]", show_xlabel=True)

    handles, labels = ax_xz.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=9, frameon=False)

    if show:
        plt.tight_layout()
        plt.show()
    return fig


if __name__ == "__main__":
    
    plot_settings()
    
    output_dir="example_outputs/snapshot_lightcone"
    # ensure output directory exists
    directory_path = Path(output_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    box_res = "L1000N0900"
    sim_name = "HYDRO_FIDUCIAL"

    halo_format = f"/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"

    shell_number = 5
    redshift_range = tuple(np.loadtxt(flamingo_shell_redshift_file("L1"), delimiter=",")[shell_number, :])
    print(f"\nshell: {shell_number}, \tredshift: {redshift_range[0]} - {redshift_range[1]}")

    ang_radius_deg = np.rad2deg(np.pi / 24)
    print(f"\nangular radius: {ang_radius_deg:.3f} [deg]\n", flush=True)

    beam_vector = (0, 0, 1)
    SB = SnapshotBeam(boxsize_resolution=box_res, simulation_name=sim_name, beam_vector=beam_vector)

    halos = SB.place_halos_in_shell(
        halo_format=halo_format,
        lightcone_redshift_range=redshift_range,
        ang_radius_deg=ang_radius_deg,
    )

    n_halos = halos["Lightcone/HaloCentre"].shape[0]
    print(f"\nPlaced {n_halos} halos in the lightcone beam")

    if n_halos > 0:
        snap_numbers = halos["Lightcone/SnapshotNumber"].value.astype(int)
        halo_index = halos["InputHalos/HaloCatalogueIndex"].value.astype(int)
        expansion_factor = halos["Lightcone/ExpansionFactor"].value

        print(f"Spanning snapshots: {sorted(np.unique(snap_numbers))}")
        print(f"HaloCatalogueIndex range: {halo_index.min()} - {halo_index.max()}")
        print(f"ExpansionFactor range: {expansion_factor.min():.4f} - {expansion_factor.max():.4f}")

        print("\nFirst 5 halos (snapshot, HaloCatalogueIndex) -- this pair is what you'd")
        print("use to look up any other SOAP property later from the original catalogue:")
        for i in range(min(5, n_halos)):
            print(f"  snapshot {snap_numbers[i]}, HaloCatalogueIndex {halo_index[i]}")

        base_dir = f"/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}"
        snapshot_dir = f"{base_dir}/snapshots"
        plot_cosmo = Snapshot_Cosmology_For_Lightcone(snapshot_dir)

        fig = plot_halos_in_beam_3panel(
            ang_radius_deg, redshift_range[0], redshift_range[1], plot_cosmo,
            halo_coords=halos["Lightcone/HaloCentre"].to_value("Mpc"),
            beam_vector=beam_vector, figsize=(7, 7), show=False,
        )
        fig_name = output_dir+"/halos_in_beam_shell{shell_nr}".format(shell_nr=shell_number)
        plt.savefig(output_dir + "/" + fig_name + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot: {output_dir}/{fig_name}.png")
    else:
        print("No halos found in this shell/beam -- nothing to plot.")