#!/bin/env python
import numpy as np
from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet.swift_snapshot_redshift_conversion import flamingo_shell_redshift_file
from pathlib import Path
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone

import lightcone_io.halo_catalogue as hc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

"""
Place haloes from the SOAP catalogues into a SnapshotBeam lightcone shell. 
Plot the central haloes, scale the size and shape of the marker used by M200c of each halo. 
"""

# halo mass shown, and the colour map of its log10: one hue, from light (low mass) to dark (high mass)
MASS_PROPERTY = "SO/200_crit/TotalMass"
MASS_LABEL = r"$\log_{10}(M_{200\mathrm{c}}\,/\,\mathrm{M}_\odot)$"
MASS_CMAP = LinearSegmentedColormap.from_list("magenta_cyan_gold", ["#8b008b", "#008c99", "#b8860b"])
MARKER_SIZE_RANGE = (1., 150.)  # marker area [pt^2] of the least and most massive halo shown
MARKER_SIZE_POWER = 2.  #


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


def lookup_soap_property(halo_format, snap_numbers, halo_index, property_name):
    
    """
    Look up a property of haloes placed in a lightcone in the SOAP catalogue of the snapshot each came from,
    matching their InputHalos/HaloCatalogueIndex.

    Returns the property of each halo, in the units given by SOAP.

    :param  halo_format:    format string for SOAP catalogue filenames (using {snap_nr})
    :type   halo_format:    str
    :param  snap_numbers:   Lightcone/SnapshotNumber of each halo from place_halos_in_shell
    :type   snap_numbers:   np.ndarray
    :param  halo_index:     InputHalos/HaloCatalogueIndex of each halo from place_halos_in_shell
    :type   halo_index:     np.ndarray
    :param  property_name:  SOAP property, e.g. "SO/200_crit/TotalMass"
    :type   property_name:  str
    """
    halo_cat = hc.SOAPCatalogue(halo_format, int(snap_numbers.min()), int(snap_numbers.max()))
    values = None
    for snap_nr in np.unique(snap_numbers):
        soap = halo_cat.read(int(snap_nr), ["InputHalos/HaloCatalogueIndex", property_name])
        soap_index = soap["InputHalos/HaloCatalogueIndex"].value.astype(int)
        order = np.argsort(soap_index)
        in_snap = snap_numbers == snap_nr
        rows = order[np.searchsorted(soap_index, halo_index[in_snap], sorter=order)]
        if values is None:
            values = np.zeros(len(halo_index)) * soap[property_name].units
        values[in_snap] = soap[property_name][rows]
    return values

def plot_halos_in_beam_3panel(angular_radius_deg, z_min, z_max, cosmo,
                               halo_coords, halo_mass, beam_vector=(0.0, 0.0, 1.0),
                               figsize=(7, 7), show=False):
    """
    Scatter plot of halo centres placed into the beam, projected onto the 
    x-y, x-z, y-z planes, coloured and sized by the log of their mass. Haloes without a mass 
    (e.g. satellites, which have no SO masses in SOAP-HBT) are not shown.

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
    half_angle = np.deg2rad(angular_radius_deg)
    chi_near = cosmo.z2r(z_min).to_value("Mpc")
    chi_far = cosmo.z2r(z_max).to_value("Mpc")

    # only the haloes with a mass, drawn from low to high mass so the most massive are on top
    has_mass = halo_mass > 0
    log_mass = np.log10(halo_mass[has_mass])
    order = np.argsort(log_mass)
    halo_coords, log_mass = halo_coords[has_mass][order], log_mass[order]
    mass_min, mass_max = log_mass.min(), log_mass.max()
    size_scale = (log_mass - mass_min) / max(mass_max - mass_min, 1e-10)
    marker_size = MARKER_SIZE_RANGE[0] + size_scale**MARKER_SIZE_POWER * (MARKER_SIZE_RANGE[1] - MARKER_SIZE_RANGE[0])
    scatter_style = dict(c=log_mass, s=marker_size, cmap=MASS_CMAP, vmin=mass_min, vmax=mass_max,
                         edgecolors="white", linewidths=0.3, zorder=10)

    fig = plt.figure(figsize=figsize, layout="constrained")
    ax_xy = fig.add_subplot(3, 1, 1)
    ax_xz = fig.add_subplot(3, 1, 2)
    ax_yz = fig.add_subplot(3, 1, 3, sharex=ax_xz)
    # space between the panels, and for the colour bar below them
    fig.get_layout_engine().set(h_pad=0.06)

    def corner_label(ax, text):
        ax.text(0.02, 0.96, text, transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", zorder=20,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

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
    
    plot_settings()
    
    output_dir="example_outputs/snapshot_lightcone"
    # ensure output directory exists
    directory_path = Path(output_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    box_res = "L1000N0900"
    sim_name = "HYDRO_FIDUCIAL"

    halo_format = f"/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"

    redshift_range=(0.05, 0.15)
    print(f"\nredshift: {redshift_range[0]} - {redshift_range[1]}")

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
        
        # look up the mass of each halo in the SOAP catalogue of its snapshot
        halo_mass = lookup_soap_property(halo_format, snap_numbers, halo_index, MASS_PROPERTY).to_value("Msun")
        print(f"\n{np.count_nonzero(halo_mass > 0)} of the halos have a {MASS_PROPERTY}")
        
        fig = plot_halos_in_beam_3panel(
            ang_radius_deg, redshift_range[0], redshift_range[1], plot_cosmo,
            halo_coords=halos["Lightcone/HaloCentre"].to_value("Mpc"), halo_mass=halo_mass,
            beam_vector=beam_vector, figsize=(7, 7), show=False,
        )
        fig_name = output_dir+"/halos_in_beam.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot:{fig_name}.png")
    else:
        print("No halos found in this shell/beam -- nothing to plot.")