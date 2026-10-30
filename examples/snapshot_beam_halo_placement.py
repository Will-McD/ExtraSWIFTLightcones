#!/bin/env python
import os
import warnings
import numpy as np
import h5py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet.simulation_suites import Simulation
from swiftlet.soap_reader import read_soap_selected_rows, comoving
from swiftlet.example_plot_settings import plot_settings, cmap_MCG


"""
Place haloes from the SOAP catalogues into a SnapshotBeam lightcone shell. 
Plot the central haloes, scale the size and shape of the marker used by M200c of each halo. 
"""

# environmental variables for the plots
# halo mass shown, and the colour map of its log10: one hue, from light (low mass) to dark (high mass)
MASS_PROPERTY = "SO/200_crit/TotalMass"
MASS_LABEL = r"$\log_{10}(M_{200\mathrm{c}}\,/\,\mathrm{M}_\odot)$"
MARKER_SIZE_RANGE = (1., 150.)  # marker area [pt^2] of the least and most massive halo shown
MARKER_SIZE_POWER = 2.  #


# remote server and FLAMINGO's directory for use when the simulation is not on local disk. 
REMOTE_SERVER = "cosma"
REMOTE_BASE_DIR = "FLAMINGO/L1_m10/L1_m10"
REMOTE_CACHE_DIR = "example_outputs/remote_cache"

# the SOAP datasets the example reads: by place_halos_in_shell, and the mass looked up for the plot
SOAP_PROPERTIES = ["InputHalos/HaloCentre", "InputHalos/HaloCatalogueIndex", MASS_PROPERTY]

# the groups of a SOAP catalogue its units are made from (see virgo.formats.swift.soap_unit_registry_from_snapshot)
SOAP_METADATA_GROUPS = ["Header", "Cosmology", "Units", "InternalCodeUnits", "PhysicalConstants"]


def lookup_soap_property(halo_format, snap_numbers, soap_index, property_name):
    
    """
    For haloes in the lightcone, look up a properties in the SOAP catalogue by matching 
    their InputHalos/SOAPIndex.

    Returns the property of each halo, in the units given by SOAP.

    :param  halo_format:    format string for SOAP catalogue filenames (using {snap_nr})
    :type   halo_format:    str
    :param  snap_numbers:   Lightcone/SnapshotNumber of each halo from place_halos_in_shell
    :type   snap_numbers:   np.ndarray
    :param  soap_index:     InputHalos/SOAPIndex of each halo from place_halos_in_shell
    :type   soap_index:     np.ndarray
    :param  property_name:  SOAP property, e.g. "SO/200_crit/TotalMass"
    :type   property_name:  str
    """
    #halo_cat = hc.SOAPCatalogue(halo_format, int(snap_numbers.min()), int(snap_numbers.max()))
    values = None
    for snap_nr in np.unique(snap_numbers):
    #    soap = halo_cat.read(int(snap_nr), ["InputHalos/HaloCatalogueIndex", property_name])
    #    soap_index = soap["InputHalos/HaloCatalogueIndex"].value.astype(int)
    #    order = np.argsort(soap_index)
    #    in_snap = snap_numbers == snap_nr
    #    rows = order[np.searchsorted(soap_index, halo_index[in_snap], sorter=order)]
        
        in_snap = np.flatnonzero(snap_numbers == snap_nr)
        rows, where = np.unique(soap_index[in_snap], return_inverse=True)
        column = comoving(read_soap_selected_rows(halo_format.format(snap_nr=int(snap_nr)), [property_name], rows)[property_name])

        if values is None:
            #values = np.zeros(len(halo_index)) * soap[property_name].units
            values = np.zeros(len(snap_numbers)) * column.units
        #values[in_snap] = soap[property_name][rows]     
        values[in_snap] = column[where]
    return values


def copy_remote_snapshot_metadata(remote_file, local_filename):
    """
    Copy the metadata of a remote snapshot file (hdfstream) to a local HDF5 file for every group but the particles
    (PartType*), so the cosmology, units, box and cell grid can be read from it.

    :param  remote_file:    the remote snapshot file
    :type   remote_file:    hdfstream.RemoteFile
    :param  local_filename: path of the local copy
    :type   local_filename: str
    """
    Path(local_filename).parent.mkdir(parents=True, exist_ok=True)
    partial = local_filename + ".partial"
    with h5py.File(partial, "w") as outfile:
        for name in remote_file.keys():
            if not name.startswith("PartType"):
                remote_file.copy(name, outfile)
    os.replace(partial, local_filename)


def copy_remote_soap_catalogue(remote_file, local_filename, property_names):
    """
    Copy part of a remote SOAP catalogue (hdfstream) to a local HDF5 file for the datasets given, with their attributes,
    and the groups the SOAP units are made from (SOAP_METADATA_GROUPS).

    :param  remote_file:    the remote SOAP catalogue
    :type   remote_file:    hdfstream.RemoteFile
    :param  local_filename: path of the local copy
    :type   local_filename: str
    :param  property_names: the datasets to copy, e.g. "InputHalos/HaloCentre"
    :type   property_names: list of str
    """
    Path(local_filename).parent.mkdir(parents=True, exist_ok=True)
    partial = local_filename + ".partial"
    with h5py.File(partial, "w") as outfile:
        for name in SOAP_METADATA_GROUPS:
            if name in remote_file:
                remote_file.copy(name, outfile)
        for name in property_names:
            group, _, dataset = name.rpartition("/")
            remote_file.copy(name, outfile.require_group(group) if group else outfile, name=dataset)
    os.replace(partial, local_filename)


def fetch_remote_simulation(box_res, sim_name, redshift_range, property_names, remote_base_dir=REMOTE_BASE_DIR,
                            server=REMOTE_SERVER, cache_dir=REMOTE_CACHE_DIR):
    """
    Copy what placing the haloes of a redshift range needs from the remote server (hdfstream) into a local cache,
    laid out as the simulation's directory, the SOAP catalogues of the snapshots covering the redshift range, and 
    the metadata of one snapshot file. 
    
    Files already in the cache are not copied again.

    Returns (simulation_base_dir_format, halo_format) of the local copy, for SnapshotBeam and place_halos_in_shell.

    :param  box_res:            box size and resolution label, e.g. "L1000N0900"
    :type   box_res:            str
    :param  sim_name:           name of the simulation, e.g. "HYDRO_FIDUCIAL"
    :type   sim_name:           str
    :param  redshift_range:     minimum and maximum redshift of the lightcone shell
    :type   redshift_range:     tuple
    :param  property_names:     SOAP datasets to copy
    :type   property_names:     list of str
    :param  remote_base_dir:    directory of the simulation on the server
    :type   remote_base_dir:    str
    :param  server:             hdfstream server
    :type   server:             str
    :param  cache_dir:          local directory to copy the files to
    :type   cache_dir:          str
    """
    import hdfstream

    root = hdfstream.open(server, "/")
    remote_sim = Simulation.from_label(box_res, sim_name, base_dir_format=remote_base_dir)
    local_base_dir = f"{cache_dir}/{remote_base_dir}"
    local_sim = Simulation.from_label(box_res, sim_name, base_dir_format=local_base_dir)
    soap_path = "SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"

    # the SOAP catalogues of the snapshots covering the shell, skipping those not on the server
    for snap_nr in remote_sim.snapshots_in_range(redshift_range, range_tol=1e-6):
        local_filename = f"{local_base_dir}/{soap_path.format(snap_nr=snap_nr)}"
        if os.path.exists(local_filename):
            with h5py.File(local_filename, "r") as f:
                if all(name in f for name in property_names):
                    continue
        try:
            remote_file = root[f"{remote_base_dir}/{soap_path.format(snap_nr=snap_nr)}"]
        except KeyError:
            print(f"\tsnapshot {snap_nr}: no SOAP catalogue on the server, skipped")
            continue
        print(f"\tcopying {property_names} of the SOAP catalogue of snapshot {snap_nr}", flush=True)
        copy_remote_soap_catalogue(remote_file, local_filename, property_names)

    # the metadata of one snapshot: the redshift 0 snapshot, or the lowest redshift snapshot on the server
    snapshot_numbers = range(len(remote_sim.output_redshifts))
    if not any(os.path.exists(local_sim.snapshot_file(n, 0)) for n in snapshot_numbers):
        try:
            on_server = set(root[f"{remote_base_dir}/snapshots"].keys())
        except KeyError:
            raise FileNotFoundError(f"no snapshots directory {remote_base_dir}/snapshots on the {server} server: check "
                                    "REMOTE_BASE_DIR") from None
        on_server = [n for n in snapshot_numbers
                     if os.path.basename(os.path.dirname(remote_sim.snapshot_file(n, 0))) in on_server]
        if not on_server:
            raise FileNotFoundError(f"no snapshots of {box_res}/{sim_name} in {remote_base_dir}/snapshots on the "
                                    f"{server} server")
        snap_nr = on_server[-1]
        print(f"\tcopying the metadata of snapshot {snap_nr}", flush=True)
        copy_remote_snapshot_metadata(root[remote_sim.snapshot_file(snap_nr, 0)], local_sim.snapshot_file(snap_nr, 0))

    return local_base_dir, f"{local_base_dir}/{soap_path}"


def plot_halos_in_beam_3panel(angular_radius_deg, z_min, z_max, cosmo,
                               halo_coords, halo_mass, beam_vector=(0.0, 0.0, 1.0),
                               figsize=(7, 7), show=False):
    """
    Scatter plot of halo centres placed into the beam, projected onto the 
    x-y, x-z, y-z planes, coloured and sized by the log of their mass. 
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
    scatter_style = dict(c=log_mass, s=marker_size, cmap=cmap_MCG, vmin=mass_min, vmax=mass_max,
                         edgecolors="white", linewidths=0.3, zorder=10)

    # make fig
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
    
    parser = argparse.ArgumentParser(description="Place SOAP haloes in a SnapshotBeam and plot them.")
    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label")
    parser.add_argument("--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation")
    parser.add_argument("--output_dir", default="example_outputs/snapshot_lightcone", help="output directory")
    args = parser.parse_args()

    plot_settings()
    
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # define the beam: redshift, radius and vector. 
    redshift_range=(0.05, 0.15)
    print(f"\nredshift: {redshift_range[0]} - {redshift_range[1]}")

    ang_radius_deg = np.rad2deg(np.pi / 24)
    print(f"angular radius: {ang_radius_deg:.3f} [deg]", flush=True)

    beam_vector = (0, 0, 1)
    print(f"beam unit vector: ({beam_vector[0]}, {beam_vector[1]}, {beam_vector[2]})\n", flush=True)

    # read the SOAP catalogues and snapshot metadata locally if they are there, otherwise copy what is needed from
    # the remote server
    sim_dir = Simulation.from_label(args.box_res, args.sim_name).directory
    missing = [d for d in (f"{sim_dir}/SOAP-HBT", f"{sim_dir}/snapshots") if not os.path.isdir(d)]
    if not missing:
        base_dir_format = None
        halo_format = f"{sim_dir}/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"
    else:
        warning_str = f"{' and '.join(missing)} not found locally: the SOAP catalogues and snapshot metadata will be accessed remotely, copied from {REMOTE_BASE_DIR} on the {REMOTE_SERVER} server with hdfstream to {REMOTE_CACHE_DIR}."
        warnings.warn(warning_str)
        base_dir_format, halo_format = fetch_remote_simulation(args.box_res, args.sim_name, redshift_range, SOAP_PROPERTIES)
    print(f"SOAP catalogues: {halo_format}")


    snap_beam = SnapshotBeam(boxsize_resolution=args.box_res, simulation_name=args.sim_name, beam_vector=beam_vector, simulation_base_dir_format=base_dir_format)

    halos = snap_beam.place_halos_in_shell(
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
        

        # look up the mass of each halo in the SOAP catalogue of its snapshot, from its row there
        soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)
        halo_mass = lookup_soap_property(halo_format, snap_numbers, soap_index, MASS_PROPERTY).to_value("Msun")

        print(f"\n{np.count_nonzero(halo_mass > 0)} of the halos have a {MASS_PROPERTY}")
        
        # the lightcone's own cosmology, from the snapshots
        plot_cosmo = snap_beam.snapshot_cosmology

        fig = plot_halos_in_beam_3panel(
            ang_radius_deg, redshift_range[0], redshift_range[1], plot_cosmo,
            halo_coords=halos["Lightcone/HaloCentre"].to_value("Mpc"), halo_mass=halo_mass,
            beam_vector=beam_vector, figsize=(7, 7), show=False,
        )
        fig_name = output_dir+"/halos_in_beam.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot: {fig_name}")
    else:
        print("No halos found in this shell/beam")

