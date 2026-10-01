#!/bin/env python
import numpy as np
import unyt
import matplotlib.pyplot as plt
from pathlib import Path

from extra_swift_lightcones.snapshot_lightcone import SnapshotBeam
from extra_swift_lightcones.lightcone_projections import BeamProjection

"""
Use SnapshotBeam to build a beam through a past lightcone of gas particles from the snapshots of the
fiducial L1_m10 FLAMINGO simulation (i.e. L1000N0900/HYDRO_FIDUCIAL). 
Then project a slice through the beam to show the gas surface density.

The beam is built for each orientation lock in orientation_locks, and the gas surface densities are
plotted side by side. Without a lock (None) every snapshot box (or tile in the lightcone) has its own 
orientation, so structures are cut at every tile face the beam crosses. With "cube" or "sphere" the tiles in each 
layer share one orientation, so structures continue across the tiles faces within a layer.
"""

# units of the properties passed to the beam projection
PROPERTY_UNITS = {
    "Coordinates": "Mpc",
    "SmoothingLengths": "Mpc",
    "Masses": "Msun",
    "ExpansionFactors": "dimensionless",
}


def lock_label(orientation_lock):
    """
    Label of an orientation lock, used in titles and output filenames, e.g. "no lock" and "none".

    Returns a tuple of (title, filename label).

    :param  orientation_lock:   orientation lock passed to SnapshotBeam, None, "cube" or "sphere"
    :type   orientation_lock:   str
    """
    if orientation_lock is None:
        return "no lock", "none"
    return f"{orientation_lock} lock", orientation_lock


def beam_properties(particle_data, property_names, ptype):
    """
    Keep the particle properties needed for the beam projection, in the units of PROPERTY_UNITS.

    Returns a dict of the particle properties.

    :param  particle_data:  particles placed in the beam by SnapshotBeam.place_snapshot_particles_in_shell
    :type   particle_data:  dict
    :param  property_names: particle properties to keep, from PROPERTY_UNITS
    :type   property_names: list
    :param  ptype:          particle type, used in the error message
    :type   ptype:          str
    """
    if not particle_data or len(particle_data["Coordinates"]) == 0:
        raise ValueError(f"no {ptype} particles found in the beam")
    return {prop: unyt.unyt_array(particle_data[prop].to_value(PROPERTY_UNITS[prop]), PROPERTY_UNITS[prop]) for prop in property_names}


if __name__ == "__main__":

    # simulation to use
    box_res = "L1000N0900"
    sim_name = "HYDRO_FIDUCIAL"
    base_dir = "/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}".format(box_res=box_res, sim_name=sim_name)

    # snapshot used for its metadata when projecting
    snapshot_filename = "{base_dir}/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.hdf5".format(base_dir=base_dir, snap_nr=77)

    # define output directory
    output_dir = "./example_outputs/snapshot_beam_projection"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # part of the sky and redshift range of the beam. The beam is tilted so that it crosses the box faces,
    # and reaches past half a box sidelength from the observer, where the
    # orientation of a locked layer first changes
    beam_vector = (1.0, 1.0, 1.0)       # direction of the beam
    ang_radius_deg = 6.0                # angular radius of the beam [deg]
    redshift_range = (0.05, 0.3)        # redshift range of the beam

    # orientation locks to compare, see SnapshotBeam. Every lock builds the beam again,
    # so remove locks from the list to save time and memory
    orientation_locks = [None, "cube", "sphere"]

    # slice through the beam to project, and the resolution of the projections
    slice_thickness = 12.5 * unyt.Mpc
    resolution = 1024

    gas_property_names = ["Coordinates", "Masses", "SmoothingLengths", "ExpansionFactors"]

    gas_surface_densities_per_locking = []
    for orientation_lock in orientation_locks:
        lock_title, lock_filename = lock_label(orientation_lock)
        print(f"\n{lock_title}")

        # build the pencil beam from the snapshots, for the gas and dark matter particles
        SB = SnapshotBeam(boxsize_resolution=box_res, simulation_name=sim_name, beam_vector=beam_vector, verbose=1,
                          orientation_lock=orientation_lock)

        # Coordinates (and SmoothingLengths for gas) are always read, and ExpansionFactors are computed
        # from the position of each particle in the beam, so only the masses need to be requested
        beam_particles = SB.place_snapshot_particles_in_shell(
            lightcone_redshift_range=redshift_range,
            ang_radius_deg=ang_radius_deg,
            property_names=["Masses"],
            particle_types=["PartType0"],
            )

        gas_particle_data = beam_properties(beam_particles["PartType0"], gas_property_names, "PartType0")
        del beam_particles
        print(f"\n{len(gas_particle_data['Masses'])} gas particles in the beam\n")

        # project a slice through the beam, using the cosmology of the snapshots
        BP = BeamProjection(
            vector=beam_vector,
            angular_diameter=2 * ang_radius_deg,
            redshift_range=redshift_range,
            slice_thickness=slice_thickness,
            cosmology=SB.cosmo,
            store_snapshot_filename=snapshot_filename,
            )

        BP.place_particles_in_slice(gas_particle_data, gas_property_names)
        del gas_particle_data

        gas_surface_density = BP.project_properties(["Masses"], ptype="Gas", assign_units=["Msun"], resolution=resolution)[0].to_value("Msun/Mpc**2")
        gas_surface_densities_per_locking.append(gas_surface_density)

    # compare the total surface density for each orientation lock
    if len(orientation_locks) > 1:
        all_totals = np.concatenate([total.ravel() for total in gas_surface_densities_per_locking])
        pix_min, pix_max = np.percentile(all_totals[all_totals > 0], [10, 99.5])

        fig, axs = plt.subplots(len(orientation_locks), 1, figsize=(7, 2.6 * len(orientation_locks)))
        for ax, orientation_lock, surface_density in zip(np.atleast_1d(axs), orientation_locks, gas_surface_densities_per_locking):
            BP.split_beam_plot(
                1, [[surface_density, (pix_min, pix_max)]], ["magma"], axs=ax,
                minor_tick_kwargs={"color": "k", "lw": 0.6}, tick_label_kwargs={"rotation": "auto"},
                titles=[rf"$\Sigma_{{\mathrm{{Gas}}}}$, {lock_label(orientation_lock)[0]}"], title_kwargs={"fontsize": 10},
                overlay_grid=(False, False, False), # turn off grids
                )

        output_filename = f"{output_dir}/snapshot_beam_surface_density_lock_comparison.png"
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Orientation lock comparison saved as: {output_filename}")

