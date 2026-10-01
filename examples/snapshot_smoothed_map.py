#!/bin/env python
import numpy as np
import h5py
import unyt
import healpy as hp
from pathlib import Path
import lightcone_io.smoothed_map as smoothed_map
import lightcone_io.kernel as kernel
import virgo.mpi.parallel_hdf5 as phdf5

from extra_swift_lightcones.snapshot_lightcone import SnapshotAllSky

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Make an all-sky HEALPix map of the smoothed gas mass (SmoothedGasMass) in a lightcone shell built from
the snapshots. Adopt the same method as  lightcone_io.smoothed_map.make_sky_map except particles are 
read from the snapshots instead of a particle lightcone. 
"""


def message(m):
    """
    Print a message on rank 0 only.

    :param  m:  message to print
    :type   m:  str
    """
    if comm_rank == 0:
        print(m, flush=True)


def make_smoothed_sky_map(part_pos, part_hsml, part_val, nside, progress=False):
    """
    Smooth the particles onto a HEALPix map distributed over the MPI ranks, as
    lightcone_io.smoothed_map.make_sky_map does for a particle lightcone.

    Returns the local pixels of the map on this rank (ring order). 

    :param  part_pos:   comoving positions of the particles on this rank, relative to the observer, shape (N, 3)
    :type   part_pos:   unyt.unyt_array
    :param  part_hsml:  smoothing lengths of the particles on this rank
    :type   part_hsml:  unyt.unyt_array
    :param  part_val:   value of each particle to add to the map, e.g. masses
    :type   part_val:   unyt.unyt_array
    :param  nside:      HEALPix resolution of the map
    :type   nside:      int
    :param  progress:   If True, show a progress bar on rank 0 while smoothing the particles
    :type   progress:   boolean
    """
    if progress and comm_rank == 0:
        from tqdm import tqdm
        progress_bar = tqdm
    else:
        progress_bar = lambda x: x

    # Split the pixels of the map between the MPI ranks, in bands of colatitude
    nr_total_pixels, nr_local_pixels, local_offset, theta_boundary = smoothed_map.distribute_pixels(comm, nside)
    max_pixrad = hp.pixelfunc.max_pixrad(nside)
    message(f"Total number of pixels = {nr_total_pixels}")

    # Angular smoothing lengths and the values to add to the map
    part_hsml = smoothed_map.find_angular_smoothing_length(part_pos, part_hsml)
    part_val = unyt.unyt_array(part_val, dtype=float)
    val_total_global = comm.allreduce(np.sum(part_val.value, dtype=float))
    map_units = part_val.units

    # Determine which ranks each particle needs to be sent to, from the range of colatitude of
    # the pixels it could update. 
    radius = np.maximum(kernel.kernel_gamma*part_hsml, max_pixrad)
    if len(part_pos) > 0:
        theta, phi = hp.pixelfunc.vec2ang(part_pos.value)
    else:
        theta = np.zeros(0, dtype=float)
    part_min_theta = np.clip(theta-radius, 0.0, np.pi)
    part_max_theta = np.clip(theta+radius, 0.0, np.pi)
    part_first_rank = np.clip(np.searchsorted(theta_boundary, part_min_theta, side="left") - 1, 0, comm_size-1)
    part_last_rank = np.clip(np.searchsorted(theta_boundary, part_max_theta, side="right") - 1, 0, comm_size-1)
    nr_copies = part_last_rank - part_first_rank + 1

    # Copy each particle once for every rank it needs to be sent to
    index = np.repeat(np.arange(len(part_pos), dtype=int), nr_copies)
    part_pos_send = part_pos[index,...]
    part_val_send = part_val[index]
    part_hsml_send = part_hsml[index]
    offset = np.cumsum(nr_copies) - nr_copies
    part_dest = -np.ones(np.sum(nr_copies), dtype=int)
    for first_rank, nr, off in zip(part_first_rank, nr_copies, offset):
        part_dest[off:off+nr] = np.arange(first_rank, first_rank+nr, dtype=int)
    del index, offset, part_first_rank, part_last_rank, nr_copies

    # Send the particles to the ranks holding the pixels they update
    part_pos_recv, part_val_recv, part_hsml_recv = smoothed_map.exchange_particles(part_dest, (part_pos_send, part_val_send, part_hsml_send))
    del part_pos_send, part_val_send, part_hsml_send, part_dest
    message("Sent the particles to the ranks holding the pixels they update")

    # Local part of the map, using unit-less views to add to the pixels
    map_data = unyt.unyt_array(np.zeros(nr_local_pixels, dtype=float), units=map_units)
    map_view = map_data.ndarray_view()
    part_pos_view = part_pos_recv.ndarray_view()
    part_val_view = part_val_recv.ndarray_view()

    # Particles smaller than a pixel update only the pixel they are in
    single_pixel = part_hsml_recv*kernel.kernel_gamma < max_pixrad
    local_pix_index = hp.pixelfunc.vec2pix(nside,
                                           part_pos_view[single_pixel, 0],
                                           part_pos_view[single_pixel, 1],
                                           part_pos_view[single_pixel, 2]) - local_offset
    local = (local_pix_index >= 0) & (local_pix_index < nr_local_pixels)
    np.add.at(map_view, local_pix_index[local], part_val_view[single_pixel][local])
    message(f"Applied {comm.allreduce(np.sum(single_pixel))} single pixel updates")

    # Particles larger than a pixel are smoothed over the pixels within their kernel
    multi_pixel = np.flatnonzero(single_pixel==False)
    for part_nr in progress_bar(multi_pixel):
        pix_index, pix_val = smoothed_map.explode_particle(nside, part_pos_view[part_nr,:], part_val_view[part_nr], part_hsml_recv[part_nr])
        local_pix_index = pix_index - local_offset
        local = (local_pix_index >= 0) & (local_pix_index < nr_local_pixels)
        # pixel indexes are unique, so np.add.at is not needed
        map_view[local_pix_index[local]] += pix_val[local]
    message(f"Applied {comm.allreduce(len(multi_pixel))} multi-pixel updates")

    # The sum over the map should equal the sum of the values added to the map
    map_sum = comm.allreduce(np.sum(map_view, dtype=float))
    message(f"Ratio (map sum / total values to add to map) = {map_sum / val_total_global} (should be 1.0)")

    return map_data


def write_map(output_filename, dataset_name, map_data, nside, redshift_range):
    """
    Write the map, distributed over the MPI ranks, to a HDF5 file in parallel.

    :param  output_filename:    path of the HDF5 file to write
    :type   output_filename:    str
    :param  dataset_name:       name of the map dataset, e.g. "SmoothedGasMass"
    :type   dataset_name:       str
    :param  map_data:           local pixels of the map on this rank, in ring ordering
    :type   map_data:           unyt.unyt_array
    :param  nside:              HEALPix resolution of the map
    :type   nside:              int
    :param  redshift_range:     minimum and maximum redshift of the shell [z_min, z_max]
    :type   redshift_range:     tuple
    """
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, dataset_name, map_data.value, comm)
        # attributes are written collectively, with the same values on every rank
        dataset = outfile[dataset_name]
        dataset.attrs["nside"] = [nside]
        dataset.attrs["number_of_pixels"] = [hp.nside2npix(nside)]
        dataset.attrs["pixel_ordering_scheme"] = "ring"
        dataset.attrs["units"] = str(map_data.units)
        dataset.attrs["redshift_range"] = list(redshift_range)


def plot_map(map_filename, dataset_name, plot_filename):
    """
    Plot a map written by write_map as an all-sky Mollweide projection of log10 of the pixel values. 

    :param  map_filename:   path of the HDF5 file with the map
    :type   map_filename:   str
    :param  dataset_name:   name of the map dataset, e.g. "SmoothedGasMass"
    :type   dataset_name:   str
    :param  plot_filename:  path of the image to write
    :type   plot_filename:  str
    """
    with h5py.File(map_filename, "r") as infile:
        dataset = infile[dataset_name]
        map_values = dataset[...]
        nside = int(dataset.attrs["nside"][0])
        units = dataset.attrs["units"]
        zmin, zmax = dataset.attrs["redshift_range"]

    # log10 of the pixel values, empty pixels are not shown
    log_map = np.full(map_values.shape, hp.UNSEEN)
    filled = map_values > 0
    log_map[filled] = np.log10(map_values[filled])
    vmin, vmax = np.percentile(log_map[filled], [1, 99.9]) if np.any(filled) else (None, None)

    hp.mollview(
        log_map,
        title=rf"{dataset_name}, ${zmin:.3f} < z < {zmax:.3f}$, nside={nside}",
        unit=rf"$\log_{{10}}$ ({units} per pixel)",
        cmap="magma", badcolor="grey", min=vmin, max=vmax,
        )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # simulation to use
    box_res = "L1000N1800"
    sim_name = "HYDRO_FIDUCIAL"

    # lightcone shell and map resolution
    redshift_range = (0.05, 0.1)
    nside = 2048

    # define output directory and file
    output_dir = "./example_outputs/smoothed_maps"
    if comm_rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    comm.barrier()
    output_filename = f"{output_dir}/smoothed_gas_mass.z_{redshift_range[0]:.3f}_{redshift_range[1]:.3f}.nside_{nside}.hdf5"

    # Place the gas particles from the snapshots in the all-sky shell. Passing comm reads the
    # snapshot files in parallel, each rank keeping the particles from its own files
    SA = SnapshotAllSky(boxsize_resolution=box_res, simulation_name=sim_name, verbose=1,orientation_lock="cube")
    shell_particles = SA.place_snapshot_particles_in_shell(
        lightcone_redshift_range=redshift_range,
        property_names=["Masses"],
        particle_types=["PartType0"],
        comm=comm,
        )
    gas_particle_data = shell_particles["PartType0"]
    if not gas_particle_data:
        raise ValueError("no gas particles found in the lightcone shell")

    part_pos = gas_particle_data["Coordinates"].to("Mpc")
    part_hsml = gas_particle_data["SmoothingLengths"].to("Mpc")
    part_mass = gas_particle_data["Masses"].to("Msun")
    message(f"\n{comm.allreduce(len(part_mass))} gas particles in the lightcone shell\n")

    # smooth the gas mass onto the map
    map_data = make_smoothed_sky_map(part_pos, part_hsml, part_mass, nside, progress=True)

    # write the map
    write_map(output_filename, "SmoothedGasMass", map_data, nside, redshift_range)
    message(f"Map written to: {output_filename}")
    
    # plot the map, read back from the file on rank 0
    comm.barrier()
    if comm_rank == 0:
        plot_filename = output_filename.replace(".hdf5", ".png")
        plot_map(output_filename, "SmoothedGasMass", plot_filename)
        message(f"Map plot saved as: {plot_filename}")