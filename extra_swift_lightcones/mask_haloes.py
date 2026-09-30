#!/bin/env python

import os
import sys
import numpy as np
import h5py
import healpy as hp
import unyt
from pathlib import Path
from lightcone_io.halo_reader import HaloLightconeFile
from lightcone_io.units import units_from_attributes
from . import swift_snapshot_redshift_conversion as nz
import psutil
import datetime as dt
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from contextlib import ExitStack

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Mask selection of haloes for a given radius and mass range. 
"""


def message(m):
    if comm_rank == 0:
        current_time=dt.datetime.now()
        time_str=current_time.strftime("%H:%M:%S")
        print("[@{print_time}]:\t".format(print_time=time_str)+m)

def rank_message(m, rank):
    """
    Print a new message on each rank.
    """
    current_time=dt.datetime.now()
    time_str=current_time.strftime("%H:%M:%S")
    print('[Rank {rank_nr:03d}] [@{print_time}]'.format(rank_nr=rank,print_time=time_str) + m)

def _rss_used():
    """
    Report this process's own resident set size (RSS)
    """
    # Do nothing if psutil is not installed
    if psutil is None:
        return None

    GB = 1024**3
    return psutil.Process().memory_info().rss / GB

def report_rss(m, comm):
    """Collective: report max and total peak RSS across all ranks."""
    rss_gb = _rss_used()
    max_rss = comm.allreduce(rss_gb, op=MPI.MAX)
    sum_rss = comm.allreduce(rss_gb, op=MPI.SUM)
    message(f"RSS [{m}]: max = {max_rss:.2f} [GB], sum = {sum_rss:.2f} [GB]")

def get_num(x):
   return int(x.split('/')[-2].lstrip().split('_')[-1])

def attr_scalar(value):
    """
    Correctly return HDF5 attributes that are conceptually scalar but are stored as a length-1 array. 
    """
    return np.asarray(value).flat[0]

def native_endian(arr):
    return arr.astype(arr.dtype.newbyteorder("="))

def exchange_particles(part_dest, arrays):
    """
    Verbatim copy of lightcone_io.smoothed_map.exchange_particles that cannot be simply imported, hence local duplication.
    """
    
    send_arrays = []
    order = np.argsort(part_dest)
    for array in arrays:
        send_arrays.append(native_endian(array[order, ...]))

    send_count = np.bincount(part_dest, minlength=comm_size)
    recv_count = np.zeros_like(send_count)
    comm.Alltoall(send_count, recv_count)
    send_offset = np.cumsum(send_count) - send_count
    recv_offset = np.cumsum(recv_count) - recv_count
    nr_recv_tot = np.sum(recv_count)

    recv_arrays = []
    for array in send_arrays:
        shape = list(array.shape)
        shape[0] = nr_recv_tot
        recv_arrays.append(np.empty_like(array, shape=shape))

    for sendbuf, recvbuf in zip(send_arrays, recv_arrays):
        if len(sendbuf.shape) == 1:
            psort.my_alltoallv(sendbuf, send_count, send_offset,
                               recvbuf, recv_count, recv_offset,
                               comm=comm)
        elif len(sendbuf.shape) == 2:
            ndims = sendbuf.shape[1]
            psort.my_alltoallv(sendbuf.ravel(), ndims * send_count, ndims * send_offset,
                               recvbuf.ravel(), ndims * recv_count, ndims * recv_offset,
                               comm=comm)
        else:
            raise RuntimeError("Can only exchange arrays with 1 or 2 dimensions")

    return recv_arrays

def get_map_rotation_angles(fn):

    with h5py.File(fn, 'r') as rotation_data:
        rot_val=rotation_data['shells']
        latitude  = (rot_val['phi'][:]*180/np.pi * unyt.deg).astype(np.float64)
        longitude = (rot_val['theta'][:]*180/np.pi * unyt.deg).astype(np.float64)
    
    return latitude, longitude

def rotate_cluster_coords(cluster_coords, map_rotator_object):
    """
    Rotate cluster 3D Cartesian comoving positions to match a sky
    rotation already applied to a HEALPix map via rotate_map_fast.

    Each clusters comoving distance from the observer is preserved. 

    Parameters
    ----------
    cluster_coords : ndarray, shape (N, 3)
        Cartesian comoving positions (Mpc), relative to the observer.
    map_rotator_object : hp.Rotator
        The SAME Rotator instance returned by rotate_map_fast for this
        shell/rotation -- do not construct a new one from theta/phi
        independently.

    Returns
    -------
    rotated_coords : ndarray, shape (N, 3)
        Cluster positions in the rotated frame, same distances, rotated
        directions.
    """
    cluster_coords = np.atleast_2d(np.asarray(cluster_coords, dtype=float))
    distances = np.linalg.norm(cluster_coords, axis=1)

    rotated_coords = np.empty_like(cluster_coords)
    for i in range(cluster_coords.shape[0]):
        r = distances[i]
        if r == 0:
            rotated_coords[i] = cluster_coords[i]
            continue
        unit_vec = cluster_coords[i] / r
        rotated_unit_vec = map_rotator_object(*unit_vec)
        rotated_coords[i] = np.array(rotated_unit_vec) * r

    return rotated_coords

def read_shell_halo_properties(halo_filenames, soap_filenames, snapshot_numbers, radius_prop, mass_prop, mass_unit_str, comm):
    """
    MPI-parallel replacement for looping over each snapshot contributing to
    a shell. 
    Per shell:
    1) Read a shell's halo lightcone files in a single collective read
    2) Cross-references each snapshot's SOAP file, within the same snapshot,
         for radius_prop/mass_prop via InputHalos/SOAPIndex, 
    3) radius and mass properties are converted from SOAP units to comoving Mpc and the given mass units

    Returns a dict of local per-halo arrays
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # halo lightcone properties to read
    halo_lightcone_datasets = ("Lightcone/HaloCentre", "Lightcone/Redshift", "InputHalos/IsCentral",
                                "Lightcone/SnapshotNumber", "InputHalos/SOAPIndex")
    # collective multifile read of halo lightcones
    mf = phdf5.MultiFile(halo_filenames, comm=comm)
    # load halo lightcone dataset
    halo_data = mf.read(halo_lightcone_datasets, group="/", read_attributes=True)
    
    # number of haloes
    nr_halos = len(halo_data["Lightcone/SnapshotNumber"])
    halo_data[radius_prop] = None
    halo_data[mass_prop] = None

    # Sort locally by snapshot number so each snapshot's halos are contiguous,
    # then find each present snapshot's slice 
    order = np.argsort(halo_data["Lightcone/SnapshotNumber"])
    for name in halo_data:
        if halo_data[name] is not None:
            halo_data[name] = halo_data[name][order, ...]

    unique_snap, snap_offset, snap_count = np.unique(halo_data["Lightcone/SnapshotNumber"],
                                                      return_index=True, return_counts=True)
    snap_slice = {int(us): (int(so), int(so) + int(sc))
                  for us, so, sc in zip(unique_snap, snap_offset, snap_count)}

    for snapnum, soap_fn in zip(snapshot_numbers, soap_filenames):
        i1, i2 = snap_slice.get(int(snapnum), (0, 0))

        message(f"\treading soap file(s):\t{snapnum}")
        
        # collective multifile read of soap catalogues
        mf_soap = phdf5.MultiFile([soap_fn], comm=comm)
        # load soap dataset
        soap_data = mf_soap.read((radius_prop, mass_prop), read_attributes=True)

        #cheeky gathering of soap files
        if comm_rank == 0:
            with h5py.File(soap_fn, "r") as f:
                a = float(attr_scalar(f["SWIFT"]["Header"].attrs["Scale-factor"]))
        else:
            a = None
        a = comm.bcast(a)

        # store attrs
        radius_attrs = soap_data[radius_prop].attrs
        mass_attrs = soap_data[mass_prop].attrs

        # convert units for mass and radius
        soap_data[radius_prop] = phdf5.AttributeArray(
            unyt.unyt_array(soap_data[radius_prop], units_from_attributes(soap_data[radius_prop])).to_value("Mpc"),
            attrs=radius_attrs)
        
        soap_data[mass_prop] = phdf5.AttributeArray(
            unyt.unyt_array(soap_data[mass_prop], units_from_attributes(soap_data[mass_prop])).to_value(mass_unit_str),
            attrs=mass_attrs)

        
        # Ensure radius is comoving
        radius_a_exponent = float(attr_scalar(radius_attrs["a-scale exponent"]))
        radius_physical = bool(attr_scalar(mass_attrs.get("Value stored as physical", 0)))
        if radius_physical:
            soap_data[radius_prop] = soap_data[radius_prop] / a
        else:
            soap_data[radius_prop] = soap_data[radius_prop] * a**(radius_a_exponent - 1.0)

        if halo_data[radius_prop] is None:
            halo_data[radius_prop] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=soap_data[radius_prop].dtype),
                                                            attrs=soap_data[radius_prop].attrs)
            halo_data[mass_prop] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=soap_data[mass_prop].dtype),
                                                          attrs=soap_data[mass_prop].attrs)

        ptr = halo_data["InputHalos/SOAPIndex"][i1:i2]
        assert np.all(ptr>=0) # All halos in the lightcone should be found in SOAP
        psort.fetch_elements(soap_data[radius_prop], ptr, result=halo_data[radius_prop][i1:i2], comm=comm)
        psort.fetch_elements(soap_data[mass_prop], ptr, result=halo_data[mass_prop][i1:i2], comm=comm)

    assert np.all(halo_data[radius_prop] >= 0)
    assert np.all(halo_data[mass_prop] >= 0)

    return halo_data

def distribute_pixels(comm, nside):
    """
    Verbatim copy of lightcone_io.smoothed_map.distribute_pixels that cannot be simply imported, hence local duplication.
    
    Returns (nr_total_pixels, nr_local_pixels, local_offset, theta_boundary).
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    nr_total_pixels = hp.pixelfunc.nside2npix(nside)
    nr_local_pixels = nr_total_pixels // comm_size
    local_offset = nr_local_pixels * comm_rank

    first_pixel = np.arange(comm_size, dtype=int) * nr_local_pixels
    last_pixel = first_pixel + nr_local_pixels - 1
    last_pixel[-1] = nr_total_pixels - 1
    nr_local_pixels = last_pixel[comm_rank] - first_pixel[comm_rank] + 1

    theta_boundary = np.ndarray(comm_size + 1, dtype=float)
    theta_boundary[0] = 0.0
    theta_boundary[-1] = np.pi
    for i in range(0, comm_size - 1):
        theta1, phi = hp.pixelfunc.pix2ang(nside, last_pixel[i])
        theta2, phi = hp.pixelfunc.pix2ang(nside, first_pixel[i + 1])
        theta_boundary[i + 1] = 0.5 * (theta1 + theta2)

    return nr_total_pixels, nr_local_pixels, local_offset, theta_boundary

def mask_cluster_apertures_local(nside, cluster_coords, radius_mpc, local_offset, nr_local_pixels):
    """
    Stamp each halo's aperture onto only a given rank's slice of the sky
    (global pixel indices [local_offset, local_offset + nr_local_pixels)). 
    If zero haloes on rank, returns an all 1's mask. 
    
    """
    cluster_coords = np.atleast_2d(np.asarray(cluster_coords, dtype=float))
    n_clusters = cluster_coords.shape[0]

    radius_mpc = np.atleast_1d(np.asarray(radius_mpc, dtype=float))
    if radius_mpc.size == 1 and n_clusters > 1:
        radius_mpc = np.full(n_clusters, radius_mpc[0])
    elif radius_mpc.size not in (0, n_clusters):
        raise ValueError(
            f"radius_mpc must be a scalar or have length {n_clusters}, got length {radius_mpc.size}")

    local_mask = np.ones(nr_local_pixels, dtype=np.int32)
    for i in range(n_clusters):
        pos = cluster_coords[i]
        r = np.linalg.norm(pos)
        if r == 0:
            continue
        sky_vec = pos / r
        half_angle = np.arctan(radius_mpc[i] / r)
        disc_pixels = hp.query_disc(nside, sky_vec, half_angle)
        local_idx = disc_pixels - local_offset
        local = (local_idx >= 0) & (local_idx < nr_local_pixels)
        local_mask[local_idx[local]] = 0
    return local_mask

def route_halos_to_pixel_owners(theta_boundary, cluster_coords, radius_mpc, extra_arrays, comm):
    """
    Determine which rank(s) each halo's angular aperture could reach. 
    Mirrors lightcone_io.smoothed_map.make_sky_map's particle-to-rank linking.

    cluster_coords: (N, 3) comoving positions, this rank's local halos only.
    radius_mpc:     (N,) comoving aperture radius for each halo AT THIS RADIUS
                        SCALING (i.e. already multiplied by whichever of [0.2, 1.0, 5.0] the
                        caller is on) - routing must be redone per radius scaling, since a
                        halo's angular footprint changes with it, but NOT per mass bin,
                        since the mass bin only filters which halos are kept, not where
                        their aperture reaches. Do that filtering on the receiving side
                        instead of re-routing once per (ri, mj).
    extra_arrays:   additional 1D/2D per-halo arrays to carry along (e.g. mass,
                        needed by the receiver for its own mass-bin filtering).

    Returns (recv_coords, recv_radius, recv_extra_arrays):  
        this rank's own halos plus every halo forwarded to it by other ranks whose aperture
        reaches this rank's band.
    """
    n = cluster_coords.shape[0]

    r = np.linalg.norm(cluster_coords, axis=1)
    safe_r = np.where(r == 0, 1.0, r)
    theta, phi = hp.pixelfunc.vec2ang(cluster_coords)
    half_angle = np.arctan(radius_mpc / safe_r)

    theta_min = np.clip(theta - half_angle, 0.0, np.pi)
    theta_max = np.clip(theta + half_angle, 0.0, np.pi)

    first_rank = np.clip(np.searchsorted(theta_boundary, theta_min, side="left") - 1, 0, comm_size - 1)
    last_rank = np.clip(np.searchsorted(theta_boundary, theta_max, side="right") - 1, 0, comm_size - 1)
    assert np.all(theta_boundary[first_rank] <= theta_min)
    assert np.all(theta_boundary[last_rank + 1] >= theta_max)

    nr_copies = last_rank - first_rank + 1
    assert np.all(nr_copies >= 1) & np.all(nr_copies <= comm_size)

    index = np.repeat(np.arange(n, dtype=int), nr_copies)
    offset = np.cumsum(nr_copies) - nr_copies
    dest = -np.ones(np.sum(nr_copies), dtype=int)
    for fr, nc, off in zip(first_rank, nr_copies, offset):
        dest[off:off + nc] = np.arange(fr, fr + nc, dtype=int)
    assert np.all(dest >= 0) & np.all(dest < comm_size)

    arrays_to_send = [cluster_coords[index, :], radius_mpc[index]] + [a[index, ...] for a in extra_arrays]
    recv = exchange_particles(dest, arrays_to_send)
    recv_coords, recv_radius = recv[0], recv[1]
    recv_extra = recv[2:]
    return recv_coords, recv_radius, recv_extra

_UNSET = object()
def write_binary_masks(halo_cat_format, soap_cat_format, 
                            mass_poperty, radius_poperty, 
                            nshell, nside, 
                            latitudes, longitudes,
                            lightcone_numbers, mask_filename, output_dir,
                            comm, 
                            bin_method="both", mass_units="Msun", 
                            log_mass_bin_range=None, log_mass_bin_width=None, scale_radius=None,
                            hdf5_dset_kwargs=_UNSET
                            ):
    """
    Write mask in parallel, Reuses read_shell_halo_properties unchanged
    (the MPI-parallel halo lightcone / SOAP cross-reference read is the same
    regardless of how the OUTPUT map is decomposed - only how the resulting
    halos get turned into a written mask differs here).

    bin_method selects which mass-bin definition(s) to build masks for, same
    meaning as write_masks_mpi_M500c's bin_method: "discrete", "cumulative",
    or "both". With "both", neither the halo lightcone/SOAP read (once per
    shell) nor the halo-to-rank routing (once per shell per radius scaling,
    see the ri loop below) is repeated per method - only the cheap mass-bin
    filter + local aperture-masking + collective write is.


    Params: 
        mass_poperty:           Path to SOAP catalogue values to use for the 
                                    selection of haloes on the sky 
                                    e.g. 'SO/500_crit/TotalMass' for M500c
        radius_poperty:         Path to values to use as the radius of each halo 
                                    in the SOAP catalogue e.g. 'SO/500_crit/SORadius' 
                                    for R500c
        mass_untis:             String showing the units of the mass property given. 
                                    Optional, default is solar masses 'Msun'
        log_mass_bin_range:     Tuple, log10 min and max values of the mass_poperty to 
                                    bin haloes by. Optional, if None, then use 
                                    default range of 12-16
        mass_bin_width:         log10 width of the log10 mass bins. Optional, if None 
                                    then default is 0.5
        scale_radius:           Array, coeficents to scale the given radius property by 
                                    when drawing apatures about each halo. If None then 
                                    assume a coefficents of 0.2, 0.5, 1.0, 2.0 and 5.0. 
        bin_method:             str,  selects which mass-bin definition(s) to build masks 
                                    with. 
                                        'discrete':     M1 < log10(M) <= M2, where M1 and M2 
                                                            are the edges of the mass bin
                                        'cumulative':   M1 < log10(M) <= Mmax, where Mmax is 
                                                            the maximum of the log_mass_bin_range 
                                                            given. 
                                        "both":         Make 2 sets of mask, one for each other 
                                                            bin method. 
        redshift_filename:      path to redshift filename
    """
    if bin_method not in ("discrete", "cumulative", "both"):
        raise ValueError(f"bin_method must be 'discrete', 'cumulative', or 'both', got {bin_method!r}")
    
    if hdf5_dset_kwargs is _UNSET:
        hdf5_dset_kwargs={
                "compression": "gzip",
                "compression_opts": 9,
                "shuffle": True,
            }
    elif hdf5_dset_kwargs is None:
        hdf5_dset_kwargs={}

    mass_unit_str = mass_units
    mass_prop=mass_poperty
    radius_prop=radius_poperty

    def radius_values(r_soap, scale_radius=None):
        if scale_radius is None:
            r_modify = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
        else:
            r_modify= np.asarray(scale_radius)

        return r_soap[:, None] * r_modify[None, :]

    def discrete_mass_bins(mass_bin_range=None, mass_bin_width=None):
        if mass_bin_range is None:
            mass_bin_range=(12,16)
        if mass_bin_width is None:
            mass_bin_width=0.5
        lo=float(mass_bin_range[0])
        hi=float(mass_bin_range[1])
        dm=float(mass_bin_width)

        return np.vstack((np.arange(lo, hi, dm), np.arange(lo+dm, hi+dm, dm))).T

    def cumulative_mass_bins(mass_bin_range=None, mass_bin_width=None):
        if mass_bin_range is None:
            mass_bin_range=(12,16)
        if mass_bin_width is None:
            mass_bin_width=0.5
        lo=float(mass_bin_range[0])
        hi=float(mass_bin_range[1])
        dm=float(mass_bin_width)
        low = np.arange(lo, hi, dm)
        high = np.full_like(low, hi)
        return np.vstack((low, high)).T

    # (method_name, log_mass_bins, output_filename_prefix) for every bin
    # method this call is asked to build
    method_specs = []
    if bin_method in ("discrete", "both"):
        method_specs.append(("discrete", discrete_mass_bins(log_mass_bin_range, log_mass_bin_width), "discrete_"))
    if bin_method in ("cumulative", "both"):
        method_specs.append(("cumulative", cumulative_mass_bins(log_mass_bin_range, log_mass_bin_width), "cumulative_"))

    radius_scalings = radius_values(np.ones(1), scale_radius).flatten()
    n_radius = len(radius_scalings)

    # Pixel decomposition 
    nr_total_pixels, nr_local_pixels, local_offset, theta_boundary = distribute_pixels(comm, nside)
    assert nr_total_pixels == hp.nside2npix(nside)

    assert nr_local_pixels > 0, f"rank {comm_rank} has 0 local pixels (nside={nside}, comm_size={comm_size})"

    # Chunk boundaries aligned to per-rank pixel range. 
    #'chunk_pixels' is the number of pixels on every rank, except the last. 
    chunk_pixels = max(1, nr_total_pixels // comm_size)

    for lightcone_nr in lightcone_numbers:
        lc_total = 0
        kept_total = 0

        if comm_rank == 0:
            # Create directory and read redshifts on rank 0 before broadcasting
            directory_path = Path(output_dir)
            directory_path.mkdir(parents=True, exist_ok=True)
            redshifts=np.loadtxt(nz.flamingo_shell_redshift_file('L2p8'), delimiter=",")[:nshell]

        else:
            redshifts = None
        redshifts = comm.bcast(redshifts)

        # With mpio driver, open files, create groups and 
        # write datasets + attributes on every rank. 
        with ExitStack() as stack: #use stack as safety net
            outfiles = {}
            MASKS = {}
            for method_name, log_mass_bins, prefix in method_specs:
                n_mass = np.shape(log_mass_bins)[0]
                output_filename = output_dir + f"/lightcone{lightcone_nr}." + prefix + mask_filename
                message(output_dir)
                message(f"lightcone:\t{lightcone_nr}\nbin_method:\t{method_name}\nwriting to outputfile:\t{output_filename}")
                
                # build output on each rank
                outfile = stack.enter_context(h5py.File(output_filename, "w", driver="mpio", comm=comm))
                #outfile = h5py.File(output_filename, "w", driver="mpio", comm=comm)
                #with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
                outfiles[method_name] = outfile
                # store mask metadata 
                meta_data = outfile.create_group("metadata")
                meta_data.attrs["shells"] = np.arange(0, nshell, 1)
                meta_data.attrs["redshift"] = np.arange(0, nshell * 0.05, 0.05)
                meta_data.attrs["nside"] = [nside]
                meta_data.attrs["log_mass_bins"] = log_mass_bins
                meta_data.attrs["radius_coef"] = radius_scalings
                meta_data.attrs["mass_property"] = [mass_prop]
                meta_data.attrs["mass_units"]=[mass_unit_str]
                meta_data.attrs["radius_property"] = [radius_prop]
                meta_data.attrs["bin_method"] = [method_name]

                mask_data = outfile.create_group("mask")

                comm.barrier() # collect ranks after building output and meta data

                MASK = {}
                for ri in range(n_radius):
                    for mj in range(n_mass):
                        mask_str = f"R{ri:d}_M{mj:d}" # name of mask based on bin indices
                        MASK[mask_str] = mask_data.create_dataset(
                            mask_str,
                            shape=(nshell, nr_total_pixels),
                            dtype=np.int32,
                            chunks=(1, chunk_pixels),
                            **hdf5_dset_kwargs
                        )
                MASKS[method_name] = MASK

            for shell_nr in range(nshell):
                message(f"\tloading haloes in shell {shell_nr}...")
                zmin = redshifts[shell_nr, 0]
                zmax = redshifts[shell_nr, 1]

                snapshot_numbers = nz.snapshot_number_in_range(
                    redshift_range=(zmin, zmax), boxsize_resolution=nz.flamingo_box_resolution(halo_cat_format),
                    redshift_buffer=(0.025, 0.025), snapshot_buffer=(0, 0),
                    bounds="strict", decimals=5)[::-1]

                message(f"\tredshift range:\t{zmin:.3f}-{zmax:.3f}, snapshot numbers:\t{snapshot_numbers[0]}-{snapshot_numbers[-1]}")
                comm.barrier() # collect ranks after establishing shell redshifts

                # generate list of halo lightcone and soap catalogue filenames from snapshot numbers
                halo_filenames = [halo_cat_format.format(lightcone_nr=lightcone_nr, snap_nr=ii) for ii in snapshot_numbers]
                soap_filenames = [soap_cat_format.format(snap_nr=ii) for ii in snapshot_numbers]

                # shared reading of all catalogues for this shell
                halo_data = read_shell_halo_properties(halo_filenames, soap_filenames, snapshot_numbers, radius_prop, mass_prop, mass_unit_str, comm)

                # get local redshifts and central masks
                local_redshift = np.asarray(halo_data["Lightcone/Redshift"])

                # if SO properties use centrals only
                if ("SO/" in mass_prop) or ("SO/" in radius_prop):
                    message("SO property detected, using centrals only in redshift range")
                    local_central = np.asarray(halo_data["InputHalos/IsCentral"])
                else:
                    message("no SO property detected, using all haloes in redshift range")
                    local_central = np.ones(len(halo_data["Lightcone/Redshift"]), dtype=int)
                m_redshift = (local_redshift >= zmin) & (local_redshift <= zmax) & (local_central == 1)

                shell_total = comm.allreduce(len(m_redshift))
                shell_kept = comm.allreduce(int(np.sum(m_redshift)))
                message(f"shell:\t{shell_nr}, keep:\t{shell_kept}/{shell_total}")

                lc_total += shell_total
                kept_total += shell_kept

                if shell_kept == 0:
                    # No kept halos anywhere in this shell so skip the
                    # rotation, routing and mass-bin work 
                    ones = np.ones(nr_local_pixels, dtype=np.int32)
                    for method_name, log_mass_bins, prefix in method_specs:
                        n_mass = np.shape(log_mass_bins)[0]
                        MASK = MASKS[method_name]
                        for ri in range(n_radius):
                            for mj in range(n_mass):
                                mask_str = f"R{ri:d}_M{mj:d}"
                                with MASK[mask_str].collective:
                                    MASK[mask_str][shell_nr, local_offset:local_offset + nr_local_pixels] = ones
                    continue

                # store local halo centres on the sky
                local_centre = np.asarray(halo_data["Lightcone/HaloCentre"])[m_redshift]
                local_radius_1d = np.asarray(halo_data[radius_prop])[m_redshift]
                local_mass = np.asarray(halo_data[mass_prop])[m_redshift]

                # if no need to rotate coordinates on the sky then skip this step, i.e., theta==phi==0 
                if latitudes[shell_nr] == 0 and longitudes[shell_nr] == 0:
                    map_rotator_object = None
                else:
                    # rotate halo coords on the sky the same as done when rotating maps for intergrated lightcones. 
                    map_rotator_object = hp.Rotator(
                        rot=[latitudes[shell_nr].to_value(unyt.deg), longitudes[shell_nr].to_value(unyt.deg)],
                        inv=True, deg=True)
                if map_rotator_object is not None and local_centre.shape[0] > 0:
                    local_centre = rotate_cluster_coords(local_centre, map_rotator_object)

                # iterate through radii to draw appatures
                for ri in range(n_radius):
                    radius_this_ri = local_radius_1d * radius_scalings[ri]
                    recv_coords, recv_radius, (recv_mass,) = route_halos_to_pixel_owners(theta_boundary, local_centre, radius_this_ri, [local_mass], comm)

                    #iterate through mass bins per bin method 
                    for method_name, log_mass_bins, prefix in method_specs:
                        n_mass = np.shape(log_mass_bins)[0]
                        MASK = MASKS[method_name]
                        for mj in range(n_mass):
                            mask_str = f"R{ri:d}_M{mj:d}" # output mask name 
                            m_mass = (recv_mass > 10**log_mass_bins[mj][0]) & (recv_mass <= 10**log_mass_bins[mj][1]) # transform bin edges from log10 to real 

                            # local mask for haloes in redshift range and mass bin
                            local_mask = mask_cluster_apertures_local(nside, recv_coords[m_mass], recv_radius[m_mass], local_offset, nr_local_pixels)

                            # Each rank writes into the same collective dataset
                            with MASK[mask_str].collective:
                                MASK[mask_str][shell_nr, local_offset:local_offset + nr_local_pixels] = local_mask

                    rank_message(f"lightcone {lightcone_nr},\t{ri+1:d}/{n_radius:d} masks completed [radius < {radius_scalings[ri]} {radius_prop}]", comm_rank)
                
                comm.barrier() # use barrier to collect ranks per shell iteration
            
            rank_message(f"Completed {mass_prop} masks for lightcone {lightcone_nr} shells 0 - {nshell}", comm_rank)
