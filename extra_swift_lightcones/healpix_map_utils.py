#!/bin/env python
import os
import sys
import numpy as np
import healpy as hp
import h5py
import unyt
import lightcone_io.healpix_maps as hm
from . import snapshot_units as sw_units
from .swift_snapshot_redshift_conversion import flamingo_shell_redshift_file

import math

"""
Utility functions for working with healpix maps. 
"""


def get_related_ipix(ipix, nside, levels, ordering="ring"):
    """
    Return the parent or child HEALPix pixel indices.

    Returns a tuple of (pixel indices at the new resolution, nside of the new map).

    :param  ipix:       pixel indices in the input map
    :type   ipix:       int or np.ndarray
    :param  nside:      nside of the input map
    :type   nside:      int
    :param  levels:     change in HEALPix resolution. Each level increases or decreases nside by a factor of 2.
                            levels > 0 returns child pixels at higher resolution 
                            levels < 0 returns parent pixels at lower resolution
                            levels == 0 returns the input pixels unchanged
    :type   levels:     int
    :param  ordering:   HEALPix pixel ordering, either 'ring' or 'nested'
    :type   ordering:   str
    """

    # Ensure array
    ipix = np.atleast_1d(ipix)

    # Check ordering
    if ordering not in ("ring", "nested"):
        raise ValueError("ordering must be either 'ring' or 'nested'")

    # Convert to nested ordering
    if ordering == "ring":
        ipix_nest = hp.ring2nest(nside, ipix)
    else:
        ipix_nest = ipix.copy()

    # Increase resolution and get child pixels
    if levels > 0:
        for _ in range(levels):
            ipix_nest = (4 * ipix_nest[:, None]+ np.arange(4)).reshape(-1)
        nside_new = nside * (2 ** levels)

    # Decrease resolution and get parent pixels
    elif levels < 0:
        for _ in range(-levels):
            ipix_nest = ipix_nest // 4
        nside_new = nside // (2 ** (-levels))

    # Same resolution, no change 
    else:
        nside_new = nside

    # Convert back to requested ordering
    if ordering == "ring":
        ipix_new = hp.nest2ring(nside_new, ipix_nest)
    else:
        ipix_new = ipix_nest
    
    return ipix_new, nside_new


def find_healpy_pixel_weights_dir(datapath=None):
    """
    Find the healpy pixel weights directory.

    Returns the first directory that contains a full_weights/ folder. 
    Check:
        1. the datapath argument
        2. the HEALPY_PIXEL_WEIGHTS environment variable
        3. <venv>/share/healpy-data
    
    Raises FileNotFoundError if none are found.

    :param  datapath:   directory to check first for the pixel weights
    :type   datapath:   str
    """
    candidates = [
        datapath,
        os.environ.get("HEALPY_PIXEL_WEIGHTS"),
        os.path.join(sys.prefix, "share", "healpy-data"),
    ]
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "full_weights")):
            return path
    tried = [p for p in candidates if p]
    raise FileNotFoundError(f"Could not find healpy pixel weights (no full_weights/folder in any of: {tried})")


def check_healpy_pixel_weights(nside, datapath=None):
    """
    Check that the healpy pixel weights file for this nside can be found.
    Looks for <datapath>/full_weights/healpix_full_weights_nside_XXXX.fits

    Returns the datapath to pass to hp.map2alm.

    :param  nside:      nside of the map
    :type   nside:      int
    :param  datapath:   directory to check first for the pixel weights
    :type   datapath:   str
    """
    datapath = find_healpy_pixel_weights_dir(datapath)

    weights_file = os.path.join(datapath, "full_weights", f"healpix_full_weights_nside_{nside:04d}.fits")
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"No healpy pixel weights for nside={nside}: expected {weights_file}")

    return datapath


def healpy_pixel_weights_available(nside=None, datapath=None):
    """
    Returns True if healpy pixel weights can be found, otherwise False.

    :param  nside:      nside of the map. If None, check for any pixel weights
    :type   nside:      int
    :param  datapath:   directory to check first for the pixel weights
    :type   datapath:   str
    """
    candidates = [
        datapath,
        os.environ.get("HEALPY_PIXEL_WEIGHTS"),
        os.path.join(sys.prefix, "share", "healpy-data"),
    ]
    for path in candidates:
        if not path:
            continue
        weights_dir = os.path.join(path, "full_weights")
        if not os.path.isdir(weights_dir):
            continue
        if nside is None:
            return True
        weights_file = os.path.join(
            weights_dir, f"healpix_full_weights_nside_{int(nside):04d}.fits"
        )
        if os.path.isfile(weights_file):
            return True
    return False


def get_common_maps(filenames):
    """
    Returns a list of all dataset names common to all files.

    :param  filenames:  paths to the .hdf5 files
    :type   filenames:  list
    """
    # check for common datasets:

    all_map_names=[[] for i in range(len(filenames))]

    for file_idx, file_name in enumerate(filenames):
        infile_names=[]
        with h5py.File(file_name, "r") as f:
            
            for name in f:
                dset=f[name]
                if "nside" in dset.attrs:
                    infile_names.append(name)


        all_map_names[file_idx] = infile_names
    common_map_names = list(set.intersection(*map(set, all_map_names)))
    
    return common_map_names


def read_rotation_angles(filename):

    """
    Read 'theta' (co-latitude) and 'phi' (longitude) angles from hdf5 file. 
    Assumes that theta and phi are given in radians. 

    Returns a tuple of (co-latitude, longitude) in degrees.

    :param  filename:   path to the .hdf5 file containing the rotation angles
    :type   filename:   str
    """
    with h5py.File(filename, 'r') as rotation_data:
        rot_val=rotation_data['shells']
        longitude  = (rot_val['phi'][:]*180/np.pi * unyt.deg).astype(np.float64)
        latitude = (rot_val['theta'][:]*180/np.pi * unyt.deg).astype(np.float64)
    
    return latitude, longitude


def rotate_map(base_map, theta, phi, ell_max=None, map_rotator_object=None):
    """
    Rotate given map using alm space via the co-latitude and longitude passed as theta and phi.

    Returns a tuple, (rotated map, rotator object)

    :param  base_map:           healpix map to rotate
    :type   base_map:           np.ndarray
    :param  theta:              co-latitude rotation angle
    :type   theta:              unyt.unyt_quantity (units of deg or equivalent)
    :param  phi:                longitude rotation angle
    :type   phi:                unyt.unyt_quantity (units of deg or equivalent)
    :param  ell_max:            maximum multipole used in the rotation
    :type   ell_max:            int
    :param  map_rotator_object: rotator to use. If None, one is created from theta and phi
    :type   map_rotator_object: healpy.Rotator
    """

    if theta ==0. and phi ==0.:
        return base_map, map_rotator_object
    if map_rotator_object is None:
        map_rotator_object = hp.Rotator(rot=[theta.to_value(unyt.deg), phi.to_value(unyt.deg)], inv=True, deg=True)

    if ell_max is None:
        return map_rotator_object.rotate_map_alms(base_map.astype(np.float64)), map_rotator_object
    elif ell_max is not None:
        return map_rotator_object.rotate_map_alms(base_map.astype(np.float64), lmax=ell_max), map_rotator_object


def rotate_map_fast(base_map, theta, phi, nside, ell_max=None, map_rotator_object=None):
    """
    Rotate a given map using alm space via the passed as theta (co-latitude) and phi (longitude). 
    Use the healpy pixel weights for the map to alm transform.

    Returns a tuple (rotated map, rotator object)

    :param  base_map:           healpix map to rotate
    :type   base_map:           np.ndarray
    :param  theta:              co-latitude rotation angle
    :type   theta:              unyt.unyt_quantity (units of deg or equivalent)
    :param  phi:                longitude rotation angle
    :type   phi:                unyt.unyt_quantity (units of deg or equivalent)
    :param  nside:              nside of the rotated map
    :type   nside:              int
    :param  ell_max:            maximum multipole used in the rotation
    :type   ell_max:            int
    :param  map_rotator_object: rotator to use. If None, one is created from theta and phi
    :type   map_rotator_object: healpy.Rotator
    """
    if theta == 0. and phi == 0.:
        return base_map, map_rotator_object
    if map_rotator_object is None:
        map_rotator_object = hp.Rotator(rot=[theta.to_value(unyt.deg), phi.to_value(unyt.deg)], inv=True, deg=True)
    
    HEALPY_PIXEL_WEIGHTS=check_healpy_pixel_weights(nside)
    
    alm = hp.map2alm(base_map.astype(np.float64), lmax=ell_max, use_pixel_weights=True, iter=3, datapath=HEALPY_PIXEL_WEIGHTS)
    alm_rot = map_rotator_object.rotate_alm(alm, lmax=ell_max)
    map_rot = hp.alm2map(alm_rot, nside=nside, lmax=ell_max)
    return map_rot, map_rotator_object



_UNSET = object()
def write_rotated_lightcone_chunks(
                                    basedir, basename, map_names, 
                                    theta_arr_deg, phi_arr_deg, 
                                    output_filename, 
                                    output_nside, input_nside, 
                                    remote_dir=None, 
                                    shell_range=None, 
                                    unit_conversion_func=None,
                                    rotate_nside=None, ell_max=None, 
                                    save_chunks=False,
                                    hdf5_dset_kwargs=_UNSET
                                    ):
    """
        For a given set of rotation angles per shell, sum all shells along the line of sight that share the same rotation angles. 
        
        Params:
            map_names:              List of map names, qunatities, datasets to sum and rotate in chunks, e.g. "TotalMass"
            theta_arr_deg:          array of theta (co-latitude) values per shell [deg]
            phi_arr_deg:            array of phi (longitude) values per shell [deg]
            remote_dir:             For reading healpix maps from a remote directory as done with lightcone_io
            shell_range:            tuple containing the shell number of the minium and maximum shell to consider
            unit_conversion_func:   Optional function for additional unit conversion prior to rotating maps.
            save_chunks:            Boolean, if true write each individual chunk to the output file. 
    """
    """
    For a given set of rotation angles per shell, sum all shells along the line of sight that share the same rotation angles. 
    Each chunk is rotated and the sum of all chunks is written to the output file.

    :param  basedir:                directory containing the lightcone healpix maps
    :type   basedir:                str
    :param  basename:               base name of the lightcone healpix map files
    :type   basename:               str
    :param  map_names:              map names (quantities or datasets) to sum and rotate in chunks, e.g. ["TotalMass"]
    :type   map_names:              list
    :param  theta_arr_deg:          theta (co-latitude) values per shell [deg]
    :type   theta_arr_deg:          np.ndarray or unyt.unyt_array
    :param  phi_arr_deg:            phi (longitude) values per shell [deg]
    :type   phi_arr_deg:            np.ndarray or unyt.unyt_array
    :param  output_filename:        path of the .hdf5 file to write
    :type   output_filename:        str
    :param  output_nside:           nside of the output maps
    :type   output_nside:           int
    :param  input_nside:            nside of the input maps
    :type   input_nside:            int
    :param  remote_dir:             for reading healpix maps from a remote directory as done with lightcone_io
    :type   remote_dir:             str
    :param  shell_range:            shell number of the minimum and maximum shell to consider. If None, use all shells
    :type   shell_range:            tuple
    :param  unit_conversion_func:   optional function for additional unit conversion prior to rotating maps
    :type   unit_conversion_func:   function
    :param  rotate_nside:           nside the maps are rotated at. If None, use input_nside
    :type   rotate_nside:           int
    :param  ell_max:                maximum multipole used in the rotation
    :type   ell_max:                int
    :param  save_chunks:            If True, write each individual chunk to the output file
    :type   save_chunks:            boolean
    :param  hdf5_dset_kwargs:       keyword arguments passed to h5py create_dataset. 
                                        If not given, use gzip compression level 9 with shuffle. If None, default h5py compression applied
    :type   hdf5_dset_kwargs:       dict
    """
    if hdf5_dset_kwargs is _UNSET:
        hdf5_dset_kwargs={
                "compression": "gzip",
                "compression_opts": 9,
                "shuffle": True,
            }
    elif hdf5_dset_kwargs is None:
        hdf5_dset_kwargs={}
    
    from lightcone_io.downsample_maps import get_power

    def write_dataset(outfile, name, data, dset_dtype=np.float64):
        """
        Write dataset to the output file, replacing any existing dataset with the same name. 
        Assume gzip=9 and shuffle=True.
        """
        if name in outfile:
            del outfile[name]
        outfile.create_dataset(
            name,
            data=data,
            dtype=dset_dtype,
            **hdf5_dset_kwargs
        )
    
    # Open the lightcone map shell array
    shell_arr = hm.ShellArray(basedir, basename, remote_dir=remote_dir)

    if shell_range is None:
        numb_shells=shell_arr.nr_shells
        shell_numbers = np.arange(0, numb_shells+1)
    else:
        shell_numbers = np.arange(shell_range[0], shell_range[1]+1)
        numb_shells=len(shell_numbers)
    
    # update rotation angles 
    theta_arr_deg = sw_units.apply_expected_units(theta_arr_deg, "degree")[shell_numbers]
    phi_arr_deg = sw_units.apply_expected_units(phi_arr_deg, "degree")[shell_numbers]
    
    theta_unique, indices_theta = np.unique(theta_arr_deg, return_index = True)
    phi_unique, indices_phi = np.unique(phi_arr_deg, return_index = True)
    chunk_num = np.hstack([indices_theta[np.argsort(indices_theta)], numb_shells])
    theta_rot = theta_unique[np.argsort(indices_theta)]
    phi_rot = phi_unique[np.argsort(indices_phi)]
    nrot = len(theta_rot)

    if input_nside==16384:
        input_nside=8192
    if rotate_nside is None:
        rotate_nside=input_nside
    
    in_npix=hp.nside2npix(rotate_nside)
    out_npix=hp.nside2npix(output_nside)
    
    # load all redshifts for any FLAMINGO lightcone
    redshifts=np.loadtxt(flamingo_shell_redshift_file('L2p8'), delimiter=",")[shell_numbers, :]
    #redshifts = np.loadtxt("/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/shell_redshifts.txt", delimiter=",")[shell_numbers, :]
    
    update_str=f"total number of rotations:\t{nrot}"+f"\ntotal range of shell numbers:\t{shell_numbers[0]}, {shell_numbers[-1]}\n" +f"total redshift range:\t{redshifts[0, 0]:.3f}, {redshifts[-1, 1]:.3f}"
    print(update_str)

    
    with h5py.File(output_filename, 'w') as outfile:
        print(f"write to file: {output_filename}\n")
        
        # add metadata attrs to output file
        metadata = outfile.create_group("metadata")
        metadata.attrs["lightcone"] =[basename]
        metadata.attrs["angles"] = np.vstack((theta_rot, phi_rot)).T
        metadata.attrs["nside_rotate"] =[rotate_nside]
        metadata.attrs["nside"] =[output_nside]
        metadata.attrs["redshift"] = redshifts
        metadata.attrs["shell_numbers"] = shell_numbers

        # runnning total
        integrated_map_stack = {str(map_quantity):np.zeros(out_npix, dtype=float) for map_quantity in map_names}

        for chunk_nr in range(0, nrot):
            
            arg_min, arg_max = int(chunk_num[chunk_nr]), int(chunk_num[chunk_nr+1])

            # runnning chunk total, reset once per rotation group
            chunk_map_stack = {str(map_quantity):np.zeros(in_npix, dtype=float) for map_quantity in map_names}
            
            # give update
            print(f"chunk {chunk_nr+1}/{nrot}, shells {arg_min + shell_numbers[0]}-{arg_max -1 + shell_numbers[0]}")

            # combine shells, rotate and then downsample
            if save_chunks:
                chunk_group = outfile.create_group(f"chunk_{chunk_nr}")
                
                # add attrs to chunk group
                for z_range_idx in range(arg_min, arg_max):
                    zmin = redshifts[z_range_idx, 0]
                    zmax = redshifts[z_range_idx, 1]
                    
                    chunk_group.attrs["redshift_range"] = (zmin, zmax)
                    chunk_group.attrs["shell_numbers"] = np.arange(arg_min, arg_max) + shell_numbers[0]
                    metadata.attrs["nside_rotate"] =[rotate_nside]
                    chunk_group.attrs["nside"] =[output_nside]
                    chunk_group.attrs["chunk_nr"] =[chunk_nr]
                    chunk_group.attrs["lightcone"] =[basename]
                    chunk_group.attrs["chunk_rotation_angles"] = (theta_rot[chunk_nr], phi_rot[chunk_nr])

            for shell_nr in range(arg_min, arg_max):
                
                shell_nr += shell_numbers[0] # account for non-zero starting shell 
                shell = shell_arr[shell_nr]   # access shell level data

                for k, map_quantity in enumerate(map_names):
                    #check map is available in shell
                    if map_quantity not in shell.map_names:
                        raise ValueError(f"{map_quantity} not found in shell\nAvailable maps: {shell.map_names}")
                    downsample_power = get_power(map_quantity)
                    map_unit = shell[map_quantity].units
                    
                    if unit_conversion_func is None:
                        data = shell[map_quantity][...].to_value(map_unit).astype(shell[map_quantity].dtype)
                    else:
                        data = unit_conversion_func(shell[map_quantity][...].astype(shell[map_quantity].dtype))
                    
                    if hp.npix2nside(len(data)) != rotate_nside:
                        print(f"\tdownsampling {map_quantity} from nside {hp.npix2nside(len(data))} -> {rotate_nside}")
                        data=hp.ud_grade(data, nside_out=rotate_nside, power=downsample_power)
                    
                    chunk_map_stack[map_quantity] += data

            for map_quantity in map_names:
                if chunk_nr == 0:
                    map_rot = chunk_map_stack[map_quantity]
                else:
                    print(f"\trotating chunk {chunk_nr}, map: {map_quantity}\n\t\ttheta, phi: {theta_rot[chunk_nr]:.3f}, {phi_rot[chunk_nr]:.3f}")
                    #check for pixel weights
                    if healpy_pixel_weights_available(nside=rotate_nside):
                        map_rot, __ = rotate_map_fast(chunk_map_stack[map_quantity], theta_rot[chunk_nr], phi_rot[chunk_nr], nside=output_nside, ell_max=ell_max, map_rotator_object=None)
                    else:
                        map_rot, __ = rotate_map(chunk_map_stack[map_quantity], theta_rot[chunk_nr], phi_rot[chunk_nr], ell_max=ell_max, map_rotator_object=None)
                
                if rotate_nside != output_nside:
                    print(f"\tdownsampling {map_quantity} from nside {rotate_nside} -> {output_nside}")
                    map_rot = hp.ud_grade(map_rot, nside_out=output_nside, power=get_power(map_quantity))
                
                # make a final write to the output file:
                if save_chunks:
                    print(f"\twriting lightcone chunk to file...")
                    write_dataset(chunk_group, map_quantity, map_rot)
                
                integrated_map_stack[map_quantity] += map_rot
                print(f"\tupdated running total: \t{np.sum(integrated_map_stack[map_quantity]):.4e}")
                
        for map_quantity in map_names:
            print(f"\nwriting dataset {map_quantity}")
            write_dataset(outfile, map_quantity, integrated_map_stack[map_quantity], dset_dtype=integrated_map_stack[map_quantity].dtype)
            print(f"{map_quantity} final total:\t{np.sum(integrated_map_stack[map_quantity]):.4e}")
            del integrated_map_stack[map_quantity]


def sum_maps(file_numbers, infile_format, outfile, map_names, required_groups=("InternalCodeUnits", "Units", "Shell", "__xrayInfo")):
    """
    Write a new .hdf5 file with the datasets being the sum total of the input files datasets.

    :param  file_numbers:       index number of the input files to sum 
    :type   file_numbers:       list
    :param  infile_format:      formatted path to the input files of a given file number: path/to/the/input/file_{file_nr}.hdf5
    :type   infile_format:      str
    :param  outfile:            path and name of the file that will be written
    :type   outfile:            str
    :param  map_names:          map names to sum together. If ['common'] use all maps that are found in every input file
    :type   map_names:          list
    :param  required_groups:    required groups with essential attributes for lightcone_io maps
    :type   required_groups:    tuple
    """
    # collect input filenames and make sure they exist
    infilenames = []
    for file_nr in file_numbers:
        if os.path.exists(infile_format.format(file_nr=file_nr)):
            infilenames.append(infile_format.format(file_nr=file_nr))
        else:
            raise Exception("input file not found")

    # gather map names
    if map_names[0]=="common":
        common_map_names = get_common_maps(infilenames)
    else:
        common_map_names=map_names

    print("Selected Maps:")
    for map_name in common_map_names:
        print(f"\t {map_name}")
    
    # create output file name
    output_filename = outfile
    print(f"\nwriting output:\t{output_filename}")
    with h5py.File(output_filename, "w") as outfile:

        running_totals = {}
        for map_name in common_map_names:
            running_totals[map_name]=0

        for file_idx, infile_name in enumerate(infilenames):
            print(f"[{file_idx+1}/{len(infilenames)}] Opening input: {infile_name}")
            infile = h5py.File(infile_name, "r")

            if file_idx==0:
                for group in required_groups:
                    if group in infile:
                        infile.copy(group, outfile)

            # Loop over maps
            for name in common_map_names:
                
                print("\tReading  %s for file %d" % (name, file_numbers[file_idx]))
                
                # Read in the full map
                input_dataset = infile[name]
                
                if "nside" in input_dataset.attrs: # do not sum map if it doesnt have nside as attribute
                    
                    nside = input_dataset.attrs["nside"][0] # nside of input map from its attributes

                    
                    if file_idx==0:
                        # copy the dataset over if inital file
                        infile.copy(name, outfile)
                        
                        # update totals 
                        running_totals[name]+=np.sum(input_dataset)
                        output_map_total=np.sum(outfile[name])
                    
                    else:
                        # sum maps if not inital file
 
                        # check maps are the same shape dataset
                        output_dataset = outfile[name]
                        assert np.shape(input_dataset) == np.shape(output_dataset)

                        # check maps being summed have matching attributes for nside and units
                        for k, v in input_dataset.attrs.items():
                            if k != "Started":

                                assert k in output_dataset.attrs # check that it does exist 
                                assert v == output_dataset.attrs[k]# check that attribute values are consistent

                        # start updating the output 
                        npix = input_dataset.shape[0]

                        # if high nside map then update by chuncks
                        if nside <= 4096:
                            chunck_size=int(npix)
                        else:
                            chunck_size = 65536
                        
                        for start in range(0, npix, chunck_size):
                            end = int(min(start+chunck_size, npix))

                            in_chunk  = input_dataset[start:end]
                            out_chunk = output_dataset[start:end]

                            output_dataset[start:end] = in_chunk + out_chunk # write over indexed values

                            running_totals[name] +=np.sum(in_chunk)

                        output_map_total = np.sum(output_dataset)
                    
                    print(f"Updated Map: {name}\n\tstored map total: {output_map_total:.3e}\n\trunning total: {running_totals[name]:.3e}\n\t{output_map_total/running_totals[name] * 100:.3f} %")

            print("\nFinished file %d\n" % (file_numbers[file_idx]))
            infile.close()
    
    return None

