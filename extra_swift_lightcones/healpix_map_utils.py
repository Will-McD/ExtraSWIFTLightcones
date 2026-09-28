#!/bin/env python
import os
import sys
import numpy as np
import healpy as hp
import h5py
import unyt
import lightcone_io.healpix_maps as hm
import math

"""
Utility functions for working with healpix maps. 
"""

def get_related_ipix(ipix, nside, levels, ordering="ring"):

    """
    Return parent or child HEALPix pixel indices.

    Params
        ipix:   Pixel indices in the input map.
        nside:  Nside of the input map.
        levels: Change in HEALPix resolution. Each level increases or decreases nside by a factor of 2.
            levels > 0:
                Return child pixels at higher resolution.
            levels < 0:
                Return parent pixels at lower resolution.
            levels == 0:
                Return the input pixels unchanged.

        ordering : HEALPix pixel ordering, either 'ring' or 'nested'.

    Returns:
        ipix_new:   Pixel indices at the new resolution.
        nside_new:  Nside of the new map.
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
    Returns the first directory that contains a full_weights/ folder.
    Find the healpy pixel weights directory, checking in order:
      1. the datapath argument
      2. the HEALPY_PIXEL_WEIGHTS environment variable
      3. <venv>/share/healpy-data
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
    Returns the datapath to pass to hp.map2alm.

    Check that the healpy pixel weights file for this nside can be found.
    Looks for <datapath>/full_weights/healpix_full_weights_nside_XXXX.fits
    """
    datapath = find_healpy_pixel_weights_dir(datapath)

    weights_file = os.path.join(datapath, "full_weights", f"healpix_full_weights_nside_{nside:04d}.fits")
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"No healpy pixel weights for nside={nside}: expected {weights_file}")

    return datapath


def healpy_pixel_weights_available(nside=None, datapath=None):
    """
    Return True if healpy pixel weights can be found, otherwise False.
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
    For a list of .hdf5 files, return a list of all common dataset names. 
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
        Read theta and phi (latitiude and longitude) angles from hdf5 file. 
    """
    with h5py.File(filename, 'r') as rotation_data:
        rot_val=rotation_data['shells']
        longitude  = (rot_val['phi'][:]*180/np.pi * unyt.deg).astype(np.float64)
        latitude = (rot_val['theta'][:]*180/np.pi * unyt.deg).astype(np.float64)
    
    return latitude, longitude


def rotate_map(base_map, theta, phi, ell_max=None, map_rotator_object=None):
    """
        Rotate given map using alm space via the co-latitude and longitude passed as theta and phi
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
        Rotate given map using alm space via the co-latitude and longitude passed as theta and phi
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


def write_rotated_lightcone_chunks(base_dir, basename, theta_arr_deg, phi_arr_deg, output_format, output_nside, input_nside, shell_range=None, unit_conversion_func=None, rotate_nside=None, ell_max=None, save_chunks=False):
    """
        For a given set of rotation angles per shell, sum all shells along the line of sight that share the same rotation angles. 
        
        Params:
            theta_arr_deg:          array of theta (co-latitude) values per shell [deg]
            phi_arr_deg:            array of phi (longitude) values per shell [deg]
            shell_range:            tuple containing the minium and maximum shells to consider
            unit_conversion_func:   Optional function for additional unit conversion prior to rotating maps.
            save_chunks:            Boolean, if true write each individual chunk to the output file. 
    """

    from lightcone_io.downsample_maps import get_power
    def write_dataset(outfile, name, data):
        """
        writes dataset with gzip=9 and shuffle=True.
        """
        if name in outfile:
            del outfile[name]
        outfile.create_dataset(
            name,
            data=data,
            dtype=np.float64,
            compression="gzip",
            compression_opts=9,
            shuffle=True
        )
    
    # Open the lightcone map shell array
    shell_arr = hm.ShellArray(basedir, basename)

    if nr_shells is None:
        numb_shells=shell_arr.nr_shells
        shell_numbers = np.arange(0, numb_shells+1)
    else:
        shell_numbers = np.arange(shell_range[0], shell_range[1]+1)
        numb_shells=len(shell_numbers)

    # update rotation angles 
    theta_arr_deg = apply_expected_units(theta_arr_deg, "degree")[shell_numbers]
    phi_arr_deg = apply_expected_units(phi_arr_deg, "degree")[shell_numbers]
    
    theta_unique, indices_theta = np.unique(theta_arr_deg, return_index = True)
    phi_unique, indices_phi = np.unique(phi_arr_deg, return_index = True)
    chunk_num = np.hstack([indices_theta[np.argsort(indices_theta)], numb_shells])
    theta_rot = theta_unique[np.argsort(indices_theta)]
    phi_rot = phi_unique[np.argsort(indices_phi)]
    nrot = len(theta_rot)

    out_npix=hp.nside2npix(output_nside)
    if input_nside==16384:
        input_nside=8192
    in_npix=hp.nside2npix(input_nside)

    if rotate_nside is None:
        rotate_nside=input_nside


    with h5py.File(output_filename, 'w') as outfile:
        
        # runnning total
        integrated_map_stack = {str(map_quantity):np.zeros(out_npix, dtype=float) for map_quantity in map_names}

        for chunk_nr in range(0, nrot):
            
            arg_min, arg_max = int(chunk_num[chunk_nr]), int(chunk_num[chunk_nr+1])

            # runnning chunk total, reset once per rotation group
            chunk_map_stack = {str(map_quantity):np.zeros(in_npix, dtype=float) for map_quantity in map_names}

            # give update
            print(f"{chunk_nr+1}/{nrot}:", (arg_min, arg_max), flush=True)

            # combine shells, rotate and then downsample
            if save_chunks:
                chunk_group = outfile.create_group(f"chunk_{chunk_nr}")

            for shell_nr in range(arg_min, arg_max):
            
                shell = shell_arr[shell_nr]   # access shell level data


                for k, map_quantity in enumerate(map_names):
                    #check map is available in shell
                    if map_quantity not in shell.map_names:
                        raise ValueError(f"{map_quantity} not found in shell\nAvailable maps: {shell.map_names}")
                    downsample_power = get_power(map_quantity)
                    map_units_str = shell[map_quantity].units.to_string()
                    input_nside
                    if unit_conversion_func is None:
                        data = shell[map_quantity][...].to_value(map_units).astype(shell[map_quantity].dtype)
                    else:
                        data = unit_conversion_func(shell[map_quantity][...]).astype(shell[map_quantity].dtype)
                    
                    if np.npix2nside(len(data))==16384:
                        data=hp.ud_grade(data, nside_out=rotate_nside, power=downsample_power)
                    
                    map_chunk[map_quantity] += data

            for map_quantity in map_names:
                if chunk_nr == 0:
                    map_rot = map_chunk[map_quantity]
                else:
                    print(f"\trotating chunk {chunk_nr}, map: {map_quantity}\n\t\tcurrent theta, phi: {theta_rot[chunk_nr]}, {phi_rot[chunk_nr]}")
                    #check for pixel weights
                    if healpy_pixel_weights_available(nside=rotate_nside):
                        map_rot, __ = rotate_map_fast(map_chunk[map_quantity], theta_rot[chunk_nr], phi_rot[chunk_nr], nside=output_nside, ell_max=ell_max, map_rotator_object=None)
                    else:
                        map_rot, __ = rotate_map(map_chunk[map_quantity], theta_rot[chunk_nr], phi_rot[chunk_nr], ell_max=ell_max, map_rotator_object=None)
                
                if rotate_nside != output_nside:
                    print(f"\tdownsampling {map_quantity} from nside {rotate_nside} -> {output_nside}")
                    map_rot = hp.ud_grade(map_rot, nside_out=output_nside, power=get_power(map_quantity))
                
                # make a final write to the output file:
                if save_chunks:
                    print(f"\twriting lightcone chunk to file...")
                    write_dataset(chunk_group, map_quantity, map_rot)
                
                integrated_map_stack[map_quantity] += map_rot
                print(f"\tupdated running total: \t{np.sum(integrated_map_stack[map_quantity]):.4e}\n", flush=True)
                

        for map_quantity in map_names:
            write_dataset(outfile, map_quantity, integrated_map_stack[map_quantity])
            print(f"{map_quantity} final total:\t{np.sum(integrated_map_stack[map_quantity]):.4e}", flush=True)
            del integrated_map_stack[map_quantity]


def sum_maps(file_numbers, infile_format, outfile, map_names, required_groups=("InternalCodeUnits", "Units", "Shell", "__xrayInfo")):
    """
    Write a new .hdf5 file with the datasets being the sum total of the input files datasets

    file_numbers:       The index number of the input files to sum 
    infile_format:      Formated path to the input files of a given file number: path/to/the/input/file_{file_nr}.hdf5
    outfile:            The path and name of the file that will be written
    map_names:          List of map names to sum together. If ['common'] use all maps that are found in every input file
    required_groups:    Required groups with essential attributes for lightcone_io maps
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

