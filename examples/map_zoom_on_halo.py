#!/bin/env python
import re
import numpy as np
import healpy as hp
import h5py
import unyt
import lightcone_io.healpix_maps as hm
import lightcone_io.halo_reader as hr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
import cmasher as cmr # noqa: F401 register "cmr.*" colour maps used
from lightcone_io.units import units_from_attributes
from swiftlet import swift_snapshot_redshift_conversion as nz
from pathlib import Path
import hdfstream

"""
Example script for reading and plotting only a section of pixels from a healpix map.
Here we will select a small region centred on a halo in the lightcone by reading a 
subset of pixels from the larger map, that are in a disk about the 
pixel the halos centre of mass is found in and then making an image 
using a Gnomonic projection. 


Note 1) This example uses Nside 16384 maps read from the remote comsa6 server, 
to reduce memory required change nside_output to 4096. 

Note 2) The centre of the zoomed image is on the halo lightcones "HaloCentre" values
and/or the centre of the pixel containing that HaloCentre (depending on if centre_vector is passed to the plotting function). 
The centre of the halo is not necesarily where it will appear to be brightest or most dense. 

Note 3) The selected halo can be split across multiple redshift shells. If the halo doesn't apprear 
in the loaded map you may have to change how the shell number used to select the HEALPix map if 
the halo is near the edge of a lightcones redshift shell. 
"""

def plot_settings():
    """
    matplotlib settings for this example. 
    """
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


def assign_shell_number(input_redshifts, redshift_filename, return_bounds=False):
    """
    Return an array of the healpix map lightcone shell number corresponding to the input redshifts. 

    :param  input_redshifts:    redshifts to assign a shell number to
    :type   input_redshifts:    np.ndarray
    :param  redshift_filename:  path/to/file/with/shell/redshift/ranges.txt
    :type   redshift_filename:  str
    :param  return_bounds:      If True, also return the minimum and maximum redshifts per shell
    :type   return_bounds:      boolean
    """
    input_redshifts = np.asarray(input_redshifts)
    redshifts = np.loadtxt(redshift_filename, delimiter=",")
    particle_shell_nr = np.zeros(len(input_redshifts), dtype=np.int16)
    particle_shell_nr += int(999) # if not updated then assigned a shell then return one that does not exist. 
    
    part_zmin=np.min(input_redshifts)
    part_zmax=np.max(input_redshifts)
    for shell_nr, zval in enumerate(redshifts):
        zmin, zmax= zval[0], zval[1]
        if zmax < part_zmin:
            continue
        elif zmin > part_zmax:
            continue
        else:
            shell_nr_mask = (input_redshifts>zmin) & (input_redshifts<=zmax)
            particle_shell_nr[shell_nr_mask] = int(shell_nr)
    if return_bounds is False:
        return particle_shell_nr.astype(np.int32)
    else:
         return particle_shell_nr.astype(np.int32), redshifts[np.min(particle_shell_nr.astype(np.int32)):np.max(particle_shell_nr.astype(np.int32))+1, :]



def read_selected_pixels(healpix_map, pixel_idx):
    """
    Read only the selected pixels of HEALPix map without reading the whole map.

    The pixels are read as runs of consecutive pixel indices, assuming ring order, grouped by the file of the map holding them. 
    
    Read remote maps with hdfstream in one request to avoid the server stopping access due to too many requests. 
    Local maps are read row by row of the selected pixels in the map. 

    Uses the file names, pixels per file and dataset name of the lightcone_io HealpixMap, as lightcone_io has
    no method to read several ranges of pixels in one request.

    Returns the values of the pixels, in the order of the sorted pixel indices.

    :param  healpix_map:    map to read from, e.g. a map of a lightcone_io.healpix_maps.Shell
    :type   healpix_map:    lightcone_io.healpix_maps.HealpixMap
    :param  pixel_idx:      indices of the pixels to read
    :type   pixel_idx:      np.ndarray
    """
    pixel_idx = np.sort(np.asarray(pixel_idx, dtype=int))
    healpix_map._set_metadata()
    pix_per_file, nr_files = healpix_map._pix_per_file, len(healpix_map._filenames)

    # get runs of consecutive pixels.  
    # To read lightcone_io maps, file n starts at pixel n * pix_per_file and the last file holds the rest of the pixels.
    runs_by_file = {}
    for run in np.split(pixel_idx, np.flatnonzero(np.diff(pixel_idx) != 1) + 1):
        start, stop = int(run[0]), int(run[-1]) + 1
        while start < stop:
            file_nr = min(start // pix_per_file, nr_files - 1)
            file_stop = stop if file_nr == nr_files - 1 else min(stop, (file_nr + 1) * pix_per_file)
            runs_by_file.setdefault(file_nr, []).append(np.s_[start - file_nr * pix_per_file:file_stop - file_nr * pix_per_file])
            start = file_stop

    values = []
    for file_nr in sorted(runs_by_file):
        with healpix_map.open_file(healpix_map._filenames[file_nr]) as infile:
            dataset = infile[healpix_map._map_name]
            if hasattr(dataset, "request_slices"):  # remote lightcone_io maps, all the runs in one request
                values.append(np.asarray(dataset.request_slices(runs_by_file[file_nr])))
            else:  # local maps
                values.append(np.concatenate([dataset[run] for run in runs_by_file[file_nr]]))
    return np.concatenate(values)


def fetch_zoom_pixels(filename, centre_pixel_idx, pixel_idx, nside, map_name, return_empty_map=False):
    """
    Retrieve selected pixels from a map to make a zoomed in 
    gnomview plot of a region on the sky.
    
    For large or high Nside maps to reduce peak memory return_empty_map=False. 
    
    Returns a tuple of (map of the selected pixels, value of the centre pixel).

    :param  filename:           path to hdf5 file of the map or lightcone_io healpix map shell object
    :type   filename:           str or lightcone_io.healpix_maps.Shell
    :param  centre_pixel_idx:   index of the pixel to centre the map on
    :type   centre_pixel_idx:   int
    :param  pixel_idx:          the indices of the pixels to include in the map
    :type   pixel_idx:          list or np.ndarray
    :param  nside:              the nside resolution of the map
    :type   nside:              int
    :param  map_name:           name of the map's dataset within the file
    :type   map_name:           str
    :param  return_empty_map:   If False, only return selected pixels. Otherwise return all-sky map with 
                                    all pixels not selected = 0. 
    :type   return_empty_map:   boolean
    """
    
    if isinstance(pixel_idx, list):
        pixel_idx = np.asarray(pixel_idx).astype(int)
    if isinstance(filename, str):# assume string implies that the map is hdf5 object 
        with h5py.File(filename, 'r') as f:
            # define map array and units
            map_units=units_from_attributes(f[map_name])
            if return_empty_map is True:
                xmap = unyt.unyt_array(np.zeros(hp.nside2npix(nside)), units=map_units)
            elif (return_empty_map==False) and (pixel_idx is not None):
                xmap = unyt.unyt_array(np.zeros(len(pixel_idx)), units=map_units)
            else:
                raise ValueError("no selected pixels and return_empty_map=False")
            
            if pixel_idx is None:
                xmap += f[map_name][:]
                centre_pix_val = f[map_name][centre_pixel_idx]*map_units
            else:
                pixel_idx=pixel_idx[np.argsort(pixel_idx)].astype(int) # order and ensure correct type
                if return_empty_map is True:
                    xmap[pixel_idx]+=f[map_name][pixel_idx]
                else:
                    # read only the selected pixels
                    xmap[:]+=f[map_name][pixel_idx]
                centre_pix_val = f[map_name][centre_pixel_idx]*map_units

    elif isinstance(filename, hm.Shell):#lightconeIO shell has been passed instead of string 
        # define map array and units
        if return_empty_map is True:
            xmap = unyt.unyt_array(np.zeros(hp.nside2npix(nside)), units=filename[map_name].units)
        elif (return_empty_map==False) and (pixel_idx is not None):
            xmap = unyt.unyt_array(np.zeros(len(pixel_idx)), units=filename[map_name].units)
        else:
            raise ValueError("no selected pixels and return_empty_map=False")

        # value of the centre pixel
        centre_pix_val = unyt.unyt_quantity( np.asarray(filename[map_name][centre_pixel_idx:centre_pixel_idx+1])[0], filename[map_name].units)

        if pixel_idx is None:
            xmap += filename[map_name][:]
        
        else:
            pixel_idx=np.asarray(pixel_idx[np.argsort(pixel_idx)]).astype(int) # order and ensure order, is np.array and int 
            
            # Now we gather the runs of consequtive pixels using read_selected_pixels instead of defining here. 
            #runs = np.split(pixel_idx, np.flatnonzero(np.diff(pixel_idx) != 1) + 1) # combine slices of pixels to reduce number of server requests
            selected_pix_vals = read_selected_pixels(filename[map_name], pixel_idx)
            
            if return_empty_map is True:
                #xmap[pixel_idx]+=np.concatenate([np.asarray(filename[map_name][run[0]:run[-1] + 1]) for run in runs])
                xmap[pixel_idx]+=selected_pix_vals
            else:
                # read only the selected pixels, in as few requests as possible
                #xmap[:]+=np.concatenate([np.asarray(filename[map_name][run[0]:run[-1] + 1]) for run in runs])
                xmap[:]+=selected_pix_vals

            # sanity check: are the selected pixels being read correctly? 
            # Compare selected pixels centre pix value against the centre pix value
            centre_in_selected = np.searchsorted(pixel_idx, centre_pixel_idx)
            if centre_in_selected < len(pixel_idx) and pixel_idx[centre_in_selected] == centre_pixel_idx:
                if not np.isclose(selected_pix_vals[centre_in_selected], centre_pix_val, rtol=1e-6, atol=0, equal_nan=True):
                    raise RuntimeError(f"{map_name}: centre pixel read with the selected pixels differs from reading it on its own ({centre_pix_val} vs {pixel_values[centre_in_selected]})")


    return xmap, centre_pix_val

def no_frac_latex(unit_obj):
    """
    Return a string for the latex expression for a unyt object for a single line plot axis label, 
    i.e. replace '\frac{}{}' with '/'.

    :param  unit_obj:   units to write as latex
    :type   unit_obj:   unyt.Unit
    """
    # latex expression from unyt object
    latex_expression = unit_obj.latex_repr
    # replace \frac{}{} with /
    one_line_str = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', latex_expression)
    return one_line_str


def find_haloes_in_shell(halo_format, soap_format, shell_redshift_range, centre_vector, radius, boxsize_resolution,
                        remote_dir=None, min_mass=1e13*unyt.Msun, extra_properties=("InputHalos/IsCentral", "BoundSubhalo/TotalMass")):
    """
    Find the haloes of the halo lightcone around a direction on the sky in a lightcone redshift shell.
    Haloes in one shell can come from mutiple halo lightcone files as each file contains haloes from only one snapshot. 

    Returns a dict of arrays, one entry per halo: 
        {   
            "HaloCentre" [cMpc],
            "Redshift", 
            "M200c" [Msun], 
            "R200c" [cMpc],
            "SnapshotNumber", 
            "vectors" (unit vectors of their directions), 
            "angular_radius" (of R200c),
            "offset" (from centre_vector) [radians], 
            **extra_properties...
        }

    :param  halo_format:            format string of the halo lightcone filenames (using {snap_nr})
    :type   halo_format:            str
    :param  soap_format:            format string of the matching SOAP catalogue filenames (using {snap_nr})
    :type   soap_format:            str
    :param  shell_redshift_range:   minimum and maximum redshift of the map's shell
    :type   shell_redshift_range:   sequence of two floats
    :param  centre_vector:          direction to search around, e.g. the centre of a zoom
    :type   centre_vector:          array-like, shape (3,)
    :param  radius:                 angular radius to search [radians]
    :type   radius:                 float
    :param  boxsize_resolution:     FLAMINGO box size and resolution label, e.g. "L1000N1800", for the snapshot redshifts
    :type   boxsize_resolution:     str
    :param  remote_dir:             remote directory of the files (hdfstream), or None for local files
    :type   remote_dir:             hdfstream.RemoteDirectory
    :param  min_mass:               smallest M200c of the haloes kept
    :type   min_mass:               unyt.unyt_quantity
    :param  extra_properties:       other halo properties to read, if the files have them
    :type   extra_properties:       sequence of str
    """
    centre_vector = np.asarray(centre_vector, dtype=float) / np.linalg.norm(centre_vector)
    z_min, z_max = shell_redshift_range

    # snapshots whose redshift ranges overlap the shell, and one on either side
    overlapping = []
    for snap_nr in range(0, 1000):
        try:
            snap_z_min, snap_z_max = nz.snapshot_redshift_range(snap_nr, boxsize_resolution)
        except (IndexError, KeyError, ValueError):
            break  # past the last snapshot
        last_snap_nr = snap_nr
        if snap_z_min < z_max and snap_z_max > z_min:
            overlapping.append(snap_nr)
    snapshots = range(max(min(overlapping) - 1, 0), min(max(overlapping) + 1, last_snap_nr) + 1)

    base_properties = ["Lightcone/HaloCentre", "Lightcone/Redshift", "SO/200_crit/TotalMass", "SO/200_crit/SORadius"]
    found = []
    for snap_nr in snapshots:
        try:
            halo_file = hr.HaloLightconeFile(filename=halo_format.format(snap_nr=snap_nr),
                                             soap_filename=soap_format.format(snap_nr=snap_nr), remote_dir=remote_dir)
        except Exception as error:  # e.g. no file for a snapshot beyond the last one
            print(f"\tno halo lightcone for snapshot {snap_nr}: {error}")
            continue
        available = [name for name in extra_properties
                     if name in halo_file._file or (halo_file._soap_file is not None and name in halo_file._soap_file)]
        haloes = halo_file.read_halos_in_radius(centre_vector, radius, base_properties + available)
        # read_halos_in_radius returns every halo in the index pixels it reads, so also cut on the angle
        redshift = haloes["Lightcone/Redshift"].value
        position = haloes["Lightcone/HaloCentre"].to_value("Mpc")
        cos_offset = position @ centre_vector / np.linalg.norm(position, axis=1)
        keep = ((redshift > z_min) & (redshift <= z_max) & (haloes["SO/200_crit/TotalMass"] >= min_mass)
                & (cos_offset >= np.cos(radius)))
        haloes = {name: values[keep] for name, values in haloes.items()}
        haloes["SnapshotNumber"] = np.full(np.count_nonzero(keep), snap_nr)
        found.append(haloes)
        print(f"\tsnapshot {snap_nr}: {np.count_nonzero(keep)} haloes in the shell within the search radius")

    if not found:
        raise FileNotFoundError("no halo lightcone files found for the snapshots covering the shell")

    def joined(name, units=None):
        # one array of a property from every file, in the given units
        return np.concatenate([np.asarray(haloes[name].to_value(units) if units else haloes[name]) for haloes in found])

    centre = joined("Lightcone/HaloCentre", "Mpc")
    result = {
        "HaloCentre": centre,
        "Redshift": joined("Lightcone/Redshift"),
        "M200c": joined("SO/200_crit/TotalMass", "Msun"),
        "R200c": joined("SO/200_crit/SORadius", "Mpc"),  # comoving
        "SnapshotNumber": joined("SnapshotNumber"),
    }
    for name in extra_properties:
        if all(name in haloes for haloes in found):
            # masses in Msun, anything else as stored
            is_mass = found[0][name].units.dimensions == unyt.dimensions.mass
            result[name] = joined(name, "Msun" if is_mass else None)
    distance = np.linalg.norm(centre, axis=1)
    result["vectors"] = centre / distance[:, None]
    result["angular_radius"] = result["R200c"] / distance
    result["offset"] = np.arccos(np.clip(result["vectors"] @ centre_vector, -1, 1))
    return result



def plot_zoom_on_pixel(filename, nside, centre_pix_idx, map_names, 
                        axes_idx=None, output_filename=None, 
                        r_npix=10,
                        f_pixels=None, show_plot=False, 
                        colormap="cubehelix", bad_colours="grey",
                        text_colour="white",
                        length_scale=1*unyt.arcmin,
                        scale_colour="white",
                        inclusive=True,
                        cmap_norm=None,
                        highlight_centre_pixel=None,
                        centre_vector=None,
                        ):
    """
    Make a gnomview plot of a disk centered on a given pixel. 

    :param  filename:               path to hdf5 file of the map or lightcone_io healpix map shell object
    :type   filename:               str or lightcone_io.healpix_maps.Shell
    :param  nside:                  the nside resolution of the map
    :type   nside:                  int
    :param  centre_pix_idx:         index of the pixel to centre the map on
    :type   centre_pix_idx:         int
    :param  map_names:              names of the maps to include in the plot
    :type   map_names:              list
    :param  axes_idx:               map names and the subplot's row and column indices
    :type   axes_idx:               dict
    :param  output_filename:        path to output plot
    :type   output_filename:        str
    :param  r_npix:                 radius of the disk in number of pixels
    :type   r_npix:                 int
    :param  f_pixels:               function to apply to selected pixels 
    :type   f_pixels:               function
    :param  show_plot:              If True, show the matplotlib plot object
    :type   show_plot:              boolean
    :param  colormap:               name of the colour map to use, or one per map
    :type   colormap:               str or list
    :param  bad_colours:            name of the colour to be assigned to bad value or missing pixels
    :type   bad_colours:            str
    :param  text_colour:            colour of the text on the image
    :type   text_colour:            str
    :param  length_scale:           the scale to be displayed on the image. If None, then no scale is included
    :type   length_scale:           unyt.unyt_quantity
    :param  scale_colour:           colour of the scale bar
    :type   scale_colour:           str
    :param  inclusive:              If True, when querying the map include pixels which overlap the search 
                                        radius, otherwise, if False, include pixels whose centres are in the search radius
    :type   inclusive:              boolean
    :param  cmap_norm:              matplotlib normalisation methods for each map. If None, assume 'log' for each map
    :type   cmap_norm:              list
    :param  highlight_centre_pixel: marker colour, shape, size (or scale) and linewidth used to highlight the 
                                        coords of the centre pixel
    :type   highlight_centre_pixel: tuple
    """
    

    plot_settings()

    # query the map to find all pixels within a radius of the centre pixel. 
    theta, phi = hp.pix2ang(nside, centre_pix_idx, lonlat=True) # lonlat=True => [degrees]
    if centre_vector is None:
        xyz = np.asarray(hp.pix2vec(nside, centre_pix_idx))
    else:
        xyz = np.asarray(centre_vector, dtype=float) / np.linalg.norm(centre_vector)
        if hp.vec2pix(nside, *xyz) != centre_pix_idx:
            raise ValueError("centre_vector is not in the pixel centre_pix_idx")
    theta, phi = hp.vec2ang(xyz, lonlat=True) # lonlat=True => longitude and latitude [degrees]
    theta, phi = float(theta[0]), float(phi[0])
    pix_sidelength = hp.nside2resol(nside, arcmin=True)*unyt.arcmin
    
    include_pix_idx=hp.query_disc(nside, xyz, r_npix*pix_sidelength.to_value(unyt.radian), inclusive=inclusive)
    
    if isinstance(colormap, str):
        colormap=[colormap]*len(map_names)

    # make dictionary to link maps to axes location
    if axes_idx is None:
        axes_idx={}
        n = len(map_names)
        if n==4:
            ncols=2
        elif n>4:
            ncols=3
        else:
            ncols=n

        nrows = int(np.ceil(n / ncols))
        for i, name in enumerate(map_names):
            row = int(i // ncols)
            col = int(i % ncols)
            axes_idx[name] = [row, col]
    else:
        nrows = 1
        ncols = 1
        for k, v in axes_idx.items():
            if v[0]+1 > nrows:
                nrows=v[0]+1
            if v[-1]+1 > ncols:
                ncols=v[-1]+1

    # determine figsize
    if nrows>=3:
        fig_ysize=7
    elif nrows==2:
        fig_ysize=5.5
    else:
        fig_ysize=3.21
    if ncols>=3:
        fig_xsize=7
    elif ncols==2:
        fig_xsize=5
    else:
        fig_xsize = 3.21
    
    fig = plt.figure(figsize=(fig_xsize, fig_ysize))
    plot_grid = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.02, hspace=0.02, width_ratios=[1]*ncols, height_ratios=[1]*nrows)
    axs=plot_grid.subplots(sharex=True, sharey=True, squeeze=False)

    if cmap_norm is None:
        cmap_norm = ['log']*len(map_names)
    
    for map_idx, map_name in enumerate(map_names):
        print(f"\tPlotting Zoom for {map_name} map")
        
        # collect pixel values
        xmap, centre_pix_val = fetch_zoom_pixels(filename, centre_pix_idx, include_pix_idx, nside, map_name, return_empty_map=False)
        selected_pix_idx = np.sort(include_pix_idx)

        def vec2index(x, y, z):
            # position of each pixel, or the last entry of values if it isn't a selected pixel
            pix = hp.vec2pix(nside, x, y, z)
            idx = np.clip(np.searchsorted(selected_pix_idx, pix), 0, len(selected_pix_idx) - 1)
            return np.where(selected_pix_idx[idx] == pix, idx, len(selected_pix_idx))
    
        # convert coordinate system 
        xmap.convert_to_base("galactic")
        centre_pix_val.convert_to_base("galactic")

        # update units for Intrinsic X-ray maps
        if xmap.units.dimensions == unyt.dimensions.energy/unyt.dimensions.time/unyt.dimensions.length**2:
            xmap.convert_to_units("erg/s/cm**2")
        elif xmap.units.dimensions == unyt.dimensions.time ** -1 * unyt.dimensions.length**-2:
            xmap.convert_to_units("photon/s/cm**2")
        if centre_pix_val.units.dimensions == unyt.dimensions.energy/unyt.dimensions.time/unyt.dimensions.length**2:
            centre_pix_val.convert_to_units("erg/s/cm**2")
        elif centre_pix_val.units.dimensions == unyt.dimensions.time ** -1 * unyt.dimensions.length**-2:
            centre_pix_val.convert_to_units("photon/s/cm**2")

        # update units for convolved X-ray maps
        if xmap.units.dimensions == unyt.dimensions.energy/unyt.dimensions.time:
            xmap.convert_to_units("erg/s")
        elif xmap.units.dimensions == unyt.dimensions.time ** -1:
            xmap.convert_to_units("photon/s")
        if centre_pix_val.units.dimensions == unyt.dimensions.energy/unyt.dimensions.time:
            centre_pix_val.convert_to_units("erg/s")
        elif centre_pix_val.units.dimensions == unyt.dimensions.time ** -1:
            centre_pix_val.convert_to_units("photon/s")
        
        # apply function to pixel values
        if f_pixels is not None:
            xmap=f_pixels(xmap)

        # define image resolution and coordinates
        img_res=hp.nside2resol(nside, arcmin=True) #[arcmin]
        img_xy=(2*r_npix+1, 2*r_npix+1)
        
        # make gnom projection of selected pixels 
        gnom_obj = hp.projector.GnomonicProj(rot=[theta,phi], xsize=img_xy[0], ysize=img_xy[1], reso=img_res)
        map_zoom = gnom_obj.projmap(np.append(np.asarray(xmap.value, dtype=np.float64), 0.), vec2index)

        axes_extent = [gnom_obj.get_extent()[0],
                       gnom_obj.get_extent()[1],
                       gnom_obj.get_extent()[2],
                       gnom_obj.get_extent()[3]]
        ax_ij = axes_idx[map_name]
        
        # select axis
        ax=axs[ax_ij[0], ax_ij[1]]

        # set colour map and bad colours in map
        cmap = mpl.colormaps.get_cmap(colormap[map_idx])
        cmap.set_bad(color=bad_colours)

        # create img
        if map_name == "DopplerB":
            cmap_norm[map_idx]="symlog"
            pixmin=np.percentile(map_zoom[map_zoom!=0], [1])[0]
            pixmax=np.percentile(map_zoom[map_zoom!=0], [99])[0]
            map_zoom[map_zoom==0]=np.nan
        else:
            pixmin=None
            pixmax=None
        map_zoom = np.ma.masked_invalid(map_zoom)
        norm = cmap_norm[map_idx]
        if norm == "log" and not np.any(map_zoom > 0):
            print(f"\t{map_name} has no positive values in the zoom, plotting it with a linear scale")
            norm = "linear"
            map_zoom[map_zoom==0]=np.nan

        if norm=="log" and (np.abs(np.log10(np.nanmax(map_zoom)) - np.log10(np.nanmin(map_zoom[map_zoom>0])))<2):
            print(np.log10(np.nanmax(map_zoom)), np.log10(np.nanmin(map_zoom[map_zoom>0])))
            norm = "linear"
            map_zoom[map_zoom==0]=np.nan

        img = ax.imshow(map_zoom, origin='lower', cmap=cmap, norm=norm, extent=axes_extent,vmin=pixmin, vmax=pixmax, interpolation="none")

        if highlight_centre_pixel is not None:
            gnom_x, gnom_y = gnom_obj.ang2xy(theta, phi,lonlat=True) #positions in the gnomietric plane
            ax.scatter(gnom_x, gnom_y, edgecolor=highlight_centre_pixel[0], facecolor="none", marker=highlight_centre_pixel[1], linewidth=highlight_centre_pixel[3], s=highlight_centre_pixel[2])

        # remove ticks
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        # add colour bar with maps units 
        if xmap.units==unyt.dimensionless:
            show_units="[dimensionless]"
        else:
            show_units=r"[${unit_label}$]".format(unit_label=no_frac_latex(xmap.units))

        if ax_ij[0]==0:
            cbar=fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.85, location='top',label=show_units,pad=0.01,)
        else:
            cbar=fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.85, location='bottom',label=show_units,pad=0.01,)
        cbar.ax.tick_params(labelsize=10) 

        # add scale 
        if length_scale is not None:

            ## need to add scale bar
            Lx=gnom_obj.get_extent()[1] - gnom_obj.get_extent()[0]
            dx=np.radians((length_scale).to_value(unyt.degree))
            #dx=np.radians(1)
            x0 = gnom_obj.get_extent()[0] + (Lx*0.1)
            x1=x0+dx

            y0=gnom_obj.get_extent()[2] + Lx*0.05
            y1=y0

            ax.plot([x0,x1], [y0,y1], linewidth=1., color=scale_colour, path_effects=[pe.withStroke(linewidth=1.3, foreground="black"), pe.Normal()])
            ax.text(
                x0+(0.5*(x1-x0)),
                (y0)+dx/10,
                f"{length_scale.to_value(length_scale.units):.1f} "+r"[${unit_label}$]".format(unit_label=length_scale.units.latex_representation()),
                ha='center',
                va='bottom',
                path_effects=[pe.withStroke(linewidth=1., foreground="black"), pe.Normal()],
                color=scale_colour,
            )
        
        # print maps name on img
        ax.text(
            0.5,
            0.975,
            map_name,
            ha='center',
            va='top',
            color=text_colour,
            path_effects=[pe.withStroke(linewidth=1., foreground="Black"), pe.Normal()],
            transform = ax.transAxes, fontsize=10 if len(map_name)<30 else 8, 
        )
        
    # save img
    plt.savefig(f"{output_filename}", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":

    # Use lightcone0 of L1_m9 fiducial model 
    boxsize_resolution="L1000N1800"
    sim_name="HYDRO_FIDUCIAL"

    # Find a halo on the sky at low redshift
    lightcone_nr=0
    shell_nr = 1

    # output nside
    output_nside=16384

    # define output directory
    output_dir="./example_outputs/zoom_on_pix_{nside}".format(nside=output_nside)
    
    # ensure output directory exists
    directory_path = Path(output_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    # output filename
    output_filename = output_dir+'/lightcone{lightcone_nr}.shell_{shell_nr}.png'.format(lightcone_nr=lightcone_nr, shell_nr=shell_nr)

    # make path to healpix lightcone maps, if the healpix maps and catalogues are local then set root=None
    root = hdfstream.open("cosma", "/")

    # Location of the lightcone output relative to the directory we opened
    basedir="FLAMINGO/L1_m9/L1_m9"

    # Specify which observer's lightcone to read
    basename="lightcone{lightcone_nr}".format(lightcone_nr=lightcone_nr)

    # redshift range of selected shell
    shell_redshift_range = np.loadtxt(nz.flamingo_shell_redshift_file('L1'), delimiter=",")[shell_nr]

    # halo lightcone and SOAP catalogue of each snapshot. 
    halo_format = basedir+"/halo_lightcone/lightcone{lc_nr}/lightcone_halos_{{snap_nr:04d}}.hdf5".format(lc_nr=lightcone_nr)
    soap_format = basedir+"/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"

    # Line of sight vector specifying a point on the sky
    vector = (1.0, 2.0, 3.0)
    
    # Angular radius (radians) around the unit vector that we will search within
    radius = np.radians(30.0)

    # Read the haloes in the shell around this point, from every halo lightcone file covering the shell
    print(f"\nHaloes within {np.degrees(radius):.0f} deg of {vector} in shell {shell_nr} ({shell_redshift_range[0]:.3f} < z <= {shell_redshift_range[1]:.3f}):")
    
    candidates = find_haloes_in_shell(halo_format, soap_format, shell_redshift_range, vector, radius, boxsize_resolution,remote_dir=root, min_mass=1e12*unyt.Msun, extra_properties=())


    ## select a halo with M200c close to 5 x 10^14 Msun
    target_log_M200c = np.log10(5e14)
    
    selected_halo_idx = np.argmin(np.abs(target_log_M200c - np.log10(candidates["M200c"])))
    halo_centre = candidates["HaloCentre"][selected_halo_idx]

    info_str="\nM200c:\t\t{m200:.4e} Msun\nRedshift:\t{z}\nShell numb:\t{shell_numb:d}\nSnapshot:\t{snap:d}\n".format(
        m200=candidates["M200c"][selected_halo_idx], 
        z=candidates["Redshift"][selected_halo_idx], 
        shell_numb=shell_nr,
        snap=int(candidates["SnapshotNumber"][selected_halo_idx])
    )
    print(info_str)
    
    # Find pixel in Nside 4096 healpix maps associated with tracer particle for the selcted halo 
    ipix = hp.vec2pix(output_nside, *halo_centre)

    # Plot a a disk with radius of 100 pixels, centred the selected pixel.
    r_pixels = 128
    zoom_radius = r_pixels * hp.nside2resol(output_nside)  # [radians]

    # Show all X-ray bands for intrinsic observations 
    map_names=[
        'XrayErositaLowIntrinsicPhotons_Recomp', 
        'DopplerB',
        'ComptonY',
        'DM',
        'SmoothedGasMass', 
        'DarkMatterMass',
    ]
    colour_maps=["cubehelix", "twilight", "cmr.ember" ,"cmr.eclipse", "cmr.lilac", "cmr.cosmic"]

    # Open the lightcone shell of healpix maps
    shell_arr = hm.Shell(basedir+"/healpix_maps/nside_{nside}".format(nside=output_nside), "lightcone{lc_nr}".format(lc_nr=lightcone_nr), shell_nr=shell_nr, remote_dir=root)
    
    plot_zoom_on_pixel(shell_arr, output_nside, ipix, map_names, 
                            axes_idx=None, output_filename=output_filename, r_npix=r_pixels, f_pixels=None, show_plot=False, 
                            colormap=colour_maps, bad_colours="grey",
                            length_scale=10*unyt.arcmin,
                            highlight_centre_pixel=("cyan", "o", 30, 1.), # highlight the centre pixel with a cyan ring. 
                            centre_vector=halo_centre, # force the centre of zoom is pointed to the centre of the halo instead of pixels centre
                            )
    

