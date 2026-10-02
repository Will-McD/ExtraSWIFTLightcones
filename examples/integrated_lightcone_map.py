#!/bin/env python
import numpy as np
from pathlib import Path
import hdfstream
from extra_swift_lightcones.healpix_map_utils import write_rotated_lightcone_chunks

# rotation angles (in radian) for flamingo simualtions with sidelength of 1000Mpc
ROTATIONS_L1=np.array(
    [
        [0., 0. ,3.26757547, 3.26757547, 3.26757547, 1.51289711, 1.51289711, 3.13885639, 3.13885639, 3.13885639, 2.17061318, 2.17061318, 2.17061318, 2.17061318, 4.59420579, 4.59420579, 4.59420579, 1.14273623, 1.14273623, 1.14273623, 1.14273623, 2.02717201, 2.02717201, 2.02717201, 2.02717201, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628], 
        [0., 0., 1.41518902, 1.41518902,  1.41518902, 0.80580058, 0.80580058, 0.71830831, 0.71830831, 0.71830831, 1.77536892, 1.77536892, 1.77536892, 1.77536892, 0.62434822, 0.62434822, 0.62434822, 2.14076603, 2.14076603, 2.14076603, 2.14076603, 0.49840908, 0.49840908, 0.49840908, 0.49840908, 2.0136344, 2.0136344, 2.0136344 , 2.0136344, 2.0136344,  2.25356928, 2.25356928, 2.25356928 , 2.25356928 , 2.25356928,  2.25356928, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078 , 1.85187078, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331]
    ]
)


"""
For the L1_M9 HYDRO_FIDCUIAL FLAMINGO simulation create an intergrated 'TotalMass' and æSmoothedGasMass' map over the redshift range 0.1-0.3. 
Write the intergrated maps a hdf5 file and store the individually rotated redshift chunks used to create the intergrated map. 
"""


if __name__ == "__main__":


    # simulation to use
    box_res='L1000N1800'
    simulation_name="HYDRO_FIDCUIAL"

    # define output directory
    output_dir="./example_outputs/{box_res}/{sim_name}/maps".format(box_res=box_res, sim_name=simulation_name)
    
    directory_path = Path(output_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    # map nside to read in and ouput
    input_nside=4096
    output_nside=1024

    # lightcone number to use
    lightcone_nr=0
    
    # optional, set maximum ell (smallest angular scale) for rotations
    # Intentionally made smaller for speed of exmaple
    ell_max=2*output_nside

    # make path to healpix lightcone maps
    # If the healpix maps are local then set root=None
    root = hdfstream.open("cosma", "/")

    # Location of the lightcone output relative to the directory we opened
    basedir="FLAMINGO/L1_m9/L1_m9/healpix_maps/nside_{nside}".format(nside=input_nside)

    # Specify which observer's lightcone to read
    basename="lightcone{lightcone_nr}".format(lightcone_nr=lightcone_nr)
    
    # output filename
    output_filename = output_dir+'/lightcone{lightcone_nr}.intergrated_{nside}.hdf5'.format(lightcone_nr=lightcone_nr, nside=output_nside)

    # get rotation angles in degree
    colatitiudes = ROTATIONS_L1[0, :]*180/np.pi
    longitudes = ROTATIONS_L1[1, :]*180/np.pi

    write_rotated_lightcone_chunks(
        basedir, basename, remote_dir=root,
        map_names=["TotalMass", "SmoothedGasMass"], # the maps to sum along the line of sight
        theta_arr_deg=colatitiudes, phi_arr_deg=longitudes,  # rotate shells by these latitudes and longitudes
        output_filename=output_filename,    # write file to this location
        output_nside=output_nside, input_nside=input_nside, # nside to store and rotate maps at
        shell_range=(1,5), # sum shells in this range
        ell_max=ell_max,  # smallest angular scale to consider when rotating the map, if None, then it is ignored.
        save_chunks=True,  # store the individual redshift chunks in the output file. 
        rotate_nside=output_nside, # forces the map to be downsampled before rotating, further speed up for example
        hdf5_dset_kwargs=None, # no additional compression parameters set for output. 
    )

