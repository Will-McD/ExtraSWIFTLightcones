#!/bin/env python
import argparse
import numpy as np
from extra_swift_lightcones.mask_haloes import write_binary_masks

# rotation angles (in radian) for flamingo simualtions with sidelength of 1000Mpc
ROTATIONS_L1=np.array(
    [
        [0., 0. ,3.26757547, 3.26757547, 3.26757547, 1.51289711, 1.51289711, 3.13885639, 3.13885639, 3.13885639, 2.17061318, 2.17061318, 2.17061318, 2.17061318, 4.59420579, 4.59420579, 4.59420579, 1.14273623, 1.14273623, 1.14273623, 1.14273623, 2.02717201, 2.02717201, 2.02717201, 2.02717201, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628], 
        [0., 0., 1.41518902, 1.41518902,  1.41518902, 0.80580058, 0.80580058, 0.71830831, 0.71830831, 0.71830831, 1.77536892, 1.77536892, 1.77536892, 1.77536892, 0.62434822, 0.62434822, 0.62434822, 2.14076603, 2.14076603, 2.14076603, 2.14076603, 0.49840908, 0.49840908, 0.49840908, 0.49840908, 2.0136344, 2.0136344, 2.0136344 , 2.0136344, 2.0136344,  2.25356928, 2.25356928, 2.25356928 , 2.25356928 , 2.25356928,  2.25356928, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078 , 1.85187078, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331]
    ]
)

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


if __name__ == "__main__":
    
    if comm_rank==0:
        argparser = argparse.ArgumentParser()
        
        argparser.add_argument('halo_lightcone_dir', type=str, help='base directory to halo lightcones')
        argparser.add_argument('soap_dir', type=str, help='base directory to soap halo catalogues')
        
        argparser.add_argument('mask_filename', type=str, help='mask filename')
        argparser.add_argument('output_dir', type=str, help='base directory to write masks .hdf5 files too')

        argparser.add_argument('lightcone_nr', type=int, help='observer number of the lightcone')

        argparser.add_argument('--nside', type=int, default=2048, help='Healpix map Nside if the output binary mask')
        argparser.add_argument('--nshell', type=int, default=59, help='number of lightcone shells to create a binary mask for')
        argparser.add_argument('--bin_method', type=str, default="both", choices=["discrete", "cumulative", "both"],
                                help='which mass-bin definition(s) to build masks for')

        args = argparser.parse_args()
        
        args.lightcone_nr=np.asarray([args.lightcone_nr])

        # formated paths to soap and halo lightcone files
        halo_cat_format = args.halo_lightcone_dir + "/lightcone{lightcone_nr}/lightcone_halos_{snap_nr:04d}.hdf5"
        soap_cat_format = args.soap_dir + "/halo_properties_{snap_nr:04d}.hdf5"

    else:
        args=None
        halo_cat_format=None
        soap_cat_format=None
    
    args=comm.bcast(args)
    halo_cat_format=comm.bcast(halo_cat_format)
    soap_cat_format=comm.bcast(soap_cat_format)

    # get rotation angles in degree
    colatitiudes = ROTATIONS_L1[0, :]*180/np.pi
    longitudes = ROTATIONS_L1[1, :]*180/np.pi

    # mask M500c and use apatures of R500c
    mass_poperty="SO/500_crit/TotalMass"
    radius_poperty="SO/500_crit/SORadius"
    mass_units="Msun"

    # create masks using the same methods as make smoothed maps
    write_binary_masks(
        halo_cat_format, soap_cat_format, 
        mass_poperty, radius_poperty, 
        nshell=args.nshell, nside=args.nside, 
        latitudes=colatitiudes, longitudes=longitudes,
        lightcone_numbers=args.lightcone_nr, mask_filename=args.mask_filename, output_dir=args.output_dir,
        comm=comm, 
        bin_method=args.bin_method, mass_units=mass_units, 
        log_mass_bin_range=None, log_mass_bin_width=None, scale_radius=None,
        hdf5_dset_kwargs=None
        )
