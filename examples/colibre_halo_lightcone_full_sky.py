#!/bin/env python
import argparse
from pathlib import Path

import numpy as np
import unyt
from mpi4py import MPI

from swiftlet.snapshot_lightcone import SnapshotAllSky, message, write_halo_lightcone_file
from swiftlet.simulation_suites import Simulation
from swiftlet.soap_reader import soap_unit_attributes, comoving
#from colibre_halo_lightcone_hmf import place_and_count_haloes, CENTRAL_PROPERTY

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()


"""
Build a full-sky halo lightcone from the snapshots and SOAP catalogues of the COLIBRE L400_m7 Thermal simulation
(L0400N3008/Thermal) for 0.001 < z < 0.1, with SnapshotAllSky.

    1. Place haloes from the SOAP catalogues into a lightcone with place_halos_in_shell.
    2. Write the haloes from each snapshot to an output .hdf5 file (lightcone_halos_<snapshot>.hdf5) in the format of
        the lightcone_io (FLAMINGO) halo lightcones
"""

def gather_snapshot_haloes(halos, properties, snap_nr):
    """
    Gather the haloes placed from one snapshot, and their SOAP properties, from every rank onto rank 0.

    Returns a tuple of ({name: values} of the haloes, {name: values} of their properties) on rank 0, (None, None)
    on the other ranks.

    :param  halos:      haloes placed by place_halos_in_shell, on this rank
    :type   halos:      LightconeOutput
    :param  properties: SOAP properties of each halo, see colibre_halo_lightcone_hmf.read_halo_properties
    :type   properties: dict
    :param  snap_nr:    snapshot number
    :type   snap_nr:    int
    """
    keep = halos["Lightcone/SnapshotNumber"].value.astype(int) == snap_nr
    local_halos = {name: values[keep] for name, values in halos.items()}
    local_properties = {name: np.asarray(values)[keep] for name, values in properties.items()}
    all_halos, all_properties = comm.gather(local_halos), comm.gather(local_properties)
    if comm_rank != 0:
        return None, None
    halos_out = {name: unyt.uconcatenate([rank[name] for rank in all_halos]) for name in local_halos}
    properties_out = {name: np.concatenate([rank[name] for rank in all_properties]) for name in local_properties}
    return halos_out, properties_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Full-sky halo lightcone of COLIBRE, its halo mass function, and a "
                                                 "halo lightcone file of the lowest redshift snapshot.")
    parser.add_argument("--box_res", default="L0400N3008", help="box size and resolution label (L400_m7)")
    parser.add_argument("--sim_name", default="Thermal", help="name of the simulation")
    parser.add_argument("--zmin", type=float, default=0.001, help="minimum redshift")
    parser.add_argument("--zmax", type=float, default=0.1, help="maximum redshift")
    parser.add_argument("--soap_format", default=None,help="SOAP catalogue filenames, using {snap_nr}. By default <simulation>/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5")
    parser.add_argument("--orientation_seed", type=int, default=0, help="seed of the box tile orientations")
    parser.add_argument("--orientation_lock", default="none", choices=["none", "cube", "sphere"],
                        help="orientation lock of the box tiles")
    parser.add_argument("--output_dir", default="./example_outputs/colibre_halo_lightcones", help="output directory")
    args = parser.parse_args()

    redshift_range = (args.zmin, args.zmax)
    simulation = Simulation.from_label(args.box_res, args.sim_name)
    soap_format = args.soap_format or f"{simulation.directory}/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"
    if comm_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    comm.barrier()

    # The haloes placed into lightcone
    snap_all_sky = SnapshotAllSky(simulation_object=simulation, orientation_seed=args.orientation_seed,
                        orientation_lock=args.orientation_lock, verbose=1 if comm_rank == 0 else 0)
    
    halos = snap_all_sky.place_halos_in_shell(soap_format, redshift_range, ang_radius_deg=None, comm=comm)
    

    # write the haloes of the lowest redshift snapshot to a in a lightcone_io halo lightcone file
    local_snapshots = halos["Lightcone/SnapshotNumber"].value.astype(int)
    lowest_z_snapshot = comm.allreduce(int(local_snapshots.max()) if len(local_snapshots) else -1, op=MPI.MAX)
    snapshot_halos, snapshot_properties = gather_snapshot_haloes(halos, {CENTRAL_PROPERTY: properties[CENTRAL_PROPERTY]},
                                                                 lowest_z_snapshot)
    if comm_rank == 0:
        output_filename = f"{args.output_dir}/lightcone_halos_{lowest_z_snapshot:04d}.hdf5"
        write_halo_lightcone_file(output_filename, snapshot_halos, soap_format.format(snap_nr=lowest_z_snapshot),
                                  snapshot_properties, halos.metadata)
    message(comm_rank, f"{len(snapshot_halos['Lightcone/SnapshotNumber'])} haloes of snapshot {lowest_z_snapshot} written to {output_filename}")

    message(comm_rank, "\nDONE :)\n")