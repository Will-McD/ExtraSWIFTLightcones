#!/bin/env python
"""
Build a full sky halo lightcone from SOAP catalogues of the COLIBRE L400_m7 Thermal simulation
(L0400N3008/Thermal) for redshifts 0.001 < z < 0.1 and then write a LightconeIO like halo lightcone .hdf5 file for a
singular snapshot.
"""

import argparse
from pathlib import Path
import numpy as np
import unyt
from mpi4py import MPI
from swiftlet.snapshot_lightcone import SnapshotAllSky, message, write_halo_lightcone_file
from swiftlet.simulation_suites import Simulation

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()


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
    if properties is not None:
        local_properties = {name: np.asarray(values)[keep] for name, values in properties.items()}

        all_halos, all_properties = comm.gather(local_halos), comm.gather(local_properties)
    else:
        all_halos = comm.gather(local_halos)

    if comm_rank != 0:
        return None, None

    halos_out = {name: unyt.uconcatenate([rank[name] for rank in all_halos]) for name in local_halos}

    if properties is not None:
        properties_out = {name: np.concatenate([rank[name] for rank in all_properties]) for name in local_properties}
    else:
        properties_out = None
    return halos_out, properties_out


def adjust_redshift_range(simulation, snap_nr, zmin, zmax):
    """Update the redshift range to always include the redshift range of the snapshot."""
    # get snapshot redshift range
    snap_redshift_range = simulation.snapshot_redshift_range(snap_nr)
    # adjust min
    if snap_redshift_range[0] < zmin:
        print(f"\tadjusting redshift range to fit snapshot, zmin: {zmin:.4f} -> {snap_redshift_range[0]:.4f}")
        zmin = snap_redshift_range[0]
    # adjust max
    if snap_redshift_range[1] > zmax:
        print(f"\tadjusting redshift range to fit snapshot, zmax: {zmax:.4f} -> {snap_redshift_range[1]:.4f}")
        zmax = snap_redshift_range[1]

    return zmin, zmax


if __name__ == "__main__":
    if comm_rank == 0:
        parser = argparse.ArgumentParser(
            description="Full-sky halo lightcone of COLIBRE, its halo mass function, and a "
            "halo lightcone file of the lowest redshift snapshot."
        )
        parser.add_argument("--box_res", default="L0400N3008", help="box size and resolution label (L400_m7)")
        parser.add_argument("--sim_name", default="Thermal", help="name of the simulation")
        parser.add_argument("--zmin", type=float, default=0.001, help="minimum redshift")
        parser.add_argument("--zmax", type=float, default=0.1, help="maximum redshift")
        parser.add_argument(
            "--snapshot_number",
            type=int,
            default=127,
            help="snapshot number to write as  a LightconeIO .hdf5 halo lightcone",
        )
        parser.add_argument(
            "--soap_format",
            default=None,
            help="SOAP catalogue filenames, using {snap_nr}. By default the simulation's, "
            "see :meth:swiftlet.simulation_suites.Simulation.soap_format",
        )
        parser.add_argument("--orientation_seed", type=int, default=0, help="seed of the box tile orientations")
        parser.add_argument(
            "--orientation_lock",
            default="cube",
            choices=["none", "cube", "sphere"],
            help="orientation lock of the box tiles",
        )
        parser.add_argument(
            "--output_dir", default="./example_outputs/colibre_halo_lightcones", help="output directory"
        )
        args = parser.parse_args()

        simulation = Simulation.from_label(args.box_res, args.sim_name)
        soap_format = args.soap_format or simulation.soap_format
        zmin, zmax = adjust_redshift_range(simulation, args.snapshot_number, args.zmin, args.zmax)
        redshift_range = (zmin, zmax)
    else:
        args = None
        simulation = None
        redshift_range = None
        soap_format = None

    args = comm.bcast(args)
    simulation = comm.bcast(simulation)
    redshift_range = tuple(comm.bcast(redshift_range))
    soap_format = comm.bcast(soap_format)

    if comm_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    comm.barrier()

    # The haloes placed into lightcone
    snap_all_sky = SnapshotAllSky(
        simulation=simulation,
        orientation_seed=args.orientation_seed,
        orientation_lock=args.orientation_lock,
        verbose=1 if comm_rank == 0 else 0,
    )

    # collect all haloes and place them into the lightcone
    halos = snap_all_sky.place_halos_in_shell(soap_format, redshift_range, ang_radius_deg=None, comm=comm)

    # write the haloes of the selected snapshot to a in a lightcone_io halo lightcone file
    local_snapshots = halos["Lightcone/SnapshotNumber"].value.astype(int)
    snapshot_halos, snapshot_properties = gather_snapshot_haloes(halos, None, args.snapshot_number)
    if comm_rank == 0:
        output_filename = f"{args.output_dir}/lightcone_halos_{args.snapshot_number:04d}.hdf5"
        write_halo_lightcone_file(
            output_filename,
            snapshot_halos,
            soap_format.format(snap_nr=args.snapshot_number),
            snapshot_properties,
            halos.metadata,
        )
        message(
            comm_rank,
            (
                f"{len(snapshot_halos['Lightcone/SnapshotNumber'])} haloes of snapshot"
                f"{args.snapshot_number} written to {output_filename}"
            ),
        )

    message(comm_rank, "\nDONE :)\n")
