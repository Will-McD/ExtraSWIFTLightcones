#!/bin/env python
import sys
import os
import argparse
from pathlib import Path
import h5py
import numpy as np
import unyt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from swiftlet.example_plot_settings import plot_settings

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

"""
Compare the number density of dark matter particles from Snapshot lightcones against the mean number density
of the snapshots. 
Iterativley repeat this comparison for multiple different orientation seeds for both FLAMINGO and COLIBRE. 

    - For each orientation seed (seeds 0 to nr_seeds - 1 by default) use SnapshotAllSky with orientation_lock="cube". 
    - The dark matter particles are counted in bins of comoving distance from the observer. The bins are equal in volume.
    - Each seed re-orients the box tiles differently, so each is the same shell built from the same snapshots with a 
        different arrangement of the cells i.e. making lightcone redhisft shells for differet observers. 

"""

def seed_row_vals(seed, edges, counts, expected, mean_density):
    """
    Rows of the .txt file for one orientation seed, one per bin of comoving distance.

    Returns a list of {column name: value}.

    :param  seed:           orientation seed
    :type   seed:           int
    :param  edges:          edges of the bins [Mpc]
    :type   edges:          np.ndarray
    :param  counts:         number of particles in each bin of the lightcone built from the snapshots
    :type   counts:         np.ndarray
    :param  expected:       number of particles expected in each bin from the snapshot's mean number density
    :type   expected:       np.ndarray
    :param  mean_density:   mean number density [comoving Mpc^-3] of the snapshot
    :type   mean_density:   float
    """
    volume = 4. / 3. * np.pi * (edges[1:]**3 - edges[:-1]**3)
    row = [
        { 
        "orientation_seed": int(seed), 
        "r_min_cMpc": float(edges[b]), 
        "r_max_cMpc": float(edges[b + 1]),
        "volume_cMpc3": float(volume[b]), 
        "count_snapshot_lightcone": int(counts[b]),
        "count_snapshot_mean": float(expected[b]), 
        "n_snapshot_lightcone_cMpc3": float(counts[b] / volume[b]),
        "n_snapshot_mean_cMpc3": float(mean_density), 
        "n_over_n_mean": float(counts[b] / expected[b])
        }
        for b in range(len(counts))]
    return row

def read_density_file(filename):
    """
    Read the .txt file written by this example.
    Returns a tuple of (the settings of the comment line as {key: value}, the columns as a structured np.ndarray).
    :param  filename:   path of the .txt file
    :type   filename:   str
    """
    with open(filename) as f:
        comment = f.readline().lstrip("#").strip()
    settings = dict(item.strip().split("=", 1) for item in comment.split(";") if "=" in item)
    data = np.genfromtxt(filename, delimiter=",", names=True, skip_header=1, dtype=None, encoding=None)
    return settings, data

def plot_seed_densities(filenames, output_filename):
    """
    Create this examples plot. 
    
    Plot n / n_mean in each bin of comoving distance, one panel per file side by side. 

    :param  filenames:          paths of the .txt files, one per simulation
    :type   filenames:          list of str
    :param  output_filename:    path of the plot
    :type   output_filename:    str
    """
    fig, axs = plt.subplots(1, len(filenames), figsize=(4.6 * len(filenames), 3.8), squeeze=False)
    for ax, filename in zip(axs[0], filenames):
        if not os.path.exists(filename):
            print(f"cannot find: {filename}")
            continue
        settings, data = read_density_file(filename)
        seeds = np.unique(data["orientation_seed"])
        r_mid = None
        ratios = []
        for i, seed in enumerate(seeds):
            rows = data[data["orientation_seed"] == seed]
            r_mid = 0.5 * (rows["r_min_cMpc"] + rows["r_max_cMpc"])
            ratios.append(rows["n_over_n_mean"])
            ax.plot(r_mid, rows["n_over_n_mean"], color="#9a9a96", lw=0.7, alpha=0.8, label="individual realisations" if i == 0 else None)
        ratios = np.array(ratios)
        #median, scatter = np.mean(ratios, axis=0), ratios.std(axis=0) # median and standard deviation
        ratio_percentiles = np.percentile(ratios, [50-34.13, 50, 50+34.13], axis=0)
        median=ratio_percentiles[1]
        scatter = (ratio_percentiles[0], ratio_percentiles[2])
        expected = data[data["orientation_seed"] == seeds[0]]["count_snapshot_mean"]
        poisson = 1. / np.sqrt(expected)

        ax.plot(r_mid, median, color="darkcyan", lw=2, label=r"median of realisations, $(n_{\mathrm{DM},\,50})$", zorder=-100)
        ax.fill_between(r_mid, scatter[0], scatter[1], color="darkcyan", alpha=0.25, lw=0, label=r"$n_{\mathrm{DM},\,50} \pm \sigma$", zorder=-300)
        ax.axhline(1., color="lightgrey", lw=0.8, ls=":", zorder=-200)

        ax.set_title(f"{settings.get('simulation', Path(filename).stem)}\n"+f"{settings.get('zmin', '?')} < z <= {settings.get('zmax', '?')}, full sky"+f"\nmedian over all bins and seeds: {np.median(ratios):.5f}", fontsize=9)
        ax.set_xlabel(r"comoving distance, $\chi$ [Mpc]")
        ax.set_axisbelow(True)
    axs[0, 0].set_ylabel(r"$n_{\rm DM}$ / snapshot mean")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0,0].legend(handles, labels, loc="lower right", ncol=1, fontsize=8, frameon=False)
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()

def snapshot_mean_number_density(simulation, ptype="PartType1"):
    """
    Mean number density of a particle type in a whole snapshot. 
    Computed from the header of the lowest redshift snapshot.

    Returns the number density [comoving Mpc^-3] and the number of particles.

    :param  simulation: the simulation
    :type   simulation: Simulation
    :param  ptype:      particle type, e.g. "PartType1"
    :type   ptype:      str
    """
    type_nr = int(ptype.replace("PartType", ""))
    with h5py.File(simulation.snapshot_file(simulation.z0_snapshot, 0), "r") as f:
        header = f["Header"].attrs
        nr_part = int(header["NumPart_Total"][type_nr])
        if "NumPart_Total_HighWord" in header:
            nr_part += int(header["NumPart_Total_HighWord"][type_nr]) << 32
        length_cgs = float(f["Units"].attrs["Unit length in cgs (U_L)"][0])
        boxsize = unyt.unyt_array(np.asarray(header["BoxSize"], dtype=float) * length_cgs, "cm").to_value("Mpc")
    return nr_part / np.prod(np.broadcast_to(boxsize, (3,))), nr_part

def equal_volume_edges(r_min, r_max, nr_bins):
    """
    Edges of the comoving distance bins, that are all equal in volume. 
    Every bin expects the same number of particles.

    Returns the edges [Mpc], nr_bins + 1 values.

    :param  r_min:      inner comoving distance [Mpc]
    :type   r_min:      float
    :param  r_max:      outer comoving distance [Mpc]
    :type   r_max:      float
    :param  nr_bins:    number of bins
    :type   nr_bins:    int
    """
    return np.cbrt(np.linspace(r_min**3, r_max**3, nr_bins + 1))

def subshell_redshift_ranges(cosmo, redshift_range, nr_shells, n_grid=20001):
    """
    Split a redshift range into consecutive thinner shells of equal comoving volume, so each holds a similar number of
    particles. Neighbouring shells share their boundary redshift, so together they cover the range exactly once.

    Returns a list of nr_shells (z_min, z_max) tuples, from low to high redshift.

    :param  cosmo:          cosmology of the simulation (SnapshotCosmology)
    :type   cosmo:          SnapshotCosmology
    :param  redshift_range: minimum and maximum redshift [z_min, z_max]
    :type   redshift_range: tuple
    :param  nr_shells:      number of shells to split the range into
    :type   nr_shells:      int
    :param  n_grid:         number of redshifts of the grid the boundaries are interpolated from
    :type   n_grid:         int
    """
    z_min, z_max = redshift_range
    if nr_shells <= 1:
        return [(z_min, z_max)]
    z_grid = np.linspace(z_min, z_max, n_grid)
    r_grid = cosmo.comoving_distance(z_grid).to_value("Mpc")
    z_edges = np.interp(equal_volume_edges(r_grid[0], r_grid[-1], nr_shells), r_grid, z_grid)
    z_edges[0], z_edges[-1] = z_min, z_max  # exactly the range asked for
    return [(float(z_edges[i]), float(z_edges[i + 1])) for i in range(nr_shells)]

def share_pieces(npart_per_piece, nr_ranks):
    """
    Share the pieces between the ranks, giving each piece in turn, from the largest, to the rank with the fewest
    particles so far.

    Returns a list of the indices of the pieces given to each rank.

    :param  npart_per_piece:    number of particles read for each piece
    :type   npart_per_piece:    np.ndarray
    :param  nr_ranks:           number of ranks
    :type   nr_ranks:           int
    """
    pieces = [[] for _ in range(nr_ranks)]
    npart_per_rank = np.zeros(nr_ranks, dtype=np.int64)
    for piece in np.argsort(npart_per_piece, kind="stable")[::-1]:
        rank = int(np.argmin(npart_per_rank))
        pieces[rank].append(int(piece))
        npart_per_rank[rank] += npart_per_piece[piece]
    return pieces

def count_in_bins(coords, edges, vector=None, ang_radius_rad=None):
    """
    Count the particles in each bin of comoving distance, keeping those within an angle of a direction if given.

    Returns the counts, len(edges) - 1 values.

    :param  coords:         comoving positions relative to the observer [Mpc], shape (N, 3)
    :type   coords:         np.ndarray
    :param  edges:          edges of the bins [Mpc]
    :type   edges:          np.ndarray
    :param  vector:         unit vector of the cone's axis, or None for the full sky
    :type   vector:         np.ndarray
    :param  ang_radius_rad: angular radius [rad] of the cone
    :type   ang_radius_rad: float
    """

    r = np.linalg.norm(coords, axis=1)
    if vector is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            keep = coords @ vector >= np.cos(ang_radius_rad) * r
        r = r[keep]
    # only r in [edges[0], edges[-1]], the last bin including its outer edge
    return np.histogram(r, bins=edges)[0].astype(np.int64)

def count_snapshot_lightcone(lightcone, redshift_range, edges, vector, ang_radius_deg, ptype="PartType1"):
    """
    Count the particles of a lightcone in each bin. 
    Share the snapshot pieces between the ranks (see snapshot_dark_matter_map.py).

    Returns the counts summed over the ranks (on every rank), and the number of particles in the largest piece.

    :param  lightcone:      serial SnapshotBeam or SnapshotAllSky, one on each rank
    :type   lightcone:      SnapshotLightcone
    :param  redshift_range: minimum and maximum redshift [z_min, z_max]
    :type   redshift_range: tuple
    :param  edges:          edges of the bins [Mpc]
    :type   edges:          np.ndarray
    :param  vector:         unit vector of the beam, or None for the full sky
    :type   vector:         np.ndarray
    :param  ang_radius_deg: angular radius [deg] of the beam, or None for the full sky
    :type   ang_radius_deg: float
    :param  ptype:          particle type
    :type   ptype:          str
    """
    beam_args = {} if vector is None else {"ang_radius_deg": ang_radius_deg}
    _, pieces = lightcone.gather_files(ptype, redshift_range, **beam_args)
    npart_per_piece = np.array([int(np.sum(piece.lengths)) for piece in pieces], dtype=np.int64)
    my_pieces = share_pieces(npart_per_piece, comm_size)[comm_rank]
    largest_piece = int(npart_per_piece.max()) if len(npart_per_piece) else 0

    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for piece in my_pieces:
        particles = lightcone.place_file_in_shell(piece, ptype, ["Coordinates"], **beam_args)
        if len(particles) == 0:
            continue
        # place_file_in_shell keeps only the particles in the beam and the redshift range of their snapshot
        counts += count_in_bins(particles["Coordinates"].to_value("Mpc"), edges)
    return comm.allreduce(counts, op=MPI.SUM), largest_piece

def write_counts(filename, rows, header):
    """
    Write the counts to a comma separated .txt file.  

    :param  filename:   path of the file
    :type   filename:   str
    :param  rows:       rows, each a dict {column name: value}
    :type   rows:       list
    :param  header:     comment line describing the file, without commas
    :type   header:     str
    """
    columns = list(rows[0])
    with open(filename, "w") as f:
        f.write(f"# {header}\n")
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(f"{row[c]:.10g}" if isinstance(row[c], float) else str(row[c]) for c in columns) + "\n")

def run_seeds(args):
    """
    Build the full-sky lightcone for each orientation seed, count its dark matter particles in each bin and write
    the counts of every seed (on rank 0).

    :param  args:   command line arguments, see __main__
    :type   args:   argparse.Namespace
    """
    from mpi4py import MPI
    from swiftlet.snapshot_lightcone import SnapshotAllSky, message
    from swiftlet.simulation_suites import Simulation
    from swiftlet.snapshot_cosmology import SnapshotCosmology

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    redshift_range = (args.zmin, args.zmax)
    simulation = Simulation.from_label(args.box_res, args.sim_name)
    cosmo = SnapshotCosmology.from_simulation(simulation)
    r_min, r_max = (cosmo.comoving_distance(z).to_value("Mpc") for z in redshift_range)
    edges = equal_volume_edges(r_min, r_max, args.nr_bins)
    shell_volume = 4. / 3. * np.pi * (edges[1:]**3 - edges[:-1]**3)  # [Mpc^3], full sky

    mean_density, nr_part = snapshot_mean_number_density(simulation)
    expected = mean_density * shell_volume
    update_str=f"\n{args.box_res}/{args.sim_name}: {nr_part:.3e} DM particles\nmean numb density: {mean_density:.3e} cMpc^-3\n{np.sum(expected):.3e} expected in the full sky for {args.zmin} < z <= {args.zmax}"
    message(comm_rank, update_str, time_date_update=True)

    # thinner shells of equal volume, each placed on its own to limit the memory of each rank (see the docstring)
    shells = subshell_redshift_ranges(cosmo, redshift_range, args.nr_shells)
    if len(shells) > 1:
        shell_ranges = []
        for z0, z1 in shells:
            shell_ranges.append(f"{z0:.4f}-{z1:.4f}")
        update_str= f"{len(shells)} shells of equal volume, z = {', '.join(shell_ranges)}"
        message(comm_rank, update_str)

    seeds = list(range(args.first_seed, args.first_seed + args.nr_seeds))
    rows = []
    for seed in seeds:
        message(comm_rank, f"\n=== orientation seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}), lock {args.orientation_lock} === ", time_date_update=True)
        lightcone = SnapshotAllSky(simulation_object=simulation, orientation_seed=seed,
                                   orientation_lock=args.orientation_lock, verbose=0)
        
        # compute counts in thinner redshift shells when indicated 
        counts = np.zeros(args.nr_bins, dtype=np.int64)
        for shell_nr, shell_z in enumerate(shells):
            shell_counts, largest_piece = count_snapshot_lightcone(lightcone, shell_z, edges, None, None)
            counts += shell_counts
            if len(shells) > 1:
                update_str = f"\t[shell {shell_nr + 1}/{len(shells)}], {shell_z[0]:.4f} < z <= {shell_z[1]:.4f}:\t{np.sum(shell_counts):.3e} particles\t(largest piece: {largest_piece:.3e} particles)"
                message(comm_rank, update_str,time_date_update=True)
        message(comm_rank, f"seed {seed}: {np.sum(counts)} particles, n / n_mean = {np.sum(counts) / np.sum(expected):.6f}, bins from {np.min(counts / expected):.5f} to {np.max(counts / expected):.5f}")
        rows += seed_row_vals(seed, edges, counts, expected, mean_density)

    if comm_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        filename = (f"{args.output_dir}/dark_matter_number_density_seeds.{args.box_res}.{args.sim_name}"
                    f".z_{args.zmin:.3f}_{args.zmax:.3f}.txt")
        
        # the comment line holds the settings as key=value; read back by read_density_file
        write_counts(filename, rows, f"simulation={args.box_res}/{args.sim_name}; zmin={args.zmin:g}; " f"zmax={args.zmax:g}; orientation_lock={args.orientation_lock}; " f"nr_bins={args.nr_bins}; nr_shells={args.nr_shells}; n in cMpc^-3")
        message(comm_rank, f"\ncounts of {len(seeds)} seeds written to {filename}", time_date_update=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dark matter number density of full-sky lightcones built from the "
                                                 "snapshots, for many orientation seeds, against the snapshot mean.")
    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label")
    parser.add_argument("--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation")
    parser.add_argument("--zmin", type=float, default=0.0, help="minimum redshift")
    parser.add_argument("--zmax", type=float, default=0.15, help="maximum redshift")
    parser.add_argument("--nr_bins", type=int, default=25, help="number of bins of comoving distance, of equal volume")
    parser.add_argument("--nr_shells", type=int, default=1,
        help="number of thinner sub shells (of equal volume) to split the redshift range into, each is placed one at a time, to limit the peak memory of each rank")
    parser.add_argument("--nr_seeds", type=int, default=10, help="number of orientation seeds, i.e. number of iterations")
    parser.add_argument("--first_seed", type=int, default=0, help="first orientation seed")
    parser.add_argument("--orientation_lock", default="cube", choices=["none", "cube", "sphere"], help="orientation lock of the box tiles")
    parser.add_argument("--output_dir", default="./example_outputs/number_density_seeds", help="output directory")
    parser.add_argument("--plot_files", nargs="+", default=None, help="only plot these files written by this example, side by side")
    parser.add_argument("--plot_output", default=None, help="plot to write, by default next to the first file")
    args = parser.parse_args()
    
    plot_settings() 

    if args.plot_files:
        output = args.plot_output or str(Path(args.plot_files[0]).with_name("dark_matter_number_density_seeds.png"))
        plot_seed_densities(args.plot_files, output)
        print(f"number densities plotted to {output}")
    else:
        run_seeds(args)