#!/bin/env python
#
import argparse
import tempfile
import warnings
from pathlib import Path
import numpy as np
from swiftlet.simulation_suites import Simulation
from swiftlet.snapshot_cosmology import SnapshotCosmology
from swiftlet.snapshot_lightcone import SnapshotAllSky, SnapshotBeam
from swiftlet.repeat_check import find_close_cell_repeats, CloseRepeatWarning
from swiftlet.example_plot_settings import plot_settings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
Check a lightcone for close repeats before building it, compare a given table for different orientation locks and seeds, 
then make a summary figure.

Every tile of a lightcone is a re-oriented copy of a snapshot box, so the same structures can re-apear multiple time 
along a line of sight when the radius of the lightcone surpases the size the one snapshot box. 
When tiles of differeing orientations are placed next to each other, two copies of the same cell can end up touching, 
or in nearly the same direction from the observer and closer than a box length apart. 

"""

# SET GLOBAL VARIABLES FOR LOCKS

LOCKS = [None, "cube", "sphere"] # selection of orientation locks to cycle through 
LOCK_COLOURS = {None: "darkmagenta", "cube": "darkcyan", "sphere": "darkgoldenrod"} # colours to display each lock with


def fake_simulation(run_dir):
    """
    Write a fake simulations snapshots with all particles in a regular lattice. 

    Returns the Simulation.

    :param  run_dir:    directory to write the simulation to
    :type   run_dir:    str
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/diagrams"))
    import snapshot_lattice_diagram 
    simulation, _ = snapshot_lattice.write_lattice_simulation(run_dir)
    return simulation


def new_lightcone(simulation, beam_vector, orientation_lock, orientation_seed, orientation_table=None):
    """
    Return the SnapshotLightcone object,
        SnapshotBeam if beam_vector is given
        SnapshotAllSky if beam_vector is None

    :param  simulation:         the simulation
    :type   simulation:         Simulation
    :param  beam_vector:        direction of the beam, or None for all-sky
    :type   beam_vector:        tuple
    :param  orientation_lock:   orientation lock of the box tiles: None, "cube" or "sphere"
    :type   orientation_lock:   str
    :param  orientation_seed:   seed of the box tile orientations
    :type   orientation_seed:   int
    :param  orientation_table:  file of orientation tables, or None for the default tables
    :type   orientation_table:  str
    """
    settings = dict(simulation_object=simulation, orientation_lock=orientation_lock, orientation_seed=orientation_seed, orientation_table=orientation_table, verbose=0)
    if beam_vector is None:
        return SnapshotAllSky(**settings)
    return SnapshotBeam(beam_vector=beam_vector, **settings)


def check_cases(simulation, redshift_range, seeds, beam_vector, ang_radius_deg, angle_deg, output_dir, orientation_table=None):
    """
    Check the lightcone for close repeats with every orientation lock and seed, 
    print a line for each combination and write each report to an HDF5 file.

    Returns {(lock, seed): CloseRepeatReport}.

    :param  simulation:         the simulation
    :type   simulation:         Simulation
    :param  redshift_range:     minimum and maximum redshift of the shell
    :type   redshift_range:     tuple
    :param  seeds:              orientation seeds to check
    :type   seeds:              list of int
    :param  beam_vector:        direction of the beam, or None for all-sky
    :type   beam_vector:        tuple
    :param  ang_radius_deg:     angular radius [deg] of the beam, None for all-sky
    :type   ang_radius_deg:     float
    :param  angle_deg:          angle [deg] on the sky within which copies are on the same line of sight
    :type   angle_deg:          float
    :param  output_dir:         directory to write the reports to
    :type   output_dir:         pathlib.Path
    :param  orientation_table:  file of orientation tables, or None for the default tables
    :type   orientation_table:  str
    """
    reports = {}
    for lock in LOCKS:
        for seed in seeds:
            lightcone = new_lightcone(simulation, beam_vector, lock, seed, orientation_table)
            filename = output_dir / f"close_repeats.lock_{lock}.seed_{seed}.hdf5"
            
            # print the report 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", CloseRepeatWarning)
                report = find_close_cell_repeats(lightcone, redshift_range, ang_radius_deg=ang_radius_deg, angle_deg=angle_deg, output_file=str(filename))
            reports[lock, seed] = report
            print(f"orientation_lock={str(lock):6s} seed={seed}: {report.nr_touching:5d} touching, "
                  f"{report.nr_line_of_sight:6d} line of sight pairs, closest copies "
                  f"{report.closest_separation:8.1f} Mpc ({report.nr_cells_repeated} of {report.nr_cell_copies} cell "
                  f"copies repeated) -> {filename}", flush=True)
    return reports


def plot_reports(reports, seeds, title, filename):
    """
    Plot the close repeats for every lock and seed. 
    The number of touching and line of sight pairs, where on the sky the line of sight pairs
    are (first seed), and the closest distance between copies of each repeated cell (first seed).

    :param  reports:    the reports, see check_cases
    :type   reports:    dict
    :param  seeds:      the orientation seeds checked
    :type   seeds:      list of int
    :param  title:      title of the figure
    :type   title:      str
    :param  filename:   path of the figure
    :type   filename:   str or pathlib.Path
    """
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.])

    # show number of pairs for every lock and seed as column graph
    for col, (attribute, label) in enumerate((("nr_touching", "touching pairs"), ("nr_line_of_sight", "line of sight pairs (< 1 [deg])"))):
        ax = fig.add_subplot(grid[0, col])
        width = 0.8 / len(LOCKS)
        for i, lock in enumerate(LOCKS):
            counts = [getattr(reports[lock, seed], attribute) for seed in seeds]
            ax.bar(np.arange(len(seeds)) + (i - 1) * width, counts, width=width, color=LOCK_COLOURS[lock], label=f"orientation_lock = {lock}")
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_xticklabels([str(seed) for seed in seeds])
        ax.set_xlabel("orientation seed")
        ax.set_ylabel(f"number of {label}")
        if col == 0:
            ax.legend(frameon=False, fontsize=8)

    # show closest distance between copies of the same cell as a histogram 
    ax = fig.add_subplot(grid[0, 2])
    first = seeds[0]
    boxsize = reports[LOCKS[0], first].boxsize
    bins = np.linspace(0., 2. * boxsize, 41)
    for lock in LOCKS:
        closest = reports[lock, first].cells["closest_separation"]
        ax.hist(np.clip(closest, 0., bins[-1]), bins=bins, histtype="step", lw=1.5, color=LOCK_COLOURS[lock])
    ax.axvline(boxsize, color="darkgray", ls="--", lw=1.)
    ax.text(boxsize, ax.get_ylim()[1] * 0.95, " box size", color="darkslategray", fontsize=8, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("closest distance between copies of a cell [Mpc]")
    ax.set_ylabel(f"number of repeated cells (seed {first})")

    # overlay the line of sight and touching pairs on the sky 
    for col, lock in enumerate(LOCKS):
        ax = fig.add_subplot(grid[1, col], projection="mollweide")
        pairs = reports[lock, first].pairs
        # line of sight pairs in the lock's colour, touching pairs as black crosses on top
        for selection, marker, size, colour in ((pairs["line_of_sight"], "o", 6, LOCK_COLOURS[lock]), (pairs["touching"], "x", 22, "k")):
            centre = 0.5 * (pairs["centre_a"][selection] + pairs["centre_b"][selection])
            if len(centre):
                r = np.linalg.norm(centre, axis=1)
                lon = np.arctan2(centre[:, 1], centre[:, 0])
                lat = np.arcsin(np.clip(centre[:, 2] / r, -1., 1.))
                ax.scatter(lon, lat, s=size, marker=marker, color=colour, linewidths=1.0, alpha=0.8,
                           zorder=3 if marker == "x" else 2)
        ax.grid(True, color="lightgray", lw=0.5)
        ax.tick_params(labelsize=7, colors="darkslategray")
        #ax.set_title(f"orientation_lock = {lock}, seed {first}: {reports[lock, first].nr_line_of_sight} line of sight, {reports[lock, first].nr_touching} touching", fontsize=8)
        report = reports[lock, first]
        
        ax.set_title(f"orientation_lock = {lock}, seed {first}: {report.nr_line_of_sight} line of sight, {report.nr_touching} touching", fontsize=9)
        
        # the size of the shell checked: tiles, cells, and every pair of copies of a cell that could be a close repeat
        ax.text(0.5, -0.06, f"tiles: {report.nr_tiles:,}, total pairs checked: {report.nr_cell_pairs:,}", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="k")

    # overlay legend on 1st projection 
    #if col==0:
    handles = [
        Line2D([], [], ls="", marker="o", color="darkgray", markersize=4, label="line of sight pair"),
        Line2D([], [], ls="", marker="x", color="k", markersize=6, label="touching pair")
        ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=0.5)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def run(simulation, redshift_range, seeds, beam_vector, ang_radius_deg, angle_deg, output_dir, orientation_table=None):
    """
    Check every lock and seed, write the reports and plot them.

    :param  simulation:     the simulation
    :type   simulation:     Simulation
    :param  redshift_range: minimum and maximum redshift of the shell
    :type   redshift_range: tuple
    :param  seeds:          orientation seeds to check
    :type   seeds:          list of int
    :param  beam_vector:    direction of the beam, or None for all-sky
    :type   beam_vector:    tuple
    :param  ang_radius_deg: angular radius [deg] of the beam, None for all-sky
    :type   ang_radius_deg: float
    :param  angle_deg:      angle [deg] on the sky within which copies are on the same line of sight
    :type   angle_deg:      float
    :param  output_dir:     directory to write the reports and figure to
    :type   output_dir:     str
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shell = "all-sky" if beam_vector is None else f"beam {tuple(beam_vector)}, {ang_radius_deg} deg"
    print(f"{simulation.box_res}/{simulation.name}: close repeats of cells in the {shell} shell {redshift_range[0]} < z < {redshift_range[1]}, within {angle_deg} deg on the sky", flush=True)
    if orientation_table=="none":
        orientation_table_filename=None
    else:
        orientation_table_filename=orientation_table
    
    reports = check_cases(simulation, redshift_range, seeds, beam_vector, ang_radius_deg, angle_deg, output_dir, orientation_table=orientation_table_filename)

    # the lock and seed with the fewest close repeats
    best = min(reports, key=lambda case: (reports[case].nr_touching, reports[case].nr_line_of_sight))
    print(f"fewest close repeats: orientation_lock={best[0]}, orientation_seed={best[1]} ({reports[best].nr_touching} touching, {reports[best].nr_line_of_sight} line of sight pairs)")

    filename = output_dir / "close_repeats.png"
    plot_reports(reports, seeds, f"Close repeats of cells: {simulation.box_res}/{simulation.name}, {shell}, {redshift_range[0]} < z < {redshift_range[1]}", filename)
    
    print(f"figure saved as {filename}")


if __name__ == "__main__":

    arg_descript_str="Check a lightcone shell for close repeats of cells, before building it, for every orientation lock and several seeds"

    parser = argparse.ArgumentParser(description=arg_descript_str)

    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label")
    parser.add_argument("--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation")
    parser.add_argument("--fake", action="store_true", help="use the fake lattice simulation of examples/snapshot_lattice.py instead (runs anywhere)")
    
    parser.add_argument("--zmin", type=float, default=0.02, help="minimum redshift of the shell")
    parser.add_argument("--zmax", type=float, default=0.4, help="maximum redshift of the shell")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="orientation seeds to check")
    parser.add_argument("--orientation_table", default="none", help="path/to/orientation/table/file.yaml , if none use default")

    parser.add_argument("--beam_vector", type=float, nargs=3, default=None, help="direction of a beam (default all-sky)")
    parser.add_argument("--ang_radius_deg", type=float, default=None, help="angular radius [deg] of the beam")
    parser.add_argument("--angle_deg", type=float, default=1., help="copies within this angle [deg] on the sky are on the same line of sight")
    parser.add_argument("--output_dir", default="./example_outputs/close_repeat_check", help="output directory")
    args = parser.parse_args()
    
    plot_settings() 

    if (args.beam_vector is None) != (args.ang_radius_deg is None):
        parser.error("give both --beam_vector and --ang_radius_deg for a beam, or neither for all-sky")

    redshift_range = (args.zmin, args.zmax)
    beam_vector = None if args.beam_vector is None else tuple(args.beam_vector)

    if args.fake:
        with tempfile.TemporaryDirectory() as run_dir:
            run(fake_simulation(run_dir), redshift_range, args.seeds, beam_vector, args.ang_radius_deg, args.angle_deg, args.output_dir, args.orientation_table)
    else:
        run(Simulation.from_label(args.box_res, args.sim_name), redshift_range, args.seeds, beam_vector, args.ang_radius_deg, args.angle_deg, args.output_dir, args.orientation_table)