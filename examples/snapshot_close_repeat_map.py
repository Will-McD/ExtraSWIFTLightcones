#!/bin/env python
#
"""
An example script for visualising close repeats of the same cell along a line-of-sight and then overlaying a planned
lightcone Beam.

For each orientation lock used with an installed orientation table, plot a Mollweide projection of each pair, with the
comoving distance separation and lock indicated by colour and then overlay a planned lightcone beam.

For further information see the more indepth snapshot_close_repeat_check.py example script.
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path
import numpy as np
from swiftlet.simulation_suites import Simulation
from swiftlet.repeat_check import find_close_cell_repeats, CloseRepeatWarning
from swiftlet.beam_plotting import plot_beam_on_sky
import unyt

# plotting packages
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import matplotlib.patheffects as pe
import cmasher as cmr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents))
from snapshot_close_repeat_check import fake_simulation, new_lightcone  # noqa: E402


def lock_colour_map(lock, darkest=0.25, lightest=0.975):
    """
    Returns the colour map used for each orientation lock.

    :param  lock:       name of orientation lock (None or "none", "cube", "sphere")
    :type   lock:       str
    :param  darkest:    min fraction of the colour map to crop
    :type   darkest:    float
    :param  lightest:   max fraction of the colour map to crop
    :type   lightest:   float
    """
    lock_colour_maps = {
        None: "cmr.emerald",
        "cube": "cmr.sapphire",
        "sphere": "cmr.ember",
    }  # the colour map of each lock
    return cmr.get_sub_cmap(lock_colour_maps[lock], darkest, lightest).reversed()


def plot_repeats_on_sky(reports, filename, seperation_upper_limit=1000 * unyt.Mpc, overlay_beam=None):
    """
    Plot the close repeats of a lightcone shell, for every orientation lock, on a Mollweide map of the sky.

    Use a marker to indicate each pair, with colour to show both the distance and orientation lock of each pair.
    Overlay a shaded region for the planned lightcone beam.

    :param  reports:        the close repeats of each lock, {lock: CloseRepeatReport} from find_close_cell_repeats
    :type   reports:        dict
    :param  filename:       path of the figure
    :type   filename:       str or pathlib.Path
    :param  overlay_beam:   the beam_vector and ang_radius_deg of the planned beam.
    :type   overlay_beam:   dict

    """

    def sky_position(pairs, selection):
        # the direction of each pair from the observer, halfway between its two copies, as longitude and latitude
        centre = 0.5 * (pairs["centre_a"][selection] + pairs["centre_b"][selection])
        r = np.linalg.norm(centre, axis=1)
        return np.arctan2(centre[:, 1], centre[:, 0]), np.arcsin(
            np.clip(centre[:, 2] / np.where(r > 0, r, 1.0), -1.0, 1.0)
        )

    limit = seperation_upper_limit.to_value(unyt.Mpc)
    shown = {
        lock: report.pairs["line_of_sight"] & (report.pairs["separation"] < limit) for lock, report in reports.items()
    }

    # one colour scale for every lock: from the closest to the furthest pair shown, of any lock
    # separations = np.concatenate([reports[lock].pairs["separation"][los] for lock, los in shown.items()])
    separations = np.concatenate([reports[lock].pairs["separation"][los] for lock, los in shown.items()])

    # norm = Normalize(vmin=separations.min(), vmax=separations.max()) if len(separations) else Normalize(0.0, limit)
    norm = Normalize(vmin=30, vmax=separations.max()) if len(separations) else Normalize(0.0, limit)

    # fig = plt.figure(figsize=(9, 5.6))
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(projection="mollweide")
    ax.grid(True, color="lightgray", lw=0.5)
    ax.tick_params(labelsize=7, colors="darkslategray")

    # one colour scale for every lock: the distance between the two copies, up to the largest reported
    # vmax = max(report.max_los_separation for report in reports.values())
    handles = []
    for lock, report in reports.items():
        pairs = report.pairs
        cmap = lock_colour_map(lock)
        los = shown[lock]

        if np.any(los):
            lon, lat = sky_position(pairs, los)
            ax.scatter(
                lon,
                lat,
                c=pairs["separation"][los],
                cmap=cmap,
                norm=norm,
                marker="o",
                s=20,
                edgecolors="k",  # a thin black outline
                path_effects=[pe.Stroke(linewidth=1.0, foreground="k"), pe.Normal()],
                linewidths=0.1,
                alpha=0.9,
                zorder=2,
            )

        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                markerfacecolor=cmap(0.6),
                markeredgecolor="k",
                markeredgewidth=0.4,
                markersize=6,
                label=f"orientation_lock = {lock} ({report.nr_line_of_sight} line of sight, "
                f"{report.nr_touching} clashing)",
            )
        )
    if any(report.nr_touching for report in reports.values()):
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="x",
                color="white",
                path_effects=[pe.Stroke(linewidth=1.0, foreground="k"), pe.Normal()],
                markersize=6,
                label="clashing pairs",
            )
        )

    # a colour bar for each lock, touching side by side to the right of the map, so they read as one bar of three
    # columns sharing one distance axis: ticks and labels on the right of the last only
    bar_width = 0.025
    for i, lock in enumerate(reports):
        cax = ax.inset_axes([1.03 + bar_width * i, 0.12, bar_width, 0.76])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=lock_colour_map(lock)), cax=cax)
        bar.outline.set_linewidth(0.6)
        cax.set_title(str(lock), fontsize=9, rotation=60, ha="left", va="bottom", rotation_mode="anchor")
        if i < len(reports) - 1:
            cax.tick_params(right=False, labelright=False)
        else:
            cax.tick_params(labelsize=9)
            bar.set_label("distance between pairs [Mpc]", fontsize=11)

    # overlay beam pointing
    if overlay_beam is not None:
        plot_beam_on_sky(
            ax,
            beam_vector=overlay_beam["beam_vector"],
            ang_radius_deg=overlay_beam["ang_radius_deg"],
            colour="grey",
            alpha=0.2,
            edge_width=1.0,
            fill=True,
            n_points=721,
            zorder=1,
        )

    # print the summary of the report.
    counts = "\n".join(
        f"{lock}: {report.nr_tiles:,} tiles, {report.nr_cell_copies:,} copies of {report.nr_cells:,} "
        f"cells, {report.nr_cell_pairs:,} pairs of copies of the same cell"
        + (
            ""
            if report.max_snapshot_difference == 0
            else f" (within {report.max_snapshot_difference} snapshots of each other)"
        )
        for lock, report in reports.items()
    )
    print(counts)

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    simulation,
    redshift_range,
    orientation_seed,
    beam_vector,
    ang_radius_deg,
    angle_deg,
    output_dir,
    overlay_beam_vector=None,
    overlay_ang_radius_deg=None,
    orientation_table=None,
    max_snapshot_difference=0,
):
    """
    Find the close repeats of one lightcone shell, for every orientation lock, and plot them on the sky.

    :param  simulation:                 the simulation
    :type   simulation:                 Simulation
    :param  redshift_range:             minimum and maximum redshift of the shell
    :type   redshift_range:             tuple
    :param  orientation_seed:           seed of the box tile orientations
    :type   orientation_seed:           int
    :param  beam_vector:                direction of the beam, or None for all-sky
    :type   beam_vector:                tuple
    :param  ang_radius_deg:             angular radius [deg] of the beam, None for all-sky
    :type   ang_radius_deg:             float
    :param  angle_deg:                  angle [deg] on the sky within which copies are on the same line of sight
    :type   angle_deg:                  float
    :param  output_dir:                 directory to write the figure to
    :type   output_dir:                 str
    :param  overlay_beam_vector:        direction of the planned lightcone beam
    :type   overlay_beam_vector:        tuple
    :param  overlay_ang_radius_deg:     angular radius [deg] of the planned lightcone beam being overlaid
    :type   overlay_ang_radius_deg:     float
    :param  orientation_table:          file of orientation tables, or None for the default tables
    :type   orientation_table:          str
    :param  max_snapshot_difference:    pair copies of a cell from snapshots at most this far apart (0: the same
                                        snapshot only)
    :type   max_snapshot_difference:    int
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shell = "all-sky" if beam_vector is None else f"beam {tuple(beam_vector)}, {ang_radius_deg} deg"
    title_str = (
        f"Close repeats: {simulation.box_res}/{simulation.name}, {shell}, "
        f"{redshift_range[0]:g} < z < {redshift_range[1]:g}, orientation_seed = {orientation_seed}"
        + (
            ""
            if max_snapshot_difference == 0
            else f"\ncopies of the same cell up to {max_snapshot_difference} snapshots apart"
        )
    )

    print(title_str.replace("\n", ", "))

    reports = {}
    for lock in [None, "cube", "sphere"]:  # iterate through each orientation locks
        lightcone = new_lightcone(
            simulation, beam_vector, lock, orientation_seed, orientation_table, 1, ang_radius_deg=ang_radius_deg
        )

        # the report is printed and plotted, so the warning is not shown as well
        if orientation_table == "no_reorientation":
            max_snapshot_difference = 30

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CloseRepeatWarning)
            reports[lock] = find_close_cell_repeats(
                lightcone,
                redshift_range,
                angle_deg=angle_deg,
                copies_per_block=1_000_000,
                max_snapshot_difference=max_snapshot_difference,
            )
        print(f"orientation_lock={str(lock):6s}: {reports[lock].summary()}", flush=True)

    # snapshot window from max difference
    window = "" if max_snapshot_difference == 0 else f".snapshots_within_{max_snapshot_difference}"

    filename = output_dir / f"close_repeats_sky{window}.seed_{orientation_seed}.png"

    max_seperation = 500 * unyt.Mpc  # only show pairs who's separation is less than this

    if (overlay_beam_vector is not None) and (overlay_ang_radius_deg is not None):
        planned_beam = {
            "beam_vector": overlay_beam_vector,
            "ang_radius_deg": overlay_ang_radius_deg,
        }
    else:
        planned_beam = None

    plot_repeats_on_sky(reports, filename, seperation_upper_limit=max_seperation, overlay_beam=planned_beam)

    print(f"figure saved as {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show the close repeats of a lightcone shell, for every orientation "
        "lock, on a single map of the sky"
    )
    parser.add_argument("--box_res", default="L1000N1800", help="box size and resolution label")
    parser.add_argument("--sim_name", default="HYDRO_FIDUCIAL", help="name of the simulation")
    parser.add_argument("--zmin", type=float, default=0.02, help="minimum redshift of the shell")
    parser.add_argument("--zmax", type=float, default=0.4, help="maximum redshift of the shell")
    parser.add_argument("--orientation_seed", type=int, default=0, help="seed of the box tile orientations")
    parser.add_argument(
        "--max_snapshot_difference",
        type=int,
        default=0,
        help="pair copies of the same cell from snapshots up to this far apart (0: the same snapshot only), e.g. 2",
    )
    parser.add_argument(
        "--planned_beam_vector", type=float, nargs=3, default=None, help="direction of the planned beam"
    )
    parser.add_argument(
        "--planned_ang_radius_deg", type=float, default=None, help="angular radius [deg] of the planned beam"
    )
    parser.add_argument(
        "--angle_deg",
        type=float,
        default=1.0,
        help="copies within this angle [deg] on the sky are on the same line of sight",
    )
    parser.add_argument(
        "--orientation_table",
        default=None,
        help="file of orientation tables (see write_orientation_tables); the default tables if not given",
    )
    parser.add_argument("--output_dir", default="./example_outputs/close_repeat_sky", help="output directory")
    args = parser.parse_args()

    redshift_range = (args.zmin, args.zmax)
    settings = dict(
        orientation_seed=args.orientation_seed,
        beam_vector=None,
        ang_radius_deg=None,
        angle_deg=args.angle_deg,
        output_dir=args.output_dir,
        orientation_table=args.orientation_table,
        max_snapshot_difference=args.max_snapshot_difference,
        overlay_beam_vector=args.planned_beam_vector,
        overlay_ang_radius_deg=args.planned_ang_radius_deg,
    )

    run(Simulation.from_label(args.box_res, args.sim_name), redshift_range, **settings)
