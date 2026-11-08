#!/bin/env python
"""
Example script to show that a cell of a lightcone, in a snapshot's shell, is a cell of that snapshot's box.

The diagram joins the snapshot shells and beam cell selection diagrams, in two panels:

1. a side view of a beam along x through the lattice of tiles, each tile with its grid of cells, the beam coloured by
   the shell of comoving distance each snapshot fills (FLAMINGO L1's output redshifts). One cell inside the beam is
   picked out;
2. the snapshot box of the snapshot whose shell holds that cell, at that snapshot's redshift, in 3D, with its grid of
   cells and the same cell at its place in the box.

The cell is moved back from the lightcone into the box by undoing its tile's orientation: the row of the default
orientation tables (default_v2) the tile takes from its lattice position and the orientation_seed, as a lightcone
picks it (no orientation_lock), with swiftlet.snapshot_orientation.rotate_coords_cartesian. The particles of that cell
of that snapshot are the ones placed in the lightcone at the cell's place there.

The side view is a slice through the centres of one layer of cells (y = L / (2 n_cells)), so the picked cell's
distance from the observer, and so its shell, is exact.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
from pathlib import Path
from swiftlet.example_plot_settings import plot_settings, cmap_MCG
from swiftlet.simulation_suites import Simulation
from swiftlet import snapshot_orientation as so
from swiftlet.snapshot_lightcone import TILE_HASH_PRIMES, TILE_HASH_SEED_PRIME

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_redshift_shells_diagram import snapshot_shells  # noqa: E402


def tile_orientation(tile, tables, orientation_seed=0):
    """
    The row of the orientation tables a read by a tile when orientation_lock="none".

    Returns (row, quarter turns, reflections, cell shift in eighths of the box).

    :param  tile:               lattice position (nx, ny, nz) of the tile
    :type   tile:               tuple
    :param  tables:             orientation tables, see swiftlet.snapshot_orientation.default_orientation_tables
    :type   tables:             dict
    :param  orientation_seed:   seed of the tile orientations
    :type   orientation_seed:   int
    """
    p1, p2, p3 = TILE_HASH_PRIMES
    tile_hash = tile[0] * p1 + tile[1] * p2 + tile[2] * p3 + orientation_seed * TILE_HASH_SEED_PRIME
    row = tile_hash % len(tables["QuarterTurns"])
    return row, tables["QuarterTurns"][row], tables["Reflections"][row], tables["CellShifts"][row]


def move_into_tile(points, tile, turns, reflections, eighths, n_cells, L, invert=False):
    """
    Move points of the snapshot box into a tile of the lightcone.

    Returns the moved points, shape (N, 3).

    :param  points:         points in the box (or, with invert, in the tile), shape (N, 3)
    :type   points:         numpy.ndarray
    :param  tile:           lattice position (nx, ny, nz) of the tile
    :type   tile:           tuple
    :param  turns:          quarter turns about x, y and z
    :type   turns:          numpy.ndarray
    :param  reflections:    reflections (+1 or -1) along x, y and z
    :type   reflections:    numpy.ndarray
    :param  eighths:        periodic shift along x, y and z, in eighths of the box
    :type   eighths:        numpy.ndarray
    :param  n_cells:        number of cells along each axis of the box
    :type   n_cells:        int
    :param  L:              box side length
    :type   L:              float
    :param  invert:         if True, move points of the tile back into the box
    :type   invert:         bool
    """
    offset = np.asarray(tile, dtype=float) * L - 0.5 * L
    shift = so.cell_shifts_in_cells(np.asarray(eighths), n_cells) * (L / n_cells)
    points = np.array(points, dtype=float) - (offset if invert else 0.0)
    moved = np.asarray(
        so.rotate_coords_cartesian(
            points,
            np.asarray(turns, dtype=float) * 90.0,
            reflections=np.asarray(reflections, dtype=float),
            periodic_shift=shift,
            sidelengths=np.full(3, L),
            degrees=True,
            invert=invert,
        )
    )
    return moved if invert else moved + offset


def draw_side_view(ax, r_edges, cmap, n_tiles_along, n_tiles_across, n_cells, half_angle_deg, y_slice, resolution=900):
    """
    Draw the beam along the x-axis for when y = y_slice and colour each shell by snapshot number.

    :param  ax:             axes to draw on
    :type   ax:             matplotlib.axes.Axes
    :param  r_edges:        comoving distance edges of the shells, from the observer out, in units of L
    :type   r_edges:        numpy.ndarray
    :param  cmap:           colour of each shell, from the innermost
    :type   cmap:           matplotlib.colors.ListedColormap
    :param  n_tiles_along:  number of tiles shown along the beam beyond the observer's tile
    :type   n_tiles_along:  int
    :param  n_tiles_across: number of tiles shown either side of the beam's axis
    :type   n_tiles_across: int
    :param  n_cells:        number of cells along each axis of a tile
    :type   n_cells:        int
    :param  half_angle_deg: angular radius [deg] of the beam
    :type   half_angle_deg: float
    :param  y_slice:        y of the slice, in units of L
    :type   y_slice:        float
    :param  resolution:     number of pixels along each axis of the image of the shells
    :type   resolution:     int
    """
    # x and z coords for when y=y_slice
    x_min, x_max = -0.5, n_tiles_along + 0.5
    z_min, z_max = -n_tiles_across - 0.5, n_tiles_across + 0.5
    # make mesh on x-z plane
    x, z = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(z_min, z_max, resolution))
    # distanc to each point in the mesh
    r = np.sqrt(x**2 + y_slice**2 + z**2)
    # is inside the beam
    inside = (x / r >= np.cos(np.deg2rad(half_angle_deg))) & (r <= r_edges[-1])

    shell = np.clip(np.searchsorted(r_edges, r, side="right") - 1, 0, len(r_edges) - 2)

    # make colour mesh for plot
    ax.pcolormesh(
        x,
        z,
        np.where(inside, shell, np.nan),
        cmap=cmap,
        vmin=-0.5,
        vmax=len(r_edges) - 1.5,
        alpha=0.8,
        shading="auto",
        rasterized=True,
        zorder=1,
    )

    # the boundaries between shells inside the beam, and the edge of the beam
    changes = np.zeros_like(inside)
    changes[:, 1:] |= (shell[:, 1:] != shell[:, :-1]) & inside[:, 1:] & inside[:, :-1]
    changes[1:, :] |= (shell[1:, :] != shell[:-1, :]) & inside[1:, :] & inside[:-1, :]

    # plot shells boundaries as contours
    ax.contour(x, z, changes.astype(float), levels=[0.5], colors="white", linewidths=0.5, zorder=2)
    ax.contour(x, z, inside.astype(float), levels=[0.5], colors="k", linewidths=0.75, zorder=5)

    # make grid of the cells and tiles
    for t in np.arange(x_min, x_max + 1e-9, 1.0 / n_cells):
        ax.axvline(t, color="grey", lw=0.375, alpha=0.7, zorder=3)
    for t in np.arange(z_min, z_max + 1e-9, 1.0 / n_cells):
        ax.axhline(t, color="grey", lw=0.75 * 0.5, alpha=0.7, zorder=3)
    for t in np.arange(x_min, x_max + 1e-9, 1.0):
        ax.axvline(t, color="k", lw=1.2, zorder=4)
    for t in np.arange(z_min, z_max + 1e-9, 1.0):
        ax.axhline(t, color="k", lw=1.2, zorder=4)

    # observer
    ax.plot(0, 0, "o", ms=3.5, mfc="white", mec="k", mew=0.7, zorder=7)

    # set axes limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$L$]")
    ax.set_ylabel(r"$z$ [$L$]")


def draw_box(ax, cell, n_cells, colour):
    """
    Draw the snapshot box in 3D with the grid of cells and the selected cell highlighted inside it.

    :param  ax:         3D axes to draw on
    :type   ax:         mpl_toolkits.mplot3d.axes3d.Axes3D
    :param  cell:       index (i, j, k) of the cell to fill
    :type   cell:       tuple
    :param  n_cells:    number of cells along each axis of the box
    :type   n_cells:    int
    :param  colour:     colour of the filled cell
    :type   colour:     str
    """
    # draw in the order of the zorders below, not matplotlib's depth sorting
    ax.computed_zorder = False
    back_corner = (0, 1, 0)  # away from screen

    ticks = np.linspace(0, 1, n_cells + 1)

    for t in ticks:
        # the grid on the back faces x = 0, y = 1 and z = 0
        tick_locations_in_3d = (
            ([0, 0], [t, t], [0, 1]),
            ([0, 0], [0, 1], [t, t]),
            ([t, t], [1, 1], [0, 1]),
            ([0, 1], [1, 1], [t, t]),
            ([t, t], [0, 1], [0, 0]),
            ([0, 1], [t, t], [0, 0]),
        )
        # iterate and draw cell grid
        for xs, ys, zs in tick_locations_in_3d:
            ax.plot(xs, ys, zs, color="grey", lw=0.75 * 0.5, zorder=1)

    # upper and lower bound of cell to highlight
    lo = np.asarray(cell, dtype=float) / n_cells
    hi = lo + 1.0 / n_cells
    mid = 0.5 * (lo + hi)

    # the locations of the highlighted cell's shadows on each possible plane
    shadows = (
        (0, [[0, lo[1], lo[2]], [0, hi[1], lo[2]], [0, hi[1], hi[2]], [0, lo[1], hi[2]]], [0, mid[1], mid[2]]),
        (1, [[lo[0], 1, lo[2]], [hi[0], 1, lo[2]], [hi[0], 1, hi[2]], [lo[0], 1, hi[2]]], [mid[0], 1, mid[2]]),
        (2, [[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]], [mid[0], mid[1], 0]),
    )
    for _, square, foot in shadows:
        ax.add_collection3d(
            Poly3DCollection([square], facecolor=colour, edgecolor=colour, lw=0.5 * 0.6, alpha=0.25, zorder=2)
        )
        # add connection lines from highlighted cell to the shadows
        ax.plot(*np.transpose([mid, foot]), color=colour, lw=0.5 * 0.9, ls="--", alpha=0.8, zorder=2)

    # draw the extra edges on top of the axes to make the 3d box
    for a in (0, 1):
        for b in (0, 1):
            for start, stop in (((0, a, b), (1, a, b)), ((a, 0, b), (a, 1, b)), ((a, b, 0), (a, b, 1))):
                hidden = back_corner in (start, stop)
                ax.plot(
                    *np.transpose([start, stop]),
                    color="k",
                    lw=0.75 if hidden else 1,
                    zorder=1 if hidden else 4,
                )

    # highlight
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(
        Poly3DCollection([corners[f] for f in faces], facecolor=colour, edgecolor="k", lw=0.4, alpha=0.9, zorder=3)
    )

    # set axes limits 0-1for x,y,z
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(0, 1)
    ax.set_box_aspect((1, 1, 1))

    # add labels
    ax.set_xlabel(r"$x$ [$L$]")
    ax.set_ylabel(r"$y$ [$L$]")
    ax.set_zlabel(r"$z$ [$L$]")

    # create viewing window
    ax.view_init(elev=22, azim=-58)

    ax.grid(False)  # only the grid of cells drawn above
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.set_ticks(np.linspace(0, 1, 5))
        axis.set_tick_params(labelsize=8)


def plot_cell_in_shell(
    tile=(2, 0, 0),
    cell_in_tile=(2, 4, 4),
    n_cells=8,
    L=1000.0,
    half_angle_deg=20.0,
    n_tiles_along=3,
    n_tiles_across=1,
    orientation_seed=0,
    label_every=4,
    figsize=(13.0, 5.6),
    filename="./snapshot_cell_in_shell.png",
):
    """
    Make the diagram of a cell in a snapshot's shell and its corresponding location in that snapshot's box.

    Returns a dict: the tile, its row of the tables, the cell in the tile and in the box, the snapshot
    and its redshift.

    :param  tile:               lattice position (nx, ny, nz) of the tile holding the picked cell (ny = 0)
    :type   tile:               tuple
    :param  cell_in_tile:       index (i, j, k) of the picked cell within the tile, along x, y and z; j = n_cells // 2,
                                    the layer the side view slices through
    :type   cell_in_tile:       tuple
    :param  n_cells:            number of cells along each axis of the box, a multiple of 8 as for a lightcone (the
                                    tables shift the box by eighths of it)
    :type   n_cells:            int
    :param  L:                  box side length [Mpc]
    :type   L:                  float
    :param  half_angle_deg:     angular radius [deg] of the beam
    :type   half_angle_deg:     float
    :param  n_tiles_along:      number of tiles shown along the beam beyond the observer's tile
    :type   n_tiles_along:      int
    :param  n_tiles_across:     number of tiles shown either side of the beam's axis
    :type   n_tiles_across:     int
    :param  orientation_seed:   seed of the tile orientations
    :type   orientation_seed:   int
    :param  box_res:            box size and resolution label of the simulation whose snapshots are drawn
    :type   box_res:            str
    :param  sim_name:           name of the simulation
    :type   sim_name:           str
    :param  label_every:        label every this many snapshots on the colour bar
    :type   label_every:        int
    :param  figsize:            size of the figure
    :type   figsize:            tuple
    :param  filename:           path to write the plot to
    :type   filename:           str
    """

    box_res = "L1000N1800"
    sim_name = "HYDRO_FIDUCIAL"
    cell_highlight_colour = "crimson"

    # sanity check the selected cell will work
    if cell_in_tile[1] != n_cells // 2:
        raise ValueError(f"the picked cell must be in the layer j = {n_cells // 2} the side view slices through")

    # size of cells in slice
    size = 1.0 / n_cells
    y_slice = 0.5 * size  # through the centres of the layer of cells j = n_cells // 2 of the tiles with ny = 0

    # the snapshot shells out to the far end of the tiles shown
    simulation = Simulation.from_label(box_res, sim_name)
    r_far = np.hypot(n_tiles_along + 0.5, n_tiles_across + 0.5) * L

    # get example cosmology for snapshot number - redshift conversion
    from astropy import units as u
    from astropy.cosmology import z_at_value
    from snapshot_redshift_shells_diagram import FLAMINGO_COSMOLOGY

    # redshift of each shell
    z_far = float(z_at_value(FLAMINGO_COSMOLOGY.comoving_distance, r_far * u.Mpc))
    snapshot_numbers, redshifts, _, r_edges = snapshot_shells(z_far, simulation)
    r_edges = r_edges / L

    # colourmap of shells
    cmap = ListedColormap(cmap_MCG(np.linspace(0, 1, len(snapshot_numbers))[::-1]))

    # the picked cell in the lightcone, its shell, and its place in the snapshot box
    centre = (np.asarray(tile, dtype=float) - 0.5) + (np.asarray(cell_in_tile) + 0.5) * size
    shell = int(np.searchsorted(r_edges, np.linalg.norm(centre), side="right") - 1)
    snapshot, redshift = int(snapshot_numbers[shell]), float(redshifts[shell])

    # get default orientation
    tables = so.default_orientation_tables()

    # how to remake the tile
    row, turns, reflections, eighths = tile_orientation(tile, tables, orientation_seed)

    # check cell is in shell and box
    in_box = move_into_tile(centre[None, :], tile, turns, reflections, eighths, n_cells, 1.0, invert=True)[0]
    cell_in_box = tuple(int(n) for n in np.floor(in_box / size))

    # moving the box's cell into the tile puts it back where it was picked
    back = move_into_tile(in_box[None, :], tile, turns, reflections, eighths, n_cells, 1.0)[0]
    assert np.allclose(back, centre), (back, centre)

    # draw the diagram
    fig = plt.figure(figsize=figsize, layout="constrained")
    # grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")

    draw_side_view(ax, r_edges, cmap, n_tiles_along, n_tiles_across, n_cells, half_angle_deg, y_slice)

    # highlight cell in the beam panel
    ax.add_patch(
        Rectangle(
            (centre[0] - 0.5 * size, centre[2] - 0.5 * size),
            size,
            size,
            facecolor=cell_highlight_colour,
            edgecolor="k",
            lw=0.6,
            alpha=0.9,
            zorder=6,
        )
    )

    # make way to complicated colour bar

    # one segment of the same width per snapshot, labelled every label_every snapshots
    n_shells = len(snapshot_numbers)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(n_shells + 1) - 0.5, cmap.N), cmap=cmap),
        ax=ax,
        location="bottom",
        shrink=0.8,
        aspect=40,
        drawedges=True,
    )
    cbar.dividers.set_color("white")
    cbar.dividers.set_linewidth(0.5)
    labelled = np.arange(0, n_shells, label_every)
    cbar.set_ticks(labelled, labels=[str(snapshot_numbers[i]) for i in labelled])
    cbar.set_label(f"snapshot number")

    # draw 3D box and show 3D highlighted cell
    draw_box(ax3d, cell_in_box, n_cells, cell_highlight_colour)

    # draw a curve linking the cell in the lightcone to the same cell in the snapshot box
    fig.canvas.draw()
    x2, y2, _ = proj3d.proj_transform(*((np.asarray(cell_in_box) + 0.5) * size), ax3d.get_proj())
    arrow_offset = (-0.01, 0.005)
    fig.add_artist(
        ConnectionPatch(
            xyA=(centre[0], centre[2]),
            coordsA=ax.transData,
            xyB=(x2 + arrow_offset[0], y2 + arrow_offset[1]),
            coordsB=ax3d.transData,
            arrowstyle="-|>",
            mutation_scale=8,
            color=cell_highlight_colour,
            lw=0.9,
            connectionstyle="arc3,rad=-0.2",
            zorder=10,
            path_effects=[pe.Stroke(linewidth=1.1, foreground="black"), pe.Normal()],
        )
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return {
        "tile": tuple(tile),
        "row": int(row),
        "cell_in_tile": tuple(cell_in_tile),
        "cell_in_box": cell_in_box,
        "snapshot": snapshot,
        "redshift": redshift,
    }


if __name__ == "__main__":
    plot_settings()  # establish plot settings

    output_dir = "example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = output_dir + "/snapshot_cell_in_shell.png"
    drawn = plot_cell_in_shell(filename=filename, figsize=(7.0, 3.0))
    print(
        f"cell {drawn['cell_in_tile']} of tile {drawn['tile']} (row {drawn['row']}) is cell {drawn['cell_in_box']} of "
        f"the box of snapshot {drawn['snapshot']} (z = {drawn['redshift']:.3g})"
    )
    print(f"Cell in shell diagram saved as: {filename}")
