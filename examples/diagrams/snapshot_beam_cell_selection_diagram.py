#!/bin/env python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from matplotlib.lines import Line2D
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from swiftlet.snapshot_lightcone import SnapshotBeam
from swiftlet.example_plot_settings import plot_settings

"""
Example script to show how SnapshotBeam selects tiles and the cells within each tile before any particle data is read:
    
    1. tiles:   find every periodic replica of the snapshot box (i.e. a tile), that the beam's cone can reach.  
                    Tiles are a built on a regular lattice.
    2. cells:   for each tile keep the cells whose centres are within the beam, with a buffer
                    of one cell length. Only the particles of those cells are then read.

Draw a diagram of a that beam points along the x-axis, showing both the tiles crossed and selected cells for 
the plane y=0 and a the beams circular cross-section. 

"""

###################################
# set colours used in the plot
COL_TILE_USED = "#86c5c0"   # shaded colour of tiles in lightcone
COL_TILE_UNUSED = "white"   # shaded colour of tiles not in the lightcone
COL_CELL_EDGE = "#6f9995"   # colour of the cell edges
COL_HATCH = "#0f4744"       # colour of the hatching of cells in the lightcone
COL_TILE_EDGE = "k"         # coluor of the tile edges (the lattice overlaid)
COL_CROSS = "#8B008B"   # cross section outlines colour
###################################


def diagram_beam(beam_vector):
    """
    A SnapshotBeam with only the attributes its tile and cell selection need (the beam direction and tile frame),
    so the diagram can use SnapshotBeam's own selection methods without a simulation on disk.

    Returns the SnapshotBeam.

    :param  beam_vector:    direction of the beam
    :type   beam_vector:    array-like, shape (3,)
    """

    beam = SnapshotBeam.__new__(SnapshotBeam)
    beam.beam_vec = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
    beam.beam_transverse_e1, beam.beam_transverse_e2 = SnapshotBeam._SnapshotBeam__transverse_basis(beam.beam_vec)
    beam._SnapshotBeam__init_tile_frame()

    return beam


def select_tiles_and_cells(beam, L, n_cells, r_max, half_angle_deg, cosmo):
    """
    Select the tiles the beam crosses and the cells kept in each tile.

    Returns a dictionary {(mx, my, mz): boolean array of shape (n_cells, n_cells, n_cells)},

    :param  beam:           the beam, see diagram_beam
    :type   beam:           SnapshotBeam
    :param  L:              box sidelength [Mpc]
    :type   L:              float
    :param  n_cells:        number of cells along each axis of the box
    :type   n_cells:        int
    :param  r_max:          maximum comoving distance [Mpc] of the beam
    :type   r_max:          float
    :param  half_angle_deg: angular radius [deg] of the beam
    :type   half_angle_deg: float
    :param  cosmo:          cosmology converting distances to redshifts
    :type   cosmo:          astropy.cosmology.FLRW
    """

    # max redshift to search
    z_max = float(z_at_value(cosmo.comoving_distance, r_max * u.Mpc))
    
    cell = L / n_cells # cell location along tile

    offsets = (np.arange(n_cells) + 0.5) * cell
    cx, cy, cz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    cell_offsets = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1)

    selected = {}
    for tile in beam._tiles_intersecting_shell(0., r_max, L, ang_radius_deg=half_angle_deg):
        corner = beam.get_snapshot_reposition_coords(tile, np.full(3, L))
        kept, __ = beam._in_shell(corner + cell_offsets, z_min=0., z_max=z_max, ang_radius_deg=half_angle_deg,
                                  buffer_length=cell, buffer_shape="cube", return_bool=True, cosmo=cosmo)
        lattice_position = tuple(int(m) for m in beam._tile_frame.T @ np.asarray(tile))
        selected[lattice_position] = kept.reshape(n_cells, n_cells, n_cells)
    return selected


def draw_tiles(ax, tiles, h_range, v_range, n_cells, project_axis):
    """
    Draw the tiles in a plane. used tiles are shaded and overlaied with the grid
    of cells of used tiles. Cells in the lightcone are hatch shaded.

    :param  ax:             axes to plot onto
    :type   ax:             matplotlib.axes._axes.Axes
    :param  tiles:          {(h, v): kept cells of the tile, shape (n_cells,)*3 or None if the tile is not used}
    :type   tiles:          dict
    :param  h_range:        lattice positions along the horizontal axis
    :type   h_range:        range
    :param  v_range:        lattice positions along the vertical axis
    :type   v_range:        range
    :param  n_cells:        number of cells along each axis of the box
    :type   n_cells:        int
    :param  project_axis:   axis (0, 1 or 2) of the kept-cell array along which columns are collapsed
    :type   project_axis:   int
    """

    cell = 1. / n_cells
    
    for h in h_range:
        for v in v_range:
            kept = tiles.get((h, v))
            x0, y0 = h - 0.5, v - 0.5
            ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=COL_TILE_USED if kept is not None else COL_TILE_UNUSED, edgecolor="none", zorder=1))
            if kept is None:
                continue
            
            # cell grid of the used tile
            for i in range(1, n_cells):
                ax.plot([x0 + i * cell] * 2, [y0, y0 + 1], color=COL_CELL_EDGE, lw=0.5, zorder=2)
                ax.plot([x0, x0 + 1], [y0 + i * cell] * 2, color=COL_CELL_EDGE, lw=0.5, zorder=2)
            
            # a cell is hatched if any cell in its column (along the projected axis) is kept
            column = kept.any(axis=project_axis)
            for i in range(n_cells):
                for j in range(n_cells):
                    if column[i, j]:
                        ax.add_patch(Rectangle((x0 + i * cell, y0 + j * cell), cell, cell, facecolor="none",
                                               edgecolor=COL_HATCH, hatch="///", lw=0., zorder=3))
    # tile edges
    for h in range(h_range.start, h_range.stop + 1):
        ax.axvline(h - 0.5, color=COL_TILE_EDGE, lw=1.2, zorder=4)
    for v in range(v_range.start, v_range.stop + 1):
        ax.axhline(v - 0.5, color=COL_TILE_EDGE, lw=1.2, zorder=4)
    ax.set_xlim(h_range.start - 0.5, h_range.stop - 0.5)
    ax.set_ylim(v_range.start - 0.5, v_range.stop - 0.5)
    ax.set_aspect("equal")


def plot_beam_cell_selection(L=1000., n_cells=4, half_angle_deg=30., r_max_L=4.2, n_along=3, x_cross=2, n_transverse=2,
                             figsize=(10, 5.6), filename="./snapshot_beam_cell_selection.png"):
    """
    Make 2 diagrams for the tiles and cells SnapshotBeam selects for a beam along tge x-axis, 
    1. a side view on the plane x-z plane at y = 0
    2. a cross-section of the beam on y-z plane at x=x_cross

    :param  L:              box sidelength [Mpc]
    :type   L:              float
    :param  n_cells:        number of cells along each axis of the box
    :type   n_cells:        int
    :param  half_angle_deg: angular radius [deg] of the beam
    :type   half_angle_deg: float
    :param  r_max_L:        maximum distance of the beam from the observer, in box sidelengths
    :type   r_max_L:        float
    :param  n_along:        number of tiles shown along the beam beyond the observer's tile
    :type   n_along:        int
    :param  x_cross:        lattice position (in box sidelengths) along x of the cross-section
    :type   x_cross:        int
    :param  n_transverse:   number of tiles shown either side of the beam's axis
    :type   n_transverse:   int
    :param  figsize:        size of the figure
    :type   figsize:        tuple
    :param  filename:       path to write the plot to
    :type   filename:       str
    """
    cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306)
    beam = diagram_beam((1, 0, 0))
    selected = select_tiles_and_cells(beam, L, n_cells, r_max_L * L, half_angle_deg, cosmo)
    half_angle = np.deg2rad(half_angle_deg)
    x_max = n_along
    transverse = range(-n_transverse, n_transverse + 1)

    fig, (ax_side, ax_cross) = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    # side view: tiles (mx, 0, mz), columns along y collapsed
    side = {(mx, mz): selected.get((mx, 0, mz)) for mx in range(0, x_max + 1) for mz in transverse}
    draw_tiles(ax_side, side, range(0, x_max + 1), transverse, n_cells, project_axis=1)
    edge = np.array([0, r_max_L])
    for sign in (1, -1):
        ax_side.plot(edge * np.cos(half_angle), sign * edge * np.sin(half_angle), color='k', lw=1.6, zorder=5)
    ax_side.plot([0, r_max_L], [0, 0], color='k', lw=1.0, ls="--", zorder=5)
    ax_side.axvline(x_cross, color=COL_CROSS, lw=2.0, ls=":", zorder=6)
    ax_side.text(x_cross, transverse.stop - 0.55, "cross-section", color=COL_CROSS, ha="center", va="top", fontsize=10,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=7)
    ax_side.plot(0, 0, "o", ms=7, mfc="white", mec='k', mew=1.4, zorder=8)
    ax_side.set_xlabel(r"$x$ [$L$]")
    ax_side.set_ylabel(r"$z$ [$L$]")

    # cross-section: tiles (x_cross, my, mz), columns along x collapsed
    cross = {(my, mz): selected.get((x_cross, my, mz)) for my in transverse for mz in transverse}
    draw_tiles(ax_cross, cross, transverse, transverse, n_cells, project_axis=0)
    ax_cross.add_patch(Circle((0, 0), x_cross * np.tan(half_angle), facecolor="none", edgecolor='k', lw=1.6, zorder=5))
    ax_cross.plot(0, 0, "+", ms=10, color='k', mew=1.4, zorder=5)
    for spine in ax_cross.spines.values():
        spine.set_edgecolor(COL_CROSS)
        spine.set_linestyle((0, (3, 2)))
        spine.set_linewidth(2.0)
        spine.set_zorder(6)
    ax_cross.set_xlabel(r"$y$ [$L$]")
    ax_cross.set_ylabel(r"$z$ [$L$]")

    handles = [
        Patch(facecolor=COL_TILE_USED, edgecolor=COL_CELL_EDGE, label="tile used"),
        Patch(facecolor=COL_TILE_UNUSED, edgecolor=COL_TILE_EDGE, label="tile not used"),
        Patch(facecolor="none", edgecolor=COL_HATCH, hatch="///", label="cell in the beam"),
        Line2D([], [], color='k', lw=1.6, label="edge of the beam"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    plot_settings() # establish plot settings

    output_dir = "example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = output_dir + "/snapshot_beam_cell_selection.png"
    plot_beam_cell_selection(L=1000., n_cells=4, half_angle_deg=30., r_max_L=4.2, n_along=3, x_cross=2, filename=filename)
    print(f"Beam cell selection diagram saved as: {filename}")