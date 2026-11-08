#!/bin/env python
"""
Example script to show when copies of the same cell clash in two neighbouring tiles.

Where two neighbouring tiles are oriented differently, a cell can end up sharing a face
with its own copy in the other tile, so the same structures sit side by side: a clashing (contacting) pair. The diagram
shows a slice through one layer of cells of two tiles sharing a face for three different instances:

    0. sreplicas:   tiles A and B share the exact same structure.
    1. clashing:    tile B is tile A reflected along the x-axis and every cell on the shared face meets its own copy
    2. no clash:    tile B is tile A with a quarter turn about the z-axis and reflected about along the y-axis
                    so no cell meets its own copy, and the closest copies are a quarter box length apart.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pathlib import Path
from swiftlet.example_plot_settings import plot_settings, cmap_MCG
from swiftlet import snapshot_orientation as so


# the two cases: tile B's orientation (quarter turns about x, y, z; reflections; shift in eighths of the box)
CASES = [
    ("periodic replicas: tile B is tile A", (0, 0, 0), (1, 1, 1), (0, 0, 0)),
    ("a clash: tile B is tile A reflected along $x$", (0, 0, 0), (-1, 1, 1), (0, 0, 0)),
    (
        "no clash: tile B is tile A with a quarter turn about $z$ and shifted by $1/4\\,L$ along $x$",
        (0, 0, 1),
        (1, -1, 1),
        (0, 0, 0),
    ),
]


def slice_cells(n_cells, L):
    """
    The centres of the cells of the first layer along z of a box, and their numbers within the layer.

    Returns (centres, numbers): centres, shape (n_cells**2, 3), in [0, L), and the number i * n_cells + j of cell
    (i, j).

    :param  n_cells:    number of cells along each axis of the box
    :type   n_cells:    int
    :param  L:          box side length
    :type   L:          float
    """
    ij = np.indices((n_cells, n_cells)).reshape(2, -1).T
    centres = np.column_stack([(ij + 0.5) * L / n_cells, np.full(len(ij), 0.5 * L / n_cells)])
    return centres, ij[:, 0] * n_cells + ij[:, 1]


def adjust_tile(centres, quarter_turns, reflections, eighths, L):
    """
    Move cell centres as a tile with this orientation moves them, inside the box.

    Returns the moved centres, shape (N, 3).

    :param  centres:        cell centres in the snapshot box, shape (N, 3), in [0, L)
    :type   centres:        numpy.ndarray
    :param  quarter_turns:  quarter turns about x, y and z
    :type   quarter_turns:  tuple
    :param  reflections:    reflections (+1 or -1) along x, y and z
    :type   reflections:    tuple
    :param  eighths:        periodic shift along x, y and z, in eighths of the box
    :type   eighths:        tuple
    :param  L:              box side length
    :type   L:              float
    """
    return np.asarray(
        so.rotate_coords_cartesian(
            centres.copy(),
            np.array(quarter_turns, dtype=float) * 90.0,
            reflections=np.array(reflections, dtype=float),
            periodic_shift=np.array(eighths, dtype=float) * L / 8,
            sidelengths=np.full(3, L),
            degrees=True,
        )
    )


def draw_tile(ax, positions, numbers, offset, n_cells, L, clashing):
    """
    Draw the cells of a tile, coloured and numbered by their number in the snapshot box.

    :param  ax:         axes to draw on
    :type   ax:         matplotlib.axes.Axes
    :param  positions:  cell centres in the tile, shape (N, 3)
    :type   positions:  numpy.ndarray
    :param  numbers:    the number of each cell in the snapshot box
    :type   numbers:    numpy.ndarray
    :param  offset:     position of the tile's corner along x
    :type   offset:     float
    :param  n_cells:    number of cells along each axis of the box
    :type   n_cells:    int
    :param  L:          box side length
    :type   L:          float
    :param  clashing:   the numbers of the cells that clash
    :type   clashing:   set
    """
    size = L / n_cells
    cmap = plt.get_cmap(cmap_MCG)
    for (x, y, _), number in zip(positions, numbers):
        clash = int(number) in clashing
        ax.add_patch(
            Rectangle(
                (offset + x - 0.5 * size, y - 0.5 * size),
                size,
                size,
                facecolor=cmap(number / (n_cells**2 - 1)),
                edgecolor="crimson" if clash else "white",
                lw=2.4 if clash else 0.6,
                alpha=0.75,  # if clash else 0.75,
                zorder=20 if clash else 1,
            )
        )
        ax.text(offset + x, y, str(number), ha="center", va="center", fontsize=11, color="white", zorder=21)
    ax.add_patch(Rectangle((offset, 0), L, L, facecolor="none", edgecolor="k", lw=2.0, zorder=5))


def finish_axes(ax, x_max, L, title):
    """
    Label the tiles, set the limits and title, and hide the axes.

    :param  ax:     axes to finish
    :type   ax:     matplotlib.axes.Axes
    :param  x_max:  right edge of the second tile
    :type   x_max:  float
    :param  L:      box side length
    :type   L:      float
    :param  title:  title of the axes
    :type   title:  str
    """
    ax.text(0.5 * L, -0.06 * L, "tile A", ha="center", va="top", fontsize=11)
    ax.text(x_max - 0.5 * L, -0.06 * L, "tile B", ha="center", va="top", fontsize=11)
    ax.set_xlim(-0.02 * L, x_max + 0.02 * L)
    ax.set_ylim(-0.16 * L, 1.12 * L)
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_neighbouring_tiles(
    case, n_cells=4, L=1.0, figsize=(7.0, 4.0), include_title=False, filename="./snapshot_clashing_cells.png"
):
    """
    Make a diagram of tiles A and B sharing a face along x, tile B re-oriented as one of CASES.

    Returns the minimum distance between the two copies of a cell, in units of L.

    :param  case:       one of CASES: (title, quarter turns, reflections, shift in eighths of the box) of tile B
    :type   case:       tuple
    :param  n_cells:    number of cells along each axis of the box, a multiple of 4 so the shift of a quarter of the box
                            is whole cells
    :type   n_cells:    int
    :param  L:          box side length
    :type   L:          float
    :param  figsize:    size of the figure
    :type   figsize:    tuple
    :param  filename:   path to write the plot to
    :type   filename:   str
    """
    title, turns, reflections, eighths = case
    centres, numbers = slice_cells(n_cells, L)
    tile_a = adjust_tile(centres, (0, 0, 0), (1, 1, 1), (0, 0, 0), L)  # tile A: the snapshot box, unchanged
    tile_b = adjust_tile(centres, turns, reflections, eighths, L) + np.array([L, 0.0, 0.0])
    size = L / n_cells
    separation = np.linalg.norm(tile_b - tile_a, axis=1)  # between the two copies of each cell
    clashing = {int(m) for m in numbers[np.isclose(separation, size)]}  # copies sharing a face

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    draw_tile(ax, tile_a, numbers, 0.0, n_cells, L, clashing)
    draw_tile(ax, tile_b - np.array([L, 0.0, 0.0]), numbers, L, n_cells, L, clashing)
    ax.plot([L, L], [0, L], color="k", lw=3.0, zorder=6)

    if clashing:
        label = f"{len(clashing)} cells meet their own copy across the shared face"
    else:
        # both copies of the closest cell outlined, joined by an arc above the tiles so no cell number is hidden
        k = int(np.argmin(separation))
        for position in (tile_a[k], tile_b[k]):
            ax.add_patch(
                Rectangle(
                    (position[0] - 0.5 * size, position[1] - 0.5 * size),
                    size,
                    size,
                    facecolor="none",
                    edgecolor="crimson",
                    lw=2.4,
                    zorder=6,
                )
            )

        label = (
            f"no cell meets its own copy: the closest copies (e.g. cell {numbers[k]}) are "
            f"{separation[k] / L:.2g} $L$ apart"
        )

    finish_axes(ax, 2.0 * L, L, f"{title}\n{label}" if include_title else None)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return float(separation.min())


def plot_clashing_cells(n_cells=4, L=1.0, output_dir="."):
    """
    Make the three diagrams, each a separate figure in output_dir.

    :param  n_cells:    number of cells along each axis of the box, a multiple of 4
    :type   n_cells:    int
    :param  L:          box side length
    :type   L:          float
    :param  output_dir: directory to write the figures to
    :type   output_dir: str
    """
    files = {
        name: str(Path(output_dir) / f"snapshot_clashing_cells_{name}.png") for name in ("replica", "clash", "no_clash")
    }

    separations = {
        name: plot_neighbouring_tiles(case, n_cells, L, filename=files[name], include_title=False)
        for name, case in zip(("replica", "clash", "no_clash"), CASES)
    }
    return files, separations


if __name__ == "__main__":
    plot_settings()  # establish plot settings

    output_dir = "example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    files, separations = plot_clashing_cells(n_cells=4, L=1.0, output_dir=output_dir)

    for path in files.values():
        print(f"Clashing cells diagram saved as: {path}")
