#!/bin/env python
"""
Draws a 3D diagram of a tile with its cells overlaid.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from swiftlet.example_plot_settings import plot_settings


def view_matrix(azim_deg, elev_deg):
    """
    Returns a 3x3 rotation matrix.

    :param  azim_deg:   azimuth of the viewer, anticlockwise about z from the x axis [deg]
    :type   azim_deg:   float
    :param  elev_deg:   elevation of the viewer above the x-y plane [deg]
    :type   elev_deg:   float
    """
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    towards_viewer = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    across = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(towards_viewer, across)
    return np.array([across, up, towards_viewer])


def face_cells(n_cells, axis, side, gap):
    """
    Returns the corners of the squares of the cells on one face of the cube.

    :param  n_cells:    number of cells along each axis
    :type   n_cells:    int
    :param  axis:       axis the face is normal to, 0, 1 or 2
    :type   axis:       int
    :param  side:       which face along the axis, 0 or 1
    :type   side:       int
    :param  gap:        width of the black gap around each cell, as a fraction of the cell's side
    :type   gap:        float
    """
    u_axis, v_axis = [i for i in range(3) if i != axis]
    size = 1.0 / n_cells
    # adjust to allow for a gap between squares
    lo, hi = gap / 2 * size, (1 - gap / 2) * size
    squares = []
    for i in range(n_cells):
        for j in range(n_cells):
            square = np.zeros((4, 3))
            for k, (du, dv) in enumerate(((lo, lo), (hi, lo), (hi, hi), (lo, hi))):
                square[k, axis] = side
                square[k, u_axis] = i * size + du
                square[k, v_axis] = j * size + dv
            squares.append(square)
    return np.array(squares)


def plot_tile_cells_cube(
    n_cells=4,
    gap=0.12,
    azim_deg=-55.0,
    elev_deg=28.0,
    label_axes=True,
    figsize=(3.21, 3.21),
    filename="example_outputs/diagrams/snapshot_tile_cells_cube.png",
):
    """
    Plot a tile as a cube of n_cells x n_cells x n_cells cells, like a black and white Rubik's cube.

    :param  n_cells:    number of cells along each axis
    :type   n_cells:    int
    :param  gap:        width of the black gap around each cell, as a fraction of the cell's side
    :type   gap:        float
    :param  azim_deg:   azimuth of the viewer, anticlockwise about z from the x axis [deg]
    :type   azim_deg:   float
    :param  elev_deg:   elevation of the viewer above the x-y plane [deg]
    :type   elev_deg:   float
    :param  label_axes: If True, draw a small triad of the x, y and z axes beside the cube
    :type   label_axes: bool
    :param  figsize:    size of the figure
    :type   figsize:    tuple
    :param  filename:   path to write the plot to
    :type   filename:   str
    """
    view = view_matrix(azim_deg, elev_deg)

    def project(points):
        return (np.asarray(points) @ view.T)[..., :2]

    fig, ax = plt.subplots(figsize=figsize, layout="none")
    # change the shade of each face to add shadows and make it look more 3D
    face_shades = {"x": "#D9D9D9", "y": "#EDEDED", "z": "#FFFFFF"}
    # the faces the viewer sees: those whose outward normal points towards the viewer
    for axis, name in enumerate("xyz"):
        for side in (0, 1):
            normal = np.zeros(3)
            normal[axis] = 1.0 if side else -1.0
            if normal @ view[2] <= 0:
                continue
            body = face_cells(1, axis, side, 0.0)[0]
            ax.add_patch(Polygon(project(body), closed=True, facecolor="k", edgecolor="k", lw=1.5, zorder=1))
            for square in face_cells(n_cells, axis, side, gap):
                ax.add_patch(
                    Polygon(project(square), closed=True, facecolor=face_shades[name], edgecolor="none", zorder=2)
                )

    corners = project(np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T)
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)

    if label_axes:
        # a small triad of the x, y and z axes beside the cube, at its lower left, in the same view
        origin = np.array([xmin - 0.12, ymin + 0.05])
        for axis, name in enumerate("xyz"):
            direction = project(np.eye(3)[axis])
            tip = origin + 0.22 * direction
            ax.annotate("", xy=tip, xytext=origin, arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2), zorder=3)
            label = origin + 0.31 * direction if np.linalg.norm(direction) > 0.3 else tip + np.array([0.05, -0.04])
            ax.text(*label, name, ha="center", va="center", fontsize=11, color="0.2")
        xmin, ymin = min(xmin, origin[0] - 0.1), min(ymin, origin[1] - 0.1)

    pad = 0.05
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot_settings()  # establish plot settings

    output_dir = "example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_cells = 8
    filename = output_dir + f"/snapshot_tile_cells_cube_{n_cells:d}.png"
    plot_tile_cells_cube(n_cells, gap=0.16, filename=filename)

    print(f"\n3D tile diagram saved as: {filename}")
