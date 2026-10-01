#!/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path

"""
Example script to show how the locking the orientation per layer with orientation_lock decides which
periodic replicas (i.e. tiles) of the snapshot boxes share an orientation:
    None or "none":         every box tile has its own orientation. Structures are cut at every tile face.
    "cube":                 every tile in a cube shell of boxes around the observer's box, max(|mx|, |my|, |mz|) = n,
                                shares one orientation. The orientation only changes on the tile faces between shells.
    "sphere":               everything between (n-1/2) and (n+1/2) box sidelengths from the observer shares one
                                orientation. The orientation only changes at those distances, cutting through boxes.

The diagram is a slice through the x-z plane, which holds the observer.
As in SnapshotLightcone, the tiles sit on a regular lattice of whole box sidelengths, centred on (mx, my, mz) * L.
"""

# build colour map for tiles
ORIENTATION_CMAP = LinearSegmentedColormap.from_list("cyan_magenta_gold", ["darkcyan", "darkmagenta", "darkgoldenrod"])
ORIENTATION_ALPHA = 0.7
MAX_LAYER = 3
GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
COL_BOX_EDGE = "#52514e"
COL_LAYER_EDGE = "black"
COL_CHANGE = "black"
COL_TEXT = "#0b0b0b"


def plot_settings():
    """
    Set the matplotlib settings for this example.
    """
    # Line Properties
    plt.rcParams["lines.linewidth"] = 1.5
    # Font options
    plt.rcParams["font.size"] = 9
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["legend.fontsize"] = 10
    # axes font settings
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    # axes tick settings
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["ytick.direction"] = 'in'
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True



def slice_to_3d(x, z):
    """
    3D position of points in the plane y = 0. 

    Returns the positions. 

    :param  x:  coordinate along x
    :type   x:  np.ndarray
    :param  z:  coordinate along z
    :type   z:  np.ndarray
    """
    return np.stack([x, np.zeros_like(x), z], axis=-1)


def box_lattice_position(pos, L):
    """

    Returns the lattice positions (mx, my, mz) of the tile holding 
    each position for a side length of L. Each tile is centred on (mx, my, mz) * L.

    :param  pos:    positions, shape (..., 3)
    :type   pos:    np.ndarray
    :param  L:      box sidelength
    :type   L:      float
    """
    return np.floor(pos / L + 0.5).astype(int)


def orientation_layer(pos, L, orientation_lock):
    """
    The orientation each position in the lattice is given.

    Returns an integer array: 
        orientation_lock=None:      a different number for every box. 
        orientation_lock="cube":    the cube shell of the box, max(|mx|, |my|, |mz|).
        orientation_lock="sphere":  the spherical shell, floor(r / L + 1/2).

    :param  pos:                positions, shape (..., 3)
    :type   pos:                np.ndarray
    :param  L:                  box sidelength
    :type   L:                  float
    :param  orientation_lock:   None, "cube" or "sphere"
    :type   orientation_lock:   str
    """
    m = box_lattice_position(pos, L)
    if orientation_lock == "cube":
        return np.max(np.abs(m), axis=-1)
    if orientation_lock == "sphere":
        return np.floor(np.linalg.norm(pos, axis=-1) / L + 0.5).astype(int)
    # a unique number for each box in the plane y = 0, where my = 0
    return (m[..., 0] + 100) * 1000 + (m[..., 2] + 100)


def in_lightcone(s, z, r_max, beam_vector=None, half_angle_deg=None):
    """


    Returns a boolean array.
    True if in the lightcone, within r_max of the observer, 
    and within half_angle_deg of the beam axis.

    :param  s:              coordinate along x
    :type   s:              np.ndarray
    :param  z:              coordinate along z
    :type   z:              np.ndarray
    :param  r_max:          maximum distance from the observer
    :type   r_max:          float
    :param  beam_vector:    direction of the beam. None for all-sky
    :type   beam_vector:    array-like, shape (3,)
    :param  half_angle_deg: angular radius [deg] of the beam
    :type   half_angle_deg: float
    """
    pos = slice_to_3d(s, z)
    r = np.linalg.norm(pos, axis=-1)
    inside = r <= r_max
    if beam_vector is not None:
        v = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_angle = np.where(r > 0, (pos @ v) / r, 1.0)
        inside &= cos_angle >= np.cos(np.deg2rad(half_angle_deg))
    return inside


def beam_in_slice(beam_vector):
    """
    Returns the direction of a beam in the slice, as a unit vector in (x, z). 
    The beam must lie in the plane y = 0.

    :param  beam_vector:    direction of the beam
    :type   beam_vector:    array-like, shape (3,)
    """
    v = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
    if not np.isclose(v[1], 0.):
        raise ValueError("the beam must lie in the plane y = 0 to be drawn in this slice")
    return np.array([v[0], v[2]])


def box_label_positions(s, z, layer, inside, min_pixels=0):
    """

    Returns the label of each tile in the lightcone. 
    Each tile is numbered as 1, 2, 3, 4, ect... 
    The label is placed as close to the centre of a tiles overlapping area with the lightcone.

    Returns a list of (label, (s, z)).

    :param  s:          coordinate along x of each pixel
    :type   s:          np.ndarray
    :param  z:          coordinate along z of each pixel
    :type   z:          np.ndarray
    :param  layer:      orientation number of each pixel, see orientation_layer
    :type   layer:      np.ndarray
    :param  inside:     If True, the pixel is in the lightcone
    :type   inside:     np.ndarray
    :param  min_pixels: boxes with fewer pixels in the lightcone are numbered but not labelled
    :type   min_pixels: int
    """
    positions = []
    for n, key in enumerate(np.unique(layer[inside])):
        in_box = inside & (layer == key)
        points = np.stack([s[in_box], z[in_box]], axis=-1)
        if len(points) < max(min_pixels, 1):
            continue # too small to label
        centre = points[np.argmin(np.sum((points - points.mean(axis=0))**2, axis=1))]
        positions.append((f"{n + 1}", centre))
    return positions


def layer_label_positions(ray_angle, L, r_max, orientation_lock):
    """
    Returns the labels for each locked layer, placed half way through 
    the layer along a ray from the observer.

    :param  ray_angle:          angle [rad] of the ray in the slice, from the s axis
    :type   ray_angle:          float
    :param  L:                  box sidelength
    :type   L:                  float
    :param  r_max:              maximum distance from the observer
    :type   r_max:              float
    :param  orientation_lock:   "cube" or "sphere"
    :type   orientation_lock:   str
    """
    t = np.linspace(0, r_max, 4000)
    ray = np.stack([t * np.cos(ray_angle), t * np.sin(ray_angle)], axis=-1)
    ray_layer = orientation_layer(slice_to_3d(ray[:, 0], ray[:, 1]), L, orientation_lock)
    return [(f"{key}", ray[ray_layer == key].mean(axis=0)) for key in np.unique(ray_layer)]


def draw_lock_panel(ax, orientation_lock, L, r_max, extent, beam_vector=None, half_angle_deg=None, resolution=900):
    """
    Draw the box tiles and their orientations in the slice y = 0, for one lightcone and orientation lock.

    :param  ax:                 axes to plot onto
    :type   ax:                 matplotlib.axes._axes.Axes
    :param  orientation_lock:   None, "cube" or "sphere"
    :type   orientation_lock:   str
    :param  L:                  box sidelength
    :type   L:                  float
    :param  r_max:              maximum distance from the observer
    :type   r_max:              float
    :param  extent:             (s_min, s_max, z_min, z_max) shown
    :type   extent:             tuple
    :param  beam_vector:        direction of the beam. None for all-sky
    :type   beam_vector:        array-like, shape (3,)
    :param  half_angle_deg:     angular radius [deg] of the beam
    :type   half_angle_deg:     float
    :param  resolution:         number of pixels along each axis
    :type   resolution:         int
    """
    s_min, s_max, z_min, z_max = extent
    s, z = np.meshgrid(np.linspace(s_min, s_max, resolution), np.linspace(z_min, z_max, resolution))
    pos = slice_to_3d(s, z)
    inside = in_lightcone(s, z, r_max, beam_vector, half_angle_deg)
    layer = orientation_layer(pos, L, orientation_lock)

    # position on the colour map of each pixel in the lightcone, one colour for each orientation
    if orientation_lock is None:
        # every box its own colour, the n-th box (as numbered by box_label_positions) at (n * golden ratio) mod 1
        box_number = np.searchsorted(np.unique(layer[inside]), layer)
        colour = np.mod(box_number / GOLDEN_RATIO, 1.0)
    else:
        colour = np.clip(layer, 0, MAX_LAYER) / MAX_LAYER
    image = np.where(inside, colour, np.nan)
    ax.pcolormesh(s, z, image, cmap=ORIENTATION_CMAP, vmin=0, vmax=1,
                  alpha=ORIENTATION_ALPHA, shading="auto", rasterized=True, zorder=1)

    # box tiles crossed by the plane y = 0: L x L squares
    for mx in range(int(np.floor(s_min / L)) - 1, int(np.ceil(s_max / L)) + 2):
        ax.axvline((mx - 0.5) * L, color=COL_BOX_EDGE, lw=0.6, zorder=2)
    for mz in range(int(np.floor(z_min / L)) - 1, int(np.ceil(z_max / L)) + 2):
        ax.axhline((mz - 0.5) * L, color=COL_BOX_EDGE, lw=0.6, zorder=2)

    # boundaries between orientations, where the layer of neighbouring pixels in the lightcone differs
    changes = np.zeros_like(inside)
    changes[:, 1:] |= (layer[:, 1:] != layer[:, :-1]) & inside[:, 1:] & inside[:, :-1]
    changes[1:, :] |= (layer[1:, :] != layer[:-1, :]) & inside[1:, :] & inside[:-1, :]
    ax.contour(s, z, changes.astype(float), levels=[0.5], colors=COL_LAYER_EDGE, linewidths=1.6, zorder=3)

    # number each region: each box without a lock, each layer with a lock
    if orientation_lock is None:
        label_positions = box_label_positions(s, z, layer, inside, min_pixels=0.002 * resolution**2)
    else:
        # layers are shells around the observer, so label them along a ray from the observer
        if beam_vector is None:
            ray_angle = np.deg2rad(235.)
        else:
            axis = beam_in_slice(beam_vector)
            ray_angle = np.arctan2(axis[1], axis[0]) - 0.55 * np.deg2rad(half_angle_deg)
        label_positions = layer_label_positions(ray_angle, L, r_max, orientation_lock)
    for text, position in label_positions:
        ax.text(*position, text, ha="center", va="center", fontsize=8, color=COL_TEXT, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))

    # outline of the lightcone and the observer
    if beam_vector is None:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), color=COL_TEXT, lw=1.0, zorder=4)
    else:
        axis = beam_in_slice(beam_vector)
        half_angle = np.deg2rad(half_angle_deg)
        angles = np.arctan2(axis[1], axis[0]) + np.linspace(-half_angle, half_angle, 100)
        edge = r_max * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        ax.plot(np.concatenate([[0], edge[:, 0], [0]]), np.concatenate([[0], edge[:, 1], [0]]), color=COL_TEXT, lw=1.0, zorder=4)

        # where the orientation changes along the beam axis
        t = np.linspace(0, r_max, 4000)
        axis_layer = orientation_layer(slice_to_3d(t * axis[0], t * axis[1]), L, orientation_lock)
        change = np.flatnonzero(np.diff(axis_layer) != 0)
        ax.plot(t * axis[0], t * axis[1], color=COL_CHANGE, lw=1.2, ls="--", zorder=4)
        ax.plot(t[change] * axis[0], t[change] * axis[1], "o", ms=5, mfc="white", mec=COL_CHANGE, mew=1.2, zorder=6)
    ax.plot(0, 0, "o", ms=6, mfc="white", mec=COL_TEXT, mew=1.2, zorder=7)

    ax.set_xlim(s_min, s_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal")


def plot_orientation_locks(L=1.0, r_max=3.2, half_angle_deg=20., figsize=(11, 11), filename="./orientation_locks.png"):
    """
    Plot the orientations of the box tiles for each orientation lock (rows), for an all-sky lightcone,
    a beam along (0, 0, 1) and a beam along (1, 0, 1) (columns).

    :param  L:              box sidelength
    :type   L:              float
    :param  r_max:          maximum distance from the observer, in units of L
    :type   r_max:          float
    :param  half_angle_deg: angular radius [deg] of the beams
    :type   half_angle_deg: float
    :param  figsize:        size of the figure
    :type   figsize:        tuple
    :param  filename:       path to write the plot to
    :type   filename:       str
    """
    locks = [None, "cube", "sphere"]
    lock_titles = {None: "orientation_lock=None", "cube": 'orientation_lock="cube"', "sphere": 'orientation_lock="sphere"'}
    lightcones = [
        ("SnapshotAllSky", None, (-r_max * 1.05, r_max * 1.05, -r_max * 1.05, r_max * 1.05)),
        ("SnapshotBeam (0, 0, 1)", (0, 0, 1), (-1.6 * L, 1.6 * L, -0.3 * L, r_max * 1.05)),
        ("SnapshotBeam (1, 0, 1)", (1, 0, 1), (-0.3 * L, r_max * 0.95, -0.3 * L, r_max * 0.95)),
    ]

    fig, axs = plt.subplots(len(locks), len(lightcones), figsize=figsize, layout="constrained",
                            gridspec_kw={"width_ratios": [1.15, 0.65, 1.0]})
    for i, lock in enumerate(locks):
        for j, (name, beam_vector, extent) in enumerate(lightcones):
            ax = axs[i, j]
            draw_lock_panel(ax, lock, L, r_max * L, extent, beam_vector=beam_vector, half_angle_deg=half_angle_deg)
            if j == 0:
                ax.set_ylabel(f"{lock_titles[lock]}\n\n" + r"$z$ [$L$]", fontsize=10)
            else:
                ax.set_ylabel(r"$z$ [$L$]")
            if i == len(locks) - 1:
                ax.set_xlabel(r"$x$ [$L$]")

    # column titles on one line, above the top row
    fig.canvas.draw()
    title_y = max(ax.get_position().y1 for ax in axs[0])
    for ax, (name, _, _) in zip(axs[0], lightcones):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), title_y + 0.008, name, ha="center", va="bottom", fontsize=11, fontweight="bold")

    handles = [
        Line2D([], [], color=COL_CHANGE, lw=1.2, ls="--", marker="o", ms=5, mfc="white", mec=COL_CHANGE, label="beam axis, dots where the orientation changes"),
        Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec=COL_TEXT, label="observer"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    plot_settings() # establish plot settings

    output_dir = "example_outputs/snapshot_lightcone"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_orientation_locks(L=1.0, r_max=3.2, half_angle_deg=20., filename=output_dir + "/snapshot_orientation_locks.png")
    print(f"Orientation lock diagram saved as: {output_dir}/snapshot_orientation_locks.png")
