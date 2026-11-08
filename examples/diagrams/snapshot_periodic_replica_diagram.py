#!/bin/env python
"""
Example script to show why the box tiles placed along a beam are reoriented to avoid exact periodic replicas.

This script draws a diagram of a slice through the beam, along its axis (x) and one transverse direction (z), with the
observer at the centre of the first tile. As in SnapshotLightcone, the tiles sit on a regular lattice centred on whole
multiples of L. The axes of each slice are exactly the tiles: their outline is the frame of the axes. Below each slice,
the objects the central line of sight passes through are shown against the distance from the observer.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from swiftlet.example_plot_settings import plot_settings


###################################

# Objects in the snapshot box: (label, position (x, z) in cells, radius in cells).
# The box is 4 x 4 cells, so the box centre is at (2, 2) cells.

OBJECTS = [
    ("A", (1.00, 2.00), 0.35),  # on the box's mid-line: in line with the observer in every unchanged tile
    # each other object has a coordinate near a different whole cell, so a reoriented tile can bring it, alone, onto
    # the line of sight
    ("B", (3.07, 3.27), 0.27),
    ("C", (2.47, 0.73), 0.32),
    ("D", (1.92, 3.40), 0.23),
    ("E", (3.40, 1.60), 0.24),
]

# Colours of the objects in the box, one per object (each is also labelled with a letter)
OBJECT_COLOURS = ["darkcyan", "darkmagenta", "darkgoldenrod", "yellowgreen", "grey"]

# The reorientation of each tile along the beam:
# periodic shift (x, z) [cells], reflection (x, z), rotation [deg]
# Tile 0, the observer's, is the box as it is.
REORIENTATIONS = [
    ((0, 0), (1, 1), 0),
    ((0, 2), (1, -1), 90),
    ((0, 1), (1, -1), 180),
    ((3, 0), (-1, 1), 90),
    ((1, 3), (1, 1), 270),
]


def reorient(position, shift, reflection, angle_deg, n_cells=4):
    """
    Move a position in the box [cells] as a tile is reoriented: a periodic shift of whole cells, then a reflection of
    each axis and a rotation, both about the box centre.

    Returns the position in the reoriented box [cells], np.ndarray, shape (2,).

    :param  position:   position (x, z) in the box [cells]
    :type   position:   array-like, shape (2,)
    :param  shift:      periodic shift (x, z) [cells]
    :type   shift:      tuple
    :param  reflection: reflection (+1 or -1) of x and z
    :type   reflection: tuple
    :param  angle_deg:  rotation [deg], a multiple of 90
    :type   angle_deg:  int
    :param  n_cells:    number of cells along each side of the box
    :type   n_cells:    int
    """
    centre = 0.5 * n_cells
    shifted = np.mod(np.asarray(position, dtype=float) + np.asarray(shift, dtype=float), n_cells)
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return centre + rotation @ (np.asarray(reflection, dtype=float) * (shifted - centre))


def tile_objects(n_tiles, reorientations=None, L=1.0, n_cells=4):
    """
    Copies of the objects in each tile along the beam, every tile in the box's orientation, or reoriented.

    Returns a list of (tile number, label, colour, centre (x, z) [L], radius [L])

    :param  n_tiles:        number of tiles along the beam
    :type   n_tiles:        int
    :param  reorientations: (shift, reflection, rotation) of each tile, see REORIENTATIONS. If None, no tile is
                                reoriented
    :type   reorientations: list
    :param  L:              box sidelength
    :type   L:              float
    :param  n_cells:        number of cells along each side of the box
    :type   n_cells:        int
    """
    cell = L / n_cells
    copies = []
    for n in range(n_tiles):
        tile_origin = np.array([n * L - 0.5 * L, -0.5 * L])
        for (label, pos, radius), colour in zip(OBJECTS, OBJECT_COLOURS):
            if reorientations is not None:
                pos = reorient(pos, *reorientations[n], n_cells=n_cells)
            copies.append((n, label, colour, tile_origin + np.asarray(pos, dtype=float) * cell, radius * cell))
    return copies


def central_sightline_hits(copies):
    """
    Returns the indices of the copies, in order of distance from the observer.

    :param  copies: copies of the objects, see tile_objects
    :type   copies: list
    """
    hits = [i for i, (_, _, _, centre, radius) in enumerate(copies) if abs(centre[1]) <= radius and centre[0] > 0]
    return sorted(hits, key=lambda i: copies[i][3][0])


def choose_reorientations(n_tiles, seed=0, n_cells=4):
    """
    Search for a reorientation of each tile after the first so the central line of sight passes through exactly one
    object in each of them, a different object in each tile, each whole inside its tile.

    Returns the reorientations

    :param  n_tiles:    number of tiles along the beam
    :type   n_tiles:    int
    :param  seed:       seed of the random search
    :type   seed:       int
    :param  n_cells:    number of cells along each side of the box
    :type   n_cells:    int
    """
    rng = np.random.default_rng(seed)
    while True:
        reorientations = [((0, 0), (1, 1), 0)] + [
            (
                tuple(int(s) for s in rng.integers(0, n_cells, 2)),
                tuple(int(r) for r in rng.choice([-1, 1], 2)),
                int(rng.choice([0, 90, 180, 270])),
            )
            for _ in range(1, n_tiles)
        ]
        copies = tile_objects(n_tiles, reorientations, n_cells=n_cells)
        hits = central_sightline_hits(copies)
        tiles = [copies[i][0] for i in hits]
        labels = [copies[i][1] for i in hits]
        inside = all(abs(copies[i][3][0] - copies[i][0]) <= 0.5 - copies[i][4] for i in hits)
        if sorted(tiles) == list(range(1, n_tiles)) and len(set(labels)) == len(labels) and inside:
            return reorientations


def draw_tiles(ax, n_tiles, L=1.0, n_cells=4):
    """
    Draw the tiles along the beam and overlay the cells.

    :param  ax:         axes to plot onto
    :type   ax:         matplotlib.axes._axes.Axes
    :param  n_tiles:    number of tiles along the beam
    :type   n_tiles:    int
    :param  L:          box sidelength
    :type   L:          float
    :param  n_cells:    number of cells along each side of the box
    :type   n_cells:    int
    """
    cell = L / n_cells
    for n in range(n_tiles):
        x0, z0 = n * L - 0.5 * L, -0.5 * L
        for k in range(1, n_cells):
            ax.plot([x0 + k * cell] * 2, [z0, z0 + L], color="gray", lw=0.4, zorder=1)
            ax.plot([x0, x0 + L], [z0 + k * cell] * 2, color="gray", lw=0.4, zorder=1)
        # tile edges: the outer edges are the frame of the axes, drawn as thick as these
        if n > 0:
            ax.axvline(x0, color="k", lw=1, zorder=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_zorder(5)


def draw_objects(ax, copies, sightline_hits, L=1.0):
    """
    Draw the copies of the objects, outlining the ones the central line of sight passes through. An object crossing
    the edge of its tile is drawn wrapped around the tile, as the box is periodic, and clipped to the tile.

    :param  ax:             axes to plot onto
    :type   ax:             matplotlib.axes._axes.Axes
    :param  copies:         copies of the objects, see tile_objects
    :type   copies:         list
    :param  sightline_hits: indices of the copies the central line of sight passes through
    :type   sightline_hits: set
    :param  L:              box sidelength
    :type   L:              float
    """
    for i, (n, label, colour, centre, radius) in enumerate(copies):
        hit = i in sightline_hits
        tile = plt.Rectangle((n * L - 0.5 * L, -0.5 * L), L, L, transform=ax.transData, visible=False)
        ax.add_patch(tile)
        for dx in (-L, 0.0, L):
            for dz in (-L, 0.0, L):
                image = centre + np.array([dx, dz])
                if np.any(np.abs(image - np.array([n * L, 0.0])) > 0.5 * L + radius):
                    continue
                circle = plt.Circle(
                    image,
                    radius,
                    facecolor=colour,
                    edgecolor="black" if hit else "none",
                    lw=1.8 if hit else 0,
                    alpha=0.9,
                    zorder=3,
                )
                ax.add_patch(circle)
                circle.set_clip_path(tile)
                # the letter only on the part of the object holding its centre
                if np.all(np.abs(image - np.array([n * L, 0.0])) <= 0.5 * L):
                    ax.text(
                        *image, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=4
                    )


def draw_beam(ax, r_max, half_angle_deg):
    """
    Draw the beam's edges, its central line of sight and the observer.

    :param  ax:             axes to plot onto
    :type   ax:             matplotlib.axes._axes.Axes
    :param  r_max:          length of the beam drawn [L]
    :type   r_max:          float
    :param  half_angle_deg: angular radius of the beam [deg]
    :type   half_angle_deg: float
    """
    w = r_max * np.tan(np.deg2rad(half_angle_deg))
    ax.plot([0, r_max], [0, w], color="k", lw=1.0, zorder=6)
    ax.plot([0, r_max], [0, -w], color="k", lw=1.0, zorder=6)
    ax.plot([0, r_max], [0, 0], color="k", lw=1.2, ls="--", zorder=6)
    ax.plot(0, 0, "o", ms=6, mfc="white", mec="k", mew=1.2, zorder=7, clip_on=False)


def draw_sightline_strip(ax, copies, hits, n_tiles, L=1.0):
    """
    Draw the objects the central line of sight passes through against distance from the observer, marking
    where an object passed through before is seen again. The arrows of the first repeated object are drawn above
    the line of sight, those of the next below it, and so on, so they do not overlap.

    :param  ax:         axes to plot onto
    :type   ax:         matplotlib.axes._axes.Axes
    :param  copies:     copies of the objects, see tile_objects
    :type   copies:     list
    :param  hits:       indices of the copies the line of sight passes through, see central_sightline_hits
    :type   hits:       list
    :param  n_tiles:    number of tiles along the beam
    :type   n_tiles:    int
    :param  L:          box sidelength
    :type   L:          float
    """
    ax.axhline(0, color="k", lw=1.2, ls="--", zorder=1)
    for n in range(1, n_tiles):
        ax.axvline(n * L - 0.5 * L, color="k", lw=0.8, zorder=0)

    last_seen = {}
    arrow_side = {}  # +1: arrows above the line of sight, -1: below
    for i in hits:
        _, label, colour, centre, _ = copies[i]
        ax.plot(centre[0], 0, "o", ms=13, mfc=colour, mec="black", mew=1.2, zorder=3)
        ax.text(centre[0], 0, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=4)
        if label in last_seen:
            # the same object again: an arrow from where it was last seen
            side = arrow_side.setdefault(label, 1 if len(arrow_side) % 2 == 0 else -1)
            ann = ax.annotate(
                "",
                xy=(centre[0] - 0.06, side * 0.32),
                xytext=(last_seen[label] + 0.06, side * 0.32),
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.6, connectionstyle=f"arc3,rad={-0.15 * side}"),
                zorder=2,
            )
            ann.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.5, foreground="k"), pe.Normal()])
        last_seen[label] = centre[0]

    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlim(-0.5 * L, (n_tiles - 0.5) * L)


def tile_label(reorientation):
    """
    The label of a reoriented tile: its shift, reflection and rotation.

    :param  reorientation:  (shift, reflection, rotation) of the tile, see REORIENTATIONS
    :type   reorientation:  tuple
    """
    shift, reflection, angle = reorientation
    return f"shift={tuple(shift)}\nrefl={tuple(reflection)}, rot={angle}°"


def draw_zoom(ax_slice, ax_strip, n_tiles, L=1.0, half_width=0.1):
    """
    Show the strip as a zoomed inset of the band around the central line of sight: the band shaded and outlined in
    the slice, the strip shaded and outlined alike, and lines joining the band's lower corners to the strip's upper
    corners.

    :param  ax_slice:   axes of the slice
    :type   ax_slice:   matplotlib.axes._axes.Axes
    :param  ax_strip:   axes of the strip
    :type   ax_strip:   matplotlib.axes._axes.Axes
    :param  n_tiles:    number of tiles along the beam
    :type   n_tiles:    int
    :param  L:          box sidelength
    :type   L:          float
    :param  half_width: half width [L] of the band
    :type   half_width: float
    """
    x0, x1 = -0.5 * L, (n_tiles - 0.5) * L
    ax_slice.add_patch(
        plt.Rectangle(
            (x0, -half_width * L),
            x1 - x0,
            2.0 * half_width * L,
            facecolor="#ECECEC",
            edgecolor="k",
            lw=1.0,
            ls="--",
            alpha=0.6,
            zorder=2,
        )
    )
    # ax_strip.set_facecolor()
    ax_strip.set_facecolor(mpl.colors.to_rgba("#ECECEC", 0.8))
    for spine in ax_strip.spines.values():
        spine.set_edgecolor("dimgray")
        spine.set_linestyle("--")


def draw_panel(ax_slice, ax_strip, n_tiles, reorientations, title, L=1.0, half_angle_deg=5.5):
    """
    Draw one panel: the slice through the beam, whose axes are exactly the tiles, with each tile's label above it,
    and the strip of the objects the central line of sight passes through below it.

    :param  ax_slice:       axes of the slice
    :type   ax_slice:       matplotlib.axes._axes.Axes
    :param  ax_strip:       axes of the strip
    :type   ax_strip:       matplotlib.axes._axes.Axes
    :param  n_tiles:        number of tiles along the beam
    :type   n_tiles:        int
    :param  reorientations: (shift, reflection, rotation) of each tile, or None for every tile in the box's orientation
    :type   reorientations: list
    :param  title:          title of the panel
    :type   title:          str
    :param  L:              box sidelength
    :type   L:              float
    :param  half_angle_deg: angular radius of the beam [deg]
    :type   half_angle_deg: float
    """
    copies = tile_objects(n_tiles, reorientations, L)
    hits = central_sightline_hits(copies)

    draw_tiles(ax_slice, n_tiles, L)
    draw_objects(ax_slice, copies, set(hits), L)
    draw_beam(ax_slice, (n_tiles - 0.5) * L, half_angle_deg)

    # each tile's label inside it, at the top, over the objects
    for n in range(n_tiles):
        label = f"tile {n}"  # if reorientations is None else tile_label(reorientations[n])

        ax_slice.text(
            n * L,
            0.52 * L,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="k",
            linespacing=1.1,
            zorder=8,
        )
        if reorientations is not None:
            ax_slice.text(
                (n - 0.45) * L,
                -0.48 * L,
                tile_label(reorientations[n]),
                ha="left",
                va="bottom",
                fontsize=9,
                color="k",
                linespacing=1.1,
                zorder=8,
                bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9, pad=1.0),
            )
    if title is not None:
        ax_slice.set_title(title, fontsize=10, loc="left")

    # the axes are exactly the tiles: the frame is the outline of the tiles
    ax_slice.set_xlim(-0.5 * L, (n_tiles - 0.5) * L)
    ax_slice.set_ylim(-0.5 * L, 0.5 * L)
    ax_slice.set_box_aspect(1.0 / n_tiles)
    ax_strip.set_box_aspect(0.25 / n_tiles)
    ax_slice.set_anchor("S")  # slice to the bottom of its cell
    ax_strip.set_anchor("N")  # strip to the top of its cell
    ax_slice.set_ylabel(r"$z$ [$L$]")
    ax_slice.tick_params(labelbottom=False)

    draw_sightline_strip(ax_strip, copies, hits, n_tiles, L)
    ax_strip.set_ylabel("line of\nsight", fontsize=10)
    draw_zoom(ax_slice, ax_strip, n_tiles, L)


def plot_panel(reorientations, filename, n_tiles=5, L=1.0, half_angle_deg=5.5, figsize=(10, 3.9)):
    """
    Plot one panel on its own: the box tiles along a beam, every tile in the box's orientation (reorientations None)
    or reoriented, with the objects the central line of sight passes through below.

    :param  n_tiles:        number of tiles along the beam
    :type   n_tiles:        int
    :param  L:              box sidelength
    :type   L:              float
    :param  half_angle_deg: angular radius of the beam [deg]
    :type   half_angle_deg: float
    :param  reorientations: (shift, reflection, rotation) of each tile in the bottom panel
    :type   reorientations: list
    :param  figsize:        size of the figure
    :type   figsize:        tuple
    :param  filename:       path to write the plot to
    :type   filename:       str
    """
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, layout="constrained", sharex=True, gridspec_kw={"height_ratios": [2.4, 0.55]}
    )
    draw_panel(axes[0], axes[1], n_tiles, reorientations, None, L, half_angle_deg)

    axes[-1].set_xlabel(r"$x$ [$L$]")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_settings()  # establish plot settings

    output_dir = "example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    height_width_ratio = 3.9 / 10
    fig_width = 7.0

    plot_panel(
        None, output_dir + "/snapshot_periodic_replicas_0.png", figsize=(fig_width, fig_width * height_width_ratio)
    )
    plot_panel(
        REORIENTATIONS,
        output_dir + "/snapshot_periodic_replicas_1.png",
        figsize=(fig_width, fig_width * height_width_ratio),
    )
    print(f"Repeated structures diagram saved as: {output_dir + '/snapshot_periodic_replicas'}")
