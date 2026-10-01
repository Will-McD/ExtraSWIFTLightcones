#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path

"""
Example script to highlilght how cells within the snaphots boxes (tiles in lightcone)
are repositioned when creating new 'unique' snapshot tiles as they placed in to the lightcone (all-sky or beam). 
"""


def plot_settings():
    """
    matplotlib settings for this example. 
    """
    import matplotlib as mpl
    
    # Line Properties
    plt.rcParams["lines.linewidth"]=1.5 
    # Font options
    plt.rcParams["font.size"]=8
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["legend.labelspacing"]=0.25
    plt.rcParams["legend.columnspacing"]=0.75
    plt.rcParams["legend.borderpad"]=0.3
    plt.rcParams["legend.borderaxespad"]=0.7
    plt.rcParams["legend.fontsize"]=10
    # Figure layour settings
    plt.rcParams["figure.constrained_layout.use"] =True
    plt.rcParams["figure.constrained_layout.h_pad"] =0.005
    plt.rcParams["figure.constrained_layout.w_pad"] =0.005
    plt.rcParams["figure.constrained_layout.hspace"]=0.005
    plt.rcParams["figure.constrained_layout.wspace"]=0.005
    # axes font settings
    mpl.rcParams['axes.labelsize']=10
    mpl.rcParams['figure.labelsize']=10
    mpl.rcParams['xtick.labelsize']=10
    mpl.rcParams['ytick.labelsize']=10
    # axes tick settings
    plt.rcParams["xtick.direction"] ='in'
    plt.rcParams["ytick.direction"] ='in'
    plt.rcParams["xtick.top"]=True
    plt.rcParams["ytick.right"]= True

###################################
# Colors for each stage of repositioning snapshot cells into the lightcone
COL_ORIG   = "darkcyan" 
COL_SHIFT  = "darkgoldenrod" 
COL_GHOST  = "darkgoldenrod" 
COL_REFLECT= "darkmagenta" 
COL_ROTATE = "yellowgreen"
COL_FINAL  = "grey" 
###################################

def rot_matrix_2d(angle_deg):
    """
    2D rotation matrix.

    :param  angle_deg:  rotation angle [deg]
    :type   angle_deg:  float
    """
    a = np.deg2rad(angle_deg)
    c, s = np.round(np.cos(a), 8), np.round(np.sin(a), 8)
    return np.array([[c, -s], [s, c]])

def transform_point(p, reflection, angle_deg, centre):
    """
    Reflect and then rotate a point about the centre.

    :param  p:          point to transform
    :type   p:          np.ndarray
    :param  reflection: per-axis reflection signs (+1/-1)
    :type   reflection: array-like, shape (2,)
    :param  angle_deg:  rotation angle [deg]
    :type   angle_deg:  float
    :param  centre:     centre of the reflection and rotation
    :type   centre:     float
    """
    S = np.diag(reflection)
    R = rot_matrix_2d(angle_deg)
    M = R @ S                      # reflect first, then rotate (established convention)
    return M @ (p - centre) + centre

def wrap_point(p, shift, L):
    return np.mod(p + shift, L)

def plot_transform_grid(examples, L=3.0, selected_cell_centre=(0.5, 0.5), ncols=3, nrows=None, figsize=(7,7), filename="./grid_transform_example.png"):
    """
    Plot a grid of examples showing how a cell is repositioned by a periodic shift, reflection and rotation.

    :param  examples:               (shift, reflection, rotation angle [deg]) of each example
    :type   examples:               list
    :param  L:                      side length of the box, in cells
    :type   L:                      float
    :param  selected_cell_centre:   centre of the cell to reposition
    :type   selected_cell_centre:   tuple
    :param  ncols:                  number of columns in the grid
    :type   ncols:                  int
    :param  nrows:                  number of rows in the grid. If None, use enough rows for all examples
    :type   nrows:                  int
    :param  figsize:                size of the figure
    :type   figsize:                tuple
    :param  filename:               path to write the plot to
    :type   filename:               str
    """
    centre = L / 2.0
    OUTLINE = [pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()]
    #OUTLINE=None
    def compute_all_steps(p0, shift, reflection, angle_deg):
        """
        Compute the position of the cell after each step of the repositioning.

        Returns a tuple of (original, shifted, wrapped, reflected, rotated, final) positions.

        :param  p0:         original position of the cell
        :type   p0:         np.ndarray
        :param  shift:      shift along each axis
        :type   shift:      np.ndarray
        :param  reflection: per-axis reflection signs (+1/-1)
        :type   reflection: array-like, shape (2,)
        :param  angle_deg:  rotation angle [deg]
        :type   angle_deg:  float
        """
        R = rot_matrix_2d(angle_deg)
        S = np.diag(reflection)

        p1_raw = p0 + shift
        p1 = np.mod(p1_raw, L)

        p2 = S @ (p1 - centre) + centre

        p3_raw = R @ (p2 - centre) + centre
        p3 = np.mod(p3_raw, L)

        return p0, p1_raw, p1, p2, p3_raw, p3

    def draw_box(ax):
        for cx in range(int(L)):
            for cy in range(int(L)):
                ax.add_patch(plt.Rectangle((cx, cy), 1, 1, facecolor='#F1EFE8',
                                            edgecolor='none', zorder=0))
        for k in range(int(L) + 1):
            ax.plot([0, L], [k, k], color='gray', linewidth=0.4, zorder=1)
            ax.plot([k, k], [0, L], color='gray', linewidth=0.4, zorder=1)
        ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, edgecolor='black',
                                    linewidth=2, zorder=5))

    def draw_rotation_arc(ax, p_start, angle_deg, color):
        if np.isclose(angle_deg % 360, 0):
            return
        radius = np.linalg.norm(p_start - centre)
        theta_start = np.arctan2(p_start[1] - centre, p_start[0] - centre)
        theta_end = theta_start + np.deg2rad(angle_deg)

        thetas = np.linspace(theta_start, theta_end, 40)
        arc_x = centre + radius * np.cos(thetas)
        arc_y = centre + radius * np.sin(thetas)

        line, = ax.plot(arc_x, arc_y, color=color, lw=1.8, zorder=4)
        line.set_path_effects(OUTLINE)

        ann = ax.annotate('', xy=(arc_x[-1], arc_y[-1]), xytext=(arc_x[-2], arc_y[-2]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                           zorder=4)
        ann.arrow_patch.set_path_effects(OUTLINE)

    def draw_step_arrow(ax, p_from, p_to, color, style='-'):
        if np.allclose(p_from, p_to):
            return
        ann = ax.annotate('', xy=p_to, xytext=p_from,
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                            linestyle=style), zorder=4)
        ann.arrow_patch.set_path_effects(OUTLINE)

    def draw_outline(ax, p, color, fill=False, lw=1.6, ls='-'):
        ax.add_patch(plt.Rectangle(p - 0.5, 1, 1, fill=fill,
                                    facecolor=color if fill else 'none',
                                    edgecolor=color, linewidth=lw, linestyle=ls,
                                    alpha=0.9 if fill else 1.0, zorder=3))

    p0_base = np.array(selected_cell_centre, dtype=float)
    n = len(examples)
    if nrows is None:
        nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.3 * ncols, 4.3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (shift, refl, angle) in enumerate(examples):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        shift_arr = np.array(shift, dtype=float)
        refl_arr = np.array(refl, dtype=float)

        p0, p1_raw, p1, p2, p3_raw, p3 = compute_all_steps(
            p0_base, shift_arr, refl_arr, angle)

        all_pts = [p0, p1_raw, p1, p2, p3_raw, p3]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = 0.6
        xmin, xmax = min(0.0, min(xs) - 0.5) - pad, max(L, max(xs) + 0.5) + pad
        ymin, ymax = min(0.0, min(ys) - 0.5) - pad, max(L, max(ys) + 0.5) + pad

        draw_box(ax)
        draw_outline(ax, p0, COL_ORIG, fill=True)

        shift_wrapped = not np.allclose(p1_raw, p1)
        if shift_wrapped:
            draw_outline(ax, p1_raw, COL_GHOST, fill=False, ls='--')
            #draw_step_arrow(ax, p0, p1_raw, COL_GHOST, style='--')
            #draw_step_arrow(ax, p1_raw, p1, COL_GHOST, style='--')
            draw_step_arrow(ax, p0, p1, COL_GHOST, style='-')
            draw_step_arrow(ax, p0, p1_raw, COL_GHOST, style='--')
        else:
            draw_step_arrow(ax, p0, p1, COL_SHIFT, style='-')
        draw_outline(ax, p1, COL_SHIFT, fill=False)

        draw_step_arrow(ax, p1, p2, COL_REFLECT, style='-')
        draw_outline(ax, p2, COL_REFLECT, fill=False)

        rotate_wrapped = not np.allclose(p3_raw, p3)
        if rotate_wrapped:
            draw_rotation_arc(ax, p2, angle, COL_GHOST)
            #draw_outline(ax, p3_raw, COL_GHOST, fill=False, ls='--')
            #draw_step_arrow(ax, p3_raw, p3, COL_GHOST, style='--')
            draw_step_arrow(ax, p3_raw, p3, COL_GHOST, style='--')
        else:
            draw_rotation_arc(ax, p2, angle, COL_ROTATE)

        draw_outline(ax, p3, COL_FINAL, fill=True)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'shift={shift}, refl={refl}, rot={angle}°', fontsize=9)

    # Hide any unused axes if examples doesn't fill the grid exactly
    for idx in range(n, nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis('off')

    legend_elems = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor=COL_ORIG,
               markersize=12, label='original'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='none',
               markeredgecolor=COL_SHIFT, markersize=12, label='[1] after shift'),
        #Line2D([0], [0], marker='s', color='none', markerfacecolor='none',
        #       markeredgecolor=COL_GHOST, markersize=12, linestyle='--',
        #       label='shift ghost'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='none',
               markeredgecolor=COL_REFLECT, markersize=12, label='[2] after reflect'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=COL_FINAL,
               markersize=12, label='[3] final (after rotation)'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])


    plt.savefig(filename,dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    plot_settings() # establish plot settings
    
    output_dir="example_outputs/snapshot_lightcone"
    # ensure output directory exists
    directory_path = Path(output_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    examples = [
        ((0, 0),   (1, 1),   0),
        ((0, 0),   (-1, 1),  0),
        ((0, 0),   (1, -1),  0),
        ((0, 0),   (-1, -1), 0),
        ((0, 0),   (1, 1),   90),
        ((0, 0),   (1, 1),   180),
        ((0, 0),   (1, 1),   270),
        ((-1, 0),  (1, 1),   0),
        ((0, -1),  (1, 1),   0),
        ((-1, -1), (1, 1),   90),
        ((-2, 0),  (1, -1),  180),
        ((-1, 0),  (-1, 1),  270),
        ((-2, -2), (-1, -1), 90),
        ((1, 0),   (-1, 1),  180),
        ((0, 1),   (1, -1),  270),
        ((-1, -2), (-1, 1),  0),
    ]
    
    plot_transform_grid(examples, L=3.0, selected_cell_centre=(0.5, 0.5),ncols=4, nrows=4, figsize=(9,9), filename=output_dir+"/snapshot_cell_repositioning.png")

