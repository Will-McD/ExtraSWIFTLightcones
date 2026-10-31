#!/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from pathlib import Path
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import cmasher as cmr # noqa: F401 registers the "cmr.*" colour maps used by name

from swiftlet.example_plot_settings import plot_settings, cmap_MCG
from swiftlet.simulation_suites import Simulation

"""
Example script to show the redshift ranges a snapshot populates within a lightcone, for both 
all-sky lightcone (SnapshotAllSky) and a beam (SnapshotBeam).

For a slice through the x-z plane of a lightcone for FLAMINGO L1_m9, overlay the redshift 
range of each snapshot on the lightcones regular lattice of tiles (the tiles sidelength is 
the same as snapshot boxes). 
"""

# use the FLAMINGO fiducial cosmology, build it as an astropy cosmology object
FLAMINGO_COSMOLOGY = FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0=0.0486, Tcmb0=2.7255, Neff=3.044, m_nu=[0.06, 0., 0.] * u.eV)

def in_lightcone(s, z, r_max, beam_vector=None, half_angle_deg=None):
    """
    Check if point is inside the lightcone by comparing against the maximum comoving
    distance from the observer and if a beam, then also compare against the
    direction of the beam and the beams angular radius. 
    
    Returns a boolean array, True if in the lightcone and False otherwise

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

    # make position on x-z plane to 3D by adding dimension of 0's
    pos = np.stack([s, np.zeros_like(s), z], axis=-1)
    
    # dist from observer
    r = np.linalg.norm(pos, axis=-1) 
    
    # check if inside max comoving dist
    inside = r <= r_max 
    
    # if a beam compare against beams radius and direction
    if beam_vector is not None: 
        v = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_angle = np.where(r > 0, (pos @ v) / r, 1.0)
        inside &= cos_angle >= np.cos(np.deg2rad(half_angle_deg))
    return inside

def beam_in_slice(beam_vector):
    """
    Returns the direction of a beam in the slice, as a unit vector in (x, z). 
    The beam must lie in the plane defined for y = 0.

    :param  beam_vector:    direction of the beam
    :type   beam_vector:    array-like, shape (3,)
    """
    v = np.asarray(beam_vector, dtype=float) / np.linalg.norm(beam_vector)
    if not np.isclose(v[1], 0.):
        raise ValueError("the beam must lie in the plane y = 0 to be drawn in this slice")
    return np.array([v[0], v[2]])

def snapshot_shells(z_max, simulation, cosmo=FLAMINGO_COSMOLOGY):
    """
    Identify the snapshots used to populate a lightcone from z=0 to z_max, from the lowest to the highest redshift, and
    the comoving distance [Mpc] range each one populates.

    Returns a tuple (snapshot numbers, redshift midpoint, redshift edges, comoving distance edges)

    :param  z_max:      maximum redshift of the lightcone
    :type   z_max:      float
    :param  simulation: the simulation, giving its snapshot redshifts
    :type   simulation: swiftlet.simulation_suites.Simulation
    :param  cosmo:      cosmology used for the comoving distances
    :type   cosmo:      astropy cosmology object
    """
    
    snapshot_numbers, redshifts = simulation.snapshot_redshifts()
    used, z_edges = [], [0.]
    
    # the lowest redshift snapshot is last
    for snap_nr in snapshot_numbers[::-1]:
        z_lo, z_hi = simulation.snapshot_redshift_range(snap_nr)
        if z_hi <= 0. or z_lo >= z_max:
            # skip if outside range
            continue 
        
        used.append(snap_nr)
        z_edges.append(min(z_hi, z_max))
    
    used = np.array(used)
    r_edges = cosmo.comoving_distance(z_edges).to_value("Mpc")
    
    return used, redshifts[used], np.array(z_edges), r_edges

def draw_shell_panel(ax, L, r_edges, extent, beam_vector=None, half_angle_deg=None, label_ray_deg=None,
                     resolution=900):
    """
    Draw the box tiles in the slice y = 0, coloured by the snapshot populating each part of the lightcone.

    Returns a tuple of (colour map used for the snapshots, angle [rad] of the ray to label the snapshots along).

    :param  ax:             axes to plot onto
    :type   ax:             matplotlib.axes._axes.Axes
    :param  L:              box sidelength [Mpc]
    :type   L:              float
    :param  r_edges:        comoving distance [Mpc] edges of each snapshot's shell, see snapshot_shells
    :type   r_edges:        np.ndarray
    :param  extent:         (s_min, s_max, z_min, z_max) shown, in units of L
    :type   extent:         tuple
    :param  beam_vector:    direction of the beam. None for all-sky
    :type   beam_vector:    array-like, shape (3,)
    :param  half_angle_deg: angular radius [deg] of the beam
    :type   half_angle_deg: float
    :param  label_ray_deg:  angle [deg] from the x axis of the ray the snapshots are labelled along.
                                If None, the beam axis, or 235 deg for all-sky
    :type   label_ray_deg:  float
    :param  resolution:     number of pixels along each axis
    :type   resolution:     int
    """
    n_snap = len(r_edges) - 1
    r_max = r_edges[-1] / L
    cmap = ListedColormap(cmap_MCG(np.linspace(0, 1, n_snap)[::-1]))

    s_min, s_max, z_min, z_max = extent
    s, z = np.meshgrid(np.linspace(s_min, s_max, resolution), np.linspace(z_min, z_max, resolution))
    
    # convert points on a x-z planeay y=0 to 3D by adding dimension of 0's
    slice_to_3d = np.stack([s, np.zeros_like(s), z], axis=-1)
    r = np.linalg.norm(slice_to_3d) # 3D distance 
    # distance of each pixel in the plane y = 0 from the observer, in units of L
    r = np.hypot(s, z)
    
    inside = in_lightcone(s, z, r_max, beam_vector, half_angle_deg)

    # index of the snapshot populating each pixel, from the lowest redshift
    shell = np.clip(np.searchsorted(r_edges / L, r, side="right") - 1, 0, n_snap - 1)

    image = np.where(inside, shell, np.nan)
    ax.pcolormesh(s, z, image, cmap=cmap, vmin=-0.5, vmax=n_snap - 0.5,
                  alpha=0.7, 
                  shading="auto", rasterized=True, zorder=1)

    # box tiles crossed by the plane y = 0: L x L squares
    for mx in range(int(np.floor(s_min)) - 1, int(np.ceil(s_max)) + 2):
        ax.axvline(mx - 0.5, color='k', lw=1.4, zorder=3)
    for mz in range(int(np.floor(z_min)) - 1, int(np.ceil(z_max)) + 2):
        ax.axhline(mz - 0.5, color='k', lw=1.4, zorder=3)

    # boundaries between snapshots, where the snapshot of neighbouring pixels in the lightcone differs
    changes = np.zeros_like(inside)
    changes[:, 1:] |= (shell[:, 1:] != shell[:, :-1]) & inside[:, 1:] & inside[:, :-1]
    changes[1:, :] |= (shell[1:, :] != shell[:-1, :]) & inside[1:, :] & inside[:-1, :]
    ax.contour(s, z, changes.astype(float), levels=[0.5], colors="white", linewidths=0.5, zorder=2)

    # outline of the lightcone and the observer
    if beam_vector is None:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), color='k', lw=1.0, zorder=4)
        ray_angle = np.deg2rad(235. if label_ray_deg is None else label_ray_deg)
    else:
        axis = beam_in_slice(beam_vector)
        half_angle = np.deg2rad(half_angle_deg)
        angles = np.arctan2(axis[1], axis[0]) + np.linspace(-half_angle, half_angle, 100)
        edge = r_max * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        ax.plot(np.concatenate([[0], edge[:, 0], [0]]), np.concatenate([[0], edge[:, 1], [0]]), color='k', lw=1.0, zorder=4)

        # beam axis, with a dot where the snapshot changes
        ax.plot([0, r_max * axis[0]], [0, r_max * axis[1]], color='k', lw=1.2, ls="--", zorder=4)
        r_change = r_edges[1:-1] / L
        ax.plot(r_change * axis[0], r_change * axis[1], "o", ms=3.5, mfc="white", mec='k', mew=1.0, zorder=6)
        ray_angle = np.arctan2(axis[1], axis[0]) if label_ray_deg is None else np.deg2rad(label_ray_deg)
    ax.plot(0, 0, "o", ms=6, mfc="white", mec='k', mew=1.2, zorder=7)

    return cmap, ray_angle

def label_snapshots(ax, snapshot_numbers, r_edges, L, ray_angle, offset_deg=0., min_radius=0.1):
    """
    Label each snapshot's shell with its snapshot number. 
    Place the half way through the shell along a ray from the observer and  neighbouring labels 
    alternate either side of the ray. 
    Shells closer to the observer than min_radius (in units of the lightcone radius) are left unlabelled

    :param  ax:                 axes to plot onto
    :type   ax:                 matplotlib.axes._axes.Axes
    :param  snapshot_numbers:   snapshot number of each shell
    :type   snapshot_numbers:   np.ndarray
    :param  r_edges:            comoving distance [Mpc] edges of each snapshot's shell
    :type   r_edges:            np.ndarray
    :param  L:                  box sidelength [Mpc]
    :type   L:                  float
    :param  ray_angle:          angle [rad] of the ray from the x axis
    :type   ray_angle:          float
    :param  offset_deg:         angle [deg] either side of the ray to alternate the labels between
    :type   offset_deg:         float
    :param  min_radius:         closest shell to the observer to label, as a fraction of the lightcone radius
    :type   min_radius:         float
    """
    
    # get mid point 
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1]) / L
    
    # iterate through snapshots
    for n, (snap_nr, r) in enumerate(zip(snapshot_numbers, r_mid)):
        if r < min_radius * r_edges[-1] / L:
            continue # too close to the observer to label
        
        # find correct angle and place label
        angle = ray_angle + np.deg2rad(offset_deg) * (1 if n % 2 else -1)
        
        ax.text(r * np.cos(angle), r * np.sin(angle), f"{snap_nr}", ha="center", va="center", fontsize=8,
                color='k', zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))

def plot_snapshot_shells(z_max=0.5, simulation=None, L=1000., half_angle_deg=15., figsize=(5,5), filename_format="./snapshot_shells_{lc_name}.png"):
    """
    Plot the tiles of an all-sky lightcone and a beam, coloured by the snapshot populating each part of the lightcone.

    :param  z_max:              maximum redshift of the lightcone
    :type   z_max:              float
    :param  simulation:         the simulation, giving its snapshot redshifts. If None, FLAMINGO L1000N1800
    :type   simulation:         swiftlet.simulation_suites.Simulation
    :param  L:                  box sidelength [Mpc]
    :type   L:                  float
    :param  beam_vector:        direction of the beam, in the plane y = 0
    :type   beam_vector:        array-like, shape (3,)
    :param  half_angle_deg:     angular radius [deg] of the beam
    :type   half_angle_deg:     float
    :param  figsize:            size of the figure
    :type   figsize:            tuple
    :param  filename:           path to write the plot to
    :type   filename:           str
    """
    
    # if no simulation given then assume fiducial flamingo L1_m9 for example
    if simulation is None:
        simulation = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", simulation_suite_name="flamingo")
    
    # make 2 different beams, one that is on-axis (0,0,1) and one that is off-axis (1,0,1)
    beam_vector=[(1, 0, 1),(0, 0, 1)]

    # get snapshot shells info: number, redshift, redshift and radius ranges
    snapshot_numbers, redshifts, z_edges, r_edges = snapshot_shells(z_max, simulation)
    r_max = r_edges[-1] / L

    # beam extent: the beam's bounding box with a margin, as for the all-sky lightcone
    axis = [beam_in_slice(beam_vector[0]), beam_in_slice(beam_vector[1])]
    angles = [
        np.arctan2(axis[0][1], axis[0][0]) + np.linspace(-1, 1, 100) * np.deg2rad(half_angle_deg), 
        np.arctan2(axis[1][1], axis[1][0]) + np.linspace(-1, 1, 100) * np.deg2rad(half_angle_deg)
    ]
    beam_s = [np.concatenate([[0], r_max * np.cos(angles[0])]), np.concatenate([[0], r_max * np.cos(angles[1])])]
    beam_z = [np.concatenate([[0], r_max * np.sin(angles[0])]), np.concatenate([[0], r_max * np.sin(angles[1])])]
    pad = 0.08 * r_max
    
    # the lightcones to draw diagram for: all-sky, on-axis beam and off-axis beam
    
    lightcones = [
        ("SnapshotAllSky", None, (-r_max - pad, r_max + pad, -r_max - pad, r_max + pad)),
        (f"SnapshotBeam {tuple(beam_vector[0])}", beam_vector[0],
         (beam_s[0].min() - pad, beam_s[0].max() + pad, beam_z[0].min() - pad, beam_z[0].max() + pad)),
        (f"SnapshotBeam {tuple(beam_vector[1])}", beam_vector[1],
         (beam_s[1].min() - pad, beam_s[1].max() + pad, beam_z[1].min() - pad, beam_z[1].max() + pad)),
    ]
    lightcone_names = ["fullsky", "beam_onaxis", "beam_offaxis"]
    ratios = [(e[1] - e[0]) / (e[3] - e[2]) for _, _, e in lightcones]

    counter=0
    
    for (name, beam, extent) in lightcones:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
        cmap, ray_angle = draw_shell_panel(ax, L, r_edges, extent, beam_vector=beam, half_angle_deg=half_angle_deg)
        label_snapshots(ax, snapshot_numbers, r_edges, L, ray_angle, offset_deg=0. if beam is None else 0.5 * half_angle_deg)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$x$ [$L$]")
        ax.set_ylabel(r"$z$ [$L$]")

        # snapshot redshift ranges, one colour per snapshot
        norm = BoundaryNorm(z_edges, cmap.N)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                            ax=ax, location="right", shrink=0.7,
                            alpha=1., spacing="proportional", drawedges=True)
        cbar.dividers.set_color("white")
        cbar.dividers.set_linewidth(0.5)
        cbar.set_ticks(redshifts, labels=[f"{snap_nr} ($z$={z:g})" for snap_nr, z in zip(snapshot_numbers, redshifts)])

        cbar.set_label(f"snapshot number (midplane redshift)")

        plt.savefig(filename_format.format(lc_name=lightcone_names[counter]), dpi=300, bbox_inches="tight")

        plt.close()
        counter+=1


if __name__ == "__main__":

    plot_settings() # establish plot settings

    output_dir = "example_outputs/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    simulation = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", simulation_suite_name="flamingo")
    
    plot_snapshot_shells(z_max=0.5, 
        simulation=simulation, 
        L=1000., 
        half_angle_deg=15., 
        filename_format=output_dir + "/snapshot_shells_{lc_name}.png", 
        )