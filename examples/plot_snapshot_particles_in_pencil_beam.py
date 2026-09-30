#!/bin/env python
import sys
import glob
import numpy as np
import lightcone_io.particle_reader as pr
import unyt
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone


from extra_swift_lightcones.snapshot_lightcone import SnapshotBeam
from extra_swift_lightcones.swift_snapshot_redshift_conversion import flamingo_shell_redshift_file


from pathlib import Path

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


"""
Change this to just regular read in the particles in serial and then plot the pencil beam
"""


def plot_settings():
    """
    matplotlib settings for this example. 
    """
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
    plt.rcParams["legend.fontsize"]=9
    # Figure layour settings
    plt.rcParams["figure.constrained_layout.use"] =True
    plt.rcParams["figure.constrained_layout.h_pad"] =0.005
    plt.rcParams["figure.constrained_layout.w_pad"] =0.005
    plt.rcParams["figure.constrained_layout.hspace"]=0.005
    plt.rcParams["figure.constrained_layout.wspace"]=0.005
    # axes font settings
    mpl.rcParams['axes.labelsize']=9
    mpl.rcParams['figure.labelsize']=9
    mpl.rcParams['xtick.labelsize']=9
    mpl.rcParams['ytick.labelsize']=9
    # axes tick settings
    plt.rcParams["xtick.direction"] ='in'
    plt.rcParams["ytick.direction"] ='in'
    plt.rcParams["xtick.top"]=True
    plt.rcParams["ytick.right"]= True


def message(m):
    if (comm_rank == 0):
        print(m)


def in_lightcone(coords, angular_radius_deg, z_min, z_max, cosmo, buffer_length=None, buffer_shape="cube", beam_vector=(0.0, 0.0, 1.0)):
    """
    Determine which points fall inside the lightcone shell defined by an angular radius and a
    redshift range [z_min, z_max].

    Returns a boolean mask, shape (N,), True for points inside the lightcone wedge.

    :param  coords:             Cartesian comoving coordinates [Mpc], shape (N, 3)
    :type   coords:             np.ndarray
    :param  angular_radius_deg: angular radius [deg] of the cone
    :type   angular_radius_deg: float
    :param  z_min:              minimum redshift of the lightcone slice
    :type   z_min:              float
    :param  z_max:              maximum redshift of the lightcone slice
    :type   z_max:              float
    :param  cosmo:              object with a .z2r(z) method returning a comoving distance
    :type   cosmo:              object
    :param  buffer_length:      size [Mpc] of the buffer about each coordinate. If None, no buffer
    :type   buffer_length:      float
    :param  buffer_shape:       'cube', buffer_length is a cube sidelength. 'sphere', buffer_length is a sphere radius
    :type   buffer_shape:       str
    :param  beam_vector:        direction of the beam
    :type   beam_vector:        array-like, shape (3,)
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3: # test for coords shape
        raise ValueError("coords must be a vector with shape (N, 3)")
    
    beam_vector = np.asarray(beam_vector, dtype=float)
    if beam_vector.shape != (3,): # test for vector shape
        raise ValueError("beam_vector must be a vector with shape (3,) ")
    vector_norm = np.linalg.norm(beam_vector)
    if vector_norm == 0: # test for non-zero vector
        raise ValueError("beam_vector must be a nonzero vector")
    axis = beam_vector / vector_norm  # ensure unit vector
    

    half_angle = np.deg2rad(angular_radius_deg)

    chi_inner = cosmo.z2r(z_min).to_value("Mpc")
    chi_outer = cosmo.z2r(z_max).to_value("Mpc")
    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2) # distance from observer
    #r_perp = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)  # transverse offset from the axis
    
    along_axis = coords @ axis                      # scalar projection
    perp_vec = coords - along_axis[:, None] * axis   # vector rejection
    r_perp = np.linalg.norm(perp_vec, axis=1)
    
    # Angle from the cone axis
    angle_from_axis = np.arctan2(r_perp, coords[:, 2])
    
    if buffer_length is not None:
        # account of a buffer zone about each coordinate by drawing the a sphere of radius r_buffer
        if buffer_shape=="cube":
            # Assume buffer_length = cube sidelength
            # Assuming buffer radius = the longest diagonal from the centre of a cube
            r_buffer = (np.sqrt(3.0) / 2.0) * buffer_length
        elif buffer_shape=="sphere":
            # Assume buffer_length = radius of sphere
            r_buffer = buffer_length
        else:
            raise ValueError("buffer shape not recognised")
    else:
        r_buffer=0

    # if r_buffer > radius of shell, then assume coordinate is automatically within the lightcone
    with np.errstate(invalid="ignore"):
        angle_buffer = np.where(
            r > r_buffer,
            np.arcsin(np.clip(r_buffer / np.where(r > 0, r, np.inf), 0.0, 1.0)),
            np.pi,  # cube may extend in any direction; don't exclude on angle
        )
    
    in_radial_range = (r > chi_inner - r_buffer) & (r < chi_outer + r_buffer)
    in_forward_hemisphere = along_axis > - r_buffer
    in_angular_range = angle_from_axis <= half_angle + angle_buffer 
    
    m = in_radial_range & in_forward_hemisphere & in_angular_range
    
    return m


def plot_lightcone_projection_3panel(angular_radius_deg, z_min, z_max, cosmo,
                                      test_coords=None, test_coords_buffer=None,
                                      tol=1e-6,
                                      beam_vector=(0.0, 0.0, 1.0), axes=None,
                                      figsize=(7, 15), show=True,
                                      contours=False, contour_bins=40,
                                      contour_levels=4, contour_min_points=10):
    """
    Returns the three axes used for plotting (ax_xy, ax_xz, ax_yz).

    :param  angular_radius_deg: angular radius [deg] of the cone
    :type   angular_radius_deg: float
    :param  z_min:              minimum redshift of the lightcone slice
    :type   z_min:              float
    :param  z_max:              maximum redshift of the lightcone slice
    :type   z_max:              float
    :param  cosmo:              object with a .z2r(z) method returning a comoving distance 
                                    (an astropy-Quantity-like object supporting .to_value("Mpc"))
    :type   cosmo:              object
    :param  test_coords:        Cartesian comoving coordinates [Mpc] of test points, shape (N, 3)
    :type   test_coords:        np.ndarray
    :param  test_coords_buffer: buffer length passed through to in_lightcone
    :type   test_coords_buffer: float
    :param  tol:                relative/absolute tolerance used to detect points exactly on the lightcone's boundary
    :type   tol:                float
    :param  beam_vector:        beam pointing direction passed through to in_lightcone. The analytic wedge 
                                    geometry drawn in each panel assumes this is (0, 0, 1). Point classification 
                                    itself still uses the general beam_vector, but the drawn arcs/circles will 
                                    not match the true geometry for a tilted beam.
    :type   beam_vector:        array-like, shape (3,)
    :param  axes:               if given, must have length 3: (ax_xy, ax_xz, ax_yz). The x-z/y-z sharex linkage 
                                    is only set up when this function creates its own axes (axes=None). If you 
                                    pass your own axes, set up sharex between them yourself beforehand if desired.
    :type   axes:               array-like of 3 matplotlib.axes.Axes
    :param  figsize:            figure size, used only when axes is None
    :type   figsize:            tuple
    :param  show:               If True, show the figure
    :type   show:               boolean
    :param  contours:           If False, test points are drawn as individual scatter markers for every category. 
                                    If True, "inside" and (for large enough groups) "outside" are instead drawn 
                                    as smoothed density contour lines, see the category rules above. "edge" is 
                                    always scattered regardless of this flag.
    :type   contours:           boolean
    :param  contour_bins:       number of bins per axis for the 2D histogram underlying the contours. 
                                    Only used when contours=True
    :type   contour_bins:       int
    :param  contour_levels:     number of contour levels drawn per category. Only used when contours=True
    :type   contour_levels:     int
    :param  contour_min_points: minimum number of points a category needs (in a given panel) to be contoured 
                                    rather than scattered. Only used when contours=True
    :type   contour_min_points: int
    """

    def classify_points(coords, angular_radius_deg, z_min, z_max, cosmo, coords_buffer,
                         buffer_shape="sphere",
                         tol=1e-6, beam_vector=(0.0, 0.0, 1.0)):
        """
        Classify Cartesian comoving points relative to the 3D
        lightcone as being "inside", "outside" or on the "edge" of the lightcone

        Returns an array of str, shape (N,), with values in {"inside", "outside", "edge"}.

        :param  coords:             Cartesian comoving coordinates [Mpc], shape (N, 3)
        :type   coords:             np.ndarray
        :param  angular_radius_deg: angular radius [deg] of the cone. If >= 180, only the radial range is tested
        :type   angular_radius_deg: float
        :param  z_min:              minimum redshift of the lightcone slice
        :type   z_min:              float
        :param  z_max:              maximum redshift of the lightcone slice
        :type   z_max:              float
        :param  cosmo:              object with a .z2r(z) method returning a comoving distance
        :type   cosmo:              object
        :param  coords_buffer:      buffer length passed through to in_lightcone
        :type   coords_buffer:      float
        :param  buffer_shape:       'cube' or 'sphere', passed through to in_lightcone
        :type   buffer_shape:       str
        :param  tol:                relative/absolute tolerance used to detect points exactly on the boundary
        :type   tol:                float
        :param  beam_vector:        direction of the beam
        :type   beam_vector:        array-like, shape (3,)
        """
        coords = np.asarray(coords, dtype=float)

        

        if angular_radius_deg>=180:
            r = np.linalg.norm(coords, axis=1)
            chi_inner = cosmo.z2r(z_min).to_value("Mpc")
            chi_outer = cosmo.z2r(z_max).to_value("Mpc")
            inside_strict = (r >= chi_inner - coords_buffer) & (r <= chi_outer + coords_buffer)
            edge = (r==chi_inner) | (r==chi_outer)
        
        else:
            inside_strict = in_lightcone(coords, angular_radius_deg, z_min, z_max, cosmo,
                                      buffer_shape=buffer_shape,
                                      beam_vector=beam_vector, 
                                      buffer_length=coords_buffer
                                      )
        
            axis_arr = np.asarray(beam_vector, dtype=float)
            axis_arr = axis_arr / np.linalg.norm(axis_arr)
            half_angle = np.deg2rad(angular_radius_deg)
            chi_near = cosmo.z2r(z_min).to_value("Mpc")
            chi_far = cosmo.z2r(z_max).to_value("Mpc")

            r = np.linalg.norm(coords, axis=1)
            along_axis = coords @ axis_arr
            perp_vec = coords - along_axis[:, None] * axis_arr
            r_perp = np.linalg.norm(perp_vec, axis=1)
            angle = np.arctan2(r_perp, along_axis)

            on_radial_edge = np.isclose(r, chi_near, rtol=tol, atol=tol) | \
                              np.isclose(r, chi_far, rtol=tol, atol=tol)
            on_angular_edge = np.isclose(angle, half_angle, rtol=tol, atol=tol)

            edge = (along_axis > 0) & (
                (on_radial_edge & (angle <= half_angle + tol)) |
                (on_angular_edge & (r >= chi_near - tol) & (r <= chi_far + tol))
            )
        
        status = np.where(edge, "edge", np.where(inside_strict, "inside", "outside"))
        return status

    def corner_label(ax, text, z_min, z_max):
        label_text = f"{text}\n" + rf"${z_min:.3f} < z \leq {z_max:.3f}$"
        ax.text(0.03, 0.95, label_text, transform=ax.transAxes, ha="left", va="top",
                 fontsize=12, fontweight="bold", zorder=20)

    status_style = {
        "inside": "darkcyan",
        "outside": "darkmagenta",
        "edge": "darkgoldenrod",
    }
    scatter_kwargs={
        "s":10,
        "marker":".",
        "alpha":0.5,
        "edgecolor":"none"
    }
    def plot_status_points_or_contours(ax, horiz_vals, vert_vals, status):
        """
        Draw each status category with category-specific rules:
          - "edge"    : always individual scatter points, regardless of
                        `contours`.
          - "inside"  : density contours when `contours=True` (falling
                        back to scatter if there are too few points for
                        a meaningful contour); plain scatter otherwise.
          - "outside" : density contours when `contours=True` AND the
                        group is large enough (>= contour_min_points);
                        otherwise scatter.
        """
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        def scatter_category(label, color, zorder, kwargs=scatter_kwargs):
            mask = status == label
            if np.any(mask):
                ax.scatter(horiz_vals[mask], vert_vals[mask], color=color,
                           zorder=zorder, label=label, **kwargs)

        def contour_category(label, color):
            """
            Returns True if a contour was actually drawn, False if it
            fell back (too few points, or a degenerate/empty histogram).
            """
            mask = status == label
            n_pts = np.count_nonzero(mask)
            if n_pts < contour_min_points:
                return False

            hist, xedges, yedges = np.histogram2d(
                horiz_vals[mask], vert_vals[mask], bins=contour_bins,
                range=[xlim, ylim])
            hist = hist.T
            x_centers = 0.5 * (xedges[:-1] + xedges[1:])
            y_centers = 0.5 * (yedges[:-1] + yedges[1:])

            if hist.max() <= 0:
                return False

            ax.contour(x_centers, y_centers, hist, levels=contour_levels,
                       colors=color, linewidths=1.2, zorder=8)
            ax.plot([], [], color=color, linewidth=1.5, label=label)  # legend proxy
            return True

        # "edge" is always plotted as points.
        scatter_category("edge", status_style["edge"], zorder=5)

        # "inside": contour when requested, with a scatter fallback.
        if contours:
            if not contour_category("inside", status_style["inside"]):
                scatter_category("inside", status_style["inside"], zorder=10)
        else:
            scatter_category("inside", status_style["inside"], zorder=10)

        # "outside": contour only for large groups.
        if contours:
            if not contour_category("outside", status_style["outside"]):
                scatter_category("outside", status_style["outside"], zorder=1)
        else:
            scatter_category("outside", status_style["outside"], zorder=1)

    def draw_axial_slice(ax, chi_near, chi_far, half_angle, vert_coords_idx,
                          horiz_label, vert_label, panel_label, test_coords, status,
                          z_min, z_max, show_xlabel=True):
        """
        Draw a wedge cross-section panel containing the beam axis
        (used for the x-z and y-z panels).
        """
        theta = np.linspace(-half_angle, half_angle, 200)

        near_arc_horiz = chi_near * np.cos(theta)
        near_arc_vert = chi_near * np.sin(theta)
        far_arc_horiz = chi_far * np.cos(theta)
        far_arc_vert = chi_far * np.sin(theta)

        wedge_horiz = np.concatenate([far_arc_horiz, near_arc_horiz[::-1]])
        wedge_vert = np.concatenate([far_arc_vert, near_arc_vert[::-1]])
        shell_str = rf"lightcone shell (${z_min:.3f}<z\leq{z_max:.3f}$)"
        ax.fill(wedge_horiz, wedge_vert, color="darkgrey", alpha=0.6, zorder=1, label=shell_str)

        ax.plot(far_arc_horiz, far_arc_vert, color="black", zorder=2, label=r"$R(z_{\mathrm{max}})$")
        ax.plot([near_arc_horiz[-1], far_arc_horiz[-1]], [near_arc_vert[-1], far_arc_vert[-1]],
                color="black", zorder=2)
        ax.plot([near_arc_horiz[0], far_arc_horiz[0]], [near_arc_vert[0], far_arc_vert[0]],
                color="black", zorder=2)

        ax.plot(near_arc_horiz, near_arc_vert, color="black", lw=1.5,
                linestyle="--", zorder=2, label=r"$R(z_{\mathrm{min}})$")

        ax.plot([0, near_arc_horiz[-1]], [0, near_arc_vert[-1]], color="black", linestyle="--", zorder=2)
        ax.plot([0, near_arc_horiz[0]], [0, near_arc_vert[0]], color="black", linestyle="--", zorder=2)

        if show_xlabel:
            ax.set_xlabel(horiz_label)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel(vert_label)
        corner_label(ax, panel_label, z_min, z_max)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.set_aspect("auto")

        max_vert_extent = np.max(np.abs(far_arc_vert))
        max_horiz_extent = chi_far
        if test_coords is not None:
            max_vert_extent = max(max_vert_extent, np.max(np.abs(test_coords[:, vert_coords_idx])))
            max_horiz_extent = max(max_horiz_extent, np.max(test_coords[:, 2]))

        pad_h = 0.08 * max_horiz_extent
        pad_v = 0.15 * max_vert_extent

        if np.rad2deg(angular_radius_deg)>=90:
            ax.set_xlim(-max_horiz_extent -pad_h, max_horiz_extent + pad_h)
        else:
            ax.set_xlim(-pad_h, max_horiz_extent + pad_h)
        ax.set_ylim(-max_vert_extent - pad_v, max_vert_extent + pad_v)

        if test_coords is not None:
            plot_status_points_or_contours(ax, test_coords[:, 2],
                                            test_coords[:, vert_coords_idx], status)

    def draw_perpendicular_disk(ax, chi_near, chi_far, half_angle, test_coords, status,
                                 z_min, z_max):
        """
        Draw the on-sky projected footprint (x-y panel, perpendicular
        to the beam axis).
        """
        r_far = chi_far * np.sin(half_angle)
        r_near = chi_near * np.sin(half_angle)

        theta_full = np.linspace(0, 2 * np.pi, 200)
        far_x = r_far * np.cos(theta_full)
        far_y = r_far * np.sin(theta_full)
        near_x = r_near * np.cos(theta_full)
        near_y = r_near * np.sin(theta_full)

        shell_str = rf"lightcone shell (${z_min:.3f}<z\leq{z_max:.3f}$)"
        ax.fill(far_x, far_y, color="darkgrey", alpha=0.6, zorder=1, label=shell_str)
        ax.plot(far_x, far_y, color="black", zorder=2, label=r"$R(z_{\mathrm{max}})$")
        ax.plot(near_x, near_y, color="black", lw=1.5, linestyle="--", zorder=2,
                label=r"$R(z_{\mathrm{min}})$")

        ax.set_xlabel("x [cMpc]")
        ax.set_ylabel("y [cMpc]")
        corner_label(ax, "x-y", z_min, z_max)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.axvline(0, color="gray", lw=0.5, zorder=0)
        ax.set_aspect("equal")

        max_extent = r_far
        if test_coords is not None:
            max_extent = max(max_extent, np.max(np.abs(test_coords[:, 0])),
                              np.max(np.abs(test_coords[:, 1])))
        pad = 0.15 * max_extent
        ax.set_xlim(-max_extent - pad, max_extent + pad)
        ax.set_ylim(-max_extent - pad, max_extent + pad)

        if test_coords is not None:
            plot_status_points_or_contours(ax, test_coords[:, 0], test_coords[:, 1], status)

    half_angle = np.deg2rad(angular_radius_deg)
    chi_near = cosmo.z2r(z_min).to_value("Mpc")
    chi_far = cosmo.z2r(z_max).to_value("Mpc")

    if axes is None:
        fig = plt.figure(figsize=figsize)
        ax_xy = fig.add_subplot(3, 1, 1)
        ax_xz = fig.add_subplot(3, 1, 2)
        ax_yz = fig.add_subplot(3, 1, 3, sharex=ax_xz)
        fig.subplots_adjust(hspace=0.08)
    else:
        ax_xy, ax_xz, ax_yz = axes
        fig = ax_xy.figure

    status = None
    if test_coords is not None:
        test_coords = np.asarray(test_coords, dtype=float)
        status = classify_points(test_coords, angular_radius_deg, z_min, z_max, cosmo,
                                  tol=tol, beam_vector=beam_vector,
                                  coords_buffer=test_coords_buffer)

    draw_perpendicular_disk(ax_xy, chi_near, chi_far, half_angle, test_coords, status,
                             z_min, z_max)
    draw_axial_slice(ax_xz, chi_near, chi_far, half_angle, vert_coords_idx=0,
                      horiz_label="z [cMpc]", vert_label="x [cMpc]",
                      panel_label="x-z", test_coords=test_coords, status=status,
                      z_min=z_min, z_max=z_max, show_xlabel=False)
    draw_axial_slice(ax_yz, chi_near, chi_far, half_angle, vert_coords_idx=1,
                      horiz_label="z [cMpc]", vert_label="y [cMpc]",
                      panel_label="y-z", test_coords=test_coords, status=status,
                      z_min=z_min, z_max=z_max, show_xlabel=True)

    combined = {}
    for ax in (ax_xy, ax_xz, ax_yz):
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in combined:
                combined[l] = h

    fig.legend(combined.values(), combined.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=False)

    if show:
        plt.tight_layout()
        plt.show()

    return ax_xy, ax_xz, ax_yz


def plot_radial_dist(local_array, comm, shell_nr, redshift_range, ang_radius_deg, ptype, beam_vec,base_rank=0, axes_plots=True, scatter_plots=True,
    output_dir='./', 
    ):
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    k=ptype
    N_local = np.shape(local_array)[0]
    N_local_all =  np.asarray(comm.gather(N_local, root=base_rank))
    
    if comm_rank == base_rank:
        N_local_all = np.asarray(N_local_all)
        # Number of elements, not number of rows
        if np.ndim(local_array)>1:
            counts = N_local_all * np.shape(local_array)[1]
            gathered = np.empty((np.sum(N_local_all), 3), dtype=local_array.dtype)
        else:
            counts = N_local_all
            gathered = np.empty(np.sum(N_local_all), dtype=local_array.dtype)
        
        displacements = np.zeros(len(counts), dtype=np.int64)
        displacements[1:] = np.cumsum(counts[:-1])
        print(f"\nnumber of {k} returned: {np.sum(N_local_all)}")
    else:
        counts = None
        displacements = None
        gathered = None
    
    comm.barrier()
    
    comm.Gatherv(
        local_array,
        [gathered, counts, displacements, MPI._typedict[local_array.dtype.char]],
        root=base_rank,
    )
    
    comm.barrier()
    if comm_rank==base_rank:
        if axes_plots:

            label_text = rf"${redshift_range[0]:.3f} < z \leq {redshift_range[1]:.3f}$"

            # Creating plot
            fig = plt.figure(figsize =(7, 4))
            z= np.linalg.norm(gathered, axis=1)
            n, edges = np.histogram(z, bins = 200)
            mids = 0.5*(edges[:-1] + edges[1:])
            plt.plot(mids, n)
            plt.xlabel("Comoving Distance")
            plt.ylabel("Number of particles")
            fig.text(0.03, 0.95, label_text, transform=fig.transFigure, ha="left", va="top",
                     fontsize=10, fontweight="bold", zorder=20)
            fig_name=f"{k}_shell{int(shell_nr)}_radius"
            
            print(f"{np.min(mids), np.max(mids)}\n{output_dir+"/"+fig_name}")
            
            plt.savefig(output_dir+"/"+fig_name+".png",dpi=300, bbox_inches='tight')
            plt.close()


            del z, n, edges

            fig = plt.figure(figsize =(7, 4))

            z_shift=0

            for i in np.array([0,1,2])[::-1]:
                z= gathered[:, i]
                n, edges = np.histogram(z, bins = 200)
                mids = 0.5*(edges[:-1] + edges[1:])
                #if i==2:
                #    z_shift = np.median(mids)
                #
                #if i<2:
                #    mids += z_shift
                plt.plot(mids, n, label=f"axis: {np.array(['x', 'y', 'z'])[int(i)]}")

            plt.xlabel("Comoving Distance")
            plt.ylabel("Number of particles")

            fig.text(0.03, 0.95, label_text, transform=fig.transFigure, ha="left", va="top",
                     fontsize=10, fontweight="bold", zorder=20)
            plt.legend(loc="best")

            fig_name=f"{k}_shell{int(shell_nr)}_axis_coords"
            plt.savefig(output_dir+"/"+fig_name+".png",dpi=300, bbox_inches='tight')
            plt.close()

    comm.barrier()
    if comm_rank==base_rank:
        
        if scatter_plots:
            box_res="L1000N0900"
            sim_name="HYDRO_FIDUCIAL"
            base_dir="/cosma8/data/dp004/flamingo/Runs/{sim}".format(sim=box_res+"/"+sim_name)
            snapshot_dir = "{base_name}/snapshots".format(base_name=base_dir)
            plot_cosmo=Snapshot_Cosmology_For_Lightcone(snapshot_dir)
            #output_dir="./example_outputs/shell_examples"
            ax=plot_lightcone_projection_3panel(ang_radius_deg, 
                                              #SB.last_snap['z_updated'][0][0], SB.last_snap['z_updated'][-1][1], 
                                              redshift_range[0], redshift_range[1],
                                              plot_cosmo,
                                              test_coords=gathered, test_coords_buffer=0.,
                                              tol=1e-6,
                                              beam_vector=beam_vec, axes=None,
                                              figsize=(7, 7), show=False,
                                              contours=False, contour_bins=100,
                                              contour_levels=5, contour_min_points=250
                                              )
            
            fig_name=f"{k}_shell{int(shell_nr)}_scatter2"
            plt.savefig(output_dir+"/"+fig_name+".png",dpi=300, bbox_inches='tight')
            plt.close()



if __name__ == "__main__":

    plot_settings()

    

    box_res="L1000N0900"
    sim_name="HYDRO_FIDUCIAL"

    if comm_rank==0:

        box_res="L1000N0900"
        sim_name="HYDRO_FIDUCIAL"

        shell_number = 0
        #snapshot_number=76

        shell_redshifts="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/shell_redshifts.txt"
        redshift_range = tuple(np.loadtxt(shell_redshifts, delimiter=",")[shell_number, :])
        redshift_range=tuple([0.05, 0.1])
        print(f"\nshell: {shell_number}, \t redshift: {redshift_range[0]} - {redshift_range[1]}")

        ell_min=4
        ang_radius_deg=np.rad2deg(np.pi/ell_min)
        print(f"\nmin ell: {ell_min}\nangular radius: {ang_radius_deg} [deg]\n",flush=True)

        beam_vector=(0,0,1)
    else:
        box_res=None
        sim_name=None
        beam_vector=None
        redshift_range=None
        ang_radius_deg=None
        shell_number=None
    
    box_res = comm.bcast(box_res)
    sim_name=comm.bcast(sim_name)
    beam_vector=comm.bcast(beam_vector)
    ang_radius_deg=comm.bcast(ang_radius_deg)
    redshift_range=comm.bcast(redshift_range)
    shell_number=comm.bcast(shell_number)
    SnapA = SnapshotAllSky(boxsize_resolution=box_res, simulation_name=sim_name)
    
    property_names = (
        "Coordinates",
        "ParticleIDs"
        )


    AllSky_particle_data=SnapA.place_snapshot_particles_in_shell(
                        lightcone_redshift_range=redshift_range, 
                        property_names=list(property_names),  
                        particle_types=["PartType5"], 
                        comm=comm,
        )
    
    
    comm.barrier()
    if comm_rank==0:
        print(AllSky_particle_data["PartType5"].keys())
    base_dir="/cosma8/data/dp004/flamingo/Runs/{sim}".format(sim=box_res+"/"+sim_name)
    snapshot_dir = "{base_name}/snapshots".format(base_name=base_dir)
    plot_cosmo=Snapshot_Cosmology_For_Lightcone(snapshot_dir)
    ptype="PartType5"
    local_array = AllSky_particle_data[ptype]["Coordinates"][AllSky_particle_data[ptype]["ParticleIDs"].value>=0, :].to_value("Mpc")
    plot_radial_dist(
        local_array, comm, shell_number, redshift_range, ang_radius_deg=180, 
        ptype="PartType5", 
        beam_vec=beam_vector, 
        base_rank=0, axes_plots=True, scatter_plots=True,
        output_dir="example_outputs/all_sky_examples",
        )
