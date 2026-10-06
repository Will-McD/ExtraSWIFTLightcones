#!/bin/env python
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.layout_engine
import matplotlib.pyplot as plt
from pathlib import Path
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone

from swiftlet.snapshot_lightcone import SnapshotBeam


"""
Place snapshot particles in pencil beam lightcone shells in serial, reading the snapshot files in chunks, 
and plot where they are relative to the geometry of the beam, for beams pointing in different directions.
The plots are drawn in the frame of each beam, with the line of sight along the horizontal axis.
"""


def plot_settings():
    """
    Set the matplotlib settings for this example. 
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


def beam_frame(beam_vector):
    """
    Orthonormal frame of a beam: two axes across the beam and the line of sight. For a beam along 
    the z-axis this is the x, y, z frame.

    Returns a tuple of (frame, axis_names). frame has shape (3, 3), with rows (e1, e2, line of sight), 
    so coords @ frame.T gives coordinates in the beam's frame. axis_names are the names of the 
    three axes, x, y or z (with a minus sign if reversed) when the axis is along one, otherwise 
    e1, e2 and "los".

    :param  beam_vector:    direction of the beam
    :type   beam_vector:    array-like, shape (3,)
    """
    los = np.asarray(beam_vector, dtype=float)
    los = los / np.linalg.norm(los)
    reference = np.array([1.0, 0.0, 0.0]) if abs(los[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = reference - (reference @ los) * los
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(los, e1)
    frame = np.array([e1, e2, los])

    axis_names = []
    for vec, generic_name in zip(frame, (r"$e_1$", r"$e_2$", "los")):
        axis = np.argmax(np.abs(vec))
        if np.isclose(abs(vec[axis]), 1.0):
            axis_names.append(("" if vec[axis] > 0 else "-") + "xyz"[axis])
        else:
            axis_names.append(generic_name)
    return frame, axis_names


def beam_label(beam_vector):
    """
    Short label of a beam direction, used in titles and output filenames, e.g. "1_1_1".

    :param  beam_vector:    direction of the beam
    :type   beam_vector:    array-like, shape (3,)
    """
    return "_".join(f"{v:g}" for v in beam_vector)


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
    Plot the lightcone's geometry across three stacked panels (one per
    row), in the frame of the beam (see beam_frame): the projection onto 
    the plane across the beam (e1-e2), and two slices along the beam (e1-los and e2-los). 
    For a beam along the z-axis these are the x-y, x-z and y-z planes.

    The two slices along the beam both use the beam's line-of-sight coordinate
    as their horizontal axis and share that x-axis (only labeled on
    the bottom panel). The panel across the beam
    shows the wedge's projected on-sky footprint as a filled disk.

    Each panel is labeled in its top-left corner with its projection
    plane and the redshift range, and all three panels share a single
    combined legend below the figure.

    Test points are classified once (inside/outside/edge of the full 3D
    lightcone) and displayed in all three panels using category-specific
    rules:
        "edge":     always individual scatter points, regardless of contours.
        "inside":   density contours when contours=True (falling back to scatter if there are 
                        too few points for a meaningful contour); plain scatter when contours=False.
        "outside":  density contours when contours=True AND the group is large enough 
                        (>= contour_min_points); otherwise scatter.

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
    :param  beam_vector:        direction of the beam. The test points are plotted in the beam's frame
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
        Classify Cartesian comoving points (N, 3) relative to the 3D
        lightcone defined by angular_radius_deg, z_min, z_max, using in_lightcone
        for the inside/outside test and a tolerance-based check for exact
        boundary membership.

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

    def corner_label(ax, text):
        """
        Label the top-left corner of a panel with its projection plane, on a white background 
        so it stays readable over the lightcone and the points.

        :param  ax:     axes to label
        :type   ax:     matplotlib.axes._axes.Axes
        :param  text:   projection plane, e.g. "x-y"
        :type   text:   str
        """
        ax.text(0.02, 0.96, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold", zorder=20,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

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
            "edge":     always individual scatter points, regardless of contours.
            "inside":   density contours when contours=True (falling back to scatter if there are 
                            too few points for a meaningful contour); plain scatter otherwise.
            "outside":  density contours when contours=True AND the group is large enough 
                            (>= contour_min_points); otherwise scatter.

        :param  ax:         axes to plot onto
        :type   ax:         matplotlib.axes._axes.Axes
        :param  horiz_vals: horizontal coordinates of the points
        :type   horiz_vals: np.ndarray
        :param  vert_vals:  vertical coordinates of the points
        :type   vert_vals:  np.ndarray
        :param  status:     status of each point, "inside", "outside" or "edge"
        :type   status:     np.ndarray
        """
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        def scatter_category(label, color, zorder, kwargs=scatter_kwargs):
            """
            Scatter the points of one status category.

            :param  label:  status category
            :type   label:  str
            :param  color:  colour of the points
            :type   color:  str
            :param  zorder: zorder of the points
            :type   zorder: int
            :param  kwargs: keyword arguments passed to scatter
            :type   kwargs: dict
            """
            mask = status == label
            if np.any(mask):
                ax.scatter(horiz_vals[mask], vert_vals[mask], color=color,
                           zorder=zorder, label=label, **kwargs)

        def contour_category(label, color):
            """
            Draw density contours of the points of one status category.

            Returns True if a contour was actually drawn, False if it
            fell back (too few points, or a degenerate/empty histogram).

            :param  label:  status category
            :type   label:  str
            :param  color:  colour of the contours
            :type   color:  str
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

        :param  ax:                 axes to plot onto
        :type   ax:                 matplotlib.axes._axes.Axes
        :param  chi_near:           comoving distance [Mpc] of the near side of the shell
        :type   chi_near:           float
        :param  chi_far:            comoving distance [Mpc] of the far side of the shell
        :type   chi_far:            float
        :param  half_angle:         angular radius [rad] of the cone
        :type   half_angle:         float
        :param  vert_coords_idx:    index of the coordinate on the vertical axis
        :type   vert_coords_idx:    int
        :param  horiz_label:        label of the horizontal axis
        :type   horiz_label:        str
        :param  vert_label:         label of the vertical axis
        :type   vert_label:         str
        :param  panel_label:        projection plane shown in the corner label
        :type   panel_label:        str
        :param  test_coords:        Cartesian comoving coordinates [Mpc] of test points, shape (N, 3)
        :type   test_coords:        np.ndarray
        :param  status:             status of each test point
        :type   status:             np.ndarray
        :param  z_min:              minimum redshift
        :type   z_min:              float
        :param  z_max:              maximum redshift
        :type   z_max:              float
        :param  show_xlabel:        If True, show the horizontal axis label and tick labels
        :type   show_xlabel:        boolean
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
        corner_label(ax, panel_label)
        ax.axhline(0, color="gray", lw=0.5, zorder=0)
        ax.set_aspect("auto")

        max_vert_extent = np.max(np.abs(far_arc_vert))
        max_horiz_extent = chi_far
        if test_coords is not None:
            max_vert_extent = max(max_vert_extent, np.max(np.abs(test_coords[:, vert_coords_idx])))
            max_horiz_extent = max(max_horiz_extent, np.max(test_coords[:, 2]))

        pad_h = 0.08 * max_horiz_extent
        pad_v = 0.15 * max_vert_extent

        if half_angle >= np.pi / 2:
            ax.set_xlim(-max_horiz_extent -pad_h, max_horiz_extent + pad_h)
        else:
            ax.set_xlim(-pad_h, max_horiz_extent + pad_h)
        ax.set_ylim(-max_vert_extent - pad_v, max_vert_extent + pad_v)

        if test_coords is not None:
            plot_status_points_or_contours(ax, test_coords[:, 2],
                                            test_coords[:, vert_coords_idx], status)

    def draw_perpendicular_disk(ax, chi_near, chi_far, half_angle, test_coords, status,
                                 z_min, z_max, axis_names):
        """
        Draw the on-sky projected footprint (x-y panel, perpendicular
        to the beam axis).

        :param  ax:             axes to plot onto
        :type   ax:             matplotlib.axes._axes.Axes
        :param  chi_near:       comoving distance [Mpc] of the near side of the shell
        :type   chi_near:       float
        :param  chi_far:        comoving distance [Mpc] of the far side of the shell
        :type   chi_far:        float
        :param  half_angle:     angular radius [rad] of the cone
        :type   half_angle:     float
        :param  test_coords:    Cartesian comoving coordinates [Mpc] of test points, shape (N, 3)
        :type   test_coords:    np.ndarray
        :param  status:         status of each test point
        :type   status:         np.ndarray
        :param  z_min:          minimum redshift
        :type   z_min:          float
        :param  z_max:          maximum redshift
        :type   z_max:          float
        :param  axis_names:     names of the beam frame axes (e1, e2, line of sight)
        :type   axis_names:     list of str
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

        ax.set_xlabel(f"{axis_names[0]} [cMpc]")
        ax.set_ylabel(f"{axis_names[1]} [cMpc]")
        corner_label(ax, f"{axis_names[0]}-{axis_names[1]}")
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
        fig = plt.figure(figsize=figsize, layout="constrained")
        ax_xy = fig.add_subplot(3, 1, 1)
        ax_xz = fig.add_subplot(3, 1, 2)
        ax_yz = fig.add_subplot(3, 1, 3, sharex=ax_xz)
    else:
        ax_xy, ax_xz, ax_yz = axes
        fig = ax_xy.figure

    # plot in the frame of the beam, where the line of sight is the third axis
    frame, (e1_name, e2_name, los_name) = beam_frame(beam_vector)

    status = None
    if test_coords is not None:
        test_coords = np.asarray(test_coords, dtype=float) @ frame.T
        status = classify_points(test_coords, angular_radius_deg, z_min, z_max, cosmo,
                                  tol=tol, beam_vector=(0.0, 0.0, 1.0),
                                  coords_buffer=test_coords_buffer)

    draw_perpendicular_disk(ax_xy, chi_near, chi_far, half_angle, test_coords, status,
                             z_min, z_max, axis_names=(e1_name, e2_name, los_name))
    draw_axial_slice(ax_xz, chi_near, chi_far, half_angle, vert_coords_idx=0,
                      horiz_label=f"{los_name} [cMpc]", vert_label=f"{e1_name} [cMpc]",
                      panel_label=f"{e1_name}-{los_name}", test_coords=test_coords, status=status,
                      z_min=z_min, z_max=z_max, show_xlabel=False)
    draw_axial_slice(ax_yz, chi_near, chi_far, half_angle, vert_coords_idx=1,
                      horiz_label=f"{los_name} [cMpc]", vert_label=f"{e2_name} [cMpc]",
                      panel_label=f"{e2_name}-{los_name}", test_coords=test_coords, status=status,
                      z_min=z_min, z_max=z_max, show_xlabel=True)

    combined = {}
    for ax in (ax_xy, ax_xz, ax_yz):
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in combined:
                combined[l] = h

    # redshift range of the shell as the title of the figure, and the legend below the panels
    beam_str = ", ".join(f"{v:g}" for v in beam_vector)
    fig.suptitle(rf"beam ({beam_str}),  ${z_min:.3f} < z \leq {z_max:.3f}$", fontsize=12, fontweight="bold")
    if isinstance(fig.get_layout_engine(), matplotlib.layout_engine.ConstrainedLayoutEngine):
        # make room for the legend outside the axes, with some space between it and the axis label
        fig.get_layout_engine().set(h_pad=0.06)
        fig.legend(combined.values(), combined.keys(), loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    else:
        fig.legend(combined.values(), combined.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.0), 
                   ncol=3, fontsize=9, frameon=False)

    if show:
        plt.show()

    return ax_xy, ax_xz, ax_yz


def plot_particles_in_beam(coords, shell_nr, redshift_range, ang_radius_deg, ptype, beam_vec, cosmo, 
    axes_plots=True, scatter_plots=True, output_dir='./'):
    """
    Plot the radial distribution, coordinate distributions (along the axes of the beam's frame) 
    and a 3-panel projection of the particles against the lightcone geometry. 
    The output filenames include the particle type, beam direction and shell number.

    :param  coords:         comoving coordinates [Mpc] of the particles in the beam, shape (N, 3)
    :type   coords:         np.ndarray
    :param  shell_nr:       shell number, used in the output filenames
    :type   shell_nr:       int
    :param  redshift_range: minimum and maximum redshift of the shell [z_min, z_max]
    :type   redshift_range: tuple
    :param  ang_radius_deg: angular radius [deg] of the beam
    :type   ang_radius_deg: float
    :param  ptype:          particle type, used in the output filenames
    :type   ptype:          str
    :param  beam_vec:       direction of the beam
    :type   beam_vec:       array-like, shape (3,)
    :param  cosmo:          object with a .z2r(z) method returning a comoving distance
    :type   cosmo:          object
    :param  axes_plots:     If True, plot the radial and coordinate distributions
    :type   axes_plots:     boolean
    :param  scatter_plots:  If True, plot the 3-panel projection
    :type   scatter_plots:  boolean
    :param  output_dir:     directory to write the plots to
    :type   output_dir:     str
    """
    print(f"\nnumber of {ptype} particles in the beam: {len(coords)}")
    if len(coords) == 0:
        return

    beam_str = ", ".join(f"{v:g}" for v in beam_vec)
    label_text = rf"beam ({beam_str}),  ${redshift_range[0]:.3f} < z \leq {redshift_range[1]:.3f}$"
    fig_prefix = f"{ptype}_beam_{beam_label(beam_vec)}_shell{int(shell_nr)}"

    if axes_plots:
        # distance from the observer
        plt.figure(figsize =(7, 4))
        n, edges = np.histogram(np.linalg.norm(coords, axis=1), bins = 200)
        mids = 0.5*(edges[:-1] + edges[1:])
        plt.plot(mids, n)
        plt.xlabel("Comoving Distance")
        plt.ylabel("Number of particles")
        plt.title(label_text, fontsize=10, fontweight="bold")
        fig_name=f"{fig_prefix}_radius"
        plt.savefig(output_dir+"/"+fig_name+".png",dpi=300, bbox_inches='tight')
        plt.close()

        # coordinates along each axis of the beam's frame
        frame, axis_names = beam_frame(beam_vec)
        frame_coords = coords @ frame.T
        plt.figure(figsize =(7, 4))
        for i in [2, 1, 0]:
            n, edges = np.histogram(frame_coords[:, i], bins = 200)
            mids = 0.5*(edges[:-1] + edges[1:])
            plt.plot(mids, n, label=f"axis: {axis_names[i]}")
        plt.xlabel("Comoving Distance")
        plt.ylabel("Number of particles")
        plt.title(label_text, fontsize=10, fontweight="bold")
        plt.legend(loc="best")
        fig_name=f"{fig_prefix}_axis_coords"
        plt.savefig(output_dir+"/"+fig_name+".png",dpi=300, bbox_inches='tight')
        plt.close()

    if scatter_plots:
        plot_lightcone_projection_3panel(ang_radius_deg, 
                                         redshift_range[0], redshift_range[1],
                                         cosmo,
                                         test_coords=coords, test_coords_buffer=0.,
                                         tol=1e-6,
                                         beam_vector=beam_vec, axes=None,
                                         figsize=(7, 7), show=False,
                                         contours=False, contour_bins=100,
                                         contour_levels=5, contour_min_points=250
                                         )
        fig_name=f"{fig_prefix}_scatter"
        plt.savefig(output_dir+"/"+fig_name+".png",dpi=300, bbox_inches='tight')
        plt.close()



# 

if __name__ == "__main__":

    plot_settings()

    # simulation to use
    box_res="L1000N0900"
    sim_name="HYDRO_FIDUCIAL"
    base_dir="/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}".format(box_res=box_res, sim_name=sim_name)

    # define output directory
    output_dir="./example_outputs/pencil_beam_examples"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # redshift range of the lightcone shell
    shell_number = 0 # label for the output filenames
    redshift_range=(0.20, 0.30)
    print(f"\nredshift: {redshift_range[0]} - {redshift_range[1]}")

    # angular radius of the beam, set by the minimum multipole

    ang_radius_deg=45
    print(f"\nangular radius: {ang_radius_deg} [deg]\n", flush=True)

    # directions of the beams to make: along an axis, along a box diagonal and along a general direction.
    # Each beam is plotted in its own frame, so the beams are drawn the same way whichever way they point
    beam_vectors = [
        (0, 0, 1), # number of PartType5 particles in the beam: 1174892, 1174892
        (1, 1, 1), # number of PartType5 particles in the beam: 1363832, 1182332
        (1, 2, 3), # number of PartType5 particles in the beam: 250440,  1192684
    ]

    # particles to place in the beam
    ptype="PartType5"
    property_names = ["Coordinates", "ParticleIDs"]

    # number of snapshot files to read before combining their particles
    files_per_chunk = 8

    plot_cosmo=Snapshot_Cosmology_For_Lightcone("{base_dir}/snapshots".format(base_dir=base_dir))

    for beam_vector in beam_vectors:
        print(f"\nbeam direction: {beam_vector}")
        SB = SnapshotBeam(boxsize_resolution=box_res, simulation_name=sim_name, beam_vector=beam_vector)

        # 1) find every snapshot file (and box tile) needed to populate the shell, without reading any particles
        numb_files, files_to_read = SB.gather_files(ptype, redshift_range, ang_radius_deg)
        print(f"\n{numb_files} snapshot files to read for {ptype}")

        # 2) read the files a chunk at a time, keeping the coordinates of the particles placed in the beam
        beam_coords = []
        for chunk_nr, first_file in enumerate(range(0, numb_files, files_per_chunk)):
            chunk_files = range(first_file, min(first_file + files_per_chunk, numb_files))

            chunk_coords = []
            for file_number in chunk_files:
                particle_data = SB.place_file_in_shell(file_number, ptype, property_names, ang_radius_deg)
                if not particle_data: # no particles from this file fall in the beam
                    continue
                # keep only particles that were read
                read = particle_data["ParticleIDs"].value >= 0
                chunk_coords.append(particle_data["Coordinates"][read, :].to_value("Mpc"))

            if len(chunk_coords) > 0:
                beam_coords.append(np.concatenate(chunk_coords))
            npart_chunk = sum(len(c) for c in chunk_coords)
            print(f"chunk {chunk_nr}: files {chunk_files.start}-{chunk_files.stop-1} of {numb_files}, {npart_chunk} particles in the beam", flush=True)

        beam_coords = np.concatenate(beam_coords) if len(beam_coords) > 0 else np.zeros((0, 3))

        # 3) plot the particles against the geometry of the beam, in the beam's frame
        plot_particles_in_beam(
            beam_coords, shell_number, redshift_range, ang_radius_deg, 
            ptype=ptype, 
            beam_vec=beam_vector, 
            cosmo=plot_cosmo,
            axes_plots=True, scatter_plots=True,
            output_dir=output_dir,
            )