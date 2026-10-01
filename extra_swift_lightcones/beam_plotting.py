#!/bin/env python
import copy
import math
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon

"""
Plot projections of a past lightcone beam as wedges, with redshift, comoving distance and angular axes.
"""


# simple rounding functions

def round_down_10(x):
    """
    Round down to the nearest multiple of 10.

    :param  x:  value to round
    :type   x:  float
    """
    return int(math.floor(x / 10) * 10)

def round_up_10(x):
    """
    Round up to 2 significant figures.

    :param  x:  value to round
    :type   x:  float
    """
    if x == 0:
        return 0
    power = 10 ** (int(math.floor(math.log10(abs(x)))) - 1)
    return math.ceil(x / power) * power

def orderOfMagnitude(number):
    """
    Order of magnitude (base 10) of a number.

    :param  number: positive number
    :type   number: float
    """
    return math.floor(math.log(number, 10))

def filter_kwargs(func, kwargs):
    """
    Keep only the keyword arguments accepted by a function.

    :param  func:   function to filter the keyword arguments for
    :type   func:   function
    :param  kwargs: keyword arguments
    :type   kwargs: dict
    """
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}

def upright_rotation(angle_deg):
    """
    Rotate text by 180 degrees if needed so that it is not upside down.

    :param  angle_deg:  rotation angle [deg]
    :type   angle_deg:  float
    """
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg


# geometry helpers

def arc_xy(comoving_dist, theta_degree, n=360):
    """
    Get points on the arc of the beam.

    Returns a tuple of (x, y) coordinates.

    :param  comoving_dist:  comoving distance [Mpc], radius of the arc
    :type   comoving_dist:  float
    :param  theta_degree:   angular radius [deg] of the arc. If None, draw a full circle
    :type   theta_degree:   float
    :param  n:              number of points on the arc
    :type   n:              int
    """
    if theta_degree is not None:
        theta_rad = (np.deg2rad(-theta_degree), np.deg2rad(theta_degree))
    else:
        theta_rad = (0, 2*np.pi)
    arc_rad = np.linspace(theta_rad[0], theta_rad[1], n)
    a = comoving_dist * np.cos(arc_rad)
    b = comoving_dist * np.sin(arc_rad)
    return a, b

def minor_tick_spacer(major_spacing, max_numb_of_minor_ticks=5, min_numb_of_minor_ticks=2):
    """
    Return minor tick spacing for preferred number of minor ticks between major ticks.

    :param  major_spacing:              spacing between the major ticks
    :type   major_spacing:              float
    :param  max_numb_of_minor_ticks:    maximum number of minor ticks between major ticks
    :type   max_numb_of_minor_ticks:    int
    :param  min_numb_of_minor_ticks:    minimum number of minor ticks between major ticks
    :type   min_numb_of_minor_ticks:    int
    """
    n_minor_divisions = np.arange(min_numb_of_minor_ticks+1, max_numb_of_minor_ticks+2, 1)[::-1]
    for divisions in n_minor_divisions:  # 5,4,3,2 minor ticks
        minor = major_spacing / divisions
        exponent = np.floor(np.log10(abs(minor)))
        mantissa = minor / 10**exponent
        if np.isclose(mantissa, (1, 2, 2.5, 5, 10, 30)).any():
            return minor
    # if all else fails return 2 minor ticks
    return major_spacing / 3

def labels_loc(ax, p1, p2, offset=12, dpi=None, tick_length=1):
    """
    Determine the position of the axes labels based on 2 points along the axes.

    Returns the (x, y) position of the label in data coordinates.

    :param  ax:             axes of the plot
    :type   ax:             matplotlib.axes._axes.Axes
    :param  p1:             first point along the axes, in data coordinates
    :type   p1:             tuple
    :param  p2:             second point along the axes, in data coordinates
    :type   p2:             tuple
    :param  offset:         distance [points] of the label from the axes
    :type   offset:         float
    :param  dpi:            dots per inch. If None, use the figure dpi
    :type   dpi:            float
    :param  tick_length:    additional distance [points] of the label from the axes, to clear the ticks
    :type   tick_length:    float
    """
    # Offset in display coordinates (points -> pixels)
    if dpi is None:
        dpi = ax.figure.dpi
    offset_px = (tick_length+offset) * dpi / 72

    # points in display coords
    t = ax.transData.transform
    x1, y1 = t(p1)
    x2, y2 = t(p2)

    # unit normal to the axes
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length

    x_disp = (x1+x2) / 2 + nx * offset_px
    y_disp = (y1+y2) / 2 + ny * offset_px

    # Back to data coordinates
    x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp))

    return x_data, y_data

def offset_point_from_tick(ax, x, y, ux, uy, offset):
    """
    Use display coordinates to move tick label away
    from the tick by shifting it in the direction (ux,uy).

    Returns the new (x, y) position in data coordinates.

    :param  ax:     axes of the plot
    :type   ax:     matplotlib.axes._axes.Axes
    :param  x:      x position of the tick, in data coordinates
    :type   x:      float
    :param  y:      y position of the tick, in data coordinates
    :type   y:      float
    :param  ux:     x component of the direction to shift in
    :type   ux:     float
    :param  uy:     y component of the direction to shift in
    :type   uy:     float
    :param  offset: distance [points] to shift by
    :type   offset: float
    """
    # Tick endpoint in display coordinates
    x_disp, y_disp = ax.transData.transform((x, y))

    # direction in display coordinates
    x2_disp, y2_disp = ax.transData.transform((x + ux, y + uy))

    dx = x2_disp - x_disp
    dy = y2_disp - y_disp
    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm

    # points -> pixels
    offset_pixels = offset * ax.figure.dpi / 72

    # apply offset and go back to data coordinates
    return ax.transData.inverted().transform(
        (x_disp + dx * offset_pixels, y_disp + dy * offset_pixels)
    )

def points_to_data(ax, x, y, dx, dy, offset):
    """
    Use display coordinates to move points on a plot.

    Returns the (dx, dy) shift in data coordinates.

    :param  ax:     axes of the plot
    :type   ax:     matplotlib.axes._axes.Axes
    :param  x:      x position of the point, in data coordinates
    :type   x:      float
    :param  y:      y position of the point, in data coordinates
    :type   y:      float
    :param  dx:     x component of the direction to move in
    :type   dx:     float
    :param  dy:     y component of the direction to move in
    :type   dy:     float
    :param  offset: distance [points] to move by
    :type   offset: float
    """
    # norm
    L = np.hypot(dx, dy)
    dx /= L
    dy /= L

    # transform points -> display coords
    x0, y0 = ax.transData.transform((x, y))

    # 1 point = dpi / 72 pixels
    offset_px = offset * ax.figure.dpi / 72

    # change position in display coords, then display coords -> data coords
    xd, yd = ax.transData.inverted().transform((x0 + offset_px * dx, y0 + offset_px * dy))

    return xd - x, yd - y


# colour functions
def colour_norm(norm="log", vmin=None, vmax=None, label="wedge"):
    """
    Colour normalisation of an image. 

    Returns a matplotlib.colors.Normalize instance. A Normalize given is copied, and its vmin and vmax 
    are set from those given only if it does not already have them.

    :param  norm:   "log" (or None) for log scaling, "linear" for linear scaling, or any matplotlib Normalize 
                        instance, e.g. SymLogNorm, TwoSlopeNorm or PowerNorm
    :type   norm:   str or matplotlib.colors.Normalize
    :param  vmin:   minimum value shown. If None, the minimum of the data
    :type   vmin:   float
    :param  vmax:   maximum value shown. If None, the maximum of the data
    :type   vmax:   float
    :param  label:  name of the image in error messages
    :type   label:  str
    """
    if (vmin is not None) and (vmax is not None) and (vmin > vmax):
        raise ValueError(f"{label}: minimum value shown ({vmin}) is greater than the maximum ({vmax})")

    if isinstance(norm, col.Normalize):
        norm = copy.copy(norm)
        if norm.vmin is None:
            norm.vmin = vmin
        if norm.vmax is None:
            norm.vmax = vmax
        return norm

    if norm is None or norm == "log":
        if (vmin is not None) and (vmin <= 0):
            raise ValueError(f"{label}: log colour scaling needs a minimum value above 0, got {vmin}. "
                             "Use a linear norm or a matplotlib Normalize such as SymLogNorm for data that can be 0 or negative")
        return col.LogNorm(vmin=vmin, vmax=vmax)
    if norm == "linear":
        return col.Normalize(vmin=vmin, vmax=vmax)

    raise ValueError(f"{label}: unknown colour norm {norm!r}, use 'log', 'linear' or a matplotlib.colors.Normalize")

# default styles, updated by any keyword arguments given

def wedge_style(wedge_kwargs=None):
    """
    Keyword arguments of the wedge outline. If color is given, it is used as the edge colour.

    :param  wedge_kwargs:   keyword arguments to control the wedge outline
    :type   wedge_kwargs:   dict
    """
    style = {
        "alpha":1.0,
        "lw":0.8,
        "ls":"-",
        "edgecolor":"k",
        "facecolor":"none",
        "path_effects":None,
        "zorder":10,
    }
    style.update(**(wedge_kwargs or {}))
    if "color" in style:
        style["edgecolor"] = style.pop("color")
    return style

def title_style(title_kwargs=None):
    """
    Keyword arguments of the wedge titles. rotation and zorder are set by the wedge.

    :param  title_kwargs:   keyword arguments to control the text of the wedge titles
    :type   title_kwargs:   dict
    """
    style = {
        "color":"white",
        "alpha":1.0,
        "fontsize":10,
        "va":"center",
        "ha":"left",
        "path_effects":[path_effects.withStroke(linewidth=1., foreground="black"), path_effects.Normal()],
        "rotation_mode":"anchor",
    }
    style.update(**(title_kwargs or {}))
    style.pop("rotation", None)
    style.pop("zorder", None)
    return style

def tick_styles(major_tick_kwargs=None, minor_tick_kwargs=None, tick_label_kwargs=None):
    """
    Keyword arguments of the major ticks, minor ticks and tick labels.
    Minor ticks default to the major tick colour and alpha with a thinner line.

    Returns a tuple of (major tick, minor tick, tick label) keyword arguments.

    :param  major_tick_kwargs:  keyword arguments to control the major tick lines
    :type   major_tick_kwargs:  dict
    :param  minor_tick_kwargs:  keyword arguments to control the minor tick lines
    :type   minor_tick_kwargs:  dict
    :param  tick_label_kwargs:  keyword arguments to control the text of the labels on the major ticks
    :type   tick_label_kwargs:  dict
    """
    major = {
        "color":"black",
        "lw":0.8,
        "alpha":1,
    }
    major.update(**(major_tick_kwargs or {}))
    minor = {
        "color":major["color"],
        "lw":major["lw"]*0.55,
        "alpha":major["alpha"],
    }
    minor.update(**(minor_tick_kwargs or {}))
    label = {
        "color":"black",
        "alpha":1,
        "fontsize":8,
        "va":"center",
        "ha":"right",
        'rotation':0,
        'rotation_mode':"anchor",
    }
    label.update(**(tick_label_kwargs or {}))
    return major, minor, label

def axes_label_style(axes_label_kwargs=None):
    """
    Keyword arguments of the axes labels.

    :param  axes_label_kwargs:  keyword arguments to control the text of the axes labels
    :type   axes_label_kwargs:  dict
    """
    style = {
        "color":"black",
        "fontsize":10,
        "weight":"bold",
        "ha":"center",
        "va":"center",
        "rotation_mode":"anchor",
    }
    style.update(**(axes_label_kwargs or {}))
    return style

def grid_style(grid_line_kwargs=None):
    """
    Keyword arguments of the grid lines.

    :param  grid_line_kwargs:   keyword arguments to control the grid lines
    :type   grid_line_kwargs:   dict
    """
    style = {
        "color":"silver",
        "lw":0.8,
        "alpha":1.0,
        'ls':'--'
    }
    style.update(**(grid_line_kwargs or {}))
    return style


# zorder of the axes components
AXES_TOP_LEVEL = 20
AXES_LOW_LEVEL = 10

class BeamPlot:
    """
    Plot 2D projections of a beam as wedges, with redshift, comoving distance and angular axes.
    """
    def __init__(self, cosmology, angular_diameter=None, redshift_range=None, axes_extent=None):
        """
        :param  cosmology:          the simulation's cosmology model
        :type   cosmology:          astropy cosmology object
        :param  angular_diameter:   total angular diameter of beam in degrees. Required by split_beam_plot
        :type   angular_diameter:   float
        :param  redshift_range:     minimum and maximum redshift of the beam shown. Required by split_beam_plot
        :type   redshift_range:     sequence of two floats [z_min, z_max]
        :param  axes_extent:        extent of the projected images, in coordinate space [xmin, xmax, ymin, ymax].
                                        Required by split_beam_plot and add_wedge
        :type   axes_extent:        list
        """
        self.cosmology = cosmology
        self.angular_diameter = angular_diameter
        self.redshift_range = redshift_range
        self.axes_extent = axes_extent

    def split_beam_plot(self, numb_wedges, projection_data, colour_maps,
            axs=None, filename=None, update_badcol=True, figsize=(7,7), titles=None, norms=None, **kwargs):
        """
        Create plot of the whole beam, split into separate wedges.

        Returns list of each projected wedge, with the axes if axs is given or the figure if axs is None.
        If filename is given, returns only the list of each projected wedge.

        :param  numb_wedges:        number of wedges or different projections to show in the beam
        :type   numb_wedges:        int
        :param  projection_data:    nested list containing different segments of the beam the 2D array to plot and
                                        a tuple with the min and max values shown in the img [2D array, (min, max)]
        :type   projection_data:    list
        :param  colour_maps:        list of colour maps for each segment of the beam
        :type   colour_maps:        list
        :param  axs:                Default=None, If None create a new axes for the plot, otherwise plot onto the axes given
        :type   axs:                matplotlib.axes._axes.Axes
        :param  filename:           None, if given, write plot to this file
        :type   filename:           str
        :param  update_badcol:      If true, modify all colour maps so that the minimum, nan and None values are set to black
        :type   update_badcol:      boolean
        :param  figsize:            size of the figure, used if axs is None
        :type   figsize:            tuple
        :param  titles:             title of each wedge
        :type   titles:             list
        :param  norms:              colour normalisation of each segment of the beam: "log", "linear" or a 
                                        matplotlib.colors.Normalize, see colour_norm. If None, all are "log"
        :type   norms:              list
        :param  kwargs:             All additional arguments to be passed onto the add_wedge and add_beam_axes functions.
        :type   kwargs:             dict
        """
        
        #define wedge angles
        beam_ang_offset_deg = 0.
        beam_radius_deg=(self.angular_diameter-beam_ang_offset_deg)/2
        wedge_diameter_deg= (self.angular_diameter-beam_ang_offset_deg)/numb_wedges # angular diameter of one wedge
        print(f"beam angular diameter:\t{self.angular_diameter:.2f} [deg]", flush=True)
        print(f"wedge angular diameter:\t{wedge_diameter_deg:.2f} [deg]", flush=True)

        # create redshift ticks for the range given
        all_redshift_major_ticks=np.arange(0.05, 5.0, 0.05)
        all_redshift_minor_ticks=np.arange(0.01, 5.0, 0.01)

        # refine range of redshift ticks for the redshift range given
        z_axis_min = max(self.redshift_range[0], 0.01)
        z_axis_max = self.redshift_range[-1]
        redshift_axis_ticks=np.concatenate((all_redshift_major_ticks[(all_redshift_major_ticks>z_axis_min)&(all_redshift_major_ticks<z_axis_max)], [z_axis_max]))
        redshift_axis_minor_ticks=np.concatenate((all_redshift_minor_ticks[(all_redshift_minor_ticks>z_axis_min)&(all_redshift_minor_ticks<z_axis_max)], [z_axis_max]))

        # define inner and outer radius
        rmax=self.cosmology.comoving_distance(redshift_axis_ticks[-1]).to_value("Mpc")
        rmin=self.cosmology.comoving_distance(redshift_axis_minor_ticks[0]).to_value("Mpc")

        wedge_imgs=[[] for i in range(numb_wedges)]

        # if not adding to other plot, then create empty subplot
        new_figure = axs is None
        if new_figure:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, gridspec_kw={'height_ratios': [1], 'width_ratios': [1], 'hspace': 0, 'wspace': 0})
        else:
            fig = axs.figure

        axs.margins(0)
        axs.set_aspect("equal")
        axs.axis("off")

        if titles is not None:
            assert len(titles) == numb_wedges
        
        if norms is None:
            norms = ["log"] * numb_wedges
        elif len(norms) != numb_wedges:
            raise ValueError(f"{len(norms)} norms given for {numb_wedges} wedges")
        if "norm" in kwargs:
            raise TypeError("split_beam_plot takes a list of norms, one per wedge, not norm")
        
        # seperate kwargs for image function and axes function
        img_kwargs = filter_kwargs(self.add_wedge, kwargs)
        axes_kwargs = filter_kwargs(self.add_beam_axes, kwargs)

        # iterate through the different split beams and add to plot
        for wedge_idx in range(numb_wedges):

            wedge_cmap = plt.colormaps[colour_maps[wedge_idx]].copy()

            if update_badcol:
                # update colour maps to ensure bad colour = 'black'
                wedge_cmap.set_bad("black", alpha=1.)
                wedge_cmap.set_under("black")

            wedge_title = titles[wedge_idx] if titles is not None else None

            # read in pixel range
            cmap_pix_range=projection_data[wedge_idx][1]
            
            cmap_pix_range = projection_data[wedge_idx][1] if len(projection_data[wedge_idx]) > 1 else None
            if cmap_pix_range is None:
                cmap_pix_range = (None, None)
            
            # add wedge
            wedge_imgs[wedge_idx], ax  = self.add_wedge(
                axs, wedge_idx, projection_data[wedge_idx][0],
                wedge_cmap, cmap_pix_range[0], cmap_pix_range[1],
                beam_radius_deg, wedge_diameter_deg,
                rmin, rmax, title=wedge_title,
                img_zorder=-1, 
                norm=norms[wedge_idx],
                **img_kwargs
                )

        # add axes to the outer edge of the whole beam
        self.add_beam_axes(axs, redshift_axis_ticks, redshift_axis_minor_ticks, beam_radius_deg,
            beam_ang_offset=beam_ang_offset_deg,
            rmin=rmin, rmax=rmax,
            **axes_kwargs
            )

        if filename is not None:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            if new_figure:
                plt.close(fig)

        return fig, axs, wedge_imgs

    def add_wedge(self, ax, wedge_idx, data_2D, cmap, pix_min, pix_max, beam_max_ang_radius_deg, wedge_ang_diameter_deg, rmin, rmax,
        title=None, img_zorder=10, wedge_kwargs=None, title_kwargs=None, norm="log", interpolation="quadric"):
        """
        Add each smaller beam or wedge onto the plot.

        Returns the updated projected image and the axes.

        :param  ax:                         axes to plot onto
        :type   ax:                         matplotlib.axes._axes.Axes
        :param  wedge_idx:                  order that the wedge is added to the plot.
        :type   wedge_idx:                  int
        :param  data_2D:                    2D histogram to plot
        :type   data_2D:                    np.ndarray
        :param  cmap:                       colour map
        :type   cmap:                       str or matplotlib colour map type object
        :param  pix_min:                    minimum pixel value shown. If None, the minimum of the data or the norm
        :type   pix_min:                    float
        :param  pix_max:                    maximum pixel value shown. If None, the maximum of the data or the norm
        :type   pix_max:                    float
        :param  beam_max_ang_radius_deg:    maximum angular radius [deg] of the beam (slice) as a whole
        :type   beam_max_ang_radius_deg:    float
        :param  wedge_ang_diameter_deg:     angular diameter [deg] of each wedge that the beam is split into
        :type   wedge_ang_diameter_deg:     float
        :param  rmin:                       minimum comoving distance
        :type   rmin:                       float
        :param  rmax:                       maximum comoving distance
        :type   rmax:                       float
        :param  title:                      title of the wedge, placed on the inner arc
        :type   title:                      str
        :param  img_zorder:                 zorder of the image
        :type   img_zorder:                 int
        :param  wedge_kwargs:               keyword arguments to control the wedge outline
        :type   wedge_kwargs:               dict
        :param  title_kwargs:               keyword arguments to control the text of the wedge title
        :type   title_kwargs:               dict
        :param  norm:                       colour normalisation, "log", "linear" or a matplotlib.colors.Normalize, see colour_norm
        :type   norm:                       str or matplotlib.colors.Normalize
        :param  interpolation:              matplotlib imshow interpolation of the image, e.g. "nearest" to show each pixel
        :type   interpolation:              str
        """
        
        norm = colour_norm(norm, pix_min, pix_max, label=f"wedge {wedge_idx}")
        
        wedge_kwargs = wedge_style(wedge_kwargs)
        title_kwargs = title_style(title_kwargs)

        theta0 = np.deg2rad(beam_max_ang_radius_deg - np.abs(wedge_idx*wedge_ang_diameter_deg))
        theta1 = np.deg2rad(beam_max_ang_radius_deg - np.abs((1+wedge_idx)*wedge_ang_diameter_deg))

        print(f"\twedge {wedge_idx}, theta 0 -> theta 1 = {np.rad2deg(theta0):>3f} ->  {np.rad2deg(theta1):.3f}")

        # outline of the wedge: inner arc then outer arc
        theta_arc = np.linspace(theta0, theta1, 720)
        verts = np.vstack([
            np.c_[rmin * np.cos(theta_arc), rmin * np.sin(theta_arc)],          # inner arc
            np.c_[rmax * np.cos(theta_arc), rmax * np.sin(theta_arc)][::-1],    # outer arc
        ])

        # plot img of beam wedge
        img = ax.imshow(
            data_2D.T,
            norm=norm, cmap=cmap,
            origin="lower", extent=self.axes_extent, zorder=img_zorder,
            interpolation=interpolation,
        )

        # add wedge to image and clip image to the wedge
        wedge = Polygon(verts, closed=True, **wedge_kwargs)
        ax.add_patch(wedge)
        img.set_clip_path(wedge)

        if title is not None:
            # Position text on the inner arc
            theta_mid = 0.5 * (theta0 + theta1)
            ax.text(
                rmin * np.cos(theta_mid),
                rmin * np.sin(theta_mid),
                title,
                **title_kwargs,
                rotation = np.rad2deg(theta_mid),
                zorder=wedge_kwargs["zorder"] + 1,
            )

        return img, ax

    def add_beam_axes(self, ax, redshift_major_ticks, redshift_minor_ticks, beam_radius_deg,
        beam_ang_offset=0.,
        rmin=None, rmax=None,
        comoving_distance_major_ticks=None, comoving_distance_minor_ticks=None,
        dtheta_major_ticks_deg=5, dtheta_minor_ticks_deg=1, theta_ticks_abs=True, major_tick_length=3.5, minor_tick_length=None,
        redshift_label_offset=(0,0,0), comoving_label_offset=(0,0,0), tick_label_offset=(0,0,0,0),
        comoving_dist_axes_label=r"$\mathrm{comoving~distance}~[{\mathrm{Mpc}}]$", redshift_axes_label=r"$\mathrm{redshift},~z$",
        overlay_grid=(True, True, True), 
        major_tick_kwargs=None,minor_tick_kwargs=None,tick_label_kwargs=None, axes_label_kwargs=None, grid_line_kwargs=None,
        ):
        """
        Add axes, labels and ticks to the split beam plot. The redshift axis is drawn along the upper
        edge of the beam, the comoving distance axis along the lower edge and the angular axis along the outer arc.

        :param  ax:                     axes to plot onto
        :type   ax:                     matplotlib.axes._axes.Axes
        :param  redshift_major_ticks:  location of the major ticks on the redshift axes
        :type   redshift_major_ticks:  np.ndarray
        :param  redshift_minor_ticks:  location of the minor ticks on the redshift axes
        :type   redshift_minor_ticks:  np.ndarray
        :param  beam_radius_deg:    maximum angular radius [deg] of the beam (slice) as a whole
        :type   beam_radius_deg:    float
        :param  beam_ang_offset:    decrease the maximum angular radius [deg] shown
        :type   beam_ang_offset:    float
        :param  rmin:   minimum comoving distance [Mpc] of the plot
        :type   rmin:   float
        :param  rmax:   maximum comoving distance [Mpc] on the plot
        :type   rmax:   float
        :param  comoving_distance_major_ticks:  location of the major ticks on the comoving radius axes
        :type   comoving_distance_major_ticks:  np.ndarray
        :param  comoving_distance_minor_ticks:  location of the minor ticks on the comoving radius axes
        :type   comoving_distance_minor_ticks:  np.ndarray
        :param  dtheta_major_ticks_deg:  spacing [deg] between the major ticks on the angular radius axes
        :type   dtheta_major_ticks_deg:  float
        :param  dtheta_minor_ticks_deg:  spacing [deg] between the minor ticks on the angular radius axes.
                                            If 'auto', choose the spacing from the major ticks
        :type   dtheta_minor_ticks_deg:  float or str
        :param  theta_ticks_abs:  If true, label the angular radius ticks with absolute values
        :type   theta_ticks_abs:  boolean
        :param  major_tick_length:  length of major ticks
        :type   major_tick_length:  float
        :param  minor_tick_length:  length of minor ticks
        :type   minor_tick_length:  float
        :param  comoving_dist_axes_label:  label placed on the comoving radius axes
        :type   comoving_dist_axes_label:  str
        :param  redshift_axes_label:  label placed on the redshift axes
        :type   redshift_axes_label:  str
        :param  redshift_label_offset:  additional offset (in coordinate space) added to the redshift axes
                                            label along the x,y axes & its rotation  (x,y,theta).
        :type   redshift_label_offset:  tuple
        :param  comoving_label_offset:  additional offset (in coordinate space) added to the comoving radius axes
                                            label along the x,y axes & its rotation  (x,y,theta).
        :type   comoving_label_offset:  tuple
        :param  tick_label_offset:  additional offset (in coordinate space) added to the tick labels on the redshift
                                        and comoving radius axes (x_redshift,y_redshift, x_radius,y_radius).
        :type   tick_label_offset:  tuple
        :param  overlay_grid:   If true overlay a grid line at each major tick from the given axes (redshift, comoving radius, angular radius)
        :type   overlay_grid:   tuple    (boolean, boolean, boolean)
        :param  major_tick_kwargs:    keyword arguments to control the major tick lines
        :type   major_tick_kwargs:    dict
        :param  minor_tick_kwargs:    keyword arguments to control the minor tick lines
        :type   minor_tick_kwargs:    dict
        :param  tick_label_kwargs:    keyword arguments to control the text of the labels on the major ticks
        :type   tick_label_kwargs:    dict
        :param  axes_label_kwargs:    keyword arguments to control the text of the axes labels
        :type   axes_label_kwargs:    dict
        :param  grid_line_kwargs:    keyword arguments to control the grid lines
        :type   grid_line_kwargs:    dict
        """
        ax.figure.canvas.draw() # require ax to be drawn so transformation from pixels to axes coords are stable

        # define basic properties of ticks and labels
        major_style, minor_style, tick_label_style = tick_styles(major_tick_kwargs, minor_tick_kwargs, tick_label_kwargs)
        label_style = axes_label_style(axes_label_kwargs)
        grid_line_style = grid_style(grid_line_kwargs)

        # tick properties
        tick_len = major_tick_length
        minor_tick_len = tick_len*0.55 if minor_tick_length is None else minor_tick_length

        # handle rotation of tick labels
        auto_rot_tick_label = tick_label_style.get("rotation") == "auto"
        if auto_rot_tick_label:
            tick_label_style.pop("rotation")

        # define wedge angles
        theta_max = np.deg2rad(beam_radius_deg)  # max theta from 0

        # comoving distances from redshifts
        comoving_distances=self.cosmology.comoving_distance(redshift_major_ticks).to_value("Mpc")
        comoving_tick_range = [np.min(comoving_distances), np.max(comoving_distances)]

        if redshift_minor_ticks is not None:
            minor_comoving_distances=self.cosmology.comoving_distance(redshift_minor_ticks).to_value("Mpc")
            comoving_tick_range[0] = min(comoving_tick_range[0], np.min(minor_comoving_distances))
            comoving_tick_range[1] = max(comoving_tick_range[1], np.max(minor_comoving_distances))

        #image coordinate max and min radius
        if rmax is None:
            rmax =comoving_distances[-1]+1
        if rmin is None:
            rmin=0

        # common settings of the radial axes
        radial_axis = dict(
            tick_len=tick_len, minor_tick_len=minor_tick_len, rmin=rmin, rmax=rmax, beam_radius_deg=beam_radius_deg,
            major_style=major_style, minor_style=minor_style, tick_label_style=tick_label_style, label_style=label_style,
            grid_line_style=grid_line_style, auto_rot_tick_label=auto_rot_tick_label,
        )

        # redshift axis along the upper edge of the beam, drop the last tick at the outer arc
        self._draw_radial_axis(
            ax, theta_max,
            major_radii=comoving_distances[:-1],
            major_labels=[f"{z:.2f}" for z in redshift_major_ticks[:-1]],
            minor_radii=None if redshift_minor_ticks is None else minor_comoving_distances[:-1],
            tick_label_offset=tick_label_offset[0:2],
            axis_label=redshift_axes_label, axis_label_offset=redshift_label_offset,
            overlay_grid=overlay_grid[0], side=1, **radial_axis,
        )

        # comoving distance axis along the lower edge of the beam
        comoving_distance_major_ticks, comoving_distance_minor_ticks = self._comoving_distance_ticks(
            comoving_tick_range, comoving_distance_major_ticks, comoving_distance_minor_ticks)

        self._draw_radial_axis(
            ax, -theta_max,
            major_radii=comoving_distance_major_ticks,
            major_labels=None if comoving_distance_major_ticks is None else [f"{math.floor(rr):d}" for rr in comoving_distance_major_ticks],
            minor_radii=comoving_distance_minor_ticks,
            tick_label_offset=tick_label_offset[2:4],
            # do not show a label if there are no major ticks
            axis_label=None if comoving_distance_major_ticks is None else comoving_dist_axes_label,
            axis_label_offset=comoving_label_offset,
            overlay_grid=overlay_grid[1], side=-1, **radial_axis,
        )

        # angular axis along the outer arc
        angle_ticks, angle_minor_ticks = self._angular_ticks(beam_radius_deg, beam_ang_offset, dtheta_major_ticks_deg, dtheta_minor_ticks_deg)

        if angle_ticks is not None:
            angle_label_style = dict(tick_label_style, va="center", ha="center")
            angle_label_style.pop("rotation", None)
            self._draw_angular_ticks(ax, angle_ticks, beam_radius_deg, rmin, rmax, tick_len, major_style, zorder=AXES_TOP_LEVEL,
                label_style=angle_label_style, theta_ticks_abs=theta_ticks_abs,
                grid_line_style=grid_line_style if overlay_grid[2] else None)

        if angle_minor_ticks is not None:
            self._draw_angular_ticks(ax, angle_minor_ticks, beam_radius_deg, rmin, rmax, minor_tick_len, minor_style, zorder=AXES_TOP_LEVEL)

    def _draw_radial_axis(self, ax, theta, major_radii, major_labels, minor_radii, tick_label_offset, axis_label, axis_label_offset,
        overlay_grid, side, tick_len, minor_tick_len, rmin, rmax, beam_radius_deg,
        major_style, minor_style, tick_label_style, label_style, grid_line_style, auto_rot_tick_label):
        """
        Draw the ticks, tick labels, grid lines and label of an axis along a straight edge of the beam.

        :param  ax:                     axes to plot onto
        :type   ax:                     matplotlib.axes._axes.Axes
        :param  theta:                  angle [rad] of the edge of the beam
        :type   theta:                  float
        :param  major_radii:            comoving distance [Mpc] of each major tick. If None, no major ticks
        :type   major_radii:            np.ndarray
        :param  major_labels:           label of each major tick
        :type   major_labels:           list
        :param  minor_radii:            comoving distance [Mpc] of each minor tick. If None, no minor ticks
        :type   minor_radii:            np.ndarray
        :param  tick_label_offset:      additional offset (in coordinate space) added to the tick labels (x, y)
        :type   tick_label_offset:      tuple
        :param  axis_label:             label of the axis. If None, no label
        :type   axis_label:             str
        :param  axis_label_offset:      additional offset (in coordinate space) added to the axis label (x, y, theta)
        :type   axis_label_offset:      tuple
        :param  overlay_grid:           If True, draw an arc at each major tick
        :type   overlay_grid:           boolean
        :param  side:                   1 for the upper edge of the beam, -1 for the lower edge. Labels are placed outside the beam
        :type   side:                   int
        """
        if auto_rot_tick_label:
            tick_label_style = dict(tick_label_style, rotation=upright_rotation(np.rad2deg(theta)))

        # direction perpendicular to the edge
        dx = -np.sin(theta)
        dy = np.cos(theta)
        tick_font_size = tick_label_style["fontsize"]

        if major_radii is not None:
            for rr, label in zip(major_radii, major_labels):
                x0 = rr * np.cos(theta)
                y0 = rr * np.sin(theta)

                tx_in, ty_in = offset_point_from_tick(ax, x0, y0, dx, dy, -side*tick_len)
                tx_out, ty_out = offset_point_from_tick(ax, x0, y0, dx, dy, side*tick_len)
                ax.plot([tx_in, tx_out], [ty_in, ty_out], **major_style, zorder=AXES_TOP_LEVEL)

                # tick label outside the beam
                lx, ly = offset_point_from_tick(ax, x0, y0, dx, dy, side * (0.5*tick_font_size + 1.*tick_len))
                ax.text(lx+tick_label_offset[0], ly+tick_label_offset[1], label, **tick_label_style, zorder=AXES_TOP_LEVEL)

                if overlay_grid:
                    x,y = arc_xy(rr, beam_radius_deg)
                    ax.plot(x, y, **grid_line_style, zorder=AXES_LOW_LEVEL)

        if minor_radii is not None:
            for rr in minor_radii:
                x0 = rr * np.cos(theta)
                y0 = rr * np.sin(theta)

                tx_in, ty_in = offset_point_from_tick(ax, x0, y0, dx, dy, -minor_tick_len)
                tx_out, ty_out = offset_point_from_tick(ax, x0, y0, dx, dy, minor_tick_len)
                ax.plot([tx_in, tx_out], [ty_in, ty_out], **minor_style, zorder=AXES_TOP_LEVEL)

        if axis_label is not None:
            # axis label, outside the tick labels
            p1 = (rmin * np.cos(theta), rmin * np.sin(theta))
            p2 = (rmax * np.cos(theta), rmax * np.sin(theta))
            px_offset = label_style["fontsize"]/2 + 3*tick_font_size + 2 # offset > fontsize/2 + tick fontsize + extra space
            label_x, label_y = labels_loc(ax, p1, p2, offset=side*px_offset, dpi=None, tick_length=side*1.75*tick_len)

            ax.text(
                label_x+axis_label_offset[0], label_y+axis_label_offset[1],
                axis_label,
                rotation=upright_rotation(np.rad2deg(theta)) + axis_label_offset[2],
                **label_style,
            )

    @staticmethod
    def _comoving_distance_ticks(comoving_tick_range, major_ticks=None, minor_ticks=None):
        """
        Choose the major and minor ticks of the comoving distance axis, if not given, and clip them to the range of the axis.

        Returns a tuple of (major ticks, minor ticks), either can be None.

        :param  comoving_tick_range:    minimum and maximum comoving distance [Mpc] of the axis
        :type   comoving_tick_range:    list
        :param  major_ticks:            comoving distance [Mpc] of the major ticks. If None, chosen from the range
        :type   major_ticks:            np.ndarray
        :param  minor_ticks:            comoving distance [Mpc] of the minor ticks. If None, chosen from the major tick spacing
        :type   minor_ticks:            np.ndarray
        """
        # order of magnitude of most distant comoving tick
        order_of_mag = orderOfMagnitude(comoving_tick_range[-1])

        if major_ticks is None:
            major_tick_spacing = round_down_10(minor_tick_spacer((comoving_tick_range[-1]-comoving_tick_range[0]), 10, 2))
            if major_tick_spacing>0:
                major_ticks = np.arange(0, 2*10**(order_of_mag+1), major_tick_spacing)

        if minor_ticks is None:
            if major_ticks is None:
                major_tick_spacing = comoving_tick_range[-1]-comoving_tick_range[0]
            else:
                major_tick_spacing = major_ticks[1]-major_ticks[0]
            minor_tick_spacing = minor_tick_spacer(major_tick_spacing, 5, 1)
            minor_ticks = np.arange(0, 2*10**(order_of_mag+1), minor_tick_spacing)

        # clip tick locations
        def clip(ticks):
            if ticks is None:
                return None
            return ticks[(ticks>=comoving_tick_range[0]) & (ticks<=comoving_tick_range[-1])]

        return clip(major_ticks), clip(minor_ticks)

    @staticmethod
    def _angular_ticks(beam_radius_deg, beam_ang_offset, dtheta_major_ticks_deg, dtheta_minor_ticks_deg):
        """
        Angles [deg] of the major and minor ticks on the angular axis.
        The major tick spacing is increased automatically for wide beams.

        Returns a tuple of (major tick angles, minor tick angles), either can be None.

        :param  beam_radius_deg:        maximum angular radius [deg] of the beam
        :type   beam_radius_deg:        float
        :param  beam_ang_offset:        decrease the maximum angular radius [deg] shown
        :type   beam_ang_offset:        float
        :param  dtheta_major_ticks_deg: spacing [deg] between the major ticks. If None, no major ticks
        :type   dtheta_major_ticks_deg: float
        :param  dtheta_minor_ticks_deg: spacing [deg] between the minor ticks. If 'auto', choose from the major ticks. If None, no minor ticks
        :type   dtheta_minor_ticks_deg: float or str
        """
        dtheta = dtheta_major_ticks_deg

        if dtheta_minor_ticks_deg is None:
            dtheta_minor = None
        elif dtheta_minor_ticks_deg == "auto":
            dtheta_minor = None if dtheta is None else minor_tick_spacer(dtheta)
        else:
            dtheta_minor = dtheta_minor_ticks_deg

        # automatically adjust cadance of ticks if angle is very large
        if (beam_radius_deg > 45) and (dtheta is not None) and (dtheta<=5):
            if (beam_radius_deg > 90) and ((beam_radius_deg*2)%30==0):
                dtheta=30 # always prefer 30degree over other seperations
            else:
                dtheta_test = minor_tick_spacer(beam_radius_deg*2, 10, 5)
                if dtheta_test < 10:
                    dtheta=10
                else:
                    dtheta = round_down_10(dtheta_test)

            dtheta_minor = minor_tick_spacer(dtheta)

        # do not include minor ticks if they are greater than major ticks in distance
        if (dtheta_minor is not None) and (dtheta is not None) and (dtheta_minor >= dtheta):
            dtheta_minor=None

        if dtheta is None:
            angle_ticks =None
        else:
            radius_ticks = np.arange(0, (beam_radius_deg+beam_ang_offset/2)+dtheta, dtheta)
            radius_ticks = radius_ticks[radius_ticks<=beam_radius_deg]
            angle_ticks  = np.concatenate((-1*radius_ticks[::-1], radius_ticks[1:]))

        if dtheta_minor is None:
            angle_minor_ticks=None
        else:
            radius_minor_ticks = np.arange(0+dtheta_minor, (beam_radius_deg+beam_ang_offset/2)+dtheta_minor, dtheta_minor)
            radius_minor_ticks = radius_minor_ticks[radius_minor_ticks<=beam_radius_deg]
            angle_minor_ticks  = np.concatenate((-1*radius_minor_ticks[::-1], radius_minor_ticks))

        return angle_ticks, angle_minor_ticks

    @staticmethod
    def _draw_angular_ticks(ax, angles_deg, beam_radius_deg, rmin, rmax, tick_len, tick_style, zorder,
        label_style=None, theta_ticks_abs=True, grid_line_style=None):
        """
        Draw ticks on the outer arc of the beam, with optional labels and radial grid lines.

        :param  ax:                 axes to plot onto
        :type   ax:                 matplotlib.axes._axes.Axes
        :param  angles_deg:         angle [deg] of each tick
        :type   angles_deg:         np.ndarray
        :param  beam_radius_deg:    maximum angular radius [deg] of the beam
        :type   beam_radius_deg:    float
        :param  rmin:               minimum comoving distance [Mpc], start of the grid lines
        :type   rmin:               float
        :param  rmax:               maximum comoving distance [Mpc], radius of the outer arc
        :type   rmax:               float
        :param  tick_len:           length of the ticks
        :type   tick_len:           float
        :param  tick_style:         keyword arguments of the tick lines
        :type   tick_style:         dict
        :param  zorder:             zorder of the ticks
        :type   zorder:             int
        :param  label_style:        keyword arguments of the tick labels. If None, no labels
        :type   label_style:        dict
        :param  theta_ticks_abs:    If true, label the ticks with absolute values
        :type   theta_ticks_abs:    boolean
        :param  grid_line_style:    keyword arguments of the grid lines. If None, no grid lines
        :type   grid_line_style:    dict
        """
        tick_str_format="{tick_val:.1f}°"

        for a in angles_deg:
            if np.abs(a) > beam_radius_deg:
                continue
            elif np.abs(a) == beam_radius_deg:
                a = a/np.abs(a) * beam_radius_deg

            th = np.deg2rad(a)

            # tick on arc, along the radial direction
            x0 = rmax * np.cos(th)
            y0 = rmax * np.sin(th)
            nx, ny = np.cos(th), np.sin(th)

            tx_in, ty_in = offset_point_from_tick(ax, x0, y0, nx, ny, -tick_len)
            tx_out, ty_out = offset_point_from_tick(ax, x0, y0, nx, ny, tick_len)
            ax.plot([tx_in, tx_out], [ty_in, ty_out], **tick_style, zorder=zorder)

            if grid_line_style is not None and np.abs(a)+1 < beam_radius_deg:
                ax.plot(
                    [rmin * np.cos(th), tx_in],
                    [rmin * np.sin(th), ty_in],
                    **grid_line_style,
                    zorder=AXES_LOW_LEVEL
                )

            if label_style is not None:
                # label slightly outside arc, rotated parallel to the tick mark
                theta_tick_val = np.abs(a) if theta_ticks_abs else a
                font_offset =  1.5*label_style["fontsize"] if len(str(theta_tick_val)) <=4 else 1.75*label_style["fontsize"]
                lx, ly = offset_point_from_tick(ax, x0, y0, nx, ny, font_offset + 1*tick_len)
                ax.text(
                    lx, ly,
                    tick_str_format.format(tick_val=theta_tick_val),
                    rotation=a,
                    **label_style
                    )
