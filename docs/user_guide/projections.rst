Lightcone projections
=====================


SWIFT includes a module: :mod:`swiftlet.lightcone_projections`, that includes a basic template for making nice 
looking projections of any particle lightcone. These tools can be useful for visualising the cosmic web and the redshift 
evolution of different particle properties within the lightcone. 

This page contains an example of how to make a projection of the surface density of a particle lightcone built from the 
snapshots of the L400_m7 COLIBRE simulation. 

.. currentmodule:: swiftlet.lightcone_projections

Project particles in a slice
----------------------------

:class:`BeamProjection` makes projections of a slice through a beam of a lightcone using the 
`swiftsimio.visualisation.projection <https://swiftsimio.readthedocs.io/en/latest/visualisation/projection.html>`_
submodule backends. The particles can come from the FLAMINGO particle lightcones (a path, or ``lightcone_io`` particle 
data) or from a snapshot using :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` and 
:class:`SnapshotAllSky <swiftlet.snapshot_lightcone.SnapshotAllSky>`.

As shown in the example below, to make a projection:

* **Define the projected slice:** defining a slice through the lightcone requires:

  1.    A beam within the lightcone (i.e. pass a ``vector``, ``angular_diameter`` and ``redshift_range`` that define the
        beam).
  2.    The thickness of the slice through the beam (``slice_thickness``).
  3.    Either a cosmology object for the simulation corresponding to the intended lightcone data, or a path to a 
        snapshot of the same simulation.

  Note that the projected beam and the corresponding particles are rotated to lie along the x-axis, so the slice is 
  always through the middle of the beam, along the z-axis.

* **Add particles to the slice:** add the lightcone particle data (or an equivalent dictionary
  object of ``unyt`` arrays) and a list of the properties you want to project to the slice via
  :meth:`BeamProjection.place_particles_in_slice`.

* **Project selected properties:** for a given particle type in the slice, create a projection
  of the selected properties (in the assigned units) using :meth:`BeamProjection.project_properties`.
  The slice is projected with the `swiftsimio.visualisation.projection <https://swiftsimio.readthedocs.io/en/latest/visualisation/projection.html>`_
  submodule backends, but in the shape of the slice. Gas particles use their own smoothing lengths, and smoothing 
  lengths are generated for dark matter.


.. code-block:: python

    
    from swiftlet import Simulation, BeamProjection, SnapshotBeam, SnapshotCosmology

    sim = Simulation.from_label("L0400N3008", "Thermal") # L400m7 COLIBRE, Thermal AGN
    cosmo = SnapshotCosmology.from_simulation(sim)        # its cosmology

    # build the particle lightcone: 
    snap_beam = SnapshotBeam(
        simulation=sim,
        beam_vector=(0,0,1),         # point the beam along the z-axis
        orientation_seed=0,          # reproduces the same box orientations
        orientation_lock="cube",     # set strict layers of the lightcones tiles
    )

    shell_particles = snap_beam.place_snapshot_particles_in_shell(
        lightcone_redshift_range=(0.001, 0.11),     # redshift range of the lightcone
        ang_radius_deg=8,                           # angular radius in degrees
        property_names=["Masses", "Temperatures"],
        particle_types=["PartType0", "PartType1"],
    )

    # define the slice so it fits within the beam
    lc_slice = BeamProjection(
        vector=(0, 0, 1),                           # direction of the beam
        angular_diameter=15,                        # [deg]
        redshift_range=(0.01, 0.11),                # redshift range of the lightcone
        slice_thickness=10,                         # [Mpc]
        cosmology=cosmo,                            # Cosmology specific to simulation used to build projection
        store_snapshot_filename=sim.snapshot_format_whole.format(snap_nr=sim.z0_snapshot)
    )

    # place the gas and dark matter particles of the beam in the slice
    lc_slice.place_particles_in_slice(
        shell_particles["PartType0"],
        ["Coordinates", "Masses", "SmoothingLengths", "ExpansionFactors", "Temperatures"],
        dm_particle_data=shell_particles["PartType1"],
        dm_property_names=["Coordinates", "Masses", "ExpansionFactors"],
    )

    # surface densities: gas, dark matter, total
    gas_surface_density = lc_slice.project_properties(["Masses"], ptype="Gas", assign_units=["Msun"])[0]
    dm_surface_density = lc_slice.project_properties(["Masses"], ptype="DM", assign_units=["Msun"])[0]
    total_surface_density = gas_surface_density + dm_surface_density
    
:meth:`project_properties <BeamProjection.project_properties>` returns one image per property (``unyt`` arrays):

* **Default:** the surface density of each property.
* **weight:** the weighted mean, :math:`\sum(q w) / \sum(w)`, and zero where there is no weight.
* **Other options:** ``resolution`` sets the number of pixels along each axis, and ``periodic`` (default ``False``) 
  treats the box as periodic. ``parallel`` uses SWIFTsimIO's parallel projection backend. Other properties can be added 
  to the slice with :meth:`BeamProjection.add_property_to_slice`.


Plotting beams
--------------

:meth:`BeamProjection.split_beam_plot` draws the beam as a wedge in comoving distance and angle, with redshift, comoving
distance and angular radius axes. The beam can be split into several wedges, each showing a different particle type 
and/or property:

.. code-block:: python

    images = [image.to_value("Msun/Mpc**2") for image in (gas_surface_density, total_surface_density, dm_surface_density)]
    pix_min, pix_max = 1e10, 1e15                       # colour range of pixels in the image [Msun / Mpc^2]
    fig, ax, wedge_images = lc_slice.split_beam_plot(
        3,
        [[image, (pix_min, pix_max)] for image in images],  # each wedge's image and colour range
        ["magma", "cubehelix", "viridis"],                  # individually set each wedge's colour map
        titles=["Gas", "Gas+DM", "DM"],                     # the title placed in each wedge
        norms=["log", "log", "log"],                        # or "linear", or any matplotlib Normalize
        overlay_grid=(False, False, False),                 # grid lines at redshift, distance and angle ticks
        filename="colibre_beam.png",                        # save image to this file
    )

It either draws a new figure or adds each wedge to an existing axes, then returns the figure, the axes and the image of 
each wedge. Other keyword arguments style the wedges, ticks, labels and grid lines (see :meth:`BeamPlot.add_wedge <swiftlet.beam_plotting.BeamPlot.add_wedge>` and 
:meth:`BeamPlot.add_beam_axes <swiftlet.beam_plotting.BeamPlot.add_wedge>`).

The plotting is done by :class:`BeamPlot <swiftlet.beam_plotting.BeamPlot>`, which can also be used on its own with a 
cosmology, angular diameter, redshift range and the extent of the images.
