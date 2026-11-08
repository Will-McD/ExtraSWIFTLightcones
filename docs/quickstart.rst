Quick start
===========

This page walks through the steps, after installation, to build a pencil beam through the fiducial ``L1_m10`` FLAMINGO 
simulation from both its snapshots and halo catalogues. We will build a pencil beam with an angular radius of
:math:`\theta=2.5^{\circ}` that extends out to redshift :math:`z=0.3`.

Building a lightcone from snapshots takes three steps:

#.  :ref:`define-lightcone`: choose the simulation, the lightcone's geometry and its internal structure (how the copies
    of the snapshot box are oriented).
#.  Gather the simulation data needed: the lightcone is first built at snapshots cell level, from which we find the 
    exact files and cells to read without ever touching the particle (of halo) data. 
#.  Read and Place: place the particles or haloes of those cells into the lightcone, following the blueprint laid out 
    by the previous gather step.


.. _define-lightcone:

Defining the lightcone 
----------------------

First, define the FLAMINGO L1_m10 simulation as a :class:`Simulation <swiftlet.simulation_suites.Simulation>`, the
class SWIFTLET uses to find the snapshots, output redshifts and SOAP halo catalogues of the FLAMINGO
and COLIBRE simulation suites (see :doc:`user_guide/new_simulations_and_suites`). 
The :class:`Simulation <swiftlet.simulation_suites.Simulation>` is then passed to the lightcone.

A lightcone is built from the snapshots with one of the two subclasses of
:class:`SnapshotLightcone <swiftlet.snapshot_lightcone.SnapshotLightcone>`, which manage the lightcone's geometry:
:class:`SnapshotAllSky <swiftlet.snapshot_lightcone.SnapshotAllSky>` for the whole sky, and
:class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` for a pencil beam, used here. A lightcone needs the
simulation, the direction of the beam on the sky (for a beam), and how to orient the copies of the snapshot box that
fill it.

The direction of the beam is given by a unit vector, here :math:`\mathbf{v}_{\mathrm{beam}}=(0,0,1)`, to point the beam 
along the z-axis of the simulations box. 

For now, we do not worry about the internal structure of the beam: we use the default orientation table (see
:doc:`user_guide/orientation_tables`) and no orientation lock, as the beam is not wider than the side length of a
snapshot box (see :doc:`user_guide/orientation`).

.. code-block:: python

    from swiftlet import SnapshotBeam
    from swiftlet import Simulation
    
    sim = Simulation.from_label("L1000N0900", "HYDRO_FIDUCIAL") # FLAMINGO L1_m10 fiducial simulation
    
    snap_beam = SnapshotBeam(
        simulation_object=sim,       # pass all relevant simluation info too the lightcone as a Simulation
        beam_vector=(0,0,1),         # unit vector giving the beams pointing on the sky
    )


.. note::

    A beam is an exact copy of the same cone within an all-sky lightcone (with the same table, ``orientation_lock`` and
    ``orientation_seed``), whatever its direction as every copy of the snapshot box takes the same
    orientation in both (see :ref:`beams-are-part-of-the-all-sky-lightcone`).

The lightcone can now be populated with haloes and particles. Both follow the same workflow, an initial gather step that 
finds the files needed by building the lightcone at the cell level before then placing the particles (or haloes) of the 
gathered files into the lightcone.

.. tip::
    Before building a lightcone in full it is often a good idea to visualise the lightcone to check for any periodic 
    replicas of the same objects that appear close together along a given line of sight 
    (see :doc:`user_guide/repeated_objects`)


.. _build-halo-lightcone:

Halo Lightcone
--------------
Here we build a halo lightcone with ``snap_beam``, and write it to a HDF5 file with the same structure as the
`LightconeIO <https://lightconeio.readthedocs.io/en/latest/halo_lightcones.html>`_ halo lightcones.

The halo lightcone is built with
:meth:`place_halos_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>`, which includes both 
the gather step and, read and place step. It takes the paths of the SOAP catalogues as a format string with a
``{snap_nr}`` field: the simulation's ``soap_format``, from its suite (``SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5``
in the simulation's directory for FLAMINGO and COLIBRE), or its own if it was given one (see
:ref:`soap-halo-catalogues`).

..
    can be build in either one or two distinct steps using :meth:`gather_halo_catalogues <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>` to gather files and 
    then :meth:`place_halos_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>` to read and place
    haloes into the lightcone. Or alternatively using only :meth:`place_halos_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>`, 
    where the gather step is skipped within :meth:`place_halos_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>`
    if :meth:`gather_halo_catalogues <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>` has already been 
    called with the exact same parameters. 

.. code-block:: python
    
    # Read catalogues and place haloes into the lightcone using the blueprint given by snap_beam.gather_halo_catalogues
    halos = snap_beam.place_halos_in_shell(
        sim.soap_format,                            # formatted path to the SOAP halo catalogues
        lightcone_redshift_range=(0.001, 0.3),      # redshift range of the lightcone
        ang_radius_deg=2.5,                         # angular radius in degrees
        soap_chunk_size=2**22,                      # rows of a catalogue read at a time
    )

    halos["Lightcone/HaloCentre"]           # lightcone positions, with the observer at the origin
    halos["Lightcone/ExpansionFactor"]      # expansion factor at each halo's distance from the observer
    halos["Lightcone/SnapshotNumber"]       # snapshot each halo was taken from
    halos["InputHalos/SOAPIndex"]           # index of each halo in its snapshot's SOAP catalogue
    halos.metadata                          # the settings that made the lightcone


The output halo lightcone contains only the properties needed to place the halo in the lightcone 
(``"Lightcone/HaloCentre"``, ``"ExpansionFactor"``) and to map each halo to its properties stored in the SOAP halo 
catalogues (``"InputHalos/SOAPIndex"``, ``"InputHalos/SOAPIndex"``). The mapping between SOAP catalogues and created 
halo lightcones is done with :func:`read_soap_selected_rows <swiftlet.soap_reader.read_soap_selected_rows>`. 

The halo lightcone holds only the properties needed to place each halo in the lightcone (``"Lightcone/HaloCentre"``,
``"Lightcone/ExpansionFactor"``) and to match it back with the SOAP catalogues (``"Lightcone/SnapshotNumber"`` and 
``"InputHalos/SOAPIndex"``). Any other property can be read from the SOAP catalogues with :func:`read_soap_selected_rows <swiftlet.soap_reader.read_soap_selected_rows>`.

For demonstration purposes we will now write a .hdf5 file for the halo lightcone using only the haloes from the highest 
redshift snapshot, in this saved catalogue we will also include, from SOAP, if the halo is a central:

As an example we can write a HDF5 file storing only the haloes in the lightcone above that are from the highest redshift 
snapshot read in. Additionally we will also include if each halo is a central, as read from SOAP:

.. code-block:: python
    
    import numpy as np
    from swiftlet.snapshot_lightcone import write_halo_lightcone_file
    from swiftlet.soap_reader import read_soap_selected_rows

    snap_nr = halos["Lightcone/SnapshotNumber"].value.astype(int)
    soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)

    snap = int(snap_nr.min())  # the highest redshift snapshot
    keep = snap_nr == snap     # keep haloes in this snapshot
    snapshot_haloes = {name: values[keep] for name, values in halos.items()} # make dict of haloes kept 
    
    # the rows of the catalogue to read, and where each halo of the lightcone is among them
    rows, where = np.unique(soap_index[keep], return_inverse=True)  
    
    # read properties for each halo from the corresponding soap catalogue
    soap_filename = soap_format.format(snap_nr=snap)
    is_central = read_soap_selected_rows(soap_filename, ["InputHalos/IsCentral"], rows)["InputHalos/IsCentral"][where]

    halo_filename = f"./lightcone_halos_{snap:04d}.hdf5" # the file to write
    write_halo_lightcone_file(
        halo_filename,                                            
        snapshot_haloes,                                          # the haloes from the lightcone
        soap_filename,                                            # the soap catalogue, for metadata 
        properties = {"InputHalos/IsCentral": is_central.value},  # SOAP properties not in the halo lightcone
        metadata = halos.metadata                                 # the settings that made the lightcone
    )

The file can be read back with `LightconeIO <https://lightconeio.readthedocs.io/en/latest/halo_lightcones.html>`_,
the same as any existing halo lightcone from FLAMINGO, with all the same functionality including the ability to 
spatially query the lightcone.

.. code-block:: python
    
    from lightcone_io.halo_reader import HaloLightconeFile
    
    # read the file back with lightcone_io
    halo_file = HaloLightconeFile(halo_filename, soap_filename=soap_filename)
    read_properties=[                   
        "Lightcone/HaloCentre",
        "Lightcone/Redshift",
        "InputHalos/IsCentral",
        "InputHalos/HaloCatalogueIndex"
        ] 
    haloes = halo_file.read_halos(read_properties)


.. _build-part-lightcone:

Particle Lightcone
------------------

The same beam is populated with particles in the same way.

Again starting from ``snap_beam``,
:meth:`place_snapshot_particles_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_snapshot_particles_in_shell>`
runs both the gather step and the read and place step.

Here the masses and temperatures of the gas particles (``PartType0``) are placed in the lightcone:

.. code-block:: python

    # Read and place particles into the lightcone
    shell_particles = snap_beam.place_snapshot_particles_in_shell(
        lightcone_redshift_range=(0.05, 0.3),
        ang_radius_deg=2.5,
        property_names=["Masses", "Temperatures"],
        particle_types=["PartType0"],
    )

    gas = shell_particles["PartType0"]
    gas["Coordinates"]       # lightcone positions, with the observer at the origin
    gas["ExpansionFactors"]  # expansion factor at each particle's distance from the observer
    gas["SnapshotNumber"]    # snapshot each particle was taken from
    gas["SmoothingLengths"]  # always returned for gas particles


.. Note::
    **``SmoothingLengths`` are always returned** for any particle that has them.


To place a large lightcone one snapshot file at a time, and therefore limit the peak memory used, see
:ref:`placing-files-one-at-a-time`.


Visualise the lightcone
^^^^^^^^^^^^^^^^^^^^^^^

A slice through a particle lightcone can be quickly projected using :class:`BeamProjection <swiftlet.lightcone_projections.BeamProjection>`.
Following a similar workflow to building the lightcones, we first define a slice of the lightcone, add particles to the 
slice before then projecting. 

A slice through a particle lightcone can be projected with
:class:`BeamProjection <swiftlet.lightcone_projections.BeamProjection>`. As when building lightcones, we define a
slice, add particles to it, and then project it (see :doc:`user_guide/projections`).

.. code-block:: python
    
    from swiftlet import BeamProjection
    from swiftlet import SnapshotCosmology

    # cosmology object from simulation
    cosmo = SnapshotCosmology.from_simulation(sim)

    # define the slice through the particle lightcone
    lc_slice = BeamProjection(
        vector=(0, 0, 1),                           # direction of the beam
        angular_diameter=4,                         # [deg] beam's diameter (2 x its radius)
        redshift_range=(0.001, 0.3),                # redshift range to project
        slice_thickness=10,                         # [Mpc] thickness of the slice across the line of sight
        cosmology=cosmo,                            # the simulation's cosmology
        store_snapshot_filename=sim.snapshot_format_whole.format(snap_nr=sim.z0_snapshot), # snapshot, for the metadata
    )

    # the properties of the particles to add to the slice
    properties_for_slice = [
            "Coordinates", 
            "Masses", 
            "SmoothingLengths", 
            "ExpansionFactors", 
            "Temperatures"
        ]
    
    # place the gas particles of the beam in the slice
    lc_slice.place_particles_in_slice(gas, properties_for_slice)
    
    # the projected surface density, with every mass smoothed
    gas_surface_density = lc_slice.project_properties(["Masses"], ptype="Gas", assign_units=["Msun"])[0]
    
    # the projected mass-weighted temperature
    gas_temperature = lc_slice.project_properties(["Temperatures"], ptype="Gas", weight="Masses")[0]

    # plot the projection for both surface density and mass weighted temperatures
    fig, ax, wedge_images = lc_slice.split_beam_plot(
        numb_wedges=2,
        projection_data=[[gas_temperature.to_value("K"), None],[gas_surface_density.to_value("Msun * Mpc**-2"), None]], 
        colour_maps=["viridis", "magma"],                   # individually set each wedges colour map
        norms=["log", "log"],                               # or "linear", or any matplotlib Normalize, "log" by default
        overlay_grid=(False, False, False),                 # grid lines at the redshift, distance and angle ticks
        filename="beam.png",                                # save image to this file
    )



