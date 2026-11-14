Quick start
===========

This page walks through the steps, after installation, to build a pencil beam through the fiducial ``L1_m10`` FLAMINGO 
simulation from both its snapshots and halo catalogues. We will build a pencil beam with an angular radius of
:math:`\theta=2.5^{\circ}` that extends out to redshift :math:`z=0.3`.

Building a lightcone from snapshots takes three steps:

#.  Defining the lightcone: choose the simulation, the lightcone's geometry and its internal structure (how the copies
    of the snapshot box are oriented).
#.  Gather the simulation data needed: the lightcone is first built at snapshots cell level, from which we find the 
    exact files and cells to read without ever touching the particle (or halo) data. 
#.  Read and place: place the particles or haloes of those cells into the lightcone, following the blueprint laid out 
    by the previous gather step.

For an overview of the main ideas and specific steps involved with building a lightcone see 
:doc:`user_guide/methods_overview` and :doc:`user_guide/building_lightcones`, respectively. 

.. _define-lightcone:

Defining the lightcone 
----------------------

A lightcone is built from the snapshots with one of the two subclasses of
:class:`SnapshotLightcone <swiftlet.snapshot_lightcone.SnapshotLightcone>`, which manage the lightcone's geometry:
:class:`SnapshotAllSky <swiftlet.snapshot_lightcone.SnapshotAllSky>` for the whole sky, and
:class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` for a beam (or cone), which we will use in this pages 
example. Unlike the a :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` requires both the direction and 
angular radius (``ang_radius_deg`` in degrees) of the beam. The direction of the beam is given by a vector 
(``beam_vector``), here :math:`\mathbf{v}_{\mathrm{beam}}=(0,0,1)`, to point the beam along the z-axis of the 
simulations box. 

In addition to the direction and angular radius of the beam , the lightcone also needs to known which simulation it is 
being built from. In SWIFTLET all the required information on how to find the snapshots, SOAP catalogues and the output 
redshifts is handled by a :class:`Simulation <swiftlet.simulation_suites.Simulation>` object, which is passed to the 
:class:`SnapshotLightcone <swiftlet.snapshot_lightcone.SnapshotLightcone>`.

For now, we will not worry about the internal structure of the beam, we will use the default orientation table (see
:doc:`user_guide/orientation_tables`) and no orientation lock, as the beam is not wider than the side length of a
snapshot box (see :doc:`user_guide/orientation`). 

.. code-block:: python

    from swiftlet import SnapshotBeam
    from swiftlet import Simulation
    
    # path to root directory of the FLAMINGO L1_m10 fiducial simulation
    base_dir = "path/to/flamingo/data/L1000N1800/HYDRO_FIDUCIAL" 

    # define the Simulation object
    sim = Simulation.from_path(base_dir)
    
    snap_beam = SnapshotBeam(
        simulation=sim,       # pass all relevant simluation info too the lightcone as a Simulation
        beam_vector=(0,0,1),         # unit vector giving the beams pointing on the sky
        ang_radius_deg=2.5,          # angular radius in degrees
    )

The lightcone can now be populated with haloes and particles. Both follow the same workflow, an initial gather step that 
finds the files needed by building the lightcone at the cell level before then placing the particles (or haloes) of the 
gathered files into the lightcone.

.. tip::
    Before building a lightcone in full it is often a good idea to check for any periodic replicas of the same objects 
    that appear close together along a given line of sight (see :doc:`user_guide/repeated_objects`).


.. _build-halo-lightcone:

Halo lightcone
--------------

Here we build a halo lightcone with ``snap_beam``, and write it to a HDF5 file with the same structure as the
`LightconeIO <https://lightconeio.readthedocs.io/en/latest/halo_lightcones.html>`_ halo lightcones.

The halo lightcone is built with
:meth:`place_halos_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>`, which includes both 
the gather step and, read and place step.

.. code-block:: python
    
    # Read catalogues and place haloes into the lightcone 
    halos = snap_beam.place_halos_in_shell(
        sim.soap_format,                            # formatted path to the SOAP halo catalogues
        lightcone_redshift_range=(0.05, 0.3),       # redshift range of the lightcone
    )

    halos["Lightcone/HaloCentre"]           # lightcone positions, with the observer at the origin
    halos["Lightcone/ExpansionFactor"]      # expansion factor at each halo's distance from the observer
    halos["Lightcone/SnapshotNumber"]       # snapshot each halo was taken from
    halos["InputHalos/SOAPIndex"]           # index of each halo in its snapshot's SOAP catalogue
    halos.metadata                          # the settings that made the lightcone


The halo lightcone holds only the properties needed to place each halo in the lightcone (``"Lightcone/HaloCentre"``,
``"Lightcone/ExpansionFactor"``) and to match it back with the SOAP catalogues (``"Lightcone/SnapshotNumber"`` and 
``"InputHalos/SOAPIndex"``). Any other property can be read from the SOAP catalogues with 
:func:`read_soap_selected_rows <swiftlet.soap_reader.read_soap_selected_rows>`.


As an example we can sort the haloes in the lightcone by select properties from SOAP and then store the selected haloes 
as a HDF5 files in the same format as the existing FLAMINGO halo lightcones: one file per snapshot (see 
:ref:`write-halo-lightcone` for more information on writing halo lightcone files). 

Here we will store one file per snapshot for only the central haloes in the lightcone:

.. code-block:: python
    
    import numpy as np
    from swiftlet.snapshot_lightcone import write_halo_lightcone_file
    from swiftlet.soap_reader import read_soap_selected_rows

    snap_nr = halos["Lightcone/SnapshotNumber"].value.astype(int)
    soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)
    numb_haloes = len(soap_index)
    
    # iterate through snapshots haloes are from 
    for snap in np.unique(snap_nr):

        keep = snap_nr == snap     # keep haloes in this snapshot
        soap_filename = sim.soap_file(snap) # SOAP catalogue for this snapshot 

        # the rows of the catalogue to read, and where each halo of the lightcone is among them
        rows, where = np.unique(soap_index[keep], return_inverse=True) 

        # "InputHalos/IsCentral" is 1 for central haloes and 0 otherwise 
        is_central = read_soap_selected_rows(soap_filename, ["InputHalos/IsCentral"], rows)["InputHalos/IsCentral"][where]
        
        # write halo lightcone file for snapshot 
        halo_filename = f"./lightcone_halos_{snap:04d}.hdf5" # the file to write
        write_halo_lightcone_file(
            halo_filename,                                            
            {name: values[keep] for name, values in halos.items()},   # the haloes from the lightcone 
            soap_filename,                                            # the soap catalogue, for metadata 
            properties = {"InputHalos/IsCentral": is_central.value},  # SOAP properties not in the halo lightcone
            metadata = halos.metadata                                 # the settings that made the lightcone
        )


The files can be read back with `LightconeIO <https://lightconeio.readthedocs.io/en/latest/halo_lightcones.html>`_,
the same as any of FLAMINGO's halo lightcones. 


.. code-block:: python
    
    from lightcone_io.halo_reader import HaloLightconeFile
    
    # read the file back with lightcone_io
    halo_filename = f"./lightcone_halos_{np.unique(snap_nr)[0]:04d}.hdf5"
    halo_file = HaloLightconeFile(halo_filename, soap_filename=soap_filename)
    read_properties=[                   
        "Lightcone/HaloCentre",
        "Lightcone/Redshift",
        "InputHalos/IsCentral",
        "InputHalos/HaloCatalogueIndex"
        ] 
    haloes = halo_file.read_halos(read_properties)


.. _build-part-lightcone:

Particle lightcone
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
        property_names=["Masses", "Temperatures"],
        particle_types=["PartType0"],
    )

    gas = shell_particles["PartType0"]
    gas["Coordinates"]       # lightcone positions, with the observer at the origin
    gas["ExpansionFactors"]  # expansion factor at each particle's distance from the observer
    gas["SnapshotNumber"]    # snapshot each particle was taken from
    gas["SmoothingLengths"]  # always returned for gas particles


.. Note::
    ``SmoothingLengths`` **are always returned** for any particle that has them.


To place a large lightcone one snapshot file at a time, and therefore limit the peak memory used, see
:ref:`placing-files-one-at-a-time`.


