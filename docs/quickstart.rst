Quick start
===========

This page walks through the steps, following installation, to build a pencil beam lightcone from both the snapshots and 
halo catalogues of the fiducial `L1_m10 FLAMINGO simulation <https://dataweb.cosma.dur.ac.uk:8443/flamingo/simulations/hydro.html#fiducial-models>`_.
We will build a pencil beam with an angular radius of :math:`\theta=2.5^{\circ}` that extends out to redshift 
:math:`z=0.3`.

:mod:`swiftlet` builds lightcones from snapshots in three steps:

#.  **Define the lightcone**: choose the simulation, the lightcone's geometry and its internal structure (how the 
    copies of the snapshot box are :term:`oriented <orientation>`).
#.  **Gather**: build the lightcone at the level of the snapshots' :term:`cells <cell>`, to find exactly which files and 
    cells to read, without reading any particle (or halo) data.
#.  **Read and place**: read the particles or haloes of those cells and place them in the lightcone, as the 
    :term:`gather step` planned.

For an overview of the main ideas and specific steps for building a lightcone with :mod:`swiftlet` see 
:doc:`user_guide/methods_overview` and :doc:`user_guide/building_lightcones`, respectively. 


Defining the lightcone 
----------------------

A lightcone is built from the snapshots with one of the two subclasses of
:class:`SnapshotLightcone <swiftlet.snapshot_lightcone.SnapshotLightcone>`, which differ only in their geometry:
:class:`SnapshotAllSky <swiftlet.snapshot_lightcone.SnapshotAllSky>` for the :term:`whole sky <all-sky lightcone>` 
(celestial sphere), and :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` for a :term:`beam` (or cone), 
which we will use in this pages. Unlike the a :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` requires 
both the direction and angular radius (``ang_radius_deg`` in degrees) of the beam. The direction of the beam is given by 
a vector (``beam_vector``), here :math:`\mathbf{v}_{\mathrm{beam}}=(0,0,1)`, to point the beam along the z-axis of the 
simulations box. 

In addition to the direction and angular radius of the beam , the lightcone also needs to know which simulation it is 
being built from. All the information on where to find the snapshots, :term:`SOAP catalogues <SOAP catalogue>` 
and the output redshifts is handled by a :class:`Simulation <swiftlet.simulation_suites.Simulation>` object. 

For now, we will not worry about the internal structure of the beam, we will use the default :term:`orientation table` 
(see:doc:`user_guide/orientation_tables`) and no :term:`orientation lock`, as the beam is not wider than the side length 
of a snapshot box (see :doc:`user_guide/orientation`). 

For further reading on defining a lightcone see :ref:`Defining a lightcone <define-lightcone>`.

.. code-block:: python

    from swiftlet import SnapshotBeam
    from swiftlet import Simulation
    
    # path to root (base) directory of the FLAMINGO L1_m10 fiducial simulation
    base_dir = "path/to/flamingo/data/L1000N1800/HYDRO_FIDUCIAL" 

    # define the Simulation object by where its snapshots, SOAP catalogues and output list are
    sim = Simulation.from_path(base_dir)
    
    snap_beam = SnapshotBeam(
        simulation=sim,       # the Simulation to build the lightcone from
        beam_vector=(0,0,1),  # unit vector giving the beams pointing on the sky
        ang_radius_deg=2.5,   # angular radius in degrees
    )


With the lightcone defined it can now be populated with haloes and particles. The same general pipeline is used for 
both particles and haloes: a :term:`gather step` that finds the files needed and then a :term:`read and place step` to 
position the particles or haloes in the lightcone.

.. tip::
    Before populating a lightcone with haloes or partiles it is often a good idea to check for any periodic replicas 
    of the same objects that appear :term:`close together <close repeat>` along a given line of sight 
    (see :doc:`user_guide/repeated_objects`).


.. _build-halo-lightcone:

Halo lightcone
--------------

Here we build a halo lightcone using the lightcone we just defined (``snap_beam``), and then write it to a set of 
:term:`HDF5 files <halo lightcone file>` with the same structure as the 
`LightconeIO <https://lightconeio.readthedocs.io/en/latest/halo_lightcones.html>`_ halo lightcones.

The halo lightcone is built with
:meth:`place_halos_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell>`, which includes both 
the :term:`gather step` and, :term:`read and place step`.

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
``"InputHalos/SOAPIndex"``). Any other property can be read from the SOAP catalogues in post, 
see :ref:`map-halo-lightcone-to-soap`.

As an example we will select only central haloes in the lightcone, as indicated by the SOAP catalogues, and then store 
the selected haloes in HDF5 files with the same format as the existing FLAMINGO halo lightcones: one file per snapshot 
(for more information see :ref:`write-halo-lightcone`). 

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

Starting from ``snap_beam``, we can populate the beam with particles using 
:meth:`place_snapshot_particles_in_shell <swiftlet.snapshot_lightcone.SnapshotLightcone.place_snapshot_particles_in_shell>`
runs both the gather step and the read and place step.

Here we will place gas particles (``PartType0``), along with their masses and temperatures into the lightcone:

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

Building a particle lightcone can be very computationally expensive, it is suggested that you build particle lightcones 
in thin redshift shells or by one snapshot file at a time to limit the peak memory used, see
:ref:`placing-files-one-at-a-time`.


