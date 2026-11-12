Halo lightcones
===============

.. currentmodule:: swiftlet.snapshot_lightcone

A halo lightcone is built from a simulation's SOAP catalogues, exactly as a particle lightcone is
built from the snapshots. Each halo from the catalogue is moved into every tile its snapshot fills, as a particle at its 
centre would be, and kept if it lies in the shell (and the beam). This page covers placing the haloes, finding their 
SOAP properties, writing them to files that LightconeIO can read, and placing them with MPI.


Placing haloes
--------------

:meth:`SnapshotLightcone.place_halos_in_shell` places the haloes of a redshift shell. It takes the paths of the SOAP
catalogues as a format string with a ``{snap_nr}`` field, usually the simulation's ``soap_format`` (see
:ref:`simulation parameter <table-of-values>`).

.. code-block:: python

    from swiftlet import Simulation, SnapshotAllSky, SnapshotBeam

    sim = Simulation.from_label("L0400N3008", "Thermal")    # COLIBRE L400_m7, Thermal AGN

    # all-sky
    all_sky = SnapshotAllSky(simulation_object=sim, orientation_lock="cube")
    halos = all_sky.place_halos_in_shell(sim.soap_format, (0.001, 0.1))

    # a beam of 5 degrees radius
    beam = SnapshotBeam(simulation_object=sim, beam_vector=(1, 2, 3),  ang_radius_deg=5., orientation_lock="cube")
    beam_halos = beam.place_halos_in_shell(sim.soap_format, (0.001, 0.3))

The result is a :class:`LightconeOutput`, one row per halo placed, holding only what is needed to place the haloes in 
the lightcones and to map them back to their position in the SOAP catalogue:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Property
     - What it is

   * - ``Lightcone/HaloCentre``
     - the halo's centre in the lightcone [comoving Mpc], with the observer at the origin, moved exactly as a
       particle's ``Coordinates``

   * - ``Lightcone/ExpansionFactor``
     - the expansion factor at the halo's comoving distance from the observer

   * - ``Lightcone/SnapshotNumber``
     - the snapshot the halo was taken from

   * - ``InputHalos/HaloCatalogueIndex``
     - the halo's index as written in its SOAP catalogue

   * - ``InputHalos/SOAPIndex``
     - the row of the halo in its snapshot's SOAP catalogue, so all other related properties can be mapped to the 
       lightcone

Its ``.metadata`` records the settings that made the halo lightcone (simulation, orientation table, lock, seed, beam and 
redshift range), as for particles (see :doc:`building_lightcones`).

A halo appears once in every tile its snapshot fills, where its copy lies in the shell, so the same halo can be in the
lightcone several times, from different tiles (see :doc:`repeated_objects`).

How the catalogues are read
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SOAP catalogues are not sorted by cell, so, unlike the particles of a snapshot, every halo of a catalogue is tested.
Each catalogue is read once, and its haloes placed in every tile of its snapshot.

*   ``soap_chunk_size`` sets the number of rows read at a time, to limit the memory used. If ``None``, each catalogue is
    read whole (in serial) or in one piece per MPI rank.
*   The gather step, :meth:`SnapshotLightcone.gather_halo_catalogues`, finds the catalogues and tiles to read without
    reading any halo, from the cell metadata of one snapshot only (the box and cell grid are the same in every
    snapshot). :meth:`place_halos_in_shell <SnapshotLightcone.place_halos_in_shell>` runs it, and reuses it if it has
    already been run with the same arguments.
*   A SOAP catalogue missing on disk is filled by the catalogues of the neighbouring snapshots, with a
    :class:`SnapshotGapWarning`, the same as for missing snapshots.

The haloes placed are the same whether the catalogues are read whole, in chunks or with MPI.

.. _map-halo-lightcone-to-soap:

Halo properties from SOAP
-------------------------

Any other property of a halo is read from its SOAP catalogue using ``Lightcone/SnapshotNumber`` and
``InputHalos/SOAPIndex``. :func:`read_soap_selected_rows <swiftlet.soap_reader.read_soap_selected_rows>` only reads the 
rows of a catalogue that have been passed to it:

.. code-block:: python

    import numpy as np
    from swiftlet.soap_reader import read_soap_selected_rows

    snap_nr = halos["Lightcone/SnapshotNumber"].value.astype(int)
    soap_index = halos["InputHalos/SOAPIndex"].value.astype(int)

    m200c = np.zeros(len(snap_nr))
    for snap in np.unique(snap_nr):
        in_snap = snap_nr == snap
        # each row once, and where each halo of the lightcone is among them
        rows, where = np.unique(soap_index[in_snap], return_inverse=True)
        soap = read_soap_selected_rows(sim.soap_file(snap), ["SO/200_crit/TotalMass"], rows)
        m200c[in_snap] = soap["SO/200_crit/TotalMass"].to_value("Msun")[where]

    massive = m200c > 1e13                  # e.g. keep the haloes above 10^13 Msun
    central = ...                           # or select on any other property, e.g. "InputHalos/IsCentral"

The properties have the same units as LightconeIO's SOAP reader, with the scale factor ``a`` dropped from comoving
properties. To find the rows of a catalogue selected by a property before building a lightcone, e.g. every halo above a
mass, use :func:`select_soap_rows <swiftlet.soap_reader.select_soap_rows>` (see
:doc:`additional_tools`).

.. _write-halo-lightcone:

Writing halo lightcone files
----------------------------

:func:`write_halo_lightcone_file` writes the haloes of one snapshot to an HDF5 file laid out like FLAMINGO's halo
lightcones, so it can be read with `LightconeIO <https://lightconeio.readthedocs.io/en/latest/halo_lightcones.html>`_:

*   the haloes are sorted by HEALPix pixel of their direction (``nside=16``, nested ordering, as LightconeIO uses), with
    an index of the haloes in each pixel, so a patch of sky can be read without reading the whole file
*   the units, ``Cosmology`` and ``Units`` groups are taken from the snapshot's SOAP catalogue
*   other SOAP properties of each halo can be written with them, as ``properties``: each name must be a dataset of
    the SOAP catalogue, whose units are copied with it
*   the lightcone's ``.metadata`` is written too, so the file records how it was made



.. code-block:: python

    from swiftlet.snapshot_lightcone import write_halo_lightcone_file

    snap_nr = halos["Lightcone/SnapshotNumber"].value.astype(int)
    for snap in np.unique(snap_nr):
        keep = snap_nr == snap
        write_halo_lightcone_file(
            f"lightcone_halos_{snap:04d}.hdf5",                             # one file per snapshot
            {name: values[keep] for name, values in halos.items()},         # the haloes of this snapshot
            sim.soap_file(snap),                                            # their SOAP catalogue, for its metadata
            {"SO/200_crit/TotalMass": m200c[keep]},                         # SOAP properties to write with them
            halos.metadata,                                                 # the settings that made the lightcone
        )


.. Note::
    When writing these files it is suggested that you write one file per snapshot, so that when read back with 
    LightconeIO's ``HaloLightconeFile`` the lightcone haloes can be paired with a SOAP catalogue. However, haloes from 
    multiple snapshots can be written to the same file and read with ``HaloLightconeFile`` however it cannot pair the 
    haloes in the file with a SOAP catalogue (one file per snapshot) when read in, instead this can be done manually, 
    snapshot-by-snapshot after reading the haloes in. 


Read a file back with LightconeIO, e.g. only the haloes within an angle of a direction:

.. code-block:: python

    from lightcone_io.halo_reader import HaloLightconeFile

    halo_file = HaloLightconeFile("lightcone_halos_0123.hdf5", soap_filename=sim.soap_file(123))
    haloes = halo_file.read_halos(["Lightcone/HaloCentre", "InputHalos/SOAPIndex"])
    patch = halo_file.read_halos_in_radius(
        np.array([1., 2., 3.]) / np.sqrt(14.), np.deg2rad(5.), ["Lightcone/HaloCentre", "InputHalos/SOAPIndex"],
    )


With MPI
--------

With MPI, the rows of the catalogues are distributed across the ranks, with each rank receiving ``soap_chunk_size`` 
rows (see :doc:`parallel_and_hpc`). Each rank returns the haloes it placed unless ``redistribute_halos=True``, then all 
haloes are shared evenly between the ranks. 
The example ``make_colibre_halo_lightcone.py`` builds an all-sky halo lightcone of COLIBRE L400_m7 (see
:doc:`../examples`).
