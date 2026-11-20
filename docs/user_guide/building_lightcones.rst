Building lightcones
===================

.. currentmodule:: swiftlet.snapshot_lightcone


This page covers how to build a lightcone with :mod:`swiftlet`, this includes an overview of how to use the
:mod:`swiftlet.snapshot_lightcone` module and a :ref:`step-by-step <lightcone-step-by-step>` 
walkthrough of the main methods in the :class:`SnapshotLightcone` class that is used to build lightcones.
See :doc:`methods_overview` for an overview of the main ideas behind how lightcones are built in :mod:`swiftlet`. 

All lightcones in :mod:`swiftlet` are classified as either an :term:`all-sky <all-sky lightcone>` or :term:`beam` 
lightcone. The former are lightcones with a footprint that covers the whole sky (has a solid angle of :math:`4 \pi` 
steradians) and are built with :class:`SnapshotAllSky` class. The latter, are lightcones whose footprint is given by a 
circular beam (with an angular radius less than 60 degrees, a solid angle of :math:`\leq \pi` steradians) and are 
constructed with the :class:`SnapshotBeam` class. 
Both :class:`SnapshotAllSky` and :class:`SnapshotBeam` are subclasses of :class:`SnapshotLightcone`, which only contain 
information specific to the geometry of the lightcone it builds: an all-sky or beam lightcone. 
The methods used for building a lightcone are held by :class:`SnapshotLightcone` and shared by both subclasses.
As shown in :doc:`methods_overview`, :mod:`swiftlet` builds lightcones by taking copies of the simulations snapshot box 
and placing them into a lattice around the observer, before then populating the lattice using slices of particle and 
haloes from snapshots chosen by their redshift (comoving distance from the observer). 

.. _lightcone_building_step_list:

In practice we use :class:`SnapshotLightcone` to build lightcones in main three steps:

#.  **Defining the lightcone**: choose the simulation, the lightcone's geometry (an :term:`all-sky lightcone` or 
    :term:`beam`) and its internal structure (how the copies of the snapshot box are :term:`oriented <orientation>`).
#.  **Gather**: build the lattice containing the lightcone at the :term:`cell` level and mapping each cell the lightcone 
    passes through to the same cell in a snapshot that is output at the redshift corresponding to the comoving distance 
    from the observer in the lightcone. This step finds the exact files and cells to read without ever touching the 
    particle or halo data. 
#.  **Read and place**: read the particles or haloes of those cells, keeping only those within the lightcone and place 
    them into the lightcone, following the blueprint laid out by the previous step.

.. important::
   **Lightcones should be built in thin redshift shells:**
   the comoving volume of a lightcone quickly grows larger than that of a snapshot. For example, with the COLIBRE
   ``L400_m7`` simulations, an all-sky lightcone for redshifts :math:`z=0 - 0.2` (a comoving radius of 
   :math:`\approx 838` Mpc) holds the comoving volume, and disk space, of :math:`\approx 39` snapshots.


.. _define-lightcone:

Defining a lightcone
--------------------

To define a lightcone it needs:

*   **the simulation**: a :class:`Simulation <swiftlet.simulation_suites.Simulation>` object, passed as 
    ``simulation`` (see :doc:`new_simulations_and_suites`)
*   **its geometry**: handled by the :class:`SnapshotLightcone` subclasses, :class:`SnapshotAllSky` and 
    :class:`SnapshotBeam`, which needs a direction (``beam_vector``) and angular radius (``ang_radius_deg``).
*   **the orientation of its tiles**: the ``orientation_table`` (the default table is used if one is not specified), the
    ``orientation_lock`` (``none``, ``"cube"`` or ``"sphere"``) and the ``orientation_seed`` (see :doc:`orientation`
    and :doc:`orientation_tables`).


.. code-block:: python

    from swiftlet import Simulation, SnapshotAllSky, SnapshotBeam

    sim = Simulation.from_label("L0400N3008", "Thermal")   # COLIBRE L400_m7, Thermal AGN

    # define an all-sky lightcone
    all_sky = SnapshotAllSky(simulation=sim, orientation_lock="cube", orientation_seed=0)
    
    # define a beam lightcone
    beam = SnapshotBeam(
        simulation=sim, 
        beam_vector=(1, 2, 3), 
        ang_radius_deg=2.0, 
        orientation_lock="cube", 
        orientation_seed=0
        )

With the same internal structure (set by the :term:`orientation tables <orientation table>`, lock and seed: see 
:doc:`orientation`), in the code block above ``beam`` gives the exact same lightcone as the same subvolume within 
``all_sky`` (see :ref:`beams-are-part-of-the-all-sky-lightcone`).

.. tip::
   Before building a large lightcone it is a good idea to check how close periodic replicas will be within the 
   lightcone. This can be done at the cell level with 
   :func:`find_close_cell_repeats <swiftlet.repeat_check.find_close_cell_repeats>` (see :doc:`repeated_objects`).


Populating a lightcone
----------------------

A lightcone is filled one redshift shell at a time, with particles or haloes, by the gather and the read and place
steps above.

A lightcone is populated with particles or haloes one shell at a time (described in :doc:`methods_overview`), by 
the :term:`gather <gather step>` and the :term:`read and place <read and place step>` steps from the 
:ref:`list <lightcone_building_step_list>` above.

The diagram below sketches out what the end result of the gather step looks like for a lightcone :term:`beam`. 

.. _fig-cell_selection:

.. figure:: images/snapshot_beam_cell_selection.png
   :figwidth: 100%
   :width: 100%
   :align: left
   :alt: The tiles and cells a lightcone beam passes through, in two slices through the lattice.

   The tiles added to the lattice and the cells that the lightcone passes through are shaded and hatched, respectively. 
   Left: The xz plane, showing the line along the beam. Right: The zy plane, showing the cross section of the beam 
   indicated by the magenta line on the left panel. 


The methods that do both steps at once are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Places

   * - :meth:`SnapshotLightcone.place_snapshot_particles_in_shell`
     - the particles of one or more particle types, with the requested properties. The particles ``Coordinates``,
       ``ExpansionFactors``, ``SnapshotNumber`` and ``ParticleIDs`` are always tracked (``SmoothingLengths`` 
       are included for every type but dark matter, ``PartType1``). ``Velocities``, if asked for, are re-oriented with 
       the coordinates

   * - :meth:`SnapshotLightcone.place_halos_in_shell`
     - the haloes of the :term:`SOAP catalogues <SOAP catalogue>`, at the paths of the simulation's ``soap_format`` (see
       :ref:`simulation parameters <table-of-values>`), with the properties needed to find each halo in its catalogue 
       (see :doc:`halo_lightcones`)

Both methods can work with MPI to share the work between ranks (see :doc:`parallel_and_hpc`).

If a **simulation's snapshot is missing on disk** then snapshots next to it cover its redshift range, as long as the 
snapshots are less than 0.1 apart in redshift by default. Any time this occurs a :class:`SnapshotGapWarning` is given. 


.. _max-snapshot-spacing:

Redshift between snapshots and missing snapshots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A lightcone is filled by the snapshots whose redshift ranges cover the lightcones comoving distance, each filling the 
redshifts between the midpoints to its neighbours (see :doc:`methods_overview` and :doc:`new_simulations_and_suites`). 
Snapshots missing on disk are filled by the available ones and you are warned each time this happens with a 
:class:`SnapshotGapWarning <swiftlet.snapshot_lightcone.SnapshotGapWarning>`. By default if the snapshots are 0.1 or 
more apart in redshift (:math:`\Delta z =0.1`), a ``ValueError`` is raised before any data is read.

:envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` sets another limit and can be changed if needed:

.. code-block:: bash

    export SWIFTLET_MAX_SNAPSHOT_SPACING=0.22     # for one shell or job script

.. code-block:: python

    import os
    os.environ["SWIFTLET_MAX_SNAPSHOT_SPACING"] = "0.22"

    from swiftlet.snapshot_redshifts import max_snapshot_spacing
    max_snapshot_spacing()                         # the limit used: 0.22


.. caution::

    **Keep the snapshot spacing small.** Increasing the distance in redshift between the snapshots filling a lightcone 
    will increase the severity of the discontinuities when one snapshot meets another.  
    Raising :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` above its default of 0.1 is not advised and there is a hard limit of
    0.25. 

The output redshifts of FLAMINGO and COLIBRE are themselves spaced more widely at high redshift, so the limit also sets
the maximum redshift of a lightcone for these simulations. For a maximum snapshot redshift spacing of 0.1
(:math:`\Delta z =0.1`), FLAMINGO and COLIBRE can build lightcones out to :math:`z=3.0`, however if 
:envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` is increased to :math:`\Delta z =0.25` then the lightcones can be extended out 
to :math:`z=5.0` and :math:`10`, respectively. 


.. _placing-files-one-at-a-time:

Placing files one at a time
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A lightcone or even the shell populated by one snapshot can hold more particles than can be fit on the disks memory. 
:meth:`SnapshotLightcone.gather_files` runs the gather step alone, for one particle type, and returns the files to read:
one :class:`FileReadSpec` for each (snapshot, tile, file). :meth:`SnapshotLightcone.place_file_in_shell` then reads and 
places one :term:`piece` at a time, so each can be added to a map and discarded:

.. code-block:: python

    nr_files, files = beam.gather_files("PartType1", (0.1, 0.2))
    for piece in range(nr_files):
        particles = beam.place_file_in_shell(piece, "PartType1", ["Masses"])
        if len(particles):
            ...  # add particles["Coordinates"] and particles["Masses"] to a map

The particles placed this way are the same as those of
:meth:`place_snapshot_particles_in_shell <SnapshotLightcone.place_snapshot_particles_in_shell>` for the same shell.
:meth:`place_file_in_shell <SnapshotLightcone.place_file_in_shell>` places the files of the last
:meth:`gather_files <SnapshotLightcone.gather_files>` call, so call it with the same particle type (a different one 
raises a ``ValueError``).


.. _lightcone-output:


The output lightcones
---------------------

Each method returns a :class:`LightconeOutput`: a dictionary of `unyt` arrays, one row per particle or halo (for
particles, one :class:`LightconeOutput` per particle type), with the positions relative to the observer. Its
``.metadata`` records every setting that made it:

*   the lightcone type, the simulation (suite, label and name)
*   the ``orientation_seed``, ``orientation_lock``, the ``orientation_table`` and the :term:`fingerprint` of its 
    contents
*   the beam vector and angular radius, and the redshift range


Given the same settings and snapshots, the same lightcone is made again (see :doc:`orientation_tables`).
:func:`write_lightcone_metadata` and :func:`read_lightcone_metadata` write and read the metadata in HDF5 files, and
:func:`write_halo_lightcone_file` writes a halo lightcone in the format of the LightconeIO halo lightcones (see the
:doc:`../quickstart`).


.. _lightcone-step-by-step:

Building lightcones: Step by step
---------------------------------

What happens when a lightcone is being populated with particles or haloes, step by step.


Particles
^^^^^^^^^

.. dropdown:: Flowchart: placing particles into a lightcone
   :icon: workflow
   :animate: fade-in

   .. mermaid:: mermaid/snapshot_lightcone_particles.mmd
      :align: center
      :caption: Placing particles: choosing the snapshots, gathering the pieces to read, and reading and placing them.

:meth:`SnapshotLightcone.place_snapshot_particles_in_shell` places every particle type asked for; or
:meth:`SnapshotLightcone.gather_files` and :meth:`SnapshotLightcone.place_file_in_shell` place one particle type, one
file at a time. For each particle type:

**Choosing the snapshots**

#.  The snapshots filling the shell are chosen from their output redshifts, unless given with ``use_snapshots``. Each
    snapshot fills the redshifts between the midpoints to its neighbours.
#.  Snapshots missing on disk are filled by the available snapshots, each nearest in redshift to its own and a warning 
    is given with :class:`SnapshotGapWarning` (see :ref:`max-snapshot-spacing`).
#.  As a safety net, any snapshot whose redshift range lies outside the shell is dropped. The files of the snapshots
    given with ``use_snapshots`` must then be on disk (``FileNotFoundError`` otherwise), and the parts of the shell
    they leave unfilled are warned about with a :class:`SnapshotGapWarning`, but not filled.

Particles and haloes choose their snapshots in exactly the same way.

**Step 1, gather** (no particle data is read)

#.  For each snapshot, its redshift range is clipped to the lightcone, and its cell :term:`metadata` (cell centres, and 
    which files hold the particle type) is loaded.
#.  For each snapshot, its redshift range is clipped to the lightcone, and its cell metadata (the cell centres, the
    file itself, offset and number of particles of each cell for each particle type) is read from its file 0, once per 
    call. A snapshot without the particle type (e.g. no stars at high redshift) adds no files and a particle type in not 
    found in any of the snapshots raises a ``ValueError``.
#.  The tiles the lightcone passes through are found. With ``"sphere"``, the tiles crossing a :term:`layer` are
    split into one :term:`piece` (tile, snapshot) per layer, each with its own redshift range.
#.  For each tile, its orientation is read from the tables, from its position on the lattice (or its layer) and the
    seed. The periodic :term:`cell shift`, reflections and quarter turns are applied on each axis. 
#.  The cell centres are transformed and moved to the tile's position, and the cells inside the tile's part of the
    lightcone are kept (with a buffer of one cell length).
#.  A :class:`FileReadSpec` (snapshot, tile, file, offsets and lengths of the cells) is recorded for each file with
    particles in those cells. A file needed by several tiles is read once for each tile (with ``"sphere"``, once for 
    each layer of a tile crossing a sphere).

Everything the gather step finds (the snapshots, the pieces, the files, the cells of each piece and the snapshots' cell
metadata) is recorded in a new :class:`ShellRead`, kept as the lightcone's ``last_shell`` until the next call. Nothing
else is kept from one call to the next.

:meth:`gather_files <SnapshotLightcone.gather_files>` stops here, and returns the 
:class:`FileReadSpec`\ s for :meth:`place_file_in_shell <SnapshotLightcone.place_file_in_shell>` to 
place one at a time.

**Step 2, read and place**


#.  The :class:`FileReadSpec`\ s are read on one process, or shared in blocks between the MPI ranks if ``comm`` is
    given.
#.  For each one, the ``Coordinates`` of the particles of its cells are read, and the tile's transform is applied to
    them.
#.  The particles exactly inside the tile's part of the lightcone (and the beam) are kept.
#.  The other properties asked for and ``ParticleIDs`` are read only for the cells keeping a particle, and kept with 
    each particle's ``SnapshotNumber``. ``Velocities``, if read, take the tile's reflection and rotation (not its 
    periodic shift or position, which move positions but not directions), so each velocity keeps its direction relative 
    to its particle's surroundings.
#.  The kept particles are joined and their ``ExpansionFactors`` are computed from their comoving distance to the 
    observer.
#.  With MPI and ``redistribute=True``, the particles are shared evenly between the ranks.

The result is a :class:`LightconeOutput` ``{particle type: data}``, with the lightcone's ``.metadata``.


Haloes
^^^^^^

.. dropdown:: Flowchart: placing haloes into a lightcone
   :icon: workflow
   :animate: fade-in

   .. mermaid:: mermaid/snapshot_lightcone_haloes.mmd
      :align: center
      :caption:  Placing haloes from the SOAP catalogues.


:meth:`SnapshotLightcone.place_halos_in_shell` (with :meth:`SnapshotLightcone.gather_halo_catalogues` for its gather
step) places the haloes of the SOAP catalogues exactly as particles are placed (See :doc:`halo_lightcones` for further 
applications and examples). 

**Step 1, gather**

#.  The snapshots are chosen exactly the same as done for the particles, from the SOAP catalogues on disk (gaps filled
    with a warning, and a ``ValueError`` if they are 0.1 or more apart in redshift; see :ref:`max-snapshot-spacing`).
#.  The cell metadata are read from one snapshot only (it is the same in every snapshot).
#.  For each snapshot, its redshift range is clipped to the lightcone, and the tiles (and spherical layers) the 
    lightcone crosses are found.
#.  A :class:`HaloCatalogueSpec` (snapshot, tile, catalogue and redshift range) is recorded for each tile, with the 
    catalogue's number of haloes, in a :class:`HaloShellRead`. If the gather step has already been run with the same 
    arguments, it is not repeated.

**Step 2, read and place**

#.  The catalogues are split into chunks of ``soap_chunk_size`` rows, read on one process or shared between the MPI 
    ranks
#.  For each chunk, the halo centres and catalogue indices of its rows are read once.
#.  For each tile of its snapshot, the tile's transform is applied to the halo centres, as for particles, and the haloes
    exactly inside the tile's part of the shell are kept, with their snapshot, ``HaloCatalogueIndex`` and
    ``SOAPIndex`` (row in the catalogue).
#.  With MPI and ``redistribute_halos=True``, the haloes are shared evenly between the ranks.
#.  Each halo is assigned a ``Lightcone/ExpansionFactor`` based on its comoving distance from the observer. 

Unlike particles, each SOAP catalogue is read once, and its haloes placed in every tile where it fills the snapshot's
redshift range. SOAP catalogues are not sorted by cell, so every halo is tested. The result, a :class:`LightconeOutput`
with ``Lightcone/HaloCentre``, ``Lightcone/ExpansionFactor``, ``Lightcone/SnapshotNumber``,
``InputHalos/HaloCatalogueIndex`` and ``InputHalos/SOAPIndex``, maps each halo back to its SOAP catalogue and all its
properties, via ``Lightcone/SnapshotNumber`` and ``InputHalos/SOAPIndex`` (see :doc:`halo_lightcones`). 



What a lightcone keeps between calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Flowchart: the record each call makes, and what is kept
   :icon: workflow
   :animate: fade-in

   .. mermaid:: mermaid/snapshot_lightcone_read_state.mmd
      :align: center
      :caption: Each gather step makes a new record of its shell; the lightcone keeps only the last one.

Each call makes its own record of what its gather step found, so nothing is carried from one call to the next:

*   :meth:`gather_files <SnapshotLightcone.gather_files>` and :meth:`place_snapshot_particles_in_shell
    <SnapshotLightcone.place_snapshot_particles_in_shell>` (once for each particle type) make a :class:`ShellRead`;
    :meth:`gather_halo_catalogues <SnapshotLightcone.gather_halo_catalogues>` makes a :class:`HaloShellRead`.
*   The last record made is the lightcone's ``last_shell``. 
    :meth:`place_file_in_shell <SnapshotLightcone.place_file_in_shell>` places its files, 
    :meth:`Snapshot2Lightcone <SnapshotLightcone.Snapshot2Lightcone>` uses its cell metadata, and
    :func:`find_close_cell_repeats <swiftlet.repeat_check.find_close_cell_repeats>` reads its pieces and cells.
*   The last :class:`HaloShellRead` is also kept, so 
    :meth:`place_halos_in_shell <SnapshotLightcone.place_halos_in_shell>` called again with the same settings doesn't 
    repeat its gather step.
*   The cell metadata of the snapshots is kept for one call only: 
    :meth:`place_snapshot_particles_in_shell <SnapshotLightcone.place_snapshot_particles_in_shell>` empties it when it 
    ends.
*   The MPI communicator (``comm``) of a call is passed to the methods that use it and never stored, so a serial call
    after an MPI one is serial.
