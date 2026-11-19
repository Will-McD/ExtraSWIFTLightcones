Glossary
========

The terms used throughout this documentation, in alphabetical order. Each links to where it is explained in more
detail.

.. glossary::
   :sorted:

   all-sky lightcone
      A lightcone covering the whole sky around the observer, built with
      :class:`SnapshotAllSky <swiftlet.snapshot_lightcone.SnapshotAllSky>`. Because of its volume, it should be built in 
      thin redshift :term:`shells <shell>` (see :doc:`user_guide/building_lightcones`).

   beam
      Also referred to as a **lightcone beam** or a **beam lightcone**: A cone within the all-sky lightcone around a 
      direction (the ``beam_vector``), with an angular radius ``ang_radius_deg``, built with 
      :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>`. A beam is exactly the part of the 
      :term:`all-sky lightcone` with the same :term:`orientation table`, :term:`orientation lock` and
      :term:`orientation seed` inside its cone (see :ref:`beams-are-part-of-the-all-sky-lightcone`).

   beam frame
      The axes a :term:`beam` labels its :term:`tiles <tile>` with, ``(nx, ny, n_los)``: ``n_los`` along the lattice
      axis closest to the beam's direction, ``nx`` and ``ny`` along the axes closest to the two directions across it.
      Each label is converted back to the tile's :term:`lattice position` to orient and place it.

   cell
      One of the cubes a SWIFT snapshot's box is divided into, with the same number of cells along each axis (a
      multiple of 8 for SWIFTLET), each with its particles stored together. SWIFTLET selects the cells inside a
      lightcone before reading any particle, and a :term:`tile` moves its cells whole.

   cell shift 
      Also referred to as a **shift** or **periodic (cell) shift**: The periodic shift of an :term:`orientation`, in 
      eighths of the box along each axis (``0``, ``±1``, ``±2``, ``±3`` or ``4``), converted to whole 
      :term:`cells <cell>` with the snapshots' number of cells per axis (see :doc:`user_guide/orientation_tables`).

   clashing pair
      Also called a **clash**, **contacting pair** or **clashing cells**: two copies (replicas) of the same :term:`cell` 
      (i.e. same ID number in the initial snapshot box) in different :term:`tiles <tile>` that are in contact with each 
      other along the face where the two neighbouring :term:`tiles <tile>` meet. The two copies of the cell are 
      populated by the same or adjacent snapshots, where the boundary between their redshift ranges passes between
      them (the same region of the box at two neighbouring output times). The default :term:`orientation table` rules
      both out across the faces between tiles; with ``"sphere"`` they can still happen across the spheres between
      :term:`layers <layer>`, inside the tiles, unless the table is ``sphere_quarter_spacing``. 
      :func:`find_close_cell_repeats <swiftlet.repeat_check.find_close_cell_repeats>` reports those from the same or 
      adjacent snapshots (see :doc:`user_guide/repeated_objects`). In the code and its reports they keep the name *touching*, 
      e.g. ``nr_touching``, the ``touching`` column of the pairs and ``report.touching``.
   
   contacting pair
      See :term:`clashing pair`.

   clash
      See :term:`clashing pair`.
   
   close repeat 
      Two copies of the same structure, from different :term:`tiles <tile>`, closer together than the periodicity of
      the box would allow: a :term:`clashing pair` or a :term:`line of sight pair`. They happen where the
      :term:`orientation` changes (see :doc:`user_guide/repeated_objects`).

   exact periodic replica
      A copy of the snapshot box that is not re-oriented (``orientation_table="no_reorientation"``), so the same 
      structures repeat every box length in every direction. Within a :term:`layer` of ``"cube"`` or ``"sphere"``, the 
      tiles are exact periodic replicas of each other (see :doc:`user_guide/repeated_objects`).

   ExpansionFactor
      The expansion factor or **scale factor**, :math:`a = 1/(1+z)`, at a particle's or halo's comoving distance from 
      the observer. ``ExpansionFactors`` for particles and ``Lightcone/ExpansionFactor`` for haloes. It is determined by 
      the comoving distance from the observer not from the snapshot the object came from. The term expansion factor has 
      been adopted over scale factor to remain consistent with 
      `LightconeIO <https://lightconeio.readthedocs.io/en/latest/>`_ . 

   fingerprint
      The unique identifier for an :term:`orientation table`. A short hash of the contents of an 
      :term:`orientation table`, recorded in the lightcone's :term:`metadata` and in table files, so the same tables 
      can be recognised whatever their file is called (see :doc:`user_guide/orientation_tables`).

   gather step
      The first step of populating a lightcone from the snapshots (particles and haloes). This step includes finding the 
      snapshots that fill the redshift range of the lightcone, the :term:`pieces <piece>` of the lightcone (the 
      :term:`tiles <tile>` each snapshot covers), and the :term:`cells <cell>` and :term:`files to read <file to read>`
      (see :doc:`user_guide/building_lightcones`). It uses only the snapshots' cell metadata, so no particle data is 
      read, and records what it found in :term:`last_shell`.

   halo lightcone file
      An HDF5 file of the haloes of one snapshot of a lightcone, laid out like FLAMINGO's halo lightcones and readable
      with LightconeIO, written by
      :func:`write_halo_lightcone_file <swiftlet.snapshot_lightcone.write_halo_lightcone_file>` (see
      :doc:`user_guide/halo_lightcones`).

   lattice position
      The position ``(nx, ny, nz)`` of a :term:`tile`, in box lengths along x, y and z from the observer's tile
      ``(0, 0, 0)``, which is centred on the observer. A tile's :term:`orientation` is read from its lattice position
      (see :doc:`user_guide/orientation`).

   layer
      The :term:`tiles <tile>` sharing one :term:`orientation` with an :term:`orientation lock`: a cubic shell of
      tiles, :math:`\max(|n_x|, |n_y|, |n_z|) = n`, with ``"cube"``, or everything between :math:`(n - \frac{1}{2})L`
      and :math:`(n + \frac{1}{2})L` from the observer with ``"sphere"``. Within a layer the box is repeated
      periodically.

   lightcone metadata
      See :term:`metadata`.

   LightconeOutput
      What the placing methods return. The :class:`LightconeOutput <swiftlet.snapshot_lightcone.LightconeOutput>` is a 
      dictionary of unyt arrays, one row per particle or halo, with the lightcone's :term:`metadata`. 

   line of sight pair
      Two copies of the same :term:`cell` that could hold copies of the same particles (and haloes) within ``angle_deg``
      (1° by default) of each other on the sky and closer than ``max_los_separation`` (a box length by default) along 
      the line of sight, reported by :func:`find_close_cell_repeats <swiftlet.repeat_check.find_close_cell_repeats>`. No
      :term:`orientation table` can rule them out (see :doc:`user_guide/repeated_objects`).

   metadata
      The settings that made a lightcone, recorded with every :term:`LightconeOutput` as ``.metadata``, which includes:
      the simulation, the :term:`orientation seed`, :term:`orientation lock`, :term:`orientation table` and its
      :term:`fingerprint`, the beam and the redshift range. With the same metadata and snapshots, the same lightcone is
      made again.

   orientation
      How a :term:`tile` is rearranged from the initial snapshot box. The cells of the snapshot box are periodically 
      shifted (:term:`cell shift`) and reflected along each axis, then quarter turns are made about each axis 
      (given by one row of the :term:`orientation table`), producing a new arrangement or orientation of the initial 
      snapshot box, which is then placed as a tile in the lightcone.
      
   orientation lock
      Or **lock**, ``orientation_lock``: which :term:`tiles <tile>` share an :term:`orientation`.  ``None`` (or
      ``"none"``) every tile has its own :term:`orientation`; ``"cube"`` and ``"sphere"``: every tile of a :term:`layer` 
      shares the same :term:`orientation` (see :doc:`user_guide/orientation`).

   orientation seed
      ``orientation_seed``: changes which row of the :term:`orientation table` each tile (or :term:`layer`) takes. The
      same seed, lock, tables and snapshots give the same lightcone.

   orientation table
      The three tables (``QuarterTurns``, ``Reflections`` and ``CellShifts``) whose rows are the 
      :term:`orientations <orientation>` the tiles take, stored as YAML files (see :term:`default tables`)

   default tables
      See :term:`installed tables`

   installed tables
      The :term:`orientation tables <orientation table>` installed with the package: ``default``,
      ``no_reorientation``, ``cube_quarter_spacing`` and ``sphere_quarter_spacing`` (see
      :doc:`user_guide/orientation_tables`). ``default`` is the default option of 
      :class:`SnapshotLightcone <swiftlet.snapshot_lightcone.SnapshotLightcone>`, that is used when a table is not 
      specified. 

   piece
      One (snapshot, tile) part of a lightcone :term:`shell`, found by the :term:`gather step`. The part of a
      snapshot's redshift range that covers one :term:`tile` (with ``"sphere"``, one layer of a tile). 
      Listed in ``last_shell.pieces``.

   file to read
      One snapshot file to read for one :term:`piece`: the snapshot, tile and file, and the offset and number of
      particles of each selected :term:`cell` (a ``FileReadSpec``). Found by the :term:`gather step`, returned by
      :meth:`gather_files <swiftlet.snapshot_lightcone.SnapshotLightcone.gather_files>` and listed in
      ``last_shell.files``.

   last_shell
      What the last :term:`gather step` of a lightcone found: a
      :class:`ShellRead <swiftlet.snapshot_lightcone.ShellRead>` for particles or a
      :class:`HaloShellRead <swiftlet.snapshot_lightcone.HaloShellRead>` for haloes, with its snapshots, 
      :term:`pieces <piece>`, :term:`files to read <file to read>` and cells. Each call makes a new one, so nothing is
      carried from one call to the next (see :doc:`user_guide/building_lightcones`).

   read and place step
      The second step of populating a lightcone: reading the particles (or haloes) of each :term:`file to read`, moving
      the particles and haloes into their :term:`tile` and keeping oly those exactly inside the lightcone.

   shell
      The range of redshift (or comoving distance from the observer) a lightcone is filled over, given as
      ``lightcone_redshift_range``. Within it, each snapshot fills its own :term:`snapshot redshift range`.

   snapshot redshift range
      The redshifts a snapshot fills in a lightcone, given by the range in redshift between the midpoints of 
      neighbouring snapshots' redshifts. The minimum redshift filled by the final snapshot is set by the snapshots own 
      output redshift (:math:`z=0`) and the inverse is true for the first snapshot. A missing snapshot's range is filled
      by its neighbours, with a warning as long as the snapshots filling a shell are less than 0.1 
      (or :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING`) apart in redshift (see :ref:`max-snapshot-spacing`).

   SOAP catalogue
      The halo catalogue of a snapshot, made by SOAP, from which haloes are placed in a lightcone. Its path is the
      simulation's ``soap_format``, with a ``{snap_nr}`` field (see :ref:`soap-halo-catalogues`).

   SOAPIndex
      ``InputHalos/SOAPIndex``: the row of a halo of the lightcone in its snapshot's :term:`SOAP catalogue`, used with
      ``Lightcone/SnapshotNumber`` to read any of its other properties (see :doc:`user_guide/halo_lightcones`).

   tile
      A copy of the snapshot box in the lightcone, re-oriented and placed at its :term:`lattice position` around the
      observer. Also called a periodic replica of the box (see :doc:`user_guide/orientation`).

   touching pair
      Or **touching cells**: see :term:`clashing pair`.

