.. currentmodule:: swiftlet.snapshot_orientation

Orientation tables
==================

Each tile of a snapshot lightcone is a re-oriented copy of the simulations snapshot box, so the same structures don't 
repeat along a line of sight. The orientation of a tile (and the cells within it) is given by the same row from three 
separate **orientation tables**. This page explains what is in the tables, how to read the tables and write new custom 
tables.


What is in a table
------------------

There are three tables, each with the same number of rows and three whole numbers per row 
(one for each of the x, y and z axes):

.. list-table::
   :header-rows: 1

   * - Table
     - Values
     - What it does

   * - ``CellShifts``
     - eighths of the box: `0`, `±1`, `±2`, `±3` or `4`
     - shifts the cells in the tile periodically along each axis

   * - ``Reflections``
     - `+1` or `-1`
     - mirrors the tile along each axis

   * - ``QuarterTurns``
     - number of 90 degree turns, e.g. `0`, `1`, `2`, `-1`
     - rotates the tile about each axis


The number of possible different tile orientations is given by the number of rows in the tables. The order of each 
operation on the tiles is: 

#. the cells are shifted periodically
#. the whole tile is reflected, 
#. the tile is rotated. 


**The shifts are in eighths of the box** and it is converted from eighths to a number of whole cells 
using a snapshots cell metadata (:func:`cell_shifts_in_cells`). A shift wraps around the box, so `4` (half the box) is
the same as `-4`; use `4`.  The advantage of using eighths of a box is that the same table can be used for all 
simulations given the number of cells per axis is a multiple of 8, which is true for all FLAMINGO and COLIBRE 
simulations. 

.. _orientation-row:

How a tile picks its row
^^^^^^^^^^^^^^^^^^^^^^^^

When placing a tile in the lightcone, the tile described by ``(nx, ny, nz)`` reads the row

.. code-block:: python

    row = (nx * p1 + ny * p2 + nz * p3 + orientation_seed * p_seed) % number_of_rows

where the ``p`` are fixed primes, so:

*   the same tile and ``orientation_seed`` always give the same orientation and the lightcone can be reproduced exactly 
*   a different ``orientation_seed`` alters the order that the tables are read, giving every tile a different row 
*   with ``orientation_lock="cube"`` or ``"sphere"``, every layer of tiles out from the observer uses the same row
    ``(nx, ny, nz) = (0, 0, layer)``, so each layer becomes one continuous perodic boundary and each tile 
    has the same orientation

The installed tables
--------------------

These tables are installed with SWIFTLET as YAML files in ``swiftlet/data/orientation_tables``, which you can open
in any text editor and are named:

.. list-table::
   :header-rows: 1

   * -  Name
     -  Rows
     -  Shifts
     -  Summary

   * -  ``default_v2``
     -  5184
     -  eighths of the box
     -  **the default.** No cell clashes with its own copy across the face between two tiles, for any seed for when 
        ``orientation_lock`` is ``"none"`` or ``"cube"``. With ``orientation_lock="sphere"`` clashes can still occur 
        inside the tiles crossing a spherical layer (see :doc:`Repeated particles and haloes <repeated_objects>` and 
        :ref:`checking-tables-for-clashes`).

   * -  ``no_reorientation``
     -  1
     -  none
     -  **no re-orientation**: one row with no quarter turns, reflections of ``+1`` and no shifts, so every tile is the
        snapshot box unchanged, an exact periodic replica. Use it to compare with a re-oriented lightcone, or to see
        where each particle comes from (see :doc:`repeated_objects`)

   * -  ``cube_quarter_spacing``
     -  49
     -  eighths of the box
     -  **for orientation_lock="cube"**: quarter turns, reflections and shifts. Every two rows keep the copies of
        each cell in neighbouring tiles at least :math:`L/4` apart, so no cell is closer than :math:`L/4` to its own copy
        in the next layer, for any seed (see :ref:`tables keeping copies apart <tables-keeping-copies-apart>`)

   * -  ``sphere_quarter_spacing``
     -  44
     -  eighths of the box
     -  **for orientation_lock="sphere"**: quarter turns, reflections and shifts. Every two rows keep the copies of each
        cell at least :math:`L/4` apart in the same tile as well as in neighbouring tiles, so no cell is closer than
        :math:`L/4` to its own copy in the next layer, for any seed, even inside the tiles a layer sphere crosses. It
        also works with ``"cube"`` and no lock (see :ref:`tables keeping copies apart <tables-keeping-copies-apart>`)


The default tables are installed with SWIFTLET and can be rebuilt at any time with the provided tools: 
`make_installed_orientation_tables <../tools/make_installed_orientation_tables.py>`_ . The default tables will always
have the same fingerprint, the unique identifier for each table, so the files hold exactly the tables lightcones have 
been made with:

.. code-block:: bash

    python tools/make_installed_orientation_tables.py                     # make every installed table
    python tools/make_installed_orientation_tables.py --check 16 32 64    # check default_v2 at 64 cells per axis (minutes)


Without an ``orientation_table`` the lightcones use ``default_v2`` otherwise a lightcone can take any installed tables 
by name:

.. code-block:: python

    from swiftlet.snapshot_orientation import default_orientation_tables

    default_table = default_orientation_tables()                      # default_v2 table
    unchanged_table = default_orientation_tables("no_reorientation")  # A one row table that leaves the box unchanged


The default tables should be selected based on the intended use of the lightcones and they should be used with the 
``orientation_locks`` that are designed to work with. 

**default_v2 is clash-free between tiles, but not for orientation_lock="sphere".** With no lock (``"none"``) and with
``"cube"``, the orientation only changes on the faces between tiles, so no cell clashes with its own copy, for any 
seed. With ``"sphere"``, the orientation changes on the spherical layers (:math:`r = (n + \frac{1}{2})L`) which pass
through the tiles. A tile crossing a sphere takes layer :math:`n`'s orientation inside it and layer :math:`n+1`'s
outside it, so the two orientations meet inside the tile, across a curved surface, where a cell there can be in contact 
with its own copy, and **default_v2 cannot rule this out** (``sphere_quarter_spacing`` does, see 
:ref:`tables keeping copies apart <tables-keeping-copies-apart>`). Check ``"sphere"`` lightcones with
:func:`find_close_cell_repeats <swiftlet.repeat_check.find_close_cell_repeats>` before building them using 
``default_v2`` (see :ref:`checking-a-lightcone-for-close-repeats-before-building-it`), and choose a seed that allows you 
to avoid clashes. 


.. _tables-keeping-copies-apart:

Tables built for cubic and spherical layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``default_v2`` table rules out copies of the same cells being in contact but copies can still be one or two cells 
apart between neighbouring tiles or layers. The ``cube_quarter_spacing`` **tables are made for** 
``orientation_lock="cube"``, where the orientation of the tiles only changes between the cubic layers 
(see :ref:`orientation lock diagram <fig-layers-example>`). Any two rows read from these tables keep **the centres** of 
all copies of the same cell in neighbouring tiles **at least a quarter of the box's sidelength** (:math:`L/4`) apart. As 
any two rows can meet between two layers, regardless of the orientation seed, no cell is ever closer than :math:`L/4` to 
its own copy in the next layer (within the same layer every cell is one sidelength apart from its own copies). There are 
very few orientations that can fulfill this requirement so the table only has 49 rows, meaning a lightcone made with this 
table can at most have 49 unique layers. However, this should be sufficient for most cases as only around 8 layers are 
required for a lightcone ranging from :math:`z =0 - 3` for the fiducial FLAMINGO L1_m9 simulation (:math:`L = 1` Gpc). 
This table **does not work for when the lock is "sphere"** as the layers are inside of a tile instead of the edges. 

**The** ``sphere_quarter_spacing`` **tables are made for** ``orientation_lock="sphere"``. Here, the sphereical layers cross 
through tiles (see :ref:`orientation lock diagram <fig-layers-example>`), meaning the tile is split into two layers,
each with its own orientation, so the copies of a cell from two rows can be in the same tile as well as in neighbouring 
ones. Any two rows from these tables keep the copies of every cell at least a quarter of the box apart in both (the 
distance is measured across the periodic box) so no cell's centre is closer than :math:`L/4` to its own copy in the
next layer, wherever the sphere passes through the tiles, for any orientation seed. As this requires more strict 
conditions than for the ``cube_quarter_spacing`` tables, the ``sphere_quarter_spacing`` tables also work for when the 
``orientation_lock ="cube"``. 

To use these tables to build lightcones:

.. code-block:: python

  lightcone = SnapshotAllSky(simulation_object=sim, orientation_lock="cube", orientation_table="cube_quarter_spacing")
  lightcone = SnapshotAllSky(simulation_object=sim, orientation_lock="sphere", orientation_table="sphere_quarter_spacing")

.. important::
  Whilst clashes can be removed by the tables copies of the same cell can appear close together along a line of sight 
  cannot be easily removed by any table (see :doc:`Repeated particles and haloes <repeated_objects>`).



Generating tables with every distinct orientation
-------------------------------------------------

There are 48 distinct ways to turn and reflect a cube (the signed permutations of the axes). With the 8 periodic shifts 
per axis (every eighth of the box) there are :math:`48 \times 8^3 = 24576` distinct orientations. 
:func:`write_orientation_tables` generates tables holding every one of them and writes them to a file:

.. code-block:: python

    from swiftlet import write_orientation_tables

    fingerprint = write_orientation_tables("orientation_tables.yaml", seed=0)


*   The first row leaves the box unchanged, so the observer's tile ``(0, 0, 0)`` is not re-oriented 
    with ``orientation_seed=0``.
*   The other rows are shuffled; ``seed`` sets the shuffle, so the same ``seed`` writes the same tables.
*   ``n_rows`` keeps only the first ``n_rows`` rows of the tables, e.g. ``n_rows=500``.
*   ``cell_shift_values`` sets the shifts each axis can take, among ``EIGHTH_SHIFT_VALUES`` (``0, ±1, ±2, ±3, 4``), 
    which must include ``0`` and have no repeats, e.g. ``(0, 2, -2, 4)`` for quarters of the box only.

These generated tables are not checked for clashing cells. Check them before use (see
:ref:`checking-tables-for-clashes`), or make clash-free tables with :func:`generate_clash_free_tables`.

To make the tables in Python without writing them, use
:func:`generate_orientation_tables(seed, n_rows, cell_shift_values) <generate_orientation_tables>`, which takes the same
arguments.



Writing tables of your own
--------------------------

To choose every row yourself, pass the tables as a dictionary. Each table can be a list of rows or an array of 
shape ``(N, 3)``:

.. code-block:: python

    from swiftlet import write_orientation_tables

    my_tables = {
    "QuarterTurns": [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
    "Reflections":  [[1, 1, 1], [-1, 1, 1], [1, 1, -1]],
    "CellShifts":   [[0, 0, 0], [1, 0, -1], [2, 2, 0]],    # eighths of the box
    }
    write_orientation_tables("my_tables.yaml", tables=my_tables)


The shifts are in eighths of the box and the tables are checked before they're written 
(:func:`check_orientation_tables`), where a ``ValueError`` is raised if:

*   a table is missing, or isn't of shape ``(N, 3)`` with at least one row
*   the tables have different numbers of rows
*   any value isn't a whole number
*   a reflection isn't `+1` or `-1`
*   a shift isn't one of `0, ±1, ±2, ±3, 4`, or the tables say their ``cell_shift_unit`` is anything but eighths.

Keep the first row unchanged (``[0, 0, 0]``, ``[1, 1, 1]``, ``[0, 0, 0]``) if the observer's tile should be the 
snapshot box as it is with ``orientation_seed=0``.


.. _checking-tables-for-clashes:

Checking tables for clashes
---------------------------

Where two neighbouring tiles are oriented differently, a cell can end up sharing a face with its own copy (the same cell
index of the snapshot box) in the other tile, so the same structures sit side by side. :func:`check_neighbouring_tiles` 
finds these clashes using only the tables rows. Every pair of tiles sharing a face, within ``extent`` box lengths of 
the observer's tile, takes its orientations from the tables as done when building a snapshot lightcone. The cells are 
moved as done in the lightcone, and each cell is compared with its own copy (by position in the snapshot box) in the 
other tile:

.. code-block:: python

    from swiftlet import check_neighbouring_tiles, default_orientation_tables

    report = check_neighbouring_tiles(default_orientation_tables(), nr_cells_axis=32, orientation_lock=None,
    orientation_seed=0)
    print(report.summary())  # e.g. "11 cells touch their own copy across the faces of 11 pairs of neighbouring tiles"
    report.touching          # the two tiles and the cell index of each touching cell

Given ``boxsize`` and a ``shell`` (minimum and maximum comoving distance in Mpc), only clashing cells overlapping the 
lightcones shell are counted, the same way :func:`swiftlet.repeat_check.find_close_cell_repeats` counts these clashes for 
a lightcone shell.
:func:`check_neighbouring_tiles` is a coarse-grained search using only the tables rows, so it is not accurate for
``orientation_lock="sphere"`` (which splits tiles between layers by distance), instead you should use 
:func:`swiftlet.repeat_check.find_close_cell_repeats`, which builds the actual lightcone at the cell level 
(see :doc:`Repeated particles and haloes <repeated_objects>`). 

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. figure:: images/snapshot_clashing_cell_clash.png
         :width: 100%
         :alt: Tile B, a mirror image of tile A, placed against it, with cells 12 to 15 meeting their own copies.

         **A clash:** tile B is tile A reflected along :math:`x`. The cells on the shared face (12 to 15) meet their
         own copies.

   .. grid-item::

      .. figure:: images/snapshot_clashing_cell_no_clash.png
         :width: 100%
         :alt: Tile B, turned and shifted, placed against tile A, with the two copies of cell 11 half a box apart.

         **No clash:** tile B is turned a quarter about :math:`z` and shifted a quarter of the box. The closest two
         copies of a cell (cell 11) are half a box length apart.

Each cell is numbered by its place in the snapshot box, so its two copies can be matched (see
:ref:`the clashing cells diagrams <fig-clashing-cells>` and :ref:`example figures <fig-pairs-on-sky>` for more).


Table files
-----------

Orientation tables are written as YAML files (ending in ``.yaml`` or ``.yml``), which can be written and read with  
:func:`write_orientation_tables` and :func:`read_orientation_tables` respectively. Each file must have the column names 
and one row from the tables per line: the quarter turns, reflections and cell shifts, each along x, y and z. 

For example, the start of ``default_v2.yaml``:

.. code-block:: yaml

    # SWIFTLET orientation tables, see docs/user_guide/orientation_tables.rst

    name: default_v2
    description: 'orientation tables of the box tiles of a snapshot lightcone: use row (tile hash % number of rows)'
    cell_shift_unit: eighths
    number_of_rows: 5184
    fingerprint: 7006e905808e256f
    generated: false
    generator_seed: 21
    columns: [QuarterTurns_x, QuarterTurns_y, QuarterTurns_z, Reflections_x, Reflections_y, Reflections_z, CellShifts_x,
     CellShifts_y, CellShifts_z]
    rows:
    - [0, 0, 0,  1,  1,  1,  0,  0,  0]
    - [1, 0, 0,  1, -1,  1,  0,  1, -1]
    ...

Given the tables are stored as a plain YAML file any YAML reader works, e.g. ``yaml.safe_load(open(path))["rows"]``. 
SWIFTLET reads rows written one per line, as above, without the YAML parser, so ``default_v2`` is read in a few 
hundredths of a second. Any other layout is read with PyYAML. 

You can write new tables in the preferred format with :func:`write_orientation_tables(..., attributes={...}) <write_orientation_tables>` 
and then read it in with :func:`read_orientation_tables`: 

.. code-block:: python

    from swiftlet import read_orientation_tables

    tables = read_orientation_tables("orientation_tables.yaml")
    tables = read_orientation_tables("default_v2")              # installed tables, by name

.. note::
  If a file holds a ``fingerprint``, reading it checks the rows still have that fingerprint, so a table changed by hand 
  is not mistaken for the previous version. Remove the ``fingerprint`` line after changing the rows on purpose.


A file of your own needs only ``columns`` and ``rows``. The optional entries written before the rows are:

.. list-table::
   :header-rows: 1

   * - Entry
     - Meaning

   * - ``name``
     - the name of the table

   * - ``number_of_rows``
     - the number of rows, N

   * - ``fingerprint``
     - a short hash of the tables' contents

   * - ``generated``
     - ``True`` if the tables were generated, ``False`` if they were given

   * - ``generator_seed``
     - the settings used to generate them (only if ``generated`` is ``True``)

   * - ``cell_shift_unit``
     - the unit of the shifts (a file without it is in eighths)

   * - ``description``
     - a string giving a short summary on what the file is



Using the tables in a lightcone
-------------------------------

Use a specific or custom orientation table when building a lightcone by passing a file, or the name of a default 
table as ``orientation_table``:

.. code-block:: python

    from swiftlet import SnapshotAllSky

    lightcone = SnapshotAllSky(simulation_object=Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL"), 
    orientation_seed=0, orientation_lock="cube", orientation_table="orientation_tables.yaml")


This works exactly the same way for :class:`swiftlet.snapshot_lightcone.SnapshotBeam`.

Reproducing a lightcone
^^^^^^^^^^^^^^^^^^^^^^^

The lightcone's ``.metadata`` records which tables were used when making it (see :ref:`lightcone output <lightcone-output>`). 
As the fingerprint depends only on the contents of the tables, if a file containing a table is moved or renamed, the 
same fingerprint shows it holds the same tables. If two files have the same name but different fingerprints, they hold 
different tables. Keep the file with the lightcones made from it. With the same tables, seed, lock and snapshots, the 
same lightcone is made again.

