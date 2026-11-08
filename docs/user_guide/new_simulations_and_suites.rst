Output redshift tables and simulations suites
=============================================

.. currentmodule:: swiftlet.simulation_suites

SWIFTLET knows about a simulation through two classes:

*   :class:`SimulationSuite`: what every simulation of a suite (FLAMINGO, COLIBRE, ...) has 
    in common. This is its box size and resolution labels, its directory, how its snapshot files are named, and its 
    output redshift tables.
*   :class:`Simulation`: one simulation of a suite, with its label (e.g. ``L1000N1800``), name 
    (e.g. ``HYDRO_FIDUCIAL``) and directory.

Every snapshot and halo catalogue path and every conversion between snapshot number and redshift comes from these. 
This page explains how to modify and add output redshift tables, simulations or suites. 


Output redshift tables
----------------------

The **output redshift table** of a simulation gives the redshift that each snapshot was output at. It is a YAML file 
that contains: a description of the table and simulations, the columns headers (``snapshot_number``, ``redshift``), 
one snapshot per line:

.. code-block:: yaml

    # Output redshifts of the snapshots of my run

    description: my run
    columns: [snapshot_number, redshift]
    rows:
    - [0, 3.0]
    - [1, 2.0]
    - [2, 1.0]
    - [3, 0.5]
    - [4, 0.0]


New files containing redshift tables can be written from a list of redshifts, in order of snapshot number using
:func:`write_output_redshifts("my_outputs.yaml", redshifts, description="my run") <write_output_redshifts>` . 

These files require: 

*   **Two columns:** ``snapshot_number`` and ``redshift``.
*   **Snapshot numbers:** `0, 1, 2, ...` in order, with none missing.
*   **Redshifts:** decreasing with snapshot number, at least two snapshots.
*   **z = 0:** must include an output at z = 0. :meth:`Simulation.z0_snapshot` looks for it. 


In the lightcone, each snapshot fills the redshifts between the midpoints to its neighbours. It is also a requirement 
that each lightcone built is continuous in redshift (doesn't have gaps in redshift), therefore if a snapshot is missing 
on disk, its redshift range is filled by its neighbours, as long as the spacing between snapshots output redshifts 
doesn't surpase the ``MAX_SNAPSHOT_SPACING``. By default  ``MAX_SNAPSHOT_SPACING=0.1``  (see 
:doc:`environment_variables`), however the limit can be changed with :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` 
(see :ref:`max-snapshot-spacing`).


The tables installed with SWIFTLET are stored in ``swiftlet/data/output_redshifts``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Used for

   * - ``flamingo_L1.yaml``
     - FLAMINGO ``L1000N0900`` and ``L1000N1800``

   * - ``flamingo_L2p8.yaml``
     - FLAMINGO ``L2800N5040``  and ``L1000N3600`` (with an extra output at z = 12.26)

   * - ``colibre.yaml``
     - Every COLIBRE simulation


Any table can be read with :func:`read_output_redshifts`:

.. code-block:: python

    from swiftlet import read_output_redshifts

    read_output_redshifts("colibre.yaml")         # a table shipped with SWIFTLET, by its file name
    read_output_redshifts("./my_outputs.yaml")    # a file of your own, by its path


A name with no directory is looked for among the tables installed with SWIFTLET. For a file of your own, always give a 
path with a directory, e.g. ``./my_outputs.yaml`` or an absolute path.

Adding new ``Simulations`` and ``Suites``
-----------------------------------------

Below we cover how to modify :class:`Simulation <swiftlet.simulation_suites.Simulation>` and 
:class:`SimulationSuite <swiftlet.simulation_suites.SimulationSuite>` for: 


An existing suite with a different output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a FLAMINGO or COLIBRE run with a different output list, make a copy of the suite with your table using 
:func:`dataclasses.replace`:

.. code-block:: python

    import os
    from dataclasses import replace
    from swiftlet import COLIBRE, Simulation, SnapshotAllSky

    my_colibre = replace(COLIBRE, output_redshift_tables={"*": os.path.abspath("my_outputs.yaml")})

    sim = Simulation.from_label("L0200N1504", "MY_RUN", simulation_suite_name=my_colibre)
    lightcone = SnapshotAllSky(simulation_object=sim, orientation_lock="cube")

The ``output_redshift_tables`` parameter maps each box size and resolution label to a table file. The entry ``"*"`` is 
used for any label not listed.


A new simulation of an existing suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new label (box size and resolution) needs:

#.  A label pattern that matches it. FLAMINGO's labels are listed one by one 
    (``L1000N0900|L1000N1800|L1000N3600|L2800N5040``). COLIBRE matches any zero padded label, e.g. ``L0200N1504``, or 
    e.g. ``L200m6``.
#.  An output redshift table for it, if it isn't covered by `"*"`.

For a one-off use case, replace the suite as above, adding the label to ``label_pattern`` and its table to 
``output_redshift_tables``. For longterm use edit the suite in :mod:`swiftlet.simulation_suites` and add the 
tables file to ``swiftlet/data/output_redshifts`` with the existing files installed with the package. 

A simulation of a known label just needs its name, e.g. :func:``Simulation.from_label("L1000N1800", "HYDRO_STRONG_AGN")``. 
If it is kept somewhere other than the suite's directory, see :doc:`environment_variables` and if its SOAP catalogues 
are somewhere else, give it its own ``soap_format`` (see :ref:`soap-halo-catalogues`).


A new simulation suite
^^^^^^^^^^^^^^^^^^^^^^

A suite is a :class:`SimulationSuite` with these fields:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Field
     - Meaning
     - Example

   * - ``name``
     - Name of the suite, lower case
     - ``"mysuite"``

   * - ``label_pattern``
     - Regular expression the suite's box size and resolution labels match in full
     - ``r"L\d{4}N\d{4}_my"``

   * - ``base_dir_format``
     - Directory of a simulation, with ``{box_res}`` and ``{sim_name}`` fields
     - ``"/data/mysuite/{box_res}/{sim_name}"``

   * - ``base_dir_env``
     - Environment variable that, if set, replaces the directory holding the simulations
     - ``"SWIFTLET_MYSUITE_DIR"``

   * - ``snapshot_format``
     - Snapshot file, relative to the simulation's directory, with ``{snap_nr}`` and ``{file_nr}`` fields
     - ``"snapshots/snap_{snap_nr:04d}/snap_{snap_nr:04d}.{file_nr}.hdf5"``

   * - ``output_redshift_tables``
     - ``{label or "*": table file}``
     - ``{"*": "/data/mysuite/outputs.yaml"}``

   * - ``soap_format``
     - SOAP halo catalogue, relative to the simulation's directory, with a ``{snap_nr}`` field. Optional: ``None``
       (the default) for a suite without SOAP catalogues
     - ``"soap/halo_properties_{snap_nr:04d}.hdf5"``

Which can be accessed as: 

.. code-block:: python

    from swiftlet import SimulationSuite, Simulation, SnapshotBeam

    MYSUITE = SimulationSuite(
    name="mysuite",
    label_pattern=r"L\d{4}N\d{4}*my",
    base_dir_format="/data/mysuite/{box_res}/{sim_name}",
    base_dir_env="SWIFTLET_MYSUITE_DIR",
    snapshot_format="snapshots/snap*{snap_nr:04d}/snap_{snap_nr:04d}.{file_nr}.hdf5",
    output_redshift_tables={"*": "/data/mysuite/outputs.yaml"},
    )

    sim = Simulation.from_label("L0100N0512_my", "Fiducial", simulation_suite_name=MYSUITE)
    sim.snapshot_file(4)        # /data/mysuite/L0100N0512_my/Fiducial/snapshots/snap_0004/snap_0004.0.hdf5
    beam = SnapshotBeam(simulation_object=sim, beam_vector=(0., 0., 1.))


To find the suite from a label without passing it each time, add it to the registry of suites:

.. code-block:: python

    from swiftlet import simulation_suites

    simulation_suites.SUITES[MYSUITE.name] = MYSUITE
    Simulation.from_label("L0100N0512_my", "Fiducial")                                   # found from the label
    Simulation.from_label("L0100N0512_my", "Fiducial", simulation_suite_name="mysuite")   # or by name

The label patterns of the suites must not overlap. A label matching more than one suite, or none, raises a 
``ValueError`` asking for ``simulation_suite_name``. To make the suite part of SWIFTLET, define it next to 
``FLAMINGO`` and ``COLIBRE`` in ``swiftlet/simulation_suites`` and add it to the existing ``SUITES`` there.



A new snapshot
^^^^^^^^^^^^^^

When adding new simulations and suites the snapshots must be consistent with the structure and metadata of the current 
COLIBRE and FLAMINGO snapshots. 
They must include:

*   the following groups: ``Header``, ``Cosmology``, ``Units``, ``InternalCodeUnits`` and ``PhysicalConstants/CGS``
*   the ``Cells`` metadata (cell centres, counts, offsets and files of each particle type), used to read only the cells 
    inside a lightcone.


For haloes, the catalogues must be in the SOAP format (see `SOAP <https://github.com/SWIFTSIM/SOAP>`_ documentation), 
at the suite's (or the simulation's own) ``soap_format``.

The cosmology is read from the :math:`z = 0` snapshot. If that file isn't there, it is read from the lowest redshift 
snapshot found in ``<simulation>/snapshots``. That search only recognises files named ``flamingo_XXXX`` or 
``colibre_XXXX``, so for a suite with other file names, keep the z = 0 snapshot's file 0.


Checking a simulation
---------------------

To check what values are stored by a :class:`Simulation <swiftlet.simulation_suites.Simulation>`, 

.. code-block:: python

   sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
   sim.directory                       # the simulation's directory
   sim.snapshot_file(77, 0)            # file 0 of snapshot 77
   sim.output_redshifts                # redshift of each snapshot
   sim.z0_snapshot                     # the snapshot at z = 0
   sim.snapshot_redshift_range(70)     # the redshifts snapshot 70 fills in a lightcone
   sim.snapshots_in_range((0.1, 0.2))  # the snapshots filling 0.1 < z < 0.2

