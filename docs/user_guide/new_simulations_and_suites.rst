Simulations and Suites
======================

.. currentmodule:: swiftlet.simulation_suites

SWIFTLET knows about a simulation through two classes:

*   :class:`SimulationSuite`: tracks what every simulation in a suite (FLAMINGO, COLIBRE, ...) has 
    in common. This is its box size and resolution labels, its directory, how its snapshot and SOAP catalogue files are 
    named, and its output lists.
*   :class:`Simulation`: one simulation of a suite, with its label (e.g. ``L1000N1800``), name 
    (e.g. ``HYDRO_FIDUCIAL``) and directory.


Accessing a simulation's snapshots, SOAP catalgoues, identifying how a simulation is named and computing the redshift 
ranges of each snapshot, is done through a :class:`Simulation` object, which accesses the information from the 
:class:`SimulationSuite`. This way each simulations key information is consistent within the same suite. This page 
explains how to use these classes and track the paths to all the simulation outs required for builiding a lightcone. 

.. _soap-halo-catalogues:

Simulation suites
-----------------

A suite is a :class:`SimulationSuite` with these fields:

.. _table-of-values:

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
     - Snapshot file, relative to the simulation's directory (or an absolute path), with a ``{snap_nr}`` field, and
       ``{file_nr}`` for snapshots of several files
     - ``"snapshots/snap_{snap_nr:04d}/snap_{snap_nr:04d}.{file_nr}.hdf5"``

   * - ``output_list``
     - The output list of a simulation, relative to its directory (or an absolute path). Optional:
       ``"output_list.txt"`` (the default)
     - ``"output_list.txt"``

   * - ``soap_format``
     - SOAP halo catalogue, relative to the simulation's directory (or an absolute path), with a ``{snap_nr}`` field.
       Optional: ``"SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"`` (the default)
     - ``"soap/halo_properties_{snap_nr:04d}.hdf5"``


.. _new-suite:

A new suite (not COLIBRE or FLAMINGO) can be built and then called through a :class:`Simulation`:

.. code-block:: python

    from swiftlet import SimulationSuite, Simulation, SnapshotBeam

    MYSUITE = SimulationSuite(
        name="mysuite",
        label_pattern=r"L\d{4}N\d{4}_my",
        base_dir_format="/data/mysuite/{box_res}/{sim_name}",
        snapshot_format="snapshots/snap_{snap_nr:04d}/snap_{snap_nr:04d}.{file_nr}.hdf5",
    )

    sim = Simulation.from_label("L0100N0512_mysuite", "Fiducial", simulation_suite_name=MYSUITE)


To find the simulations in a new suite using only the labels, without passing it each time, add the suites to the 
registry:

.. code-block:: python

    from swiftlet import simulation_suites

    simulation_suites.SUITES[MYSUITE.name] = MYSUITE
    Simulation.from_label("L0100N0512_mysuite", "Fiducial")            # found from the label

The label patterns of the suites must not overlap. A label matching more than one suite, or none, raises a 
``ValueError`` asking for ``simulation_suite_name``. To make the suite part of SWIFTLET, define it next to 
``FLAMINGO`` and ``COLIBRE`` in ``swiftlet/simulation_suites.py`` and add it to the existing ``SUITES`` there.


.. _simulation-paths:

Finding Simulations
-------------------

Each suite looks for its simulations in ``<directory>/<box_res>/<sim_name>``, as given by the ``base_dir_format`` 
parameter passed to :class:`SimulationSuite` e.g. ``/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL``. From
this base directory it then looks for the ``snapshots``, ``SOAP catalogues`` and ``output lists`` using the relative 
paths shown the :ref:`table <table-of-values>` above. 

If the location of a simulations directory is given by ``base_dir_format`` in a defined :class:`SimulationSuite`,  
``snapshots/``, ``SOAP-HBT/`` and ``output_list.txt`` are in the simulation's directory 
(``<directory>/<box_res>/<sim_name>``) and all files follow the same format as given :ref:`above <table-of-values>`  
then :class:`Simulation <swiftlet.simulation_suites.Simulation>` can be called from only its labels:

.. code-block:: python
 
  from swiftlet import Simulation
  
  sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")

However for a simulation with **the same layout in a different directory** then you can pass
:class:`Simulation <swiftlet.simulation_suites.Simulation>` a new ``base_dir_format``, or give the simulation's 
directory to :meth:`Simulation.from_path <swiftlet.simulation_suites.Simulation.from_path>` (it must end in 
``<box_res>/<sim_name>``, so the label and name can be read from it):

.. code-block:: python

   sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format="/data/me/{sim_name}_{box_res}")
   sim = Simulation.from_path("/data/me/L1000N1800/HYDRO_FIDUCIAL")

Then if a simulation's **snapshots, SOAP catalogues or output list have other names**, these new paths can be passed to 
the simulation as ``snapshot_format``, ``soap_format`` and ``output_list``. Each is relative to the simulation's 
directory, or absolute, so they can be on different disks:

.. code-block:: python

   sim = Simulation.from_label(
       "L0025N0376", "MY_RUN",
       base_dir_format="/sims/{sim_name}",
       snapshot_format="snaps/run_{snap_nr:03d}.{file_nr}.hdf5",   # {snap_nr}, {file_nr} for snapshots of several files
       soap_format="/haloes/need//soap_{snap_nr:03d}.hdf5",    # {snap_nr}
       output_list="/home/my/lists/outputs.txt",
   )
   sim.snapshot_file(5, 2)     # /scratch/me/MY_RUN/snaps/run_005.2.hdf5
   sim.soap_file(5)            # /scratch/me/haloes/soap_005.hdf5


:meth:`Simulation.from_path <swiftlet.simulation_suites.Simulation.from_path>` takes the same three parameters, and
:meth:`Simulation.with_paths <swiftlet.simulation_suites.Simulation.with_paths>` makes a copy of a simulation you 
already have, with another directory or other paths:

.. code-block:: python

   sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
   local = sim.with_paths(directory="/data/me/hydro", soap_format="SOAP/halo_properties_{snap_nr:04d}.hdf5")


The formatted paths (``snapshot_format``, ``soap_format`` and ``output_list``) without the fields it needs, or with any 
other field (e.g. ``{snapnr}``), raises a ``ValueError`` when the simulation is made. If a path is not given then the 
default from the suite is assumed. 

If **every simulation of a suite has new paths**, make a copy of the suite with those paths, and pass it as
``simulation_suite_name``:

.. code-block:: python

  from swiftlet import FLAMINGO, Simulation

  # SOAP catalogues in SOAP/ instead of SOAP-HBT/
  my_flamingo = FLAMINGO.with_paths(soap_format="SOAP/halo_properties_{snap_nr:04d}.hdf5")
  sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", simulation_suite_name=my_flamingo)
 
:meth:`SimulationSuite.with_paths` calls :func:`dataclasses.replace` for the three paths, with their fields checked;
``replace`` changes any other field, e.g. ``label_pattern``. The copy is then recognised with the labels of the suite 
it was made from.
 


New Simulations
----------------

A new simulation from an existing suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a simulation of your own with the existing FLAMINGO and COLIBRE suites it needs:

#.  A label pattern that matches it. FLAMINGO's labels are listed one by one 
    (``L1000N0900|L1000N1800|L1000N3600|L2800N5040``). COLIBRE matches any zero padded label, e.g. ``L0200N1504``, or 
    e.g. ``L200m6``.
#.  An output list in each simulation's directory, as SWIFT writes it.

For a one-off use case, replace the suite as :ref:`above <new-suite>`, adding the label to ``label_pattern``. For long 
term use, edit the suite in :mod:`swiftlet.simulation_suites`.


New snapshots
^^^^^^^^^^^^^
To add new snapshots or swap out existing snapshots for a given simulation, they must be consistent with the 
structure and metadata of the current COLIBRE and FLAMINGO snapshots. 
So they must include:

*   the following groups: ``Header``, ``Cosmology``, ``Units``, ``InternalCodeUnits`` and ``PhysicalConstants/CGS``
*   the ``Cells`` metadata (cell centres, counts, offsets and files of each particle type), used to read only the cells 
    inside a lightcone.

For haloes, the catalogues must be in the same format as the SOAP catalogues (see 
`SOAP <https://github.com/SWIFTSIM/SOAP>`_ documentation), at the suite's (or the simulation's own) ``soap_format``.

The cosmology is read from the :math:`z = 0` snapshot. If that file isn't there, it is read from the lowest redshift 
snapshot of the output list that is, found with the simulation's ``snapshot_format`` (its file 0, or its virtual file),
so it works whatever the snapshots are named.



Output lists
------------

The **output list** (``output_list.txt``) of a SWIFT simulation gives the redshift that each snapshot was output at.  
There is one output per line, in order of snapshot number, with the redshift in the first column. 

For FLAMINGO simulations the top of the file will look like: 

.. code-block:: text
 
  # Redshift
  15.0
  10.38
  9.51


The output lists for COLIBRE have a second column, ``Select Output``, which is ignored. Snapshots and  snipshots are 
numbered together, and we treat snipshots as snapshots because both are used to populate the lightcone the same way:
 
.. code-block:: text

  # Redshift, Select Output
  30.000000, Snapshot
  25.000000, Snipshot
  22.500000, Snipshot

These output lists can be accessed through a :class:`Simulation` object or directly if you pass the path to the file. 

.. code-block:: python

    from swiftlet import Simulation, output_redshifts, read_output_list
    
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
    sim.output_list                       # <simulation directory>/output_list.txt
    sim.output_redshifts                  # read once from it, then kept
    output_redshifts(sim)                 # the same
    output_redshifts("/path/to/output_list.txt")   # or from the path of an output list
    read_output_list("/path/to/output_list.txt")   # the same
 

Alternatively a list of scale factors (header ``# Scale Factor``) can also be read. The redshifts must decrease with 
snapshot number, with at least two outputs. A simulation without an output list in its directory raises a 
``FileNotFoundError`` when its output redshifts are needed and the error points out the path it looked for.

A new output list can be written with :func:`write_output_list("output_list.txt", redshifts) <write_output_list>`.


Output lists with another name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simulations within an existing suite (COLIBRE and FLAMINGO)that  keep their output list under a name other than 
``output_list.txt``, give the suite its path, relative to a simulation's directory, as ``output_list``.
For example if a new COLIBRE simulation has an output list stored as ``/settings/redshift_list.txt`` the Simulation can 
be pointed towards it:

.. code-block:: python
 
  from dataclasses import replace
  from swiftlet import Simulation, SnapshotAllSky
 
  sim = Simulation.from_label("L1000N3600", "NEW_SIM", output_list="/settings/redshift_list.txt")

  lightcone = SnapshotAllSky(simulation_object=sim, orientation_lock="cube")
 