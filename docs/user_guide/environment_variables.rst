Environment Variables
=====================

SWIFTLET reads a few environment variables. They set how many threads it uses and where it finds the simulations and its 
data files. This page lists each one and how to change it.

SWIFTLET utilises the following environmental variables:

.. envvar:: SWIFTLET_MAX_THREADS

  Maximum number of threads used by SWIFTLET, 30 by default. 

.. envvar:: SWIFTLET_FLAMINGO_DIR

  The directory holding the FLAMINGO simulations: ``/cosma8/data/dp004/flamingo/Runs``.

.. envvar:: SWIFTLET_COLIBRE_DIR

  The directory holding the COLIBRE simulations: ``/cosma8/data/dp004/colibre/Runs``.

.. envvar:: L1_REDSHIFTS_FILENAME

  The FLAMINGO lightcone shell redshift files for the  1 Gpc (L1) boxsizes, set by ``swiftlet-configure``. 

.. envvar:: L2P8_REDSHIFTS_FILENAME

  The FLAMINGO lightcone shell redshift files for the  2.8 Gpc (L2p8) boxsizes, set by ``swiftlet-configure``. 

.. envvar:: HEALPY_PIXEL_WEIGHTS

  The directory of the HEALPix pixel weights: ``<environment>/share/healpy-data``

.. envvar:: SWIFTLET_MAX_SNAPSHOT_SPACING

  The largest redshift spacing allowed between the snapshots filling a lightcone shell: a gap of this much or more
  (from missing snapshots, or between the outputs themselves) raises a ``ValueError``. Set to 0.1 by default (see
  :ref:`max-snapshot-spacing`).

Setting a variable
------------------

There are four ways, from the most permanent to the most temporary:

*   **In the environment's activate script**, so it is set whenever the environment is activated. ``swiftlet-configure`` 
    does this for :envvar:`SWIFTLET_MAX_THREADS` and the shell redshift files, in ``bin/activate`` for a venv or 
    ``etc/conda/activate.d`` for conda. For the other variables, add an ``export`` line to the same script yourself.

* **In your shell**, for that shell session:

  .. code-block:: bash
    
    export SWIFTLET_MAX_THREADS=16

* **In a job script**, before the python line, for that job.

* **In Python**, before the value is used:

  .. code-block:: python

    import os
    os.environ["SWIFTLET_FLAMINGO_DIR"] = "/my/flamingo/runs"


Maximum number of threads 
-------------------------

:envvar:`SWIFTLET_MAX_THREADS` sets the maximum number of threads for SWIFTLET's threaded code:

* numba's parallel functions, which re-orient the box tiles (and swiftsimio's parallel projections);
* healpy's spherical harmonic transforms (``map2alm``, ``rotate_alm``, ``alm2map``), used to rotate HEALPix maps.

The possible values of :envvar:`SWIFTLET_MAX_THREADS`. are shown in the table below, 

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Value
     - Threads used

   * - Not set
     - 30

   * - A whole number, e.g. `16`
     - That number

   * - 'all'
     - Every core the process is allowed

Whatever the value, a process never uses more threads than the cores it is allowed, e.g. the cores given to a job, or 
to one MPI rank. numba is also capped by ``NUMBA_NUM_THREADS`` (set before Python starts).

Set it with any of:

.. code-block:: bash

    swiftlet-configure --max_threads 16     # saved in the environment's activate script ("all" also works)
    export SWIFTLET_MAX_THREADS=16          # for one shell or job script

.. code-block:: python

    import swiftlet
    swiftlet.set_max_threads(16)            # while running, from then on
    swiftlet.max_threads()                  # the maximum: 16, or None for "all"
    swiftlet.threads.thread_limit()         # the threads actually used: the maximum, capped by the cores allowed

The value is read each time threaded code runs, so changing it while running takes effect from the next call.

**In job scripts:**

*   **One process with several cores:** match the cores given to the job,
    e.g. ``export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}``.
*   **With MPI:** use `1` (or the cores per rank), so the ranks on a node don't compete for its cores.

healpy's transforms are limited with ``threadpoolctl``, if it is installed. Without it, set ``OMP_NUM_THREADS`` instead.


.. _max-snapshot-spacing:

Largest spacing between snapshots
---------------------------------

A lightcone is filled by the snapshots whose redshift ranges cover the lightcones comoving distance, each filling the 
redshifts between the midpoints to its neighbours (see :doc:`new_simulations_and_suites`). Snapshots missing on disk are 
filled by the available ones, with a :class:`SnapshotGapWarning <swiftlet.snapshot_lightcone.SnapshotGapWarning>`. By 
default if the snapshots filling the shell are then 0.1 or more apart in redshift, a ``ValueError`` is raised before any 
data is read.

:envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` sets another limit (it is read each time a shell is filled):

.. code-block:: bash

    export SWIFTLET_MAX_SNAPSHOT_SPACING=0.26     # for one shell or job script

.. code-block:: python

    import os
    os.environ["SWIFTLET_MAX_SNAPSHOT_SPACING"] = "0.26"

    from swiftlet.snapshot_redshifts import max_snapshot_spacing
    max_snapshot_spacing()                         # the limit used: 0.26

The output redshifts of FLAMINGO and COLIBRE are themselves spaced more widely at high redshift, so the limit also sets
how far out a lightcone can be built from them:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Limit
     - FLAMINGO (``flamingo_L1``) reaches
     - COLIBRE reaches

   * - 0.1 (the default)
     - :math:`z = 3`
     - :math:`z = 2.5`

   * - 0.26
     - :math:`z = 5`
     - :math:`z = 10`

   * - 0.51
     - :math:`z = 5.5`
     - :math:`z = 15`


.. caution::

    **Keep the snapshot spacing small.** Each snapshot fills the redshifts around its own, so increasing the distance in 
    redshift between the snapshots filling a lightcone will increase the severity of the discontinuities where one 
    snapshot hands over to the next. Raising :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` above its default of 0.1 is not 
    advised and SWIFTLET warns whenever the limit is raised.


Where to look for the simulations
---------------------------------

Each suite looks for its simulations in ``<directory>/<box_res>/<sim_name>``, 
e.g. ``/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL``. 

The directory of a simulation is chosen in this order:

#.  the base directory (``base_dir_format``), if given, 
    e.g. ``Simulation.from_label(..., base_dir_format="/my/runs/{box_res}/{sim_name}")``, 
    or ``simulation_base_dir_format=`` for the lightcone classes
#.  the suite's environment variable, :envvar:`SWIFTLET_FLAMINGO_DIR` or :envvar:`SWIFTLET_COLIBRE_DIR`, the directory holding the 
    simulations
#.  the suite's directory on COSMA.


For simulations copied elsewhere, keeping the ``<box_res>/<sim_name>`` layout:

.. code-block:: bash

    export SWIFTLET_FLAMINGO_DIR=/data/flamingo        # finds /data/flamingo/L1000N1800/HYDRO_FIDUCIAL
    export SWIFTLET_COLIBRE_DIR=/data/colibre

The variable is read when the ``Simulation`` is made (by ``Simulation.from_label``, or by a lightcone class given a 
label and name). Set it before, then a new suite reads the variable named by its ``base_dir_env`` 
(see :mod:`swiftlet.simulation_suites`).

Within a simulation's directory, the snapshots are at the suite's ``snapshot_format``. For FLAMINGO this 
is ``snapshots/flamingo_XXXX/flamingo_XXXX.N.hdf5``.


.. _soap-halo-catalogues:

SOAP halo catalogues
^^^^^^^^^^^^^^^^^^^^

Each suite has a ``soap_format``: the path of the
catalogues relative to a simulation's directory, with a ``{snap_nr}`` field (``SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5``
for FLAMINGO and COLIBRE). A simulation's catalogues are then in its own directory, so they follow the simulation
wherever :envvar:`SWIFTLET_FLAMINGO_DIR` or :envvar:`SWIFTLET_COLIBRE_DIR` points:

.. code-block:: python

   sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
   sim.soap_format         # <simulation directory>/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5
   sim.soap_file(77)       # the SOAP catalogue of snapshot 77
   halos = beam.place_halos_in_shell(sim.soap_format, lightcone_redshift_range=(0.05, 0.15), ang_radius_deg=5.)


For catalogues somewhere else, give the simulation its own ``soap_format``, as a path relative to the simulation's
directory or an absolute path:

.. code-block:: python

   sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", soap_format="/my/soap/halo_properties_{snap_nr:04d}.hdf5")

:meth:`Simulation.from_path <swiftlet.simulation_suites.Simulation.from_path>` takes ``soap_format`` too. Each method 
that reads the catalogues also takes the format string directly, so any other path can be passed there.


FLAMINGO shell redshift files
-----------------------------

The environment variables that set the paths to the redshift output tables: :envvar:`L1_REDSHIFTS_FILENAME` and 
:envvar:`L2P8_REDSHIFTS_FILENAME`

These give the redshift bounds of each shell of the FLAMINGO lightcones, for the 1 Gpc (L1) and 2.8 Gpc (L2p8) 
boxes. ``swiftlet-configure`` downloads them into ``<environment>/share/swiftlet/redshifts`` and sets both variables in 
the activate script. Each file is looked for in this order:

#. The variable, if set;
#. ``<environment>/share/swiftlet/redshifts/``;
#. ``~/.cache/swiftlet/redshifts/``, used when the environment can't be written to;
#. The repository's ``data/redshifts/`` (an editable install).

If none has it, it is downloaded. To use your own copies, set the variables to their paths. 

HEALPix pixel weights
---------------------

Rotating HEALPix maps is faster and more precise with healpy's pixel weights. 
Use the prebuilt script `venv_scripts/healpix_pixel_weights.sh <../venv_scripts/make_cosma_env.sh>` to 
download and add :envvar:`HEALPY_PIXEL_WEIGHTS` to the environment's activate script. The weights are looked for in this order, in a 
directory holding a ``full_weights/`` folder:

#. The ``datapath`` argument, where a function takes one;
#. :envvar:`HEALPY_PIXEL_WEIGHTS`;
#. ``<environment>/share/healpy-data``.

Without them, the maps are rotated without pixel weights (``rotate_map`` instead of ``rotate_map_fast``).
