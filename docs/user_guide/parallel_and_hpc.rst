Use with HPC systems
====================

.. currentmodule:: swiftlet.snapshot_lightcone

SWIFTLET runs in serial, with threads, or across MPI ranks. This page explains which parts of SWIFTLET use threads or
MPI, and how to run them in a job script. The examples come with job scripts for COSMA (the ``.sh`` files in
`examples <https://github.com/Will-McD/ExtraSWIFTLightcones/tree/main/examples>`_), which can be adapted for other
systems.


Threads
-------

Re-orienting the box tiles (numba), the projections of :class:`BeamProjection <swiftlet.lightcone_projections.BeamProjection>` 
and the rotation of HEALPix maps (healpy) use threads. The maximum threads used is set by :envvar:`SWIFTLET_MAX_THREADS` 
(see :doc:`environment_variables`). For a job of one process with several cores, match the cores given to the job:

.. code-block:: bash

    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16

    export SWIFTLET_MAX_THREADS=${SLURM_CPUS_PER_TASK}
    python3 -u ./my_script.py


.. _max-threads:

Maximum number of threads
^^^^^^^^^^^^^^^^^^^^^^^^^

:envvar:`SWIFTLET_MAX_THREADS` sets the maximum number of threads for SWIFTLET's threaded code:

* numba's parallel functions, which re-orient 50,000 or more points at a time (and swiftsimio's parallel projections);
* healpy's spherical harmonic transforms (``map2alm``, ``rotate_alm``, ``alm2map``), used to rotate HEALPix maps.

The possible values of :envvar:`SWIFTLET_MAX_THREADS` are shown in the table below.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Value
     - Threads used

   * - Not set
     - 30

   * - A whole number, e.g. 16
     - That number

   * - ``all``
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
*   **With MPI:** use 1 (or the cores per rank), so the ranks on a node don't compete for its cores (see `MPI`_ below)

healpy's transforms are limited with ``threadpoolctl``, if it is installed. Without it, set ``OMP_NUM_THREADS`` instead.


MPI
---

MPI needs mpi4py and an MPI-enabled build of h5py (see :doc:`../installation`). The lightcone methods that read and
place particles or haloes take an MPI communicator:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - With ``comm``

   * - :meth:`SnapshotLightcone.place_snapshot_particles_in_shell`
     - every rank only reads and places particles from the (snapshots, tiles, files) that it is allocated. Each rank 
       returns the particles it placed; with ``redistribute=True`` they are shared evenly between the ranks 
       before being returned. 

   * - :meth:`SnapshotLightcone.place_halos_in_shell`
     - the SOAP catalogues are split into chunks of ``soap_chunk_size`` rows (if ``None``, one chunks per rank), shared
       between the ranks. Each rank returns the haloes it placed; with ``redistribute_halos=True`` they are then shared
       evenly between the ranks. Without ``comm``, ``MPI.COMM_WORLD`` is used if there is more than one rank.

The particles and haloes placed are the same whichever way they are read, serially or on any number of ranks (the tests
check this, see :doc:`../tests/index`). The communicator is used for that call only and is not kept, so a later call
without ``comm`` runs in serial.

:meth:`SnapshotLightcone.gather_files` and :meth:`SnapshotLightcone.place_file_in_shell` are serial only. To place a 
large lightcone file by file on several ranks, give each rank its own share of files from 
:meth:`gather_files <SnapshotLightcone.gather_files>` (every rank finds the same pieces).


.. code-block:: python

    from mpi4py import MPI
    from swiftlet import Simulation, SnapshotAllSky

    comm = MPI.COMM_WORLD
    sim = Simulation.from_label("L0400N3008", "Thermal")
    lightcone = SnapshotAllSky(simulation=sim, orientation_lock="cube", verbose=1 if comm.Get_rank() == 0 else 0)

    particles = lightcone.place_snapshot_particles_in_shell(
        (0.05, 0.1), ["Masses"], particle_types="PartType1", comm=comm,
    )
    haloes = lightcone.place_halos_in_shell(sim.soap_format, (0.05, 0.1), comm=comm)


.. caution::

    With MPI, adjust the number of threads (or cores) used per rank, so the ranks on a node don't compete for its cores. 
    Run Python through mpi4py, so an error on one rank stops them all:

    .. code-block:: bash

        #SBATCH --ntasks=64
        #SBATCH --cpus-per-task=1

        export SWIFTLET_MAX_THREADS=1
        mpirun -- python3 -m mpi4py ./my_script.py



Example job scripts
-------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Job script
     - Runs

   * - ``snapshot_beam_projection.sh``, ``snapshot_beam_zoom_on_haloes.sh``, ``snapshot_beam_halo_placement.sh``,
       ``snapshot_close_repeat_check.sh``
     - one process, with threads

   * - ``make_colibre_halo_lightcone.sh``, ``snapshot_smoothed_map.sh``, ``snapshot_number_density_comparison.sh``,
       ``lightcone_halo_M500c_binary_masks.sh``
     - MPI, one thread per rank


Each of these examples loads COSMA's modules and the environment made by ``venv_scripts/make_cosma_env.sh`` (see
:ref:`use-with-cosma`); change the paths, partition and account for other systems. 

