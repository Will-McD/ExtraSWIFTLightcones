Tests
=====


SWIFTLET's tests are in the `tests <https://github.com/Will-McD/ExtraSWIFTLightcones/tree/main/tests>`_ directory of
the repository. These test do not need simulation data: they write small fake snapshots and SOAP catalogues (by placing 
a particles in a lattice, with a halo at every lattice point) and check SWIFTLET against what the lightcone must hold.


Running the tests
-----------------

Install the test requirements and run pytest from the top directory of the repository:

.. code-block:: bash

    pip install -e ".[test]"                        # install requirements
    python -m pytest tests                          # every test
    python -m pytest tests/test_repeat_check.py     # one file
    python -m pytest tests -k "beam_is_part"        # tests whose name matches


Useful pytest options include:

* ``-v`` to list every test case
* ``-x`` to stop at the first failure
* ``--durations=10`` to show the slowest tests
* ``-p no:warnings`` to hide the deprecation warnings from unyt and healpy


The MPI tests (``test_mpi_matches_serial`` and ``test_mpi_haloes_match_serial``) run ``tests/mpi_place_particles.py``
and ``tests/mpi_place_haloes.py`` under ``mpiexec`` on 2 and 3 ranks, and are skipped if mpi4py or ``mpiexec`` is not
available. 
The tests use at most 8 threads wherever :envvar:`SWIFTLET_MAX_THREADS` would be used. This can be changed 
in ``tests/conftest.py``.




Test files
----------

The following table summarises the test files and what they check.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * -  Test file
     -  What it checks:

   * -  ``test_config.py``
     -  Downloading the shell redshift files, adding their paths and the maximum number of threads to the environment's 
        activate script, and finding them when needed.

   * -  ``test_threads.py``
     -  The maximum number of threads (:envvar:`SWIFTLET_MAX_THREADS`) can be set, and that numba and healpy alm 
        parallel functions abide by it.

   * -  ``test_init.py``
     -  The public classes and functions are imported from the package when first used, and importing the package 
        doesn't load the modules that need MPI.

   * -  ``test_orientation_tables.py``
     -  The default orientation tables are consistent; generated tables have every distinct orientation; the writing, 
        reading and checking of files holding the tables.

   * -  ``test_snapshot_lightcone_placement.py``
     -  Where :class:`swiftlet.snapshot_lightcone.SnapshotBeam` and :class:`swiftlet.snapshot_lightcone.SnapshotAllSky` 
        place particles and haloes.

   * -  ``test_snapshot_cosmology.py``
     -  :class:`swiftlet.snapshot_cosmology.SnapshotCosmology` functionality; reads FLAMINGO and COLIBRE snapshots, 
        H0 is transformed to km/s from SWIFT units, all required cosmological parameters are present, the neutrinos 
        masses (checked against swiftsimio), comoving distances, computing lightcone volumes, finding the snapshot 
        files, and the lightcones can manage different cosmologies. 

   * -  ``test_snapshot_redshifts.py``
     -  The redshift range each snapshot covers, :func:`swiftlet.snapshot_redshifts.snapshots_in_range` returns every 
        snapshot overlapping a redshift range within a given tolerance, and 
        :func:`swiftlet.snapshot_redshifts.snapshot_coverage` fills gaps in redshift due to missing snapshots and 
        fails when the snapshots are ``MAX_SNAPSHOT_SPACING`` (0.1) or more apart in redshift, a limit
        :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` can change this upper limit. 

   * -  ``test_simulation_suites.py``
     -  Each simulation's paths and snapshot redshifts come from its suite, FLAMINGO or COLIBRE. Making a 
        :class:`swiftlet.simulation_suites.Simulation` object from its label or path, finding the suite of a label, the 
        directories (including :envvar:`SWIFTLET_FLAMINGO_DIR` and :envvar:`SWIFTLET_COLIBRE_DIR`), the output redshift 
        files, and the lightcone classes taking a :class:`swiftlet.simulation_suites.Simulation` or a label and name.

   * -  ``test_soap_reader.py``
     -  The SOAP readers of :mod:`swiftlet.soap_reader` give the same rows and units whether reading in chunks or 
        selecting rows by a property, and correctly drop the expansion factor.

   * -  ``test_repeat_check.py``
     -  :func:`swiftlet.repeat_check.find_close_cell_repeats`, nothing is reported for unchanged tiles (every lock,
        all-sky and a beam) or within a ``"cube"`` layer; every close repeat of a particle on the fake lattice is in a
        reported pair of cells; reflected tiles make the cells on their faces clash, as
        :func:`swiftlet.snapshot_orientation.check_neighbouring_tiles` finds; the example runs; the warning and the HDF5
        file.

.. note::
    All contributions to SWIFTLET need to pass the whole automated ``pytest`` suite (see :doc:`developers`).


Snapshot lightcone placement tests
----------------------------------

The lightcone placement tests are particularly extensive and can take a while (around 30 minutes) to complete.

The following describes the various tests within `test_snapshot_lightcone_placement.py`:

*   A beam is exactly the part of the all-sky lightcone it sits within, with the same tables and seed, for every
    orientation lock and none, and for beams along and off the box axes (:func:`!test_beam_is_part_of_all_sky`, see
    :ref:`beams-are-part-of-the-all-sky-lightcone`).
*   Particles on a uniform lattice, and haloes from SOAP catalogues at the same lattice points, are placed exactly once.
    The reconstructed lattice in the lightcone has no gaps or overlaps between them, with each orientation lock, for 
    all-sky lightcones and for beams on and off the box axes, over many box tiles and several snapshots.
*   Consecutive shells together give the same particles as one shell over the whole range.
*   Placing the particles file by file (:meth:`swiftlet.snapshot_lightcone.SnapshotLightcone.gather_files` and 
    :meth:`swiftlet.snapshot_lightcone.SnapshotLightcone.place_file_in_shell`) or in parallel with MPI gives exactly 
    the particles of placing them in serial.
*   The number of particles placed matches the snapshots' number density times the volume they fill.
*   Re-orienting points on cell faces keeps them on cell faces.
*   An all-sky lightcone out to half a box length places every snapshot particle within it at its box position less than 
    half a box length away, as the observer's box tile is not re-oriented. i.e. for no re-orientation the initial 
    snapshot gets rebuilt.  
*   A missing snapshot (and SOAP catalogue) is filled by the available ones with a warning (every lattice point is still
    placed exactly once), and a gap of more than ``MAX_SNAPSHOT_SPACING`` fails before any data is read.
*   Haloes placed from SOAP catalogues end up exactly where particles at the same positions in the snapshots are placed,
    including haloes on cell faces.
*   The haloes placed are the same, in the same order, whether the SOAP catalogues are read whole or in chunks, and with
    MPI (on 1, 2 and 3 ranks, also with ``redistribute_halos```) the 
    same as in serial.
*   The halo dry run (:meth:`swiftlet.snapshot_lightcone.SnapshotLightcone.gather_halo_catalogues`) reads no haloes and 
    the cell metadata of one snapshot only, and :meth:`swiftlet.snapshot_lightcone.SnapshotLightcone.place_halos_in_shell` 
    reuses it, reading only the catalogues it found. With no snapshot in the shell, no halo is placed.
*   :class:`swiftlet.snapshot_lightcone.SnapshotBeam` needs an angular radius and 
    :class:`swiftlet.snapshot_lightcone.SnapshotAllSky` takes none, so arguments given by position in a beam's order are
    caught.
*   Outside of when ``orientation_lock="sphere"``, no particle or halo is placed twice in the same tile, and across the 
    lightcone none are placed more often than the number of tiles its snapshot fills.
*   The shortest distance between particles placed into the lightcone from the same tile is the same as in the snapshot,
    so each tile is the snapshot's box moved as a whole.
*   The ``ExpansionFactors`` of the particles and haloes placed are :math:`1 / (1 + z)` at their comoving distance, with 
    redshift (:math:`z`) found independently from the comovign distance to the observer. 
*   Non-essential properties read with the particles from a snapshot are carried through unchanged with each particle.
*   Particle velocities, when read, are turned and reflected the same as the coordinates are, in every tile, every lock,
    all-sky and beams, serial and MPI (:func:`!test_velocities_reoriented_with_the_coordinates`, 
    :func:`!test_mpi_matches_serial`).
*   Every tile's orientation is an exact signed permutation of the axes with a periodic shift of a whole number of 
    eighths of the box. Neighbouring tiles differ without a lock, tiles in the same layer share an orientation with the 
    ``"cube"`` or ``"sphere"`` lock, and the same ``orientation_seed`` gives the same orientations.
*   A lightcone made with the default tables written to a file is the same as one made with the hard-coded default 
    table.
*   The ``.metadata`` of the constructed lightcone (``orientation_seed``, ``orientation_lock``, orientation table, beam 
    and redshift range) is correctly written to and read back from ``.hdf5`` file dataset attributes.


MPI tests
---------
The MPI tests, :func:`!test_mpi_matches_serial` and :func:`!test_mpi_haloes_match_serial`, run ``mpi_place_particles.py`` 
and ``mpi_place_haloes.py`` with ``mpiexec -n 2`` and ``-n 3``.
They use the ``mpiexec`` next to the Python executable (where the ``mpich`` wheel installs it) if there is one, otherwise 
the one on the ``PATH``. They are skipped if there is no ``mpiexec`` or if mpi4py cannot be imported.

The ``mpiexec`` executable must belong to the MPI library mpi4py was built with. On a cluster, run these tests on a 
compute node (e.g. in an interactive ``salloc`` or ``srun`` session), as login nodes may not allow ``mpiexec``.

