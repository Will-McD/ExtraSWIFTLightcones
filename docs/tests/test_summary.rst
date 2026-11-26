Tests
=====


SWIFTLET's tests are in the `tests <https://github.com/Will-McD/ExtraSWIFTLightcones/tree/main/tests>`_ directory of
the repository. These tests do not need simulation data: they write small fake snapshots and SOAP catalogues (by placing 
particles in a lattice, with a halo at every lattice point) and check SWIFTLET against what the lightcone must hold.


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
* ``-p no:warnings`` to hide the warnings, e.g. healpy's deprecation warnings



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
     -  The installed orientation tables are consistent; generated tables have every distinct orientation; the writing,
        reading and checking of files holding the tables; and the NumPy and numba rotations give identical results.

   * -  ``test_snapshot_lightcone_placement.py``
     -  Where :class:`swiftlet.snapshot_lightcone.SnapshotBeam` and :class:`swiftlet.snapshot_lightcone.SnapshotAllSky` 
        place particles and haloes.

   * -  ``test_snapshot_cosmology.py``
     -  :class:`swiftlet.snapshot_cosmology.SnapshotCosmology` functionality; reads FLAMINGO and COLIBRE snapshots, 
        H0 is transformed to km/s from SWIFT units, all required cosmological parameters are present, the neutrinos 
        masses (checked against swiftsimio), comoving distances, computing lightcone volumes, finding the snapshot 
        files (including a simulation's lowest redshift snapshot, whatever its snapshots are named), and the
        lightcones can manage different cosmologies. 

   * -  ``test_snapshot_redshifts.py``
     -  The redshift range each snapshot covers, :func:`swiftlet.snapshot_redshifts.snapshots_in_range` returns every 
        snapshot overlapping a redshift range within a given tolerance, and 
        :func:`swiftlet.snapshot_redshifts.snapshot_coverage` fills gaps in redshift due to missing snapshots and 
        fails when the snapshots are ``MAX_SNAPSHOT_SPACING`` (0.1) or more apart in redshift, a limit
        :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` can change this upper limit. 

   * -  ``test_simulation_suites.py``
     -  Each simulation's paths and snapshot redshifts come from its suite, FLAMINGO or COLIBRE. Making a 
        :class:`swiftlet.simulation_suites.Simulation` object from its label or path, finding the suite of a label, the 
        directories (including :envvar:`SWIFTLET_FLAMINGO_DIR` and :envvar:`SWIFTLET_COLIBRE_DIR`), reading output lists 
        and refusing other files. The lightcone classes take a :class:`swiftlet.simulation_suites.Simulation` 
        and refuse a label, with an error showing how to make a 
        :class:`Simulation <swiftlet.simulation_suites.Simulation>`. A suite written to a .yaml file is read back the
        same, moreover the files in :envvar:`SWIFTLET_SUITES_DIR` are preferentially used over the installed ones and a 
        file that isn't a suite fails.

   * -  ``test_soap_reader.py``
     -  The SOAP readers of :mod:`swiftlet.soap_reader` give the same rows and units whether reading in chunks or 
        selecting rows by a property, and correctly drop the expansion factor.

   * -  ``test_repeat_check.py``
     -  :func:`swiftlet.repeat_check.find_close_cell_repeats`, nothing is reported for unchanged tiles (every lock,
        all-sky and a beam) or within a ``"cube"`` layer; every close repeat of a particle on the fake lattice is in a
        reported pair of cells; reflected tiles make the cells on their faces clash, as
        :func:`swiftlet.snapshot_orientation.check_neighbouring_tiles` finds; the example runs; the warning and the HDF5
        file.

   * - ``test_read_counts.py``
     - ``tests/read_counter.py`` counts each dataset read in one request for a file from a fake simulation. Check that 
       placing particles into a lightcone reads exactly the cells planned from each file and each snapshot's cell 
       metadata once, opens each snapshot file once per call and reads no particle row twice.

   * -  ``test_remote_reading.py``
     -  Reading snapshots remotely with hdfstream (``swiftlet.hdf5stream_reader``), against the stand-in for the remote
        service, check that ``read_slices`` gives the same rows from local and remote datasets with one request for all 
        of the slices and refuses bad slices (out of order, overlapping,
        negative, unmatched) on disk and remotely before any request. Ensure that files and text (the output list)
        are found and read remotely. For the reading, check each particle file is opened once and read through before 
        the next, with one request per dataset per batch. Moreover, with MPI, only 2-4 ranks (``mpi_remote_reads.py``), 
        at most are making requests to read the remote data at once and the ranks place exactly the same particles as 
        if this was done in serial. Repeat the same checks for SOAP catalogues. 

.. note::
    All contributions to SWIFTLET need to pass the whole automated ``pytest`` suite (see :doc:`developers`).


Snapshot lightcone placement tests
----------------------------------

The lightcone placement tests are particularly extensive and take around 10 minutes to complete.

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
    MPI (on 1, 2 and 3 ranks, also with ``redistribute_halos``) the same as in serial.
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
*   Each call makes its own record of what its gather step found (``last_shell``), which a later call doesn't change;
    the cell metadata kept is only that of the current call; a serial call after an MPI one runs in serial.
*   Particles and haloes choose the same snapshots for the same shell, with and without ``use_snapshots``.
*   A particle type missing from some snapshots adds none from them, and one in none of the snapshots raises a
    ``ValueError`` naming the types there are.
*   ``Snapshot2Lightcone`` needs the tile, and moves every particle of a snapshot to where the lightcone places it.
*   The ``"fast"`` and ``"precise"`` redshift interpolations (``redshift_interp_method``) place the same particles.
*   When a new observer position is given it is adjusted to the nearest cell corner, that with the unchanged table 
    every lattice point is still placed exactly once, a beam is part of the all-sky lightcone, placing file by file 
    gives the particles placed at once, the placement on 3 MPI ranks matches serial and the metadata records the
    position, corner and shift. 


MPI tests
---------
The MPI tests, :func:`!test_mpi_matches_serial` and :func:`!test_mpi_haloes_match_serial`, run ``mpi_place_particles.py`` 
and ``mpi_place_haloes.py`` with ``mpiexec -n 2`` and ``-n 3``.
They use the ``mpiexec`` next to the Python executable (where the ``mpich`` wheel installs it) if there is one, otherwise 
the one on the ``PATH``. They are skipped if there is no ``mpiexec`` or if mpi4py cannot be imported.

The ``mpiexec`` executable must belong to the MPI library mpi4py was built with. On a cluster, run these tests on a 
compute node (e.g. in an interactive ``salloc`` or ``srun`` session), as login nodes may not allow ``mpiexec``.

