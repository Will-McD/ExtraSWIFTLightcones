Developers
==========


Guidelines
----------

Contributions are encouraged and always welcome.
If you would like to contribute please read the `contribution guidelines <https://github.com/Will-McD/ExtraSWIFTLightcones/blob/main/CONTRIBUTING.md>`_
first and feel free to use the additional developers tools summarised below. 

In short, a contribution must:

*   pass the automated test suite (see :doc:`test_summary`), with new tests for new functionality
*   have Sphinx/reStructuredText docstrings that pass the style checks, and be formatted with ``ruff format`` and pass
    ``ruff check`` (see `Style check`_)
*   be documented, both as docstrings and in ``docs/``
*   declare the use of AI-assisted development, and be submitted, reviewed and understood by a human


Set up and maintenance
-----------------------


Below is a list of tools (found in the ``tools/`` directory) used for setting up and maintaining the package:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Tool
     - What it does

   * - ``swiftlet-configure``
     - downloads the FLAMINGO shell redshift files into the environment and exports their paths and
       :envvar:`SWIFTLET_MAX_THREADS` in its activate script (see :doc:`../installation` and
       :doc:`../user_guide/environment_variables`)

   * - ``tools/make_installed_orientation_tables.py``
     - Python script to generate all default orientation tables (installed with SWIFTLET) and write them to their 
       respective YAML files. This script can be edited to build and write new default tables.

   * - ``tools/style_check.sh``
     - runs the style checks of the repository's GitHub workflow (ruff, codespell, interrogate, flake8 and pydoclint)
       and writes their output, which includes a report card to a log file (``./logs/style_check_<date>_<time>.log``)

   * - ``tools/check_docstring_summaries.py``
     - checks the docstring conventions the other checks have no rules for: the summary line and the ``:type:`` lines. 
       Specifically it checks that it is a full sentence ending with a full stop, followed by a blank line and every 
       ``:type:``, ``:vartype:`` and ``:rtype:`` line names types the docs can link to

   * - ``venv_scripts/make_venv.sh``, ``venv_scripts/make_cosma_env.sh``
     - make a virtual environment with SWIFTLET and its requirements (see :doc:`../installation`)

   * - ``venv_scripts/shell_redshifts.sh``, ``venv_scripts/healpix_pixel_weights.sh``
     - add the shell redshift files or the HEALPix pixel weights to an environment (see :ref:`additional-data`)


Style check
^^^^^^^^^^^

Run every style check:

.. code-block:: bash

   bash tools/style_check.sh            # the style checks
   bash tools/style_check.sh --docs     # also build the docs with -W -n



Testing how snapshots are read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tests check not only what is placed but how files are read: each file is opened once, each cell is read
once, and that there is a cap of 4 concurrent requests for accessing remote data. 
The tests of not need to access any simulation data on disk or remotely:
``test_snapshot_lightcone_placement.py`` writes small fake simulation data products (snapshots, SOAP catalogues and 
output lists) to a temporary directory for each test session, and these helpers in ``tests/`` do the rest.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Helper
     - Use

   * - ``read_counter.py``: ``ReadCounter``
     - records every HDF5 file opened for reading and every dataset read on disk, by any method
       (h5py's reads, and SWIFTLET's one read of many slices, ``hdf5stream_reader.read_slices``), as the rows read
       along each dataset's first axis. ``rows()``, ``totals()`` and ``format_table()`` report the opens, requests,
       rows read, distinct rows and rows read again, per (file, dataset) or in total. ``ReadCounter.particle_datasets``
       and ``ReadCounter.cell_datasets`` select the particle data or the cell metadata.

   * - ``read_counter.py``: ``FakeRemoteDirectory``
     - a stand-in for an ``hdfstream.RemoteDirectory``: its datasets have ``request_slices``,
       which checks the slices of one request (``check_request_slices``), and every file
       opened, request made and directory listed is logged in its ``log`` (a ``RequestLog``), with each request's start
       and end times. ``max_in_flight`` finds the most in flight at once, to test the :term:`request cap`.

   * - ``lightcone_helpers.py``
     - ``remote_simulation`` makes a fake simulation read through a ``FakeRemoteDirectory``. ``rows_of`` and
       ``assert_same_rows`` compare placed particles whatever their order. ``tile_file_pieces`` lists a shell's
       (snapshot, tile, file) pieces without the file plans, as an independent reference for them.


For example, to test that each row of particles in a file is only read once when building a lightcone:

.. code-block:: python

    from read_counter import ReadCounter

    with ReadCounter() as counter:
        lightcone.place_snapshot_particles_in_shell((0.1, 0.2), ["Coordinates"], "PartType1")
    totals = counter.totals(ReadCounter.particle_datasets)
    assert totals["rows_read_again"] == 0, counter.format_table(ReadCounter.particle_datasets)


Changing the installed orientation tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installed (and default) tables are listed in several places. If you add, remove or change any of these tables please 
take care to update where they are mentioned in the documentation:

*   ``swiftlet/snapshot_orientation.py``: ``INSTALLED_ORIENTATION_TABLES`` and, for a new default,
    ``DEFAULT_ORIENTATION_TABLE``
*   ``tools/make_installed_orientation_tables.py``: the table's rows and its entry in ``FINGERPRINTS``, then write the
    YAML files in ``swiftlet/data/orientation_tables`` with the tool, not by hand, and check them with ``--check``
*   ``tests/test_orientation_tables.py``: the tests of what the table promises (e.g. clash-free, or the separation of a
    cell's copies)
*   ``docs/user_guide/orientation_tables.rst``: the table of installed tables and the text about the default
*   ``docs/glossary.rst``: the :term:`orientation table` entry
*   ``docs/user_guide/repeated_objects.rst``, ``docs/user_guide/orientation.rst`` and ``docs/quickstart.rst``: anything
    that names the default or a table


