Developers
==========


Guidelines
----------

Contributions are encouraged and always welcome.
If you would like to contribute please read the `contribution guidelines <https://github.com/Will-McD/ExtraSWIFTLightcones/main/CONTRIBUTING.md>`_. 
first and feel free to use the additional developers tools summarised below. 

In short, a contribution must:

*   pass the automated test suite (see :doc:`test_summary`), with new tests for new functionality
*   have Sphinx/reStructuredText docstrings that pass the style checks, and be formatted with ``ruff format`` and pass
    ``ruff check`` (see `Style check`_)
*   be documented, both as docstrings and in ``docs/``
*   declare the use of AI-assisted development, and be submitted, reviewed and understood by a human


Set up and maintenance
-----------------------

.. Note::
    The tools listed here are not installed with the package, but found in the ``tools/`` directory of the repository. 

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
     - Python script to generate all default oritentation tables (installed with SWIFTLET) and write them too their 
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

