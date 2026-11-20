Environment variables
=====================

SWIFTLET reads a few environment variables. They set how many threads it uses and where it finds the simulations and its 
data files. This page lists each one and how to change it.


.. envvar:: SWIFTLET_MAX_THREADS

  Maximum number of threads used by SWIFTLET, 30 by default (see :ref:`max-threads`). 

.. envvar:: SWIFTLET_MAX_SNAPSHOT_SPACING

  The largest redshift spacing allowed between the snapshots filling a lightcone :term:`shell`: a gap of this much or 
  more (from missing snapshots, or between the outputs themselves) raises a ``ValueError``. 
  :envvar:`SWIFTLET_MAX_SNAPSHOT_SPACING` is set to 0.1 by default (see :ref:`max-snapshot-spacing`).

.. envvar:: SWIFTLET_SUITES_DIR

  Directories of simulation suite files (``*.yaml``), separated by ``:``, read after the installed FLAMINGO and COLIBRE
  suites when SWIFTLET is imported. A file there adds a new suite, or replaces the suite of the same name (see
  :ref:`new-suite`). Not set by default.

.. envvar:: L1_REDSHIFTS_FILENAME

  The FLAMINGO lightcone shell redshift files for the  1 Gpc (L1) boxsizes, set by ``swiftlet-configure``. 

.. envvar:: L2P8_REDSHIFTS_FILENAME

  The FLAMINGO lightcone shell redshift files for the  2.8 Gpc (L2p8) boxsizes, set by ``swiftlet-configure``. 

.. envvar:: SWIFTLET_FLAMINGO_DIR

  The directory holding the FLAMINGO simulations: ``/cosma8/data/dp004/flamingo/Runs``.

.. envvar:: SWIFTLET_COLIBRE_DIR

  The directory holding the COLIBRE simulations: ``/cosma8/data/dp004/colibre/Runs``.

.. envvar:: HEALPY_PIXEL_WEIGHTS

  The directory of the HEALPix pixel weights: ``<environment>/share/healpy-data``



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


.. _install-flamingo-redshifts:

FLAMINGO redshift shells
------------------------

The environment variables that set the paths to the files containing the redshift bounds of `each shell in FLAMINGO's lightcones <https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html>`_ 
: :envvar:`L1_REDSHIFTS_FILENAME` and :envvar:`L2P8_REDSHIFTS_FILENAME`.
These give the redshift bounds of each shell of the FLAMINGO lightcones, for the 1 Gpc (L1) and 2.8 Gpc (L2p8) 
boxes. ``swiftlet-configure`` downloads them into ``<environment>/share/swiftlet/redshifts`` and sets both variables to 
the environment's activate script (``bin/activate`` for a venv, ``etc/conda/activate.d`` for conda). Each file is looked 
for in this order:

#. The variable, if set;
#. ``<environment>/share/swiftlet/redshifts/``;
#. ``~/.cache/swiftlet/redshifts/``, used when the environment can't be written to;
#. The repository's ``data/redshifts/`` (an editable install).


If ``swiftlet-configure`` is not run, the files are downloaded the first time they are needed instead. 
To use your own copies, set the variables to their paths. 


.. _healpix-weights:

HEALPix pixel weights
---------------------

Rotating HEALPix maps is faster and more precise with healpy's pixel weights. 
Use the prebuilt script `venv_scripts/healpix_pixel_weights.sh <https://github.com/Will-McD/ExtraSWIFTLightcones/blob/main/venv_scripts/healpix_pixel_weights.sh>`_
to download and add :envvar:`HEALPY_PIXEL_WEIGHTS` to the environment's activate script. The weights are looked for in 
this order, in a directory holding a ``full_weights/`` folder:

#. The ``datapath`` argument, where a function takes one;
#. :envvar:`HEALPY_PIXEL_WEIGHTS`;
#. ``<environment>/share/healpy-data``.

Without them, the maps are rotated without pixel weights (``rotate_map`` instead of ``rotate_map_fast``).
