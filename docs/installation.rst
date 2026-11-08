Installation
============

Clone the repository, then either build a new virtual environment with everything SWIFTLET needs, or install it into an 
environment you already have. Both routes download two small .txt files containing the FLAMINGO lightcone shell redshifts 
into the environment, and export their paths whenever the environment is activated.

New virtual environment
-----------------------

From anywhere, make a virtual environment containing SWIFTLET (in editable mode), its dependencies, the dependencies of 
the `examples <https://github.com/Will-McD/ExtraSWIFTLightcones/tree/main/examples>`_ and tests, and the shell redshift 
files:

.. code-block:: bash

   git clone https://github.com/Will-McD/ExtraSWIFTLightcones.git
   bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment
   source /path/to/environment/bin/activate

If no path is given, the environment is made in ``ExtraSWIFTLightcones/swiftlet_env``.

.. note::

   If you have access to COSMA, build the environment as described in :ref:`use-with-cosma` instead.

Existing environment
--------------------

With your environment (venv or conda) activated, install the package and its requirements from the repository directory,
then download the shell redshift files:

.. code-block:: bash

   cd ExtraSWIFTLightcones
   pip install -e ".[all]"          # the package and its requirements
   swiftlet-configure

To install only the additional requirements of the examples, tests or building the documentation:

.. code-block:: bash

   pip install -e ".[examples]"     # requirements of the examples
   pip install -e ".[test]"         # requirements of the tests
   pip install -e ".[docs]"         # requirements of building the documentation (see docs/requirements.txt)

Keep the ``-e`` (editable install): without it, pip copies the package into the environment, and changes to the 
repository (e.g. after ``git pull``) are not used until it is installed again.



.. _use-with-cosma:

Use with COSMA
--------------

This step only applies if you have access to `COSMA <https://cosma.readthedocs.io/en/latest/index.html>`_. On COSMA, 
build the virtual environment with COSMA's pre-built wheels of mpi4py and h5py, which match its MPI and parallel HDF5 
libraries:

.. code-block:: bash

   bash ExtraSWIFTLightcones/venv_scripts/make_cosma_env.sh /path/to/environment


MPI support
-----------

MPI is not required to build new lightcones from snapshots with the :class:`swiftlet.snapshot_lightcone.SnapshotLightcone` 
subclasses. It is needed for the faster parallel methods of :class:`swiftlet.snapshot_lightcone.SnapshotLightcone`, for 
the :class:`swiftlet.lightcone_projections.BeamProjection` class, and for generating binary masks of haloes
(:mod:`swiftlet.mask_haloes`).

MPI support requires mpi4py and an MPI-enabled build of h5py. The mpi4py package installed from PyPI needs an MPI 
library at run time; if your system has none, install one into the environment with ``pip install mpich`` 
(or ``pip install openmpi``). ``swiftlet-configure`` does not need MPI.

On COSMA, the environment built by ``make_cosma_env.sh`` (see :ref:`use-with-cosma`) is already configured to use 
COSMA's mpi4py and h5py.

.. seealso::

   Instructions for installing `LightconeIO with MPI support <https://lightconeio.readthedocs.io/en/latest/installation.html#with-mpi-support>`_.


.. _number-of-threads:


Number of threads
-----------------

The threaded parts of SWIFTLET, numba's parallel functions (used to re-orient snapshots and in 
:class:`swiftlet.lightcone_projections.BeamProjection`) and healpy's ``map2alm``, ``rotate_alm`` and ``alm2map``, use at
most :envvar:`SWIFTLET_MAX_THREADS` threads (see :doc:`user_guide/environment_variables`). ``swiftlet-configure`` adds 
:envvar:`SWIFTLET_MAX_THREADS` to the environment's activate script, to change it:

.. code-block:: bash

   swiftlet-configure --max_threads 16    # saved in the environment; re-activate it to apply
   export SWIFTLET_MAX_THREADS=16         # for one shell or job script


To set the limit when building a new environment, for example to 16 instead of 30:

.. code-block:: bash

   SWIFTLET_MAX_THREADS=16 bash ExtraSWIFTLightcones/venv_scripts/make_venv.sh /path/to/environment

To have no fixed maximum, so that each job (or MPI rank) uses all the cores it is allowed, use ``all`` instead of a 
number, e.g. ``swiftlet-configure --max_threads all``. This suits jobs where the scheduler sets the number of cores, 
e.g. ``--cpus-per-task`` in Slurm.

.. _additional-data:

Additional data
---------------

.. _shell-redshifts:

Lightcone shell redshift files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The redshift bounds of `each shell in FLAMINGO's lightcones <https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/index.html>`_
are stored in two files, that can be installed with SWIFTLET. They are needed to map between FLAMINGO's HEALPix maps, 
which are built in concentric redshift shells, and the corresponding halo lightcones, which are built per snapshot, and 
to find which HEALPix maps lie within a given redshift range.

See :ref:`install-flamingo-redshifts` for further information on installing the redshift shells. 

.. note::

   The `COLIBRE <https://colibre.strw.leidenuniv.nl/index.html>`_ simulations do not have lightcones, and have no
   shell redshift files.


HEALPIx weights
^^^^^^^^^^^^^^^

To speed up the rotation of HEALPix maps, download the HEALPix pixel weights and add their paths to your environment:

.. code-block:: bash

   cd ExtraSWIFTLightcones
   bash venv_scripts/healpix_pixel_weights.sh


See :ref:`healpix-weights` for further information. 