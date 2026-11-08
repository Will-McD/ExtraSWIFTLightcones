API reference
=============

.. module:: swiftlet


Building lightcones from snapshots
----------------------------------

The classes that build lightcones of particles and haloes from the snapshots of a simulation. Each links to its
documentation in :mod:`swiftlet.snapshot_lightcone`.

.. autosummary::
   :nosignatures:

   ~snapshot_lightcone.SnapshotLightcone
   ~snapshot_lightcone.SnapshotBeam
   ~snapshot_lightcone.SnapshotAllSky
   ~snapshot_lightcone.LightconeOutput


Simulations 
-----------

The main classes and functions for accessing the snapshots and halo catalogues of the COLIBRE or FLAMINGO simulations. 

.. autosummary::
   :nosignatures:

   ~simulation_suites.Simulation
   ~simulation_suites.SimulationSuite
   ~simulation_suites.output_redshifts
   ~snapshot_cosmology.SnapshotCosmology


Internal structure of a lightcone
---------------------------------

The main classes and functions related the orientation tables, which determine the internal structure of the lightcone, 
check for repeated structures and clashing (in contact) along a given line of sight. 

.. autosummary::
   :nosignatures:

   ~repeat_check.find_close_cell_repeats
   ~repeat_check.write_close_repeats
   ~snapshot_orientation.default_orientation_tables
   ~snapshot_orientation.generate_orientation_tables
   ~snapshot_orientation.read_orientation_tables
   ~snapshot_orientation.write_orientation_tables



All Modules
-----------

Below are lists of all the modules, with the classes and functions used for building lightcones
(see :ref:`lightcones-sims`), creating projections and other HEALPix map related utility functions
(see :ref:`visualisation`), and building required tables and config (see :ref:`set-up`).


.. _lightcones-sims:

Lightcones and simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   snapshot_lightcone
   snapshot_orientation
   repeat_check
   simulation_suites
   snapshot_redshifts
   snapshot_cosmology
   snapshot_units
   soap_reader

.. _visualisation:

Projections and HEALPix maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   lightcone_projections
   beam_plotting
   healpix_map_utils
   mask_haloes
   flamingo_lightcones


.. _set-up:

Set-up and files
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:
   
   table_files
   config
   threads
