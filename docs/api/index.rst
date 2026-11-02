API reference
=============


Building lightcones from snapshots
----------------------------------

The classes that build lightcones of particles and haloes from the snapshots of a simulation. Each links to its
documentation in :mod:`swiftlet.snapshot_lightcone`.

.. autosummary::
   :nosignatures:

   ~swiftlet.snapshot_lightcone.SnapshotLightcone
   ~swiftlet.snapshot_lightcone.SnapshotBeam
   ~swiftlet.snapshot_lightcone.SnapshotAllSky
   ~swiftlet.snapshot_lightcone.LightconeOutput


Simulations 
-----------

The main classes and functions for accessing a COLIBRE or FLAMINGO simulations snapshots and halo catalogues

.. autosummary::
   :nosignatures:

   ~swiftlet.simulation_suites.Simulation
   ~swiftlet.simulation_suites.SimulationSuite
   ~swiftlet.simulation_suites.output_redshifts
   ~swiftlet.snapshot_cosmology.SnapshotCosmology


Internal structure of a lightcone
---------------------------------

The main classes and functions related the orientation tables, which determine the internal structure of the lightcone, 
and to check for replicated structure along a given line of sight. 

.. autosummary::
   :nosignatures:

   ~swiftlet.repeat_check.find_close_cell_repeats
   ~swiftlet.repeat_check.write_close_repeats
   ~swiftlet.snapshot_orientation.default_orientation_tables
   ~swiftlet.snapshot_orientation.generate_orientation_tables
   ~swiftlet.snapshot_orientation.read_orientation_tables
   ~swiftlet.snapshot_orientation.write_orientation_tables



All Modules
-----------

Below are lists of all the modules containing with classes and functions utilised for building lightcones
(see :ref:`lightcones-sims`), creating projections and other HEALPix map related utility functions 
(see :ref:`visualisation`), and building required tables and config (see :ref:`set-up`). 


.. _lightcones-sims:
Lightcones and Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   swiftlet.snapshot_lightcone
   swiftlet.snapshot_orientation
   swiftlet.repeat_check
   swiftlet.simulation_suites
   swiftlet.snapshot_redshifts
   swiftlet.snapshot_cosmology
   swiftlet.snapshot_units
   swiftlet.soap_reader

.. _visualisation:
Projections and HEALPix maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   swiftlet.lightcone_projections
   swiftlet.beam_plotting
   swiftlet.healpix_map_utils
   swiftlet.mask_haloes
   swiftlet.flamingo_lightcones


.. _set-up:
Set-up and files
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:
   
   swiftlet.table_files
   swiftlet.config
   swiftlet.threads
