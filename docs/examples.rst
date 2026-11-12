Examples
========

The `examples <https://github.com/Will-McD/ExtraSWIFTLightcones/tree/main/examples>`_ directory of the repository holds
worked examples, most with a job script (``.sh``) to run them on COSMA (see :doc:`user_guide/parallel_and_hpc`).
Their extra requirements are installed with ``pip install -e ".[examples]"``. Run them from the top directory of the
repository; on another machine, change the paths of the simulations (see :doc:`user_guide/environment_variables`).


Below are lists of examples for both the existing FLAMINGO lightcones and generating new snapshot lightcones. 


FLAMINGO's lightcones
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - What it does

   * - ``lightcone_beam_projection.py``
     - projects slices through a beam of a FLAMINGO particle lightcone

   * - ``integrated_lightcone_map.py``
     - integrates the ``TotalMass`` and ``SmoothedGasMass`` HEALPix maps of FLAMINGO L1_m9 over :math:`0.1 < z < 0.3`

   * - ``map_zoom_on_halo.py``
     - reads and plots only a subsection of pixels around a halo in a HEALPix map

   * - ``lightcone_halo_M500c_binary_masks.py`` / ``.sh``
     - makes binary masks of a full-sky map of FLAMINGO L1_m9, masking every pixel within :math:`R_{500c}` of a halo,
       with MPI


Lightcones from snapshots
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - What it does

   * - ``snapshot_beam_particle_placement.py``
     - places the particles from snapshots into a pencil beam over a set redshift range and plots where each particle 
       is. Repeated for beams pointing in different directions

   * - ``snapshot_beam_halo_placement.py`` / ``.sh``
     - places the haloes from SOAP catalogues in a :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>`
       lightcone for a set redshift shell

   * - ``make_colibre_halo_lightcone.py`` / ``.sh``
     - builds an all-sky halo lightcone of COLIBRE L400_m7 (``L0400N3008/Thermal``) for :math:`0.001 < z < 0.1`, with
       MPI, and writes it in the format of the LightconeIO halo lightcones

   * - ``snapshot_beam_projection.py`` / ``.sh``
     - builds beams of gas particles from the snapshots of FLAMINGO L1_m10 and COLIBRE L400_m7, and projects a slice
       through them (see :doc:`../user_guide/projections`)

   * - ``snapshot_beam_zoom_on_haloes.py`` / ``.sh``
     - zooms in on haloes of a :class:`SnapshotBeam <swiftlet.snapshot_lightcone.SnapshotBeam>` lightcone

   * - ``snapshot_smoothed_map.py`` / ``.sh``
     - makes an all-sky HEALPix map of the smoothed gas mass for :math:`0.05 < z < 0.1` from snapshot lightcones of
       COLIBRE L400_m7 and FLAMINGO L1_m9, with MPI

   * - ``snapshot_number_density_comparison.py`` / ``.sh``
     - compares the number density of dark matter particles in the snapshot lightcones with the expected 
       number density from the initial snapshots

   * - ``snapshot_close_repeat_check.py`` / ``.sh``
     - checks a lightcone for close repeats of the same SWIFT cells before building it, for every orientation lock and 
       several seeds. Generates a summary plot of the occurrence on-sky positions of these repeated cells 
       (see :doc:`../user_guide/repeated_objects`)

   * - ``snapshot_close_repeat_map.py`` / ``.sh``
     - checks and plots the close line-of-sight repeats of the same SWIFT cells, for every orientation lock for each 
       installed orientation table with a set orientation seed. Show each pair on a Mollweide and then overlays 
       the footprint of a planned beam lightcone beam (see :doc:`../user_guide/repeated_objects`). 


.. seealso::
  The ``examples/diagrams`` directory holds scripts that draw diagrams of how a lightcone is built: the lattice of 
  tiles, their re-orientation, the orientation locks, the redshift shells of the snapshots and the cells a beam selects.

