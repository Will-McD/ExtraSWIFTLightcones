Additional tools
================

Besides building lightcones, SWIFTLET has tools to work with FLAMINGO's own lightcone
products and read SOAP catalogues. This page lists them with each links to the corresponding API
documentation.


Reading SOAP halo catalogues
----------------------------

:mod:`swiftlet.soap_reader` reads SOAP catalogues, in chunks, with h5py  with the same units as
LightconeIO's ``SOAPCatalogue``:

*   :func:`read_soap_rows <swiftlet.soap_reader.read_soap_rows>`: read only a range of rows within the catalogue
*   :func:`read_soap_in_chunks <swiftlet.soap_reader.read_soap_in_chunks>`: read the whole catalogue, a chunk of rows at 
    a time
*   :func:`read_soap_selected_rows <swiftlet.soap_reader.read_soap_selected_rows>`: read only the given rows, e.g. the
    ``InputHalos/SOAPIndex`` of the haloes of a lightcone (see the :doc:`../quickstart`)
*   :func:`select_soap_rows <swiftlet.soap_reader.select_soap_rows>`: read only the rows whose value of a property is 
    selected, e.g. haloes above a given mass 

The paths of a simulation's catalogues come from its ``soap_format`` (see :ref:`soap-halo-catalogues`).


FLAMINGO lightcones and HEALPix maps
------------------------------------

*   :mod:`swiftlet.flamingo_lightcones`: functions the read in the redshift bounds of the shells of FLAMINGO's 
    lightcones, :func:`flamingo_shell_redshift_file <swiftlet.flamingo_lightcones.flamingo_shell_redshift_file>` (see
    :ref:`shell-redshifts`)
*   :mod:`swiftlet.healpix_map_utils`: functions to rotate the HEALPix maps of FLAMINGO's lightcone shells
    (:func:`rotate_map <swiftlet.healpix_map_utils.rotate_map>`, or
    :func:`rotate_map_fast <swiftlet.healpix_map_utils.rotate_map_fast>` with the pixel weights), and sum them
    (:func:`write_rotated_lightcone_chunks <swiftlet.healpix_map_utils.write_rotated_lightcone_chunks>`,
    :func:`sum_maps <swiftlet.healpix_map_utils.sum_maps>`)
*   :mod:`swiftlet.mask_haloes`: methods to build binary masks of the haloes of in a shell, in bins of mass, with MPI
    (:func:`write_binary_masks <swiftlet.mask_haloes.write_binary_masks>`)


Redshifts, cosmology and units
------------------------------

*   :mod:`swiftlet.snapshot_redshifts`: the redshifts each snapshot fills in a lightcone
    (:func:`redshift_edges <swiftlet.snapshot_redshifts.redshift_edges>`,
    :func:`snapshots_in_range <swiftlet.snapshot_redshifts.snapshots_in_range>`);
*   :mod:`swiftlet.snapshot_cosmology`: determine the cosmology of a simulation, from its snapshots
    (:class:`SnapshotCosmology <swiftlet.snapshot_cosmology.SnapshotCosmology>`);
*   :mod:`swiftlet.snapshot_units`: functions to work with the units of SWIFT snapshots and SOAP catalogues, as unyt 
    units.


