#!/bin/env python
import builtins

import h5py
import numpy as np
import unyt
import virgo.formats.swift as virgo_swift


"""
Functions for reading SOAP halo catalogues, in serial, with h5py.
These reading functions give soap properties the units lightcone_io.halo_catalogue.SOAPCatalogue.read gives them
(virgo.formats.swift), so the values are the same whichever reader is used:

    read_soap_rows:             a range of rows [start, stop)
    read_soap_in_chunks:        the whole catalogue, chunk_size rows at a time
    read_soap_selected_rows:    the given rows only
    select_soap_rows:           the rows where one property passes a test, the property read in chunks and the other
                                    properties only for the rows kept
    soap_catalogue_length:      the number of haloes, without reading any

Comoving properties keep the expansion factor a in their units (e.g. a*snap_length); comoving() drops it, giving the
comoving values (e.g. comoving Mpc).
"""


# FORCE A PATCH FOR VIRGODC WITH NUMPY>2.4
# unknown when raised issue will be corrected with virgodc and cannot revert to numpy<2.4
# implement an intermediate patch for virgodc float values with soap catalogues.
def _float(value):
    """
    float() that also accepts one-element arrays, as numpy < 2.4 did.

    virgo.formats.swift (virgodc <= 1.0.5) calls float() on the one-element array attributes of SWIFT and SOAP
    files to build their units, which numpy >= 2.4 refuses, so lightcone_io's SOAPCatalogue cannot read SOAP
    catalogues. Used as float inside virgo.formats.swift only, until virgodc is fixed.

    :param  value:  number or one-element array
    :type   value:  float or np.ndarray
    """
    if np.size(value) == 1:
        return builtins.float(np.asarray(value).reshape(-1)[0])
    return builtins.float(value)

virgo_swift.float = _float


def _soap_datasets(infile, property_names):
    """
    The datasets of a SOAP catalogue, with the units lightcone_io.halo_catalogue.SOAPCatalogue.read gives them
    (virgo.formats.swift).

    Returns (unit registry, {name: dataset}, {name: units}, number of rows).

    :param  infile:         open SOAP catalogue
    :type   infile:         h5py.File
    :param  property_names: names of the datasets, e.g. "InputHalos/HaloCentre"
    :type   property_names: list of str
    """
    registry = virgo_swift.soap_unit_registry_from_snapshot(infile)
    datasets = {name: infile[name] for name in property_names}
    units = {name: virgo_swift.soap_units_from_attributes(dataset.attrs, registry)
             for name, dataset in datasets.items()}
    nr_rows = {dataset.shape[0] for dataset in datasets.values()}
    if len(nr_rows) != 1:
        raise ValueError(f"{infile.filename}: {property_names} have different numbers of rows")
    return registry, datasets, units, nr_rows.pop()


def comoving(arr):
    """
    Drop the expansion factor a from the units of a SOAP property without changing its values, e.g. lengths in
    a*snap_length become comoving snap_length. Properties without a in their units are returned as they are.

    Returns a unyt.unyt_array.

    :param  arr:    property read from a SOAP catalogue
    :type   arr:    unyt.unyt_array
    """
    expr = arr.units.expr
    a_symbols = [symbol for symbol in expr.free_symbols if str(symbol) == "a"]
    if not a_symbols:
        return arr
    apply_units = unyt.Unit(expr.subs(a_symbols[0], 1), registry=arr.units.registry)
    return unyt.unyt_array(arr.value, apply_units)


def soap_catalogue_length(filename, property_name="InputHalos/HaloCatalogueIndex"):
    """
    Number of haloes (rows) in a SOAP catalogue, without reading any of them.

    :param  filename:       path of the SOAP catalogue
    :type   filename:       str
    :param  property_name:  dataset whose length is taken
    :type   property_name:  str
    """
    with h5py.File(filename, "r") as infile:
        return infile[property_name].shape[0]


def read_soap_rows(filename, property_names, start=0, stop=None):
    """
    Read rows [start, stop) of properties of a SOAP catalogue, in serial, with the same units as reading the whole
    catalogue with lightcone_io.halo_catalogue.SOAPCatalogue.

    Returns {name: unyt.unyt_array of the rows}.

    :param  filename:       path of the SOAP catalogue
    :type   filename:       str
    :param  property_names: names of the datasets to read, e.g. "InputHalos/HaloCentre"
    :type   property_names: list of str
    :param  start:          first row to read
    :type   start:          int
    :param  stop:           row after the last one to read. If None, the end of the catalogue
    :type   stop:           int
    """
    with h5py.File(filename, "r") as infile:
        registry, datasets, units, nr_rows = _soap_datasets(infile, property_names)
        stop = nr_rows if stop is None else min(int(stop), nr_rows)
        return {name: unyt.unyt_array(dataset[int(start):stop], units=units[name], registry=registry)
                for name, dataset in datasets.items()}


def read_soap_in_chunks(filename, property_names, chunk_size):
    """
    Read properties of a SOAP catalogue chunk_size rows at a time, in serial, so only one chunk of the catalogue
    is in memory at once. 
    The units are made exactly as lightcone_io.halo_catalogue.SOAPCatalogue.read makes them
    (virgo.formats.swift), so the values are the same as reading the whole catalogue.

    Returns a tuple for each chunk: 
        (index of its first row in the catalogue, {name: unyt.unyt_array of its rows}).

    :param  filename:       path of the SOAP catalogue
    :type   filename:       str
    :param  property_names: names of the datasets to read, e.g. "InputHalos/HaloCentre"
    :type   property_names: list of str
    :param  chunk_size:     number of rows read at a time
    :type   chunk_size:     int
    """
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, not {chunk_size}")
    with h5py.File(filename, "r") as infile:
        registry, datasets, units, nr_rows = _soap_datasets(infile, property_names)
        for start in range(0, nr_rows, chunk_size):
            stop = min(start + chunk_size, nr_rows)
            yield start, {name: unyt.unyt_array(dataset[start:stop], units=units[name], registry=registry)
                          for name, dataset in datasets.items()}


def _read_rows(dataset, rows, batch_size):
    """
    Read only the given rows of a dataset, in batches.

    Returns the values of the rows.

    :param  dataset:    dataset
    :type   dataset:    h5py.Dataset
    :param  rows:       sorted, unique row indices
    :type   rows:       np.ndarray
    :param  batch_size: number of rows read at a time
    :type   batch_size: int
    """
    if len(rows) == 0:
        return np.zeros((0,) + dataset.shape[1:], dtype=dataset.dtype)
    return np.concatenate([dataset[rows[i:i + batch_size]] for i in range(0, len(rows), batch_size)])


def read_soap_selected_rows(filename, property_names, rows, batch_size=100000):
    """
    Read only the given rows of properties of a SOAP catalogue, in batches, with the same units as
    lightcone_io.halo_catalogue.SOAPCatalogue.

    Returns {name: unyt.unyt_array of the rows}, in the order of rows.

    :param  filename:       path of the SOAP catalogue
    :type   filename:       str
    :param  property_names: names of the datasets to read, e.g. "InputHalos/HaloCentre"
    :type   property_names: list of str
    :param  rows:           sorted, unique row indices
    :type   rows:           np.ndarray
    :param  batch_size:     number of rows read at a time
    :type   batch_size:     int
    """
    rows = np.asarray(rows, dtype=np.int64)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be sorted and unique")
    with h5py.File(filename, "r") as infile:
        registry, datasets, units, _ = _soap_datasets(infile, property_names)
        return {name: unyt.unyt_array(_read_rows(dataset, rows, batch_size), units=units[name], registry=registry)
                for name, dataset in datasets.items()}


def select_soap_rows(filename, select_name, select, property_names, chunk_size=2**22):
    """
    Index the SOAP catalogue, read the data from 'select_name' in chunks and then read the other properties only for the rows kept.
    The whole catalogue is never read or held in memory.

    Returns {name: unyt.unyt_array of the kept rows} for property_names (and select_name), with "row" holding the
    indices of the kept rows.

    :param  filename:       path of the SOAP catalogue
    :type   filename:       str
    :param  select_name:    property to select the rows with, e.g. "SO/200_crit/TotalMass"
    :type   select_name:    str
    :param  select:         function of a chunk of select_name (unyt.unyt_array, comoving, see comoving) returning
                                True for the rows to keep, e.g. lambda m: m.to_value("Msun") > 1e14
    :type   select:         function
    :param  property_names: names of the other datasets to read for the kept rows
    :type   property_names: list of str
    :param  chunk_size:     number of rows of select_name read at a time
    :type   chunk_size:     int
    """
    rows = [start + np.flatnonzero(select(comoving(chunk[select_name])))
            for start, chunk in read_soap_in_chunks(filename, [select_name], chunk_size)]
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    found = read_soap_selected_rows(filename, list(dict.fromkeys([select_name, *property_names])), rows)
    found["row"] = rows
    return found
