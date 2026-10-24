#!/bin/env python
import os
import sys

import numpy as np
import pytest
import unyt

h5py = pytest.importorskip("h5py")
soap_reader = pytest.importorskip("swiftlet.soap_reader")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_snapshot_lightcone_placement as placement  # noqa: E402

"""
Tests swiftlet.soap_reader. 
Use a small fake SOAP catalogue written to a temporary directory. 
Every reader gives the same rows and values, matching the units from lightcone_io SOAPCatalogue, also drop the comoving expansion factors.
"""

SNAP_NR = 75
NHALOS = 5000
MSUN_G = unyt.unyt_quantity(1., "Msun").to_value("g")  # unyt's solar mass, so masses read in Msun are as written
# the fake catalogues' Mpc (placement.MPC_CM) differs from unyt's by 3e-10, so values converted to Mpc agree to this
RTOL = 1e-9


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    """
    Build a fake SOAP catalogue of haloes, with centres, indices, M200c [1e10 Msun] and comoving R200c. 
    Include the attrs for the conversion factors to physical CGS units.
    
    Returns (filename, {name: values written, in Msun and comoving Mpc}).
    
    """

    rng = np.random.default_rng(1) # set random seed to use consistent sampling
    filename = tmp_path_factory.mktemp("soap") / f"halo_properties_{SNAP_NR:04d}.hdf5" # get temp filename
    redshift = placement.OUTPUT_REDSHIFTS[SNAP_NR] 
    
    # set catalogue values
    # randomly sample coords, M200c and R200c values
    values = {"InputHalos/HaloCentre": rng.uniform(0, placement.BOXSIZE, (NHALOS, 3)),
              "InputHalos/HaloCatalogueIndex": np.arange(NHALOS),
              "SO/200_crit/TotalMass": 10**rng.uniform(12, 15, NHALOS),
              "SO/200_crit/SORadius": rng.uniform(0.1, 2., NHALOS)}
    # write fake SOAP catalogue
    placement.write_soap_catalogue(filename, SNAP_NR, values["InputHalos/HaloCatalogueIndex"],
                                   values["InputHalos/HaloCentre"])
    
    with h5py.File(filename, "a") as f:
        mass = f.create_dataset("SO/200_crit/TotalMass", data=values["SO/200_crit/TotalMass"] / 1e10)
        placement.set_unit_attrs(mass, 0, 0, conversion=1e10 * MSUN_G)
        mass.attrs["U_M exponent"] = [1.]
        placement.set_unit_attrs(f.create_dataset("SO/200_crit/SORadius", data=values["SO/200_crit/SORadius"]), 1, 1,
                                 conversion=placement.MPC_CM / (1. + redshift))
    return str(filename), values


def test_comoving_drops_the_expansion_factor(catalogue):
    """
    Test that we remove the scalefactor symbol without affecting the values. 
    
    Comoving lengths read in as a*snap_length become comoving Mpc, without changing their values; properties without a
    in their units are unchanged.

    """
    filename, values = catalogue
    data = soap_reader.read_soap_rows(filename, ["InputHalos/HaloCentre", "SO/200_crit/TotalMass"], 0, 10)
    assert str(data["InputHalos/HaloCentre"].units) == "a*snap_length"
    centre = soap_reader.comoving(data["InputHalos/HaloCentre"])
    assert str(centre.units) == "snap_length"
    np.testing.assert_allclose(centre.to_value("Mpc"), values["InputHalos/HaloCentre"][:10], rtol=RTOL)
    assert soap_reader.comoving(data["SO/200_crit/TotalMass"]) is data["SO/200_crit/TotalMass"]


def test_readers_give_the_same_rows(catalogue):
    """
    A range of rows, the catalogue in chunks and the selected rows are the same values, with the same units.
    """
    filename, _ = catalogue
    names = ["InputHalos/HaloCentre", "SO/200_crit/SORadius"]
    whole = soap_reader.read_soap_rows(filename, names)
    assert soap_reader.soap_catalogue_length(filename) == NHALOS == len(whole[names[0]])
    chunks = list(soap_reader.read_soap_in_chunks(filename, names, 777))
    rows = np.sort(np.random.default_rng(2).choice(NHALOS, 300, replace=False))
    selected = soap_reader.read_soap_selected_rows(filename, names, rows, batch_size=64)
    part = soap_reader.read_soap_rows(filename, names, 100, 250)
    for name in names:
        joined = unyt.uconcatenate([chunk[name] for _, chunk in chunks])
        assert joined.units == whole[name].units == selected[name].units
        np.testing.assert_array_equal(joined.value, whole[name].value)
        np.testing.assert_array_equal(selected[name].value, whole[name].value[rows])
        np.testing.assert_array_equal(part[name].value, whole[name].value[100:250])
    with pytest.raises(ValueError, match="sorted and unique"):
        soap_reader.read_soap_selected_rows(filename, names, rows[::-1])


@pytest.mark.parametrize("chunk_size", [1000, NHALOS - 1, 10 * NHALOS])
def test_select_soap_rows(catalogue, chunk_size):
    """
    select_soap_rows keeps the rows whose M200c passes the test, whatever the chunk size, and gives their comoving
    values in the units asked for.
    """
    filename, values = catalogue
    units = {"SO/200_crit/TotalMass": "Msun", "InputHalos/HaloCentre": "Mpc", "SO/200_crit/SORadius": "Mpc",
             "InputHalos/HaloCatalogueIndex": "dimensionless"}
    found = soap_reader.select_soap_rows(filename, "SO/200_crit/TotalMass",
                                         lambda mass: (mass.to_value("Msun") >= 1e14) & (mass.to_value("Msun") < 3e14),
                                         list(units), chunk_size=chunk_size, units=units)
    expected = np.flatnonzero((values["SO/200_crit/TotalMass"] >= 1e14) & (values["SO/200_crit/TotalMass"] < 3e14))
    assert len(expected) > 0
    np.testing.assert_array_equal(found["row"], expected)
    for name in units:
        np.testing.assert_allclose(found[name], values[name][expected], rtol=RTOL)