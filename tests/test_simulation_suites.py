#!/bin/env python
import numpy as np
import pytest

from swiftlet import snapshot_lightcone
from swiftlet.simulation_suites import (COLIBRE, FLAMINGO, Simulation, get_suite, output_redshifts, read_output_redshifts,
                                  suite_for_label)

"""
Tests of Simulations and SimulationSuite classes.
Test each simulation's paths and snapshot redshifts come from its suite and the 
that lightcone classes take a Simulation or the box size and resolution label and name.
"""

FLAMINGO_L1_REDSHIFTS = read_output_redshifts("flamingo_L1.txt")
FLAMINGO_L2P8_REDSHIFTS = read_output_redshifts("flamingo_L2p8.txt")
COLIBRE_REDSHIFTS = read_output_redshifts("colibre.txt")


@pytest.fixture(autouse=True)
def no_suite_directories(monkeypatch):
    # each suite's own directory, unless a test sets the environment variable
    monkeypatch.delenv("SWIFTLET_FLAMINGO_DIR", raising=False)
    monkeypatch.delenv("SWIFTLET_COLIBRE_DIR", raising=False)


def test_flamingo_from_label():
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
    assert sim.suite is FLAMINGO
    assert sim.directory == "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL"
    assert sim.snapshot_file(77, 3) == ("/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.3.hdf5")
    assert sim.z0_snapshot == 77
    assert sim.snapshot_redshift(73) == 0.2
    assert sim.snapshot_redshift_range(73) == pytest.approx((0.175, 0.225))


def test_colibre_from_label():
    sim = Simulation.from_label("L0200N1504", "Thermal")
    assert sim.suite is COLIBRE
    assert sim.directory == "/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal"
    assert sim.snapshot_file(127) == ("/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/snapshots/colibre_0127/colibre_0127.0.hdf5")
    assert sim.z0_snapshot == 127
    assert np.array_equal(sim.snapshot_redshifts()[1], COLIBRE_REDSHIFTS)
    assert np.array_equal(sim.snapshots_in_range((0.1, 0.2)), np.arange(114, 120))


def test_suite_given():
    # a label the patterns don't know, with the suite given
    sim = Simulation.from_label("L0800N6000", "NEW_RUN", simulation_suite_name="colibre")
    assert sim.suite is COLIBRE
    assert Simulation.from_label("L0800N6000", "NEW_RUN", simulation_suite_name=COLIBRE) == sim
    with pytest.raises(ValueError, match="is a FLAMINGO simulation"):
        Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", simulation_suite_name="colibre")
    with pytest.raises(ValueError, match="must be one of"):
        get_suite("eagle")


def test_directories(monkeypatch, tmp_path):
    # an explicit directory format first, then the suite's environment variable, then COSMA
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format=str(tmp_path) + "/{sim_name}")
    assert sim.directory == str(tmp_path / "HYDRO_FIDUCIAL")
    monkeypatch.setenv("SWIFTLET_COLIBRE_DIR", str(tmp_path))
    assert Simulation.from_label("L0200N1504", "Thermal").directory == str(tmp_path / "L0200N1504" / "Thermal")
    # the other suite is not moved
    assert Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL").directory.startswith("/cosma8/data/dp004/flamingo")


@pytest.mark.parametrize("path, suite, box_res, name", [
    ("/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL", FLAMINGO, "L1000N1800", "HYDRO_FIDUCIAL"),
    ("/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/", COLIBRE, "L0200N1504", "Thermal"),
    # a label the patterns don't know, the suite from "colibre" in the path
    ("/data/colibre/L0800N6000/NEW_RUN", COLIBRE, "L0800N6000", "NEW_RUN"),
])
def test_from_path(path, suite, box_res, name):
    sim = Simulation.from_path(path)
    assert (sim.suite, sim.box_res, sim.name) == (suite, box_res, name)
    assert sim.directory == path.rstrip("/")


FLAMINGO_LABELS = ["L1000N0900", "L1000N1800", "L1000N3600", "L2800N5040"]
COLIBRE_LABELS = ["L0025N0188", "L0200N1504", "L0400N3008", "L200m6", "L025_m5"]


@pytest.mark.parametrize("label, suite", [(label, FLAMINGO) for label in FLAMINGO_LABELS]
                         + [(label, COLIBRE) for label in COLIBRE_LABELS])
def test_suite_for_label(label, suite):
    assert suite_for_label(label) is suite
    assert suite_for_label(label, suite.name.upper()) is suite
    assert suite_for_label(label, suite) is suite


@pytest.mark.parametrize("kwargs, match", [
    ({"boxsize_resolution": "L1000N1800", "simulation_suite_name": "colibre"}, "is a FLAMINGO simulation"),
    ({"boxsize_resolution": "L0200N1504", "simulation_suite_name": "flamingo"}, "is a COLIBRE simulation"),
    ({"simulation_suite_name": "eagle"}, "must be one of"),
    ({"boxsize_resolution": "L0100N9999X"}, "can't tell which simulation suite"),
    ({}, "can't tell which simulation suite"),
])
def test_suite_for_label_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        suite_for_label(**kwargs)


def test_output_redshifts():
    assert np.array_equal(output_redshifts("L1000N1800"), FLAMINGO_L1_REDSHIFTS)
    assert np.array_equal(output_redshifts("L2800N5040"), FLAMINGO_L2P8_REDSHIFTS)
    assert np.array_equal(output_redshifts("L0200N1504"), COLIBRE_REDSHIFTS)
    # a label the patterns don't know, with the suite given: COLIBRE has one table for every label, FLAMINGO doesn't
    assert np.array_equal(output_redshifts("L0800N6000", "colibre"), COLIBRE_REDSHIFTS)
    with pytest.raises(ValueError, match="not recognised"):
        output_redshifts("L5600N5040", "flamingo")
    # a label of one suite asked for as the other
    with pytest.raises(ValueError, match="is a COLIBRE simulation"):
        output_redshifts("L0800N6000", "flamingo")


@pytest.mark.parametrize("label, redshifts", [
    ("L1000N1800", FLAMINGO_L1_REDSHIFTS), ("L2800N5040", FLAMINGO_L2P8_REDSHIFTS), ("L0200N1504", COLIBRE_REDSHIFTS),
])
def test_snapshot_redshifts_of_a_simulation(label, redshifts):
    sim = Simulation.from_label(label, "TEST")
    numbers, table = sim.snapshot_redshifts()
    assert np.array_equal(numbers, np.arange(len(redshifts))) and np.array_equal(table, redshifts)
    # snapshot number <-> redshift, for every snapshot
    for snap_nr, z in enumerate(redshifts):
        assert sim.snapshot_redshift(snap_nr) == z
        assert sim.snapshot_number(z) == snap_nr
    # the redshift ranges tile the redshifts from the first output to z = 0
    ranges = np.array([sim.snapshot_redshift_range(snap_nr) for snap_nr in range(len(redshifts))])
    assert ranges[0, 1] == redshifts[0] and ranges[-1, 0] == 0.
    assert np.array_equal(ranges[:-1, 0], ranges[1:, 1])


def test_colibre_and_flamingo_differ():
    flamingo, flamingo_l2p8 = Simulation.from_label("L1000N1800", "X"), Simulation.from_label("L2800N5040", "X")
    colibre = Simulation.from_label("L0200N1504", "X")
    # z = 0 is snapshot 127 in COLIBRE, 77 in FLAMINGO L1 and 78 in FLAMINGO L2p8
    assert (colibre.z0_snapshot, flamingo.z0_snapshot, flamingo_l2p8.z0_snapshot) == (127, 77, 78)
    # COLIBRE's outputs are closer together at low redshift
    assert colibre.snapshot_redshift_range(126) == pytest.approx((0.005, 0.015))
    assert flamingo.snapshot_redshift_range(76) == pytest.approx((0.025, 0.075))
    assert np.array_equal(colibre.snapshots_in_range((0.1, 0.2)), np.arange(114, 120))
    assert np.array_equal(flamingo.snapshots_in_range((0.1, 0.2)), np.arange(73, 76))
    # a snapshot number past the end of FLAMINGO's tables is a COLIBRE snapshot
    assert colibre.snapshot_redshift(100) == 0.6
    with pytest.raises(IndexError):
        flamingo.snapshot_redshift(100)


@pytest.mark.parametrize("filename, nr_snapshots, first, last", [
    ("flamingo_L1.txt", 78, 15., 0.), ("flamingo_L2p8.txt", 79, 15., 0.), ("colibre.txt", 128, 30., 0.),
])
def test_output_redshift_files(filename, nr_snapshots, first, last):
    redshifts = read_output_redshifts(filename)
    assert len(redshifts) == nr_snapshots and (redshifts[0], redshifts[-1]) == (first, last)
    # each call gives its own copy, so changing it doesn't change the table
    redshifts[0] = -1.
    assert read_output_redshifts(filename)[0] == first


def test_output_redshifts_from_a_file_of_your_own(tmp_path):
    good = tmp_path / "my_run.txt"
    good.write_text("# my run\n# snapshot_number, redshift\n0, 2.0\n1, 1.0\n2, 0.0\n")
    assert read_output_redshifts(good).tolist() == [2., 1., 0.]
    # a suite of your own, with its output redshifts from the file
    from dataclasses import replace
    suite = replace(COLIBRE, output_redshift_tables={"*": str(good)})
    sim = Simulation(suite, "L0050N0752", "MY_RUN", str(tmp_path))
    assert sim.z0_snapshot == 2 and sim.snapshot_redshift_range(1) == (0.5, 1.5)

    one_column = tmp_path / "one_column.txt"
    one_column.write_text("2.0\n1.0\n0.0\n")
    with pytest.raises(ValueError, match="expected 2 comma separated columns"):
        read_output_redshifts(one_column)
    out_of_order = tmp_path / "out_of_order.txt"
    out_of_order.write_text("0, 2.0\n2, 1.0\n1, 0.0\n")
    with pytest.raises(ValueError, match="snapshot numbers must be 0, 1, 2"):
        read_output_redshifts(out_of_order)


class _CosmologyFrom:
    """
    Stand in for swiftlet.cosmology.SnapshotCosmology, recording the file the cosmology was read from.
    """
    filenames = []

    def __init__(self, filename):
        from astropy import cosmology
        self.filenames.append(filename)
        self.COSMO = cosmology.FlatLambdaCDM(H0=68, Om0=0.3)

    @classmethod
    def from_simulation(cls, simulation):
        return cls(simulation.snapshot_file(simulation.z0_snapshot))


@pytest.fixture
def cosmology_from(monkeypatch):
    monkeypatch.setattr(snapshot_lightcone, "SnapshotCosmology", _CosmologyFrom)
    _CosmologyFrom.filenames.clear()
    return _CosmologyFrom.filenames


def test_lightcone_from_a_simulation(cosmology_from):
    sim = Simulation.from_label("L0200N1504", "Thermal")
    
    for lightcone in (
        snapshot_lightcone.SnapshotAllSky(simulation_object=sim, verbose=0),
        snapshot_lightcone.SnapshotBeam(simulation_object=sim, beam_vector=[0., 0., 1.], verbose=0)
    ):
        assert lightcone.simulation is sim
        assert (lightcone.box_res, lightcone.sim_name, lightcone.simulation_suite_name) == ("L0200N1504", "Thermal", "colibre")
        assert lightcone.lightcone_metadata()["simulation_suite_name"] == "colibre"
    # the cosmology is read from COLIBRE's z = 0 snapshot
    assert all(filename == sim.snapshot_file(127) for filename in cosmology_from)


def test_lightcone_from_a_label(cosmology_from, tmp_path):
    # the call before Simulation still works, with the same paths as Simulation.from_label
    lightcone = snapshot_lightcone.SnapshotAllSky(boxsize_resolution="L1000N1800", simulation_name="HYDRO_FIDUCIAL", verbose=0)
    assert lightcone.simulation == Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
    assert cosmology_from[-1] == lightcone.simulation.snapshot_file(77)
    beam = snapshot_lightcone.SnapshotBeam(
                                    boxsize_resolution="L0200N1504", 
                                    simulation_name="Thermal",
                                    beam_vector=[1., 0., 0.],
                                    verbose=0,
                                    simulation_base_dir_format=str(tmp_path) + "/{box_res}/{sim_name}"
                                    )
    assert beam.simulation_dir == str(tmp_path / "L0200N1504" / "Thermal")
    assert beam.simulation_suite_name == "colibre"


def test_lightcone_argument_errors(cosmology_from):
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")

    with pytest.raises(TypeError, match="needs a Simulation, or boxsize_resolution and simulation_name"):
        snapshot_lightcone.SnapshotAllSky(boxsize_resolution="L1000N1800", verbose=0)
    with pytest.raises(TypeError, match="needs a beam_vector"):
        snapshot_lightcone.SnapshotBeam(simulation_object=sim, verbose=0)
    with pytest.raises(TypeError, match=f"simulation_object must be a swiftlet.Simulation"):
        snapshot_lightcone.SnapshotAllSky(simulation_object="L1000N1800", verbose=0)

