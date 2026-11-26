#!/bin/env python
"""
Tests of Simulations and SimulationSuite classes.

Test each simulation's paths and snapshot redshifts come from its suite and the
that lightcone classes take a Simulation or the box size and resolution label and name.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pytest
import dataclasses
from swiftlet import simulation_suites, snapshot_lightcone
from swiftlet.simulation_suites import (
    COLIBRE,
    FLAMINGO,
    Simulation,
    SimulationSuite,
    add_suite,
    get_suite,
    load_suites,
    read_suite_file,
    suite_directories,
    write_suite_file,
    output_redshifts,
    read_output_list,
    suite_for_label,
    write_output_list,
)

# the output lists of FLAMINGO's 1 Gpc and 2.8 Gpc boxes and of COLIBRE, as in each simulation's directory
OUTPUT_LISTS = Path(__file__).resolve().parent / "data" / "output_lists"
OUTPUT_LIST_OF_LABEL = {
    "L1000N1800": "flamingo_L1.txt",
    "L2800N5040": "flamingo_L2p8.txt",
    "L0200N1504": "colibre.txt",
}
FLAMINGO_L1_REDSHIFTS = read_output_list(OUTPUT_LISTS / "flamingo_L1.txt")
FLAMINGO_L2P8_REDSHIFTS = read_output_list(OUTPUT_LISTS / "flamingo_L2p8.txt")
COLIBRE_REDSHIFTS = read_output_list(OUTPUT_LISTS / "colibre.txt")


@pytest.fixture(autouse=True)
def no_suite_directories(monkeypatch):
    # each suite's own directory, unless a test sets the environment variable
    monkeypatch.delenv("SWIFTLET_FLAMINGO_DIR", raising=False)
    monkeypatch.delenv("SWIFTLET_COLIBRE_DIR", raising=False)

    # only the installed suite files, unless a test sets the directories of its own
    monkeypatch.delenv(simulation_suites.SUITES_DIR_ENV, raising=False)


@pytest.fixture
def simulations_dir(tmp_path, monkeypatch):
    """
    Test the simulations are in a temporary directory and each have an output list that is with the environmental
    variables for this test.

    Returns a function making the Simulation of a label and name, with the output list of its label.
    """
    monkeypatch.setenv("SWIFTLET_FLAMINGO_DIR", str(tmp_path))
    monkeypatch.setenv("SWIFTLET_COLIBRE_DIR", str(tmp_path))

    def simulation(label, name):
        sim = Simulation.from_label(label, name)
        Path(sim.directory).mkdir(parents=True, exist_ok=True)
        shutil.copy(OUTPUT_LISTS / OUTPUT_LIST_OF_LABEL[label], sim.output_list)
        return sim

    return simulation


def test_flamingo_from_label(tmp_path):
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")
    assert sim.suite is FLAMINGO
    assert sim.directory == "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL"
    assert sim.snapshot_file(77, 3) == (
        "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.3.hdf5"
    )
    assert (
        sim.soap_file(77)
        == "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5"
    )
    assert sim.soap_dir == "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT"
    assert sim.snapshot_format_whole.format(snap_nr=77) == (
        "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.hdf5"
    )

    assert sim.output_list == "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/output_list.txt"

    # output redshifts, from the output list in its directory
    sim = dataclasses.replace(sim, directory=str(tmp_path))
    shutil.copy(OUTPUT_LISTS / "flamingo_L1.txt", sim.output_list)

    assert sim.z0_snapshot == 77
    assert sim.snapshot_redshift(73) == 0.2
    assert sim.snapshot_redshift_range(73) == pytest.approx((0.175, 0.225))


def test_colibre_from_label(tmp_path):
    sim = Simulation.from_label("L0200N1504", "Thermal")
    assert sim.suite is COLIBRE
    assert sim.directory == "/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal"
    assert sim.snapshot_file(127) == (
        "/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/snapshots/colibre_0127/colibre_0127.0.hdf5"
    )
    assert sim.soap_file(127) == "/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/SOAP-HBT/halo_properties_0127.hdf5"
    assert sim.snapshot_format_whole == (
        "/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/snapshots/colibre_{snap_nr:04d}/colibre_{snap_nr:04d}.hdf5"
    )
    assert sim.output_list == "/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/output_list.txt"
    # its output redshifts, from the output list in its directory (here a copy, in a directory of the test's)
    sim = dataclasses.replace(sim, directory=str(tmp_path))
    shutil.copy(OUTPUT_LISTS / "colibre.txt", sim.output_list)
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


@pytest.mark.parametrize(
    "path, suite, box_res, name",
    [
        ("/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL", FLAMINGO, "L1000N1800", "HYDRO_FIDUCIAL"),
        ("/cosma8/data/dp004/colibre/Runs/L0200N1504/Thermal/", COLIBRE, "L0200N1504", "Thermal"),
        # a label the patterns don't know, the suite from "colibre" in the path
        ("/data/colibre/L0800N6000/NEW_RUN", COLIBRE, "L0800N6000", "NEW_RUN"),
    ],
)
def test_from_path(path, suite, box_res, name):
    sim = Simulation.from_path(path)
    assert (sim.suite, sim.box_res, sim.name) == (suite, box_res, name)
    assert sim.directory == path.rstrip("/")


FLAMINGO_LABELS = ["L1000N0900", "L1000N1800", "L1000N3600", "L2800N5040"]
COLIBRE_LABELS = ["L0025N0188", "L0200N1504", "L0400N3008", "L200m6", "L025_m5"]


@pytest.mark.parametrize(
    "label, suite", [(label, FLAMINGO) for label in FLAMINGO_LABELS] + [(label, COLIBRE) for label in COLIBRE_LABELS]
)
def test_suite_for_label(label, suite):
    assert suite_for_label(label) is suite
    assert suite_for_label(label, suite.name.upper()) is suite
    assert suite_for_label(label, suite) is suite


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"boxsize_resolution": "L1000N1800", "simulation_suite_name": "colibre"}, "is a FLAMINGO simulation"),
        ({"boxsize_resolution": "L0200N1504", "simulation_suite_name": "flamingo"}, "is a COLIBRE simulation"),
        ({"simulation_suite_name": "eagle"}, "must be one of"),
        ({"boxsize_resolution": "L0100N9999X"}, "can't tell which simulation suite"),
        ({}, "can't tell which simulation suite"),
    ],
)
def test_suite_for_label_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        suite_for_label(**kwargs)


def test_output_redshifts(simulations_dir):
    """
    check that output_redshifts takes a Simulation, read from its output list, or the path of an output list and that a
    box size and resolution label alone is refused.
    """
    sim = simulations_dir("L1000N1800", "HYDRO_FIDUCIAL")
    assert sim.output_list == str(Path(sim.directory) / "output_list.txt")
    assert np.array_equal(output_redshifts(sim), FLAMINGO_L1_REDSHIFTS)
    assert np.array_equal(output_redshifts(sim.output_list), FLAMINGO_L1_REDSHIFTS)
    assert np.array_equal(output_redshifts(Path(sim.output_list)), FLAMINGO_L1_REDSHIFTS)
    with pytest.raises(ValueError, match="takes a Simulation or the path of an output list, not the label"):
        output_redshifts("L1000N1800")


def test_no_output_list(tmp_path):
    """
    Check that a simulation without an output list in its directory can't give its output redshifts and an error gives
    the path.
    """
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format=str(tmp_path / "{box_res}/{sim_name}"))
    with pytest.raises(FileNotFoundError, match="no output list at .*L1000N1800/HYDRO_FIDUCIAL/output_list.txt"):
        sim.output_redshifts
    with pytest.raises(FileNotFoundError, match="no output list"):
        output_redshifts(tmp_path / "output_list.txt")


@pytest.mark.parametrize(
    "label, redshifts",
    [
        ("L1000N1800", FLAMINGO_L1_REDSHIFTS),
        ("L2800N5040", FLAMINGO_L2P8_REDSHIFTS),
        ("L0200N1504", COLIBRE_REDSHIFTS),
    ],
)
def test_snapshot_redshifts_of_a_simulation(label, redshifts, simulations_dir):
    sim = simulations_dir(label, "TEST")
    numbers, table = sim.snapshot_redshifts()
    assert np.array_equal(numbers, np.arange(len(redshifts))) and np.array_equal(table, redshifts)
    # snapshot number <-> redshift, for every snapshot
    for snap_nr, z in enumerate(redshifts):
        assert sim.snapshot_redshift(snap_nr) == z
        assert sim.snapshot_number(z) == snap_nr
    # the redshift ranges tile the redshifts from the first output to z = 0
    ranges = np.array([sim.snapshot_redshift_range(snap_nr) for snap_nr in range(len(redshifts))])
    assert ranges[0, 1] == redshifts[0] and ranges[-1, 0] == 0.0
    assert np.array_equal(ranges[:-1, 0], ranges[1:, 1])


def test_colibre_and_flamingo_differ(simulations_dir):
    flamingo, flamingo_l2p8 = simulations_dir("L1000N1800", "X"), simulations_dir("L2800N5040", "X")
    colibre = simulations_dir("L0200N1504", "X")

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


@pytest.mark.parametrize(
    "filename, nr_snapshots, first, last",
    [
        ("flamingo_L1.txt", 78, 15.0, 0.0),
        ("flamingo_L2p8.txt", 79, 15.0, 0.0),
        ("colibre.txt", 128, 30.0, 0.0),
    ],
)
def test_output_lists(filename, nr_snapshots, first, last):
    redshifts = read_output_list(OUTPUT_LISTS / filename)
    assert len(redshifts) == nr_snapshots and (redshifts[0], redshifts[-1]) == (first, last)
    # each call gives its own copy, so changing it doesn't change the table
    redshifts[0] = -1.0
    assert read_output_list(OUTPUT_LISTS / filename)[0] == first


def test_output_list_formats(tmp_path):
    """
    Test the output lists of FLAMINGO and COLIBRE, and of scale factors, are read the same way and files that aren't
    output lists fail.
    """
    flamingo = tmp_path / "flamingo.txt"
    flamingo.write_text("# Redshift\n15.0\n10.38\n9.51\n0.0\n")
    assert read_output_list(flamingo).tolist() == [15.0, 10.38, 9.51, 0.0]

    # every row is an output, snapshot or snipshot alike, so snipshots count in the snapshot numbers
    colibre = tmp_path / "colibre.txt"
    colibre.write_text(
        "# Redshift, Select Output\n30.000000, Snapshot\n25.000000, Snipshot\n22.500000, Snipshot\n"
        "20.000000, Snapshot\n0.000000, Snapshot\n"
    )
    assert read_output_list(colibre).tolist() == [30.0, 25.0, 22.5, 20.0, 0.0]

    scale_factors = tmp_path / "scale_factors.txt"
    scale_factors.write_text("# Scale Factor\n0.25\n0.5\n1.0\n")
    np.testing.assert_allclose(read_output_list(scale_factors), [3.0, 1.0, 0.0])

    for name, text, match in (
        ("no_header.txt", "2.0\n1.0\n0.0\n", "starts with a header line"),
        ("times.txt", "# Time\n0.1\n0.2\n", "must be 'Redshift' or 'Scale Factor', not 'Time'"),
        ("increasing.txt", "# Redshift\n0.0\n1.0\n2.0\n", "must decrease with snapshot number"),
        ("one_output.txt", "# Redshift\n0.0\n", "at least 2 snapshots"),
        ("not_numbers.txt", "# Redshift\n2.0\ntwo\n", "could not convert"),
    ):
        (tmp_path / name).write_text(text)
        with pytest.raises(ValueError, match=match):
            read_output_list(tmp_path / name)


def test_write_output_list(tmp_path):
    """Test that the output list written reads back exactly as written, with or without a Select Output column."""
    redshifts = [15.0, 2.5, 0.25, 0.0]

    write_output_list(tmp_path / "plain.txt", redshifts)

    assert (tmp_path / "plain.txt").read_text().splitlines()[0] == "# Redshift"
    assert read_output_list(tmp_path / "plain.txt").tolist() == redshifts

    write_output_list(tmp_path / "selected.txt", redshifts, ["Snapshot", "Snipshot", "Snipshot", "Snapshot"])

    assert (tmp_path / "selected.txt").read_text().splitlines()[:2] == [
        "# Redshift, Select Output",
        "15.000000, Snapshot",
    ]

    assert read_output_list(tmp_path / "selected.txt").tolist() == redshifts

    with pytest.raises(ValueError, match="one entry for each redshift"):
        write_output_list(tmp_path / "bad.txt", redshifts, ["Snapshot"])

    # a suite whose simulations keep their output list under another name, relative to their directory
    suite = dataclasses.replace(COLIBRE, output_list="config/outputs.txt")
    sim = Simulation(suite, "L0050N0752", "MY_RUN", str(tmp_path / "MY_RUN"))
    (tmp_path / "MY_RUN" / "config").mkdir(parents=True)

    write_output_list(tmp_path / "MY_RUN" / "config" / "outputs.txt", redshifts)

    assert sim.output_list == str(tmp_path / "MY_RUN" / "config" / "outputs.txt")
    assert sim.output_redshifts.tolist() == redshifts and sim.z0_snapshot == 3
    assert sim.snapshot_redshift_range(2) == (0.125, 1.375)


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


def test_lightcone_from_a_simulation(cosmology_from, simulations_dir):
    sim = simulations_dir("L0200N1504", "Thermal")

    for lightcone in (
        snapshot_lightcone.SnapshotAllSky(simulation=sim, verbose=0),
        snapshot_lightcone.SnapshotBeam(simulation=sim, beam_vector=[0.0, 0.0, 1.0], ang_radius_deg=5.0, verbose=0),
    ):
        assert lightcone.simulation is sim
        assert (lightcone.box_res, lightcone.sim_name, lightcone.simulation_suite_name) == (
            "L0200N1504",
            "Thermal",
            "colibre",
        )

        assert lightcone.lightcone_metadata()["simulation_suite_name"] == "colibre"

    # the cosmology is read from COLIBRE's z = 0 snapshot

    assert all(filename == sim.snapshot_file(127) for filename in cosmology_from)


@pytest.mark.parametrize("lightcone_type", ["SnapshotAllSky", "SnapshotBeam"])
def test_lightcone_takes_only_a_simulation(cosmology_from, simulations_dir, lightcone_type):
    """
    Check that SnapshotLightcone, SnapshotAllSky and SnapshotBeam classes take a Simulation, positionally or as
    simulation and then fail if given a non Simulation object.
    """
    sim = simulations_dir("L1000N1800", "HYDRO_FIDUCIAL")
    lightcone_class = getattr(snapshot_lightcone, lightcone_type)
    if lightcone_type == "SnapshotBeam":
        beam = {"beam_vector": [1.0, 0.0, 0.0], "ang_radius_deg": 10.0}
    else:
        beam = {}

    assert lightcone_class(sim, verbose=0, **beam).simulation is sim
    assert lightcone_class(simulation=sim, verbose=0, **beam).simulation is sim

    with pytest.raises(TypeError, match=rf"{lightcone_type} takes a Simulation, not the label 'L1000N1800'"):
        lightcone_class("L1000N1800", verbose=0, **beam)

    with pytest.raises(TypeError, match=rf"got an unexpected keyword argument"):
        lightcone_class(simulation=sim, simulation_name="HYDRO_FIDUCIAL", verbose=0, **beam)

    with pytest.raises(TypeError, match=rf"{lightcone_type} takes a swiftlet.Simulation, not int"):
        lightcone_class(3, verbose=0, **beam)

    with pytest.raises(TypeError, match=rf"{lightcone_type} takes a swiftlet.Simulation, not dict"):
        lightcone_class(simulation={"test_str_input": np.zeros(10)}, verbose=0, **beam)

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'simulation'"):
        lightcone_class(verbose=0, **beam)


def test_lightcone_argument_errors(cosmology_from, simulations_dir):
    sim = simulations_dir("L1000N1800", "HYDRO_FIDUCIAL")

    with pytest.raises(TypeError, match="needs a beam_vector"):
        snapshot_lightcone.SnapshotBeam(sim, verbose=0)


def test_soap_format(tmp_path, monkeypatch):
    # the SOAP catalogues follow the simulation's directory, wherever it is
    monkeypatch.setenv("SWIFTLET_FLAMINGO_DIR", str(tmp_path))
    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")

    assert sim.soap_format == f"{tmp_path}/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_{{snap_nr:04d}}.hdf5"
    assert sim.soap_file(np.int64(5)) == sim.soap_format.format(snap_nr=5)
    assert Simulation.from_path(sim.directory).soap_file(77) == sim.soap_file(77)

    # a suite whose SOAP catalogues are elsewhere in the simulation's directory
    suite = dataclasses.replace(FLAMINGO, soap_format="SOAP/halo_properties_{snap_nr:04d}.hdf5")
    other = Simulation(suite, "L1000N1800", "HYDRO_FIDUCIAL", str(tmp_path))

    assert other.soap_file(3) == f"{tmp_path}/SOAP/halo_properties_0003.hdf5"
    assert other.soap_dir == f"{tmp_path}/SOAP"


# for variable locations of sims catalogues and output lists


def test_paths_given_to_a_simulation(tmp_path):
    """
    Test that snapshot, SOAP catalogue and output list paths given to from_label, from_path or with_paths replace the
    suite's, relative to the simulation's directory or absolute, and the suite is still taken for its labels.
    """
    snapshot_format = "outputs/snap_{snap_nr:03d}/snap_{snap_nr:03d}.{file_nr}.hdf5"
    soap_format = str(tmp_path / "haloes" / "soap_{snap_nr:03d}.hdf5")
    sim = Simulation.from_label(
        "L0025N0376",
        "MY_RUN",
        base_dir_format=str(tmp_path) + "/{sim_name}",
        snapshot_format=snapshot_format,
        soap_format=soap_format,
        output_list="config/outputs.txt",
    )
    assert sim.suite.name == "colibre" and sim.directory == str(tmp_path / "MY_RUN")
    assert sim.snapshot_file(5, 2) == str(tmp_path / "MY_RUN" / "outputs" / "snap_005" / "snap_005.2.hdf5")
    assert sim.snapshot_format_whole == str(
        tmp_path / "MY_RUN" / "outputs" / "snap_{snap_nr:03d}" / "snap_{snap_nr:03d}.hdf5"
    )
    assert sim.snapshot_dir == str(tmp_path / "MY_RUN" / "outputs")
    assert sim.soap_file(5) == str(tmp_path / "haloes" / "soap_005.hdf5")
    assert sim.output_list == str(tmp_path / "MY_RUN" / "config" / "outputs.txt")

    # the output redshifts come from that output list
    write_output_list(sim.output_list, [3.0, 1.0, 0.0])
    assert sim.output_redshifts.tolist() == [3.0, 1.0, 0.0] and sim.z0_snapshot == 2

    # the built-in suite is unchanged, and the paths not given are the suite's
    assert COLIBRE.output_list == "output_list.txt"

    plain = Simulation.from_label("L0025N0376", "MY_RUN", soap_format="SOAP/halo_properties_{snap_nr:04d}.hdf5")
    assert plain.suite.snapshot_format == COLIBRE.snapshot_format and plain.suite.output_list == COLIBRE.output_list

    # from a directory, and a copy of a simulation with other paths
    found = Simulation.from_path(tmp_path / "L1000N1800" / "HYDRO_FIDUCIAL", soap_format="SOAP/{snap_nr}.hdf5")
    assert found.suite.name == "flamingo" and found.soap_file(7) == str(
        tmp_path / "L1000N1800" / "HYDRO_FIDUCIAL" / "SOAP" / "7.hdf5"
    )

    moved = found.with_paths(directory=tmp_path / "copy", output_list="outputs.txt")
    assert (
        moved.directory == str(tmp_path / "copy") and moved.box_res == "L1000N1800" and moved.name == "HYDRO_FIDUCIAL"
    )
    assert moved.output_list == str(tmp_path / "copy" / "outputs.txt")
    assert moved.soap_file(7) == str(tmp_path / "copy" / "SOAP" / "7.hdf5")
    assert found.with_paths() == found

    # a suite with other paths, as with dataclasses.replace
    assert FLAMINGO.with_paths(soap_format="SOAP/{snap_nr}.hdf5") == found.suite

    # still refused for another suite's labels
    with pytest.raises(ValueError, match="is a FLAMINGO simulation"):
        Simulation.from_label(
            "L1000N1800", "MY_RUN", simulation_suite_name="colibre", soap_format="SOAP/{snap_nr}.hdf5"
        )


@pytest.mark.parametrize(
    "paths, match",
    [
        ({"soap_format": "SOAP/halo_properties.hdf5"}, r"soap_format=.* needs \{snap_nr\}"),
        ({"snapshot_format": "snaps/run.{file_nr}.hdf5"}, r"snapshot_format=.* needs \{snap_nr\}"),
        ({"snapshot_format": "snaps/run_{snap_nr}.{fil_nr}.hdf5"}, r"may not have \{fil_nr\}"),
        ({"soap_format": "SOAP/halo_{snap_nr:04d.hdf5"}, "is not a valid format"),
    ],
)
def test_path_formats_checked(paths, match):
    """Test that a path format without the fields it needs, or with others, is refused when the simulation is made."""
    with pytest.raises(ValueError, match=match):
        Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", **paths)

    with pytest.raises(ValueError, match=match):
        dataclasses.replace(FLAMINGO, **paths)


def test_directory_format_checked():
    with pytest.raises(ValueError, match=r"base_dir_format=.* may not have \{name\}"):
        Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format="/data/{name}")

    # a directory with no fields is the same for every simulation
    assert Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format="/data/run").directory == "/data/run"

    with pytest.raises(TypeError, match="output_list must be a path"):
        dataclasses.replace(FLAMINGO, output_list=None)


@pytest.mark.parametrize(
    "snapshot_format, snapshot_dir",
    [
        ("snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5", "/sim/snapshots"),
        ("snap_{snap_nr:03d}.hdf5", "/sim"),
        ("data/out/run_{snap_nr}/part.{file_nr}.hdf5", "/sim/data/out"),
        ("/scratch/me/snaps/run_{snap_nr:03d}.hdf5", "/scratch/me/snaps"),
    ],
)
def test_snapshot_dir(snapshot_format, snapshot_dir):
    """The snapshot directory is the directory of the snapshot format up to its first field."""
    sim = Simulation(FLAMINGO, "L1000N1800", "HYDRO_FIDUCIAL", "/sim").with_paths(snapshot_format=snapshot_format)
    assert sim.snapshot_dir == snapshot_dir


# simulation suites read from YAML files

# FLAMINGO's boxes on the hdfstream server, and the layout of its snapshots there
FLAMINGO_ALIASES = {"L1000N0900": "L1_m10", "L1000N1800": "L1_m9", "L1000N3600": "L1_m8", "L2800N5040": "L2p8_m9"}
SERVER_SNAPSHOTS = "snapshots/flamingo_{snap_nr:04d}/swift_snapshot_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"


def test_installed_suite_files():
    """
    Test that the installed suite files give FLAMINGO and COLIBRE as they were defined in the code before.
    """
    assert set(load_suites()) == {"flamingo", "colibre"}

    assert FLAMINGO == SimulationSuite(
        name="flamingo",
        label_pattern=r"L1000N0900|L1000N1800|L1000N3600|L2800N5040",
        base_dir_format="/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}",
        base_dir_env="SWIFTLET_FLAMINGO_DIR",
        snapshot_format="snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5",
        description="FLAMINGO simulations on COSMA",
        box_res_aliases=FLAMINGO_ALIASES,
        alternative_snapshot_formats=[SERVER_SNAPSHOTS],
    )

    assert COLIBRE == SimulationSuite(
        name="colibre",
        label_pattern=r"L0\d{3}N\d{4}|L\d{3}_?m\d",
        base_dir_format="/cosma8/data/dp004/colibre/Runs/{box_res}/{sim_name}",
        base_dir_env="SWIFTLET_COLIBRE_DIR",
        snapshot_format="snapshots/colibre_{snap_nr:04d}/colibre_{snap_nr:04d}.{file_nr}.hdf5",
        description="COLIBRE simulations on COSMA",
    )

    assert (FLAMINGO.output_list, FLAMINGO.soap_format) == (
        "output_list.txt",
        "SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5",
    )

    assert suite_for_label("L1000N1800") is FLAMINGO and suite_for_label("L200_m6") is COLIBRE


# make new custom suite to test for reading and writing of new suites
MYSUITE = SimulationSuite(
    name="mysuite",
    label_pattern=r"L\d{4}N\d{4}_my",
    base_dir_format="/data/my 'suite'/{box_res}/{sim_name}",
    soap_format="SOAP/{snap_nr:03d}.hdf5",
    description="my simulations, with a ' and a \\ in the text",
)


@pytest.mark.parametrize("suite", [FLAMINGO, COLIBRE, MYSUITE], ids=["flamingo", "colibre", "mysuite"])
def test_suite_file_round_trip(tmp_path, suite):
    """
    Test that a suite written with write_suite_file is read back as the same suite, regular expression included.
    """
    path = tmp_path / "suites" / f"{suite.name}.yaml"
    write_suite_file(suite, path)
    assert read_suite_file(path) == suite


def test_suite_files_add_and_replace_suites(tmp_path, monkeypatch):
    """
    Test that the .yaml files in SWIFTLET_SUITES_DIR add new suites and replace the installed ones of the same name and
    that a file's name is its suite's name if it doesn't give one.
    """

    first, second = tmp_path / "first", tmp_path / "second"

    # write new suite .yaml file
    write_suite_file(MYSUITE, first / "mysuite.yaml")

    # update flamingo .yaml file
    write_suite_file(
        FLAMINGO.with_paths(soap_format="SOAP/halo_properties_{snap_nr:04d}.hdf5"), first / "flamingo.yaml"
    )

    second.mkdir()

    (second / "flamingo.yaml").write_text(
        "label_pattern: 'L1000N1800'\nbase_dir_format: '/elsewhere/{box_res}/{sim_name}'\n"
    )

    monkeypatch.setenv(simulation_suites.SUITES_DIR_ENV, f"{first}{os.pathsep}{second}")

    assert suite_directories()[1:] == [str(first), str(second)]

    # load all suites
    suites = load_suites()

    # check all suites are found by name
    assert set(suites) == {"flamingo", "colibre", "mysuite"}

    # check that read suites are the same as the existing ones
    assert suites["mysuite"] == MYSUITE and suites["colibre"] == COLIBRE

    # the last directory's flamingo, with the defaults of the fields it doesn't give the same as the updated suite
    assert suites["flamingo"] == SimulationSuite(
        name="flamingo", label_pattern="L1000N1800", base_dir_format="/elsewhere/{box_res}/{sim_name}"
    )

    assert suites["flamingo"].base_dir_env is None

    # check correct updated path is found
    assert suites["flamingo"].directory_format() == "/elsewhere/{box_res}/{sim_name}"


def test_add_suite(tmp_path, monkeypatch):
    """
    Test that a suite added is found from the labels of its simulations.
    """

    monkeypatch.setattr(simulation_suites, "SUITES", dict(simulation_suites.SUITES))

    with pytest.raises(ValueError, match="can't tell which simulation suite"):
        Simulation.from_label("L0100N0512_my", "Fiducial")

    assert add_suite(MYSUITE) is MYSUITE

    sim = Simulation.from_label("L0100N0512_my", "Fiducial")
    assert sim.suite is MYSUITE and sim.soap_file(5) == "/data/my 'suite'/L0100N0512_my/Fiducial/SOAP/005.hdf5"

    other = dataclasses.replace(MYSUITE, base_dir_format="/other/{box_res}/{sim_name}")

    write_suite_file(other, tmp_path / "mysuite.yaml")

    assert add_suite(tmp_path / "mysuite.yaml") == other
    assert Simulation.from_label("L0100N0512_my", "Fiducial").directory == "/other/L0100N0512_my/Fiducial"


@pytest.mark.parametrize(
    "text, error",
    [
        ("- not\n- a suite\n", "holds the fields of a suite"),
        ("label_pattern: 'L1'\nbase_dir_format: '/d/{box_res}/{sim_name}'\nsoap: 'x'\n", "unknown fields \\['soap'\\]"),
        ("base_dir_format: '/d/{box_res}/{sim_name}'\n", "needs label_pattern"),
        ("label_pattern: 'L(1'\nbase_dir_format: '/d/{box_res}/{sim_name}'\n", "not a regular expression"),
        ("name: 'My Suite'\nlabel_pattern: 'L1'\nbase_dir_format: '/d/{box_res}/{sim_name}'\n", "lower case"),
        (
            "label_pattern: 'L1'\nbase_dir_format: '/d/{box_res}/{sim_name}'\nsoap_format: 'soap.hdf5'\n",
            "needs \\{snap_nr\\}",
        ),
        ("label_pattern: 'L1'\nbase_dir_format: {box_res}/{sim_name}\n", "not a YAML file|must be a path"),
    ],
    ids=["not_a_mapping", "unknown_field", "missing_field", "bad_pattern", "bad_name", "bad_path", "unquoted"],
)
def test_suite_file_errors(tmp_path, text, error):
    """
    Test that a file that isn't a suite raises a ValueError naming the file and what is wrong.
    """
    path = tmp_path / "broken.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=error) as raised:
        read_suite_file(path)

    assert str(path) in str(raised.value)


def test_suite_directory_errors(tmp_path, monkeypatch):
    """
    Test that a directory with two suites of the same name, or a path to a file in SWIFTLET_SUITES_DIR that isn't a
    directory, raises a ValueError.
    """
    # write test files
    write_suite_file(MYSUITE, tmp_path / "a.yaml")
    write_suite_file(MYSUITE, tmp_path / "b.yaml")

    with pytest.raises(ValueError, match="are both the suite 'mysuite'"):
        load_suites([str(tmp_path)])

    monkeypatch.setenv(simulation_suites.SUITES_DIR_ENV, str(tmp_path / "missing"))

    with pytest.raises(ValueError, match="is not a directory"):
        load_suites()


# test with remote reading


def server(tmp_path, box="L1_m9", sim="L1_m9", snapshots=SERVER_SNAPSHOTS):
    """
    Return a FLAMINGO simulation as a FakeRemoteDirectory laid out as on the hdfstream cosma6 server, in tmp_path: its
    directory <box>/<sim> with the output list and file 0 of the redshift 0 snapshot (empty: only found, not read).
    """
    from read_counter import FakeRemoteDirectory

    directory = tmp_path / box / sim
    write_output_list(directory / "output_list.txt", read_output_list(OUTPUT_LISTS / "flamingo_L1.txt"))
    z0 = len(read_output_list(directory / "output_list.txt")) - 1
    if snapshots is not None:
        snapshot = directory / snapshots.format(snap_nr=z0, file_nr=0)
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(b"")

    return FakeRemoteDirectory(tmp_path)


@pytest.mark.parametrize("box_res", ["L1000N1800", "L1_m9"])
def test_remote_simulation_found_by_either_box_name(tmp_path, box_res):
    """
    Test that a remote simulation is found by the box's label or alias, its directory is the one on
    the server, and its snapshot_format is the layout its snapshots have there.
    """
    remote = server(tmp_path)
    sim = Simulation.from_label(box_res, "L1_m9", remote=remote)  # the boxes in the remote directory
    assert (sim.suite.name, sim.box_res, sim.name, sim.directory) == ("flamingo", "L1000N1800", "L1_m9", "L1_m9/L1_m9")
    assert sim.snapshot_file(77, 63) == "L1_m9/L1_m9/snapshots/flamingo_0077/swift_snapshot_0077/flamingo_0077.63.hdf5"
    assert sim.soap_file(77) == "L1_m9/L1_m9/SOAP-HBT/halo_properties_0077.hdf5"
    assert sim.output_redshifts[-1] == 0.0


def test_remote_simulation_directory_not_found(tmp_path):
    """
    A directory that isn't there is no error here (the lightcone checks it): the directory is the label's, and the
    snapshot layout the suite's.
    """
    remote = server(tmp_path)
    sim = Simulation.from_label("L1_m9", "HYDRO_FIDUCIAL", remote=remote)  # the name not as on the server
    assert sim.directory == "L1000N1800/HYDRO_FIDUCIAL"
    assert sim.suite.snapshot_format == FLAMINGO.snapshot_format


@pytest.mark.parametrize(
    "snapshots, given, expected",
    [
        (SERVER_SNAPSHOTS, None, SERVER_SNAPSHOTS),  # the alternative layout, found
        (FLAMINGO.snapshot_format, None, FLAMINGO.snapshot_format),  # the suite's own, found first
        (None, None, FLAMINGO.snapshot_format),  # no snapshots: the suite's, so the missing files are named as usual
        (SERVER_SNAPSHOTS, "snaps/{snap_nr}.{file_nr}.hdf5", "snaps/{snap_nr}.{file_nr}.hdf5"),  # given: used
    ],
)
def test_snapshot_layout_found(tmp_path, snapshots, given, expected):
    """Test that unless snapshot_format is given, the simulation's defaults are used to finds its snapshots."""
    remote = server(tmp_path, snapshots=snapshots)
    sim = Simulation.from_label(
        "L1000N1800", "L1_m9", remote=remote, base_dir_format="{box_res}/{sim_name}", snapshot_format=given
    )
    assert sim.suite.snapshot_format == expected


def test_box_names_on_disk(tmp_path):
    """
    Check that simulations stored on disk are found with the box's name and/or alias, the directory is looked for only
    if asked, and then found with either name.
    """
    sim = Simulation.from_label("L1_m9", "HYDRO_FIDUCIAL")
    assert (sim.box_res, sim.name) == ("L1000N1800", "HYDRO_FIDUCIAL")
    assert sim.directory == "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL"  # not checked

    server(tmp_path)  # laid out with the other names

    base = str(tmp_path) + "/{box_res}/{sim_name}"
    found = Simulation.from_label("L1000N1800", "L1_m9", base_dir_format=base, find_directory=True)
    assert (found.box_res, found.directory) == ("L1000N1800", str(tmp_path / "L1_m9" / "L1_m9"))
    assert found.suite.snapshot_format == SERVER_SNAPSHOTS

    missing = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format=base, find_directory=True)
    assert missing.directory == str(tmp_path / "L1000N1800" / "HYDRO_FIDUCIAL")  # not there: the label's

    from_path = Simulation.from_path("/mirror/flamingo/L1_m9/fgas-8sigma")
    assert (from_path.suite.name, from_path.box_res, from_path.name) == ("flamingo", "L1000N1800", "fgas-8sigma")


def test_box_res_aliases_by_hand():
    """Test that both names of a box (the regular name and alias) are kept as tuples for a suite."""

    assert suite_for_label("L1_m9") is FLAMINGO and suite_for_label("L2p8_m9") is FLAMINGO

    assert FLAMINGO.label_of("L1_m8") == "L1000N3600" and FLAMINGO.label_of("L1000N3600") == "L1000N3600"
    assert FLAMINGO.label_of("L1_m11") is None and COLIBRE.label_of("L1_m9") is None
    assert FLAMINGO.box_res_names("L1_m10") == ["L1000N0900", "L1_m10"]
    assert COLIBRE.box_res_names("L200_m6") == ["L200_m6"]

    hash(FLAMINGO)

    with pytest.raises(ValueError, match="same other name"):
        SimulationSuite(
            name="x", label_pattern="A|B", base_dir_format="/d/{box_res}", box_res_aliases={"A": "C", "B": "C"}
        )

    with pytest.raises(ValueError, match="snapshot_format"):
        SimulationSuite(name="x", label_pattern="A", base_dir_format="/d", alternative_snapshot_formats=["snaps.hdf5"])


def test_suite_file_with_aliases(tmp_path):
    """Test a suite file's box_res_aliases and alternative_snapshot_formats are read the same."""
    suite = SimulationSuite(
        name="mirror",
        label_pattern=r"L\d{4}N\d{4}_x",
        base_dir_format="/data/{box_res}/{sim_name}",
        box_res_aliases={"L1000N1800_x": "big"},
        alternative_snapshot_formats=["a/{snap_nr}.{file_nr}.hdf5", "b/{snap_nr}.hdf5"],
    )
    path = tmp_path / "mirror.yaml"
    write_suite_file(suite, path)

    assert read_suite_file(path) == suite
    assert "box_res_aliases:\n  'L1000N1800_x': 'big'\n" in path.read_text()

    plain = tmp_path / "plain.yaml"

    write_suite_file(SimulationSuite(name="plain", label_pattern="A", base_dir_format="/d"), plain)
    assert read_suite_file(plain).box_res_aliases == ()
