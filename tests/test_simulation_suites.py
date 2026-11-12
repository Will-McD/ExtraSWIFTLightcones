#!/bin/env python
"""
Tests of Simulations and SimulationSuite classes.

Test each simulation's paths and snapshot redshifts come from its suite and the
that lightcone classes take a Simulation or the box size and resolution label and name.
"""

import shutil
from pathlib import Path
import numpy as np
import pytest
import dataclasses
from swiftlet import snapshot_lightcone
from swiftlet.simulation_suites import (
    COLIBRE,
    FLAMINGO,
    Simulation,
    get_suite,
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
        snapshot_lightcone.SnapshotBeam(simulation=sim, beam_vector=[0.0, 0.0, 1.0], verbose=0),
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
    beam = {"beam_vector": [1.0, 0.0, 0.0]} if lightcone_type == "SnapshotBeam" else {}

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
    """A path format without the fields it needs, or with others, is refused when the simulation is made."""
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
