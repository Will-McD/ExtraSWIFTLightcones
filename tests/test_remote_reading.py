#!/bin/env python
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from lightcone_helpers import assert_same_rows, remote_simulation, rows_of
from read_counter import FakeRemoteDirectory, max_in_flight
from test_snapshot_lightcone_placement import (  # noqa: F401 the fixtures, used by name
    BOX_RES,
    HALO_SIM,
    LIGHTCONES,
    PTYPE,
    fake_run_dir,
    fake_runs,
    mpiexec_command,
)

from swiftlet import hdf5stream_reader as reader
from swiftlet import snapshot_lightcone
from swiftlet.simulation_suites import Simulation
from swiftlet.snapshot_cosmology import SnapshotCosmology
from swiftlet import soap_reader

h5py = pytest.importorskip("h5py")

PROPERTIES = ["ParticleIDs", "Coordinates", "Velocities", "TileIndex"]


def lightcone_of(simulation, beam_vector, orientation_lock, ang_radius_deg=None):
    """SnapshotBeam along beam_vector of a simulation, or SnapshotAllSky if beam_vector is None."""
    if beam_vector is None:
        return snapshot_lightcone.SnapshotAllSky(simulation, verbose=0, orientation_lock=orientation_lock)
    return snapshot_lightcone.SnapshotBeam(
        simulation, beam_vector=beam_vector, ang_radius_deg=ang_radius_deg, verbose=0, orientation_lock=orientation_lock
    )


@pytest.fixture
def small_file(tmp_path):
    """A file with a 1D and a 2D dataset of 100 rows, and the FakeRemoteDirectory as its directory."""
    path = tmp_path / "small.hdf5"

    with h5py.File(path, "w") as f:
        f["ids"] = np.arange(100, dtype=np.int32)
        f["coords"] = np.arange(300, dtype=float).reshape(100, 3)

    return path, FakeRemoteDirectory(tmp_path)


def test_read_slices_local_and_remote(small_file):
    """Check that the same rows are read for a local and a remote dataset."""
    path, remote = small_file
    starts, lengths = [5, 20, 21, 90], [3, 1, 0, 10]  # a slice of no rows is left out

    with reader.open_file(path) as local, reader.open_file("small.hdf5", remote) as far:
        for name in ("ids", "coords"):
            mine = reader.read_slices(local[name], starts, lengths)
            theirs = reader.read_slices(far[name], starts, lengths)
            expected = np.concatenate([local[name][5:8], local[name][20:21], local[name][90:100]])
            np.testing.assert_array_equal(mine, expected)
            np.testing.assert_array_equal(theirs, expected)
            assert theirs.dtype == local[name].dtype

        # another dtype, as for ParticleIDs
        assert reader.read_slices(far["ids"], starts, lengths, np.int64).dtype == np.int64

    assert [(r.dataset, r.nr_slices, r.rows) for r in remote.log.requests] == [
        ("/ids", 3, 14),
        ("/coords", 3, 14),
        ("/ids", 3, 14),
    ]
    assert remote.log.opens["small.hdf5"] == 1


BAD_SLICES = [
    pytest.param([20, 5], [1, 1], "ascending", id="descending"),
    pytest.param([5, 7], [5, 1], "overlap", id="overlapping"),
    pytest.param([5, 5], [2, 2], "overlap", id="the_same_slice_twice"),
    pytest.param([5, 7], [5], "starts for", id="more_starts_than_lengths"),
    pytest.param([-1, 7], [2, 1], "at least 0", id="negative_start"),
    pytest.param([1, 7], [2, -1], "at least 0", id="negative_length"),
]


@pytest.mark.parametrize("remote_dataset", [False, True], ids=["on_disk", "remote"])
@pytest.mark.parametrize("starts, lengths, rule", BAD_SLICES)
def test_bad_slices_refused(small_file, remote_dataset, starts, lengths, rule):
    """Check that slices breaking the rules are refused,before any request, for reading data on disk and remotely."""
    path, remote = small_file

    with reader.open_file("small.hdf5" if remote_dataset else path, remote if remote_dataset else None) as infile:
        for name in ("ids", "coords"):
            with pytest.raises(ValueError, match=rule):
                reader.read_slices(infile[name], starts, lengths)

    assert remote.log.requests == []


def test_no_slices_make_no_request(small_file):
    """Test that no slices give no rows or that any requests are empty."""
    path, remote = small_file

    with reader.open_file("small.hdf5", remote) as far:
        assert reader.read_slices(far["coords"], [], []).shape == (0, 3)
        assert reader.read_slices(far["coords"], [4], [0]).shape == (0, 3)
        assert reader.read_slices(far["ids"], [4, 9], [0, 0], np.int64).dtype == np.int64

    assert remote.log.requests == []


@pytest.mark.parametrize("remote_sim", [False, True], ids=["on_disk", "remote"])
def test_pipeline_refuses_overlapping_cells(fake_runs, fake_run_dir, remote_sim):
    """
    Check that metadata giving two cells that overlap is refused when the file is read
    (on disk and remotely), instead of reading rows twice or sending the service slices it doesn't take.
    """
    simulation = Simulation.from_label(BOX_RES, HALO_SIM)
    if remote_sim:
        simulation = remote_simulation(simulation, fake_run_dir)

    lightcone = lightcone_of(simulation, (0, 0, 1), "cube", ang_radius_deg=20.0)
    _, plans = lightcone.gather_files(PTYPE, (0.05, 0.25))
    plan = next(plan for plan in plans if np.sum(plan.lengths > 1) >= 2)
    first, second = plan.cells[plan.lengths > 1][:2]
    offsets = lightcone.last_shell.cell_data[plan.snap_nr]["offsets"][PTYPE]  # the cached metadata, used again
    offsets[second] = offsets[first] + 1  # the second cell now starts inside the first

    with pytest.raises(ValueError, match="overlap"):
        lightcone.place_snapshot_particles_in_shell((0.05, 0.25), ["ParticleIDs", "Coordinates"], PTYPE)

    lightcone.clear_cache()  # the fake simulation's metadata as it was, for the next tests


#  the simulation's directory and files, checked by the lightcone


@pytest.mark.parametrize("remote_sim", [False, True], ids=["on_disk", "remote"])
def test_lightcone_checks_the_directory(fake_runs, fake_run_dir, remote_sim):
    """Test that a lightcone for a simulation whose base directory isn't found raises an error at once."""
    remote = FakeRemoteDirectory(fake_run_dir) if remote_sim else None
    simulation = Simulation.from_label(BOX_RES, "NOT_A_RUN", remote=remote)

    where = "on the remote directory" if remote_sim else "on disk"

    # with pytest.raises(FileNotFoundError, match=f"NOT_A_RUN {where}.*the box can be L1000N0900 or L1_m10"):
    with pytest.raises(FileNotFoundError, match=f"{BOX_RES}/NOT_A_RUN: no simulation"):
        lightcone_of(simulation, (0, 0, 1), "cube", ang_radius_deg=20.0)


def test_gather_steps_need_their_files(fake_runs, tmp_path):
    """Test that a lightcone notes, without raising and error, which snapshots and SOAP catalogues exist."""
    on_disk = Simulation.from_label(BOX_RES, HALO_SIM)
    empty = tmp_path / BOX_RES / HALO_SIM  # the output list, but no snapshots or SOAP catalogues
    empty.mkdir(parents=True)
    (empty / "output_list.txt").symlink_to(on_disk.output_list)
    simulation = on_disk.with_paths(directory=empty)
    lightcone = lightcone_of(simulation, (0, 0, 1), "cube", ang_radius_deg=20.0)  # (the tests' stand-in cosmology)

    assert set(lightcone._available_snapshot_cache) == {None, simulation.soap_format}

    with pytest.raises(FileNotFoundError, match="snapshot files not found"):
        lightcone.gather_files(PTYPE, (0.05, 0.25))

    with pytest.raises(FileNotFoundError, match="SOAP catalogues not found"):
        lightcone.gather_halo_catalogues((0.05, 0.25), halo_format=simulation.soap_format)

    with pytest.raises(FileNotFoundError, match="SOAP catalogues not found"):
        lightcone.gather_halo_catalogues((0.05, 0.25), halo_format=str(tmp_path / "elsewhere_{snap_nr:04d}.hdf5"))


def test_files_and_text(small_file, tmp_path):
    """Test file_exists and read_text on disk and remote, where a missing remote file is a FileNotFoundError."""
    path, remote = small_file
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.txt").write_text("# Redshift\n1.0\n0.5\n")

    for where, root in ((None, tmp_path), (remote, "")):
        assert reader.directory_exists(os.path.join(root, "sub"), where)
        assert not reader.directory_exists(os.path.join(root, "small.hdf5"), where)  # a file
        assert not reader.directory_exists(os.path.join(root, "nowhere"), where)
        assert reader.file_exists(os.path.join(root, "small.hdf5"), where)
        assert reader.file_exists(os.path.join(root, "sub", "notes.txt"), where)
        assert not reader.file_exists(os.path.join(root, "sub", "nothing.txt"), where)
        assert not reader.file_exists(os.path.join(root, "nowhere", "nothing.txt"), where)
        assert reader.read_text(os.path.join(root, "sub", "notes.txt"), where) == ["# Redshift", "1.0", "0.5"]

    with pytest.raises(FileNotFoundError):
        reader.read_text("sub/nothing.txt", remote)

    with pytest.raises(FileNotFoundError):
        with reader.open_file("nothing.hdf5", remote):
            pass


def test_request_cap_on_one_rank():
    """
    Check that a cap on one rank works correctly.

    Slots must be taken one at a time, counted and released on an error.
    """
    MPI = pytest.importorskip("mpi4py.MPI")

    with pytest.raises(ValueError, match="at least 1"):
        reader.RequestCap(MPI.COMM_SELF, 0)

    cap = reader.RequestCap(MPI.COMM_SELF, 2)

    try:
        with cap.slot():
            with cap.slot():
                pass

        with pytest.raises(RuntimeError), cap.slot():
            raise RuntimeError("a failed request gives its slot back")

        with cap.slot():
            pass
        assert cap.requests == 4
        assert cap.most_in_flight() == 2

    finally:
        cap.free()

    cap.free()  # freeing twice does nothing


# test with a remote simulation


@pytest.fixture
def remote_halo_sim(fake_runs, fake_run_dir):
    """The fake halo simulation, read remotely (a new FakeRemoteDirectory, with an empty log, for each test)."""
    return remote_simulation(Simulation.from_label(BOX_RES, HALO_SIM), fake_run_dir)


def test_remote_simulation(remote_halo_sim):
    """
    Test that a remote simulation reads the same output list and cosmology; the remote directory doesn't change
    equality.
    """
    local = Simulation.from_label(BOX_RES, HALO_SIM)
    np.testing.assert_array_equal(remote_halo_sim.output_redshifts, local.output_redshifts)
    assert remote_halo_sim.remote.log.listings  # read remotely
    assert remote_halo_sim.with_paths(snapshot_format=local.suite.snapshot_format).remote is remote_halo_sim.remote
    assert remote_halo_sim == remote_halo_sim.with_paths(remote=FakeRemoteDirectory("/"))
    hash(remote_halo_sim)
    with pytest.raises(FileNotFoundError, match="no output list"):
        remote_halo_sim.with_paths(output_list="nothing.txt").output_redshifts


def test_remote_cosmology(tmp_path):
    """
    Check that the cosmology of a remote simulation is read from its lowest redshift snapshot, the same as on disk.
    """
    from test_snapshot_cosmology import OUTPUT_LISTS, write_snapshot

    from swiftlet.simulation_suites import read_output_list, write_output_list

    sim = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL", base_dir_format=str(tmp_path) + "/{box_res}/{sim_name}")
    write_output_list(sim.output_list, read_output_list(OUTPUT_LISTS / "flamingo_L1.txt"))
    write_snapshot(Path(sim.snapshot_file(sim.z0_snapshot)))
    remote = sim.with_paths(directory=os.path.relpath(sim.directory, tmp_path), remote=FakeRemoteDirectory(tmp_path))

    local_cosmology = SnapshotCosmology.from_simulation(sim)
    remote_cosmology = SnapshotCosmology.from_simulation(remote)
    assert remote_cosmology.snapshot_filename == os.path.relpath(local_cosmology.snapshot_filename, tmp_path)
    assert remote_cosmology.cosmology == local_cosmology.cosmology
    assert remote_cosmology.internal_units == local_cosmology.internal_units
    assert remote_cosmology.Hubble_const == local_cosmology.Hubble_const
    assert remote.remote.log.opens[remote_cosmology.snapshot_filename] == 1


@pytest.mark.parametrize("max_particles_per_read", [None, 100])
@pytest.mark.parametrize("orientation_lock", [None, "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_remote_places_the_same_particles(
    remote_halo_sim, monkeypatch, beam_vector, ang_radius_deg, redshift_range, orientation_lock, max_particles_per_read
):
    """
    Check that a remote simulation places exactly the particles of the simulation on disk, with the same velocities, and
    opens nothing on disk to do it.
    """
    local = lightcone_of(
        Simulation.from_label(BOX_RES, HALO_SIM), beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg
    )
    expected = local.place_snapshot_particles_in_shell(
        redshift_range, PROPERTIES, PTYPE, max_particles_per_read=max_particles_per_read
    )[PTYPE]

    class NoFilesOnDisk:
        @staticmethod
        def File(*args, **kwargs):
            raise AssertionError(f"a remote simulation opened a file on disk: {args}")

    monkeypatch.setattr(reader, "h5py", NoFilesOnDisk)  # only the layer's h5py: the stand-in reads with its own
    remote = lightcone_of(remote_halo_sim, beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg)
    placed = remote.place_snapshot_particles_in_shell(
        redshift_range, PROPERTIES, PTYPE, max_particles_per_read=max_particles_per_read
    )[PTYPE]

    assert_same_rows(rows_of(placed), rows_of(expected))
    order = np.lexsort((placed["Coordinates"].value[:, 0], placed["ParticleIDs"].value))
    order_expected = np.lexsort((expected["Coordinates"].value[:, 0], expected["ParticleIDs"].value))
    np.testing.assert_allclose(placed["Velocities"].value[order], expected["Velocities"].value[order_expected])
    assert placed["Coordinates"].units == expected["Coordinates"].units
    assert remote.last_shell.max_requests_in_flight is None  # serial: no cap


def server_layout(simulation, root):
    """
    Lay out a fake FLAMINGO simulation in root as the hdfstream server does: the box's other name (L1_m10 for
    L1000N0900) and the snapshot files one directory deeper (the suite's alternative layout), as links to the files on
    disk. Returns the simulation's directory in root.
    """
    directory = Path(root) / simulation.suite.box_res_names(simulation.box_res)[-1] / simulation.name
    directory.mkdir(parents=True)
    (directory / "output_list.txt").symlink_to(simulation.output_list)
    layout = simulation.suite.alternative_snapshot_formats[0]
    for snap_nr in range(len(simulation.output_redshifts)):
        for file_nr in range(64):
            local = Path(simulation.snapshot_file(snap_nr, file_nr))
            if not local.is_file():
                break
            remote = directory / layout.format(snap_nr=snap_nr, file_nr=file_nr)
            remote.parent.mkdir(parents=True, exist_ok=True)
            remote.symlink_to(local)
    return directory


@pytest.mark.parametrize("box_res", [BOX_RES, "L1_m10"])
def test_remote_names_and_layout(fake_runs, tmp_path, box_res):
    """
    Test that a simulation found by the name of its box, in the server's layout, places exactly the particles of the
    simulation on disk, and keeps its labels in the metadata.
    """
    on_disk = Simulation.from_label(BOX_RES, HALO_SIM)
    server_layout(on_disk, tmp_path / "server")
    remote = Simulation.from_label(
        box_res, HALO_SIM, remote=FakeRemoteDirectory(tmp_path / "server"), base_dir_format="{box_res}/{sim_name}"
    )
    assert (remote.box_res, remote.name, remote.directory) == (BOX_RES, HALO_SIM, f"L1_m10/{HALO_SIM}")

    expected = lightcone_of(on_disk, (0, 0, 1), "cube", ang_radius_deg=20.0)
    expected = expected.place_snapshot_particles_in_shell((0.05, 0.25), PROPERTIES, PTYPE)[PTYPE]
    lightcone = lightcone_of(remote, (0, 0, 1), "cube", ang_radius_deg=20.0)
    placed = lightcone.place_snapshot_particles_in_shell((0.05, 0.25), PROPERTIES, PTYPE)
    assert_same_rows(rows_of(placed[PTYPE]), rows_of(expected))

    assert (placed.metadata["boxsize_resolution"], placed.metadata["simulation_name"]) == (BOX_RES, HALO_SIM)

    opened = set(remote.remote.log.opens)
    assert opened and all("/swift_snapshot_" in name for name in opened)


@pytest.mark.parametrize("max_particles_per_read", [None, 100])
def test_remote_requests_follow_the_service_tips(remote_halo_sim, max_particles_per_read):
    """
    Check that each particle file is opened once and read through before the next file per batch, in one request.
    """
    lightcone = lightcone_of(remote_halo_sim, None, "cube")
    lightcone.place_snapshot_particles_in_shell(
        (0.1, 0.2), ["ParticleIDs", "Coordinates"], PTYPE, max_particles_per_read=max_particles_per_read
    )
    log = remote_halo_sim.remote.log
    particle_requests = [r for r in log.requests if r.dataset.startswith(f"/{PTYPE}/")]
    plans = lightcone.last_shell.plans
    files = [remote_halo_sim.snapshot_file(plan.snap_nr, plan.file_nr) for plan in plans]  # relative to the remote

    # files read one after the other, in plan order, each read through before the next
    order = [r.file for r in particle_requests]
    read_order = [name for i, name in enumerate(order) if i == 0 or order[i - 1] != name]
    assert read_order == files

    for plan, name in zip(plans, files):
        assert log.opens[name] == 1 + (plan.file_nr == 0)  # file 0 also for the snapshot's cell metadata
        nr_batches = len(lightcone._cell_batches(plan.lengths, max_particles_per_read))
        coordinates = log.requests_of(name, f"/{PTYPE}/Coordinates")
        ids = log.requests_of(name, f"/{PTYPE}/ParticleIDs")

        assert len(coordinates) == nr_batches
        assert 1 <= len(ids) <= nr_batches  # none for a batch keeping no particle
        assert sum(r.rows for r in coordinates) == plan.nr_particles  # each cell once
        assert all(r.nr_slices >= 1 for r in coordinates + ids)

    if max_particles_per_read is None:
        assert any(r.nr_slices > 1 for r in particle_requests)  # many slices in one request


# with mpi the cap on requests is abided by


def run_remote_reads(fake_run_dir, tmp_path, nr_ranks, **args):
    """
    Run mpi_remote_reads.py on nr_ranks ranks.

    Returns (each rank's entry, the particles placed by every rank).
    """
    pytest.importorskip("mpi4py")
    mpiexec = mpiexec_command()
    if mpiexec is None:
        pytest.skip("mpiexec is not available")
    args = dict(args, run_dir=str(fake_run_dir), output=str(tmp_path / "remote"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])

    run = subprocess.run(
        [
            mpiexec,
            "-n",
            str(nr_ranks),
            sys.executable,
            str(Path(__file__).parent / "mpi_remote_reads.py"),
            json.dumps(args),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    entries = json.loads((tmp_path / "remote.json").read_text())
    particles = dict(np.load(tmp_path / "remote.npz"))

    return entries, particles


def serial_rows(beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """The rows of the particles placed in serial from the simulation on disk."""
    lightcone = lightcone_of(
        Simulation.from_label(BOX_RES, HALO_SIM), beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg
    )
    placed = lightcone.place_snapshot_particles_in_shell(redshift_range, PROPERTIES, PTYPE)[PTYPE]
    return rows_of(placed)


# (ranks, remote, max_concurrent_requests ("default": not given), the cap expected)
MPI_CASES = [
    pytest.param(2, True, "default", 2, id="remote_2_ranks_default"),
    pytest.param(4, True, "default", 2, id="remote_4_ranks_default"),
    pytest.param(3, True, 1, 1, id="remote_3_ranks_cap_1"),
    pytest.param(4, True, None, None, id="remote_4_ranks_no_cap"),
    pytest.param(3, False, 1, None, id="on_disk_3_ranks_cap_1_ignored"),
]


@pytest.mark.parametrize("nr_ranks, remote, max_concurrent_requests, cap", MPI_CASES)
def test_mpi_requests_capped(fake_run_dir, fake_runs, tmp_path, nr_ranks, remote, max_concurrent_requests, cap):
    """
    Test, with several MPI ranks that the request cap is abided by.

    Use the default cap of 2 for a remote simulation and no cap on disk, and the ranks together place exactly the same
    particles as in serial.
    """
    case = dict(beam_vector=None, ang_radius_deg=None, redshift_range=(0.1, 0.2), orientation_lock="cube")

    entries, particles = run_remote_reads(
        fake_run_dir,
        tmp_path,
        nr_ranks,
        remote=remote,
        delay=0.01 if remote else 0.0,
        max_concurrent_requests=max_concurrent_requests,
        max_particles_per_read=200,
        **case,
    )

    expected = serial_rows(**case)
    placed = (particles["SnapshotNumber"], particles["TileIndex"], particles["ParticleIDs"], particles["Coordinates"])

    order = np.lexsort(
        [placed[3][:, 2], placed[3][:, 1], placed[3][:, 0], placed[2]]
        + [placed[1][:, i] for i in reversed(range(placed[1].shape[1]))]
        + [placed[0]]
    )
    assert_same_rows(tuple(column[order] for column in placed), expected)

    most = [entry["max_requests_in_flight"] for entry in entries]
    assert len(set(most)) == 1  # every rank reads the same counter

    if cap is None:
        assert most[0] is None

        if remote:  # without a cap the ranks' requests overlap (so the capped cases show the cap holding them)
            assert max_in_flight([tuple(i) for entry in entries for i in entry["particle_requests"]]) >= 2
        return

    assert 1 <= most[0] <= cap

    intervals = [tuple(interval) for entry in entries for interval in entry["particle_requests"]]
    assert intervals
    assert max_in_flight(intervals) <= cap


# SOAP catalogues

SOAP_PROPERTIES = ["InputHalos/HaloCentre", "InputHalos/HaloCatalogueIndex"]


def halo_rows(haloes):
    """Rows of (snapshot, catalogue index, SOAP row, centre [Mpc]) of placed haloes, sorted, to compare two sets."""
    snap_nr = haloes["Lightcone/SnapshotNumber"].value.astype(int)
    index = haloes["InputHalos/HaloCatalogueIndex"].value.astype(int)
    soap_index = haloes["InputHalos/SOAPIndex"].value.astype(int)
    centre = haloes["Lightcone/HaloCentre"].to_value("Mpc")
    order = np.lexsort((centre[:, 2], centre[:, 1], centre[:, 0], soap_index, index, snap_nr))
    return snap_nr[order], index[order], soap_index[order], centre[order]


def assert_same_haloes(rows, other):
    """Two sets of halo rows are the same haloes at the same positions."""
    assert len(rows[0]) == len(other[0]) > 0
    for mine, theirs in zip(rows[:3], other[:3]):
        np.testing.assert_array_equal(mine, theirs)
    np.testing.assert_allclose(rows[3], other[3], atol=1e-6)


def test_soap_reader_remote(fake_runs, fake_run_dir):
    """Check every SOAP reader gives the same rows and units remotely."""
    local = fake_runs.format(snap_nr=72)
    remote = FakeRemoteDirectory(fake_run_dir)
    far = os.path.relpath(local, fake_run_dir)

    assert soap_reader.soap_catalogue_length(far, remote=remote) == soap_reader.soap_catalogue_length(local) > 10

    nr_rows = soap_reader.soap_catalogue_length(local)

    def same(mine, theirs):
        assert set(mine) == set(theirs)
        for name in mine:
            np.testing.assert_array_equal(np.asarray(mine[name]), np.asarray(theirs[name]))
            if hasattr(mine[name], "units"):
                assert mine[name].units == theirs[name].units

    same(
        soap_reader.read_soap_rows(far, SOAP_PROPERTIES, 3, 9, remote=remote),
        soap_reader.read_soap_rows(local, SOAP_PROPERTIES, 3, 9),
    )

    assert len(remote.log.requests) == 2  # one per dataset
    chunks = list(soap_reader.read_soap_in_chunks(far, SOAP_PROPERTIES, 4, remote=remote))

    for (start, mine), (other_start, theirs) in zip(chunks, soap_reader.read_soap_in_chunks(local, SOAP_PROPERTIES, 4)):
        assert start == other_start
        same(mine, theirs)

    rows = np.array([0, 1, 2, 5, 7, 8, nr_rows - 1])
    before = len(remote.log.requests)

    same(
        soap_reader.read_soap_selected_rows(far, SOAP_PROPERTIES, rows, remote=remote),
        soap_reader.read_soap_selected_rows(local, SOAP_PROPERTIES, rows),
    )

    selected = remote.log.requests[before:]
    assert [r.nr_slices for r in selected] == [4, 4]  # runs 0-2, 5, 7-8, last: one request per dataset

    big = lambda index: index.value > nr_rows // 2  # noqa: E731

    same(
        soap_reader.select_soap_rows(
            far, "InputHalos/HaloCatalogueIndex", big, ["InputHalos/HaloCentre"], 5, remote=remote
        ),
        soap_reader.select_soap_rows(local, "InputHalos/HaloCatalogueIndex", big, ["InputHalos/HaloCentre"], 5),
    )


@pytest.mark.parametrize("soap_chunk_size", [None, 700])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_remote_places_the_same_haloes(
    fake_runs, fake_run_dir, monkeypatch, beam_vector, ang_radius_deg, redshift_range, soap_chunk_size
):
    """
    Check a remote simulation places the haloes of its SOAP catalogues on the server exactly as the simulation on disk.
    """
    local = lightcone_of(Simulation.from_label(BOX_RES, HALO_SIM), beam_vector, "cube", ang_radius_deg=ang_radius_deg)
    expected = local.place_halos_in_shell(redshift_range, soap_chunk_size=soap_chunk_size, halo_format=fake_runs)

    class NoFilesOnDisk:
        @staticmethod
        def File(*args, **kwargs):
            raise AssertionError(f"a remote simulation opened a file on disk: {args}")

    simulation = remote_simulation(Simulation.from_label(BOX_RES, HALO_SIM), fake_run_dir)
    simulation = simulation.with_paths(soap_format="SOAP/halo_properties_{snap_nr:04d}.hdf5")  # as the fake runs

    monkeypatch.setattr(reader, "h5py", NoFilesOnDisk)
    remote = lightcone_of(simulation, beam_vector, "cube", ang_radius_deg=ang_radius_deg)
    placed = remote.place_halos_in_shell(redshift_range, soap_chunk_size=soap_chunk_size)

    assert_same_haloes(halo_rows(placed), halo_rows(expected))
    soap_requests = [r for r in simulation.remote.log.requests if r.dataset.startswith("/InputHalos/")]

    assert soap_requests and all(r.nr_slices == 1 for r in soap_requests)  # one slice per chunk of rows
    assert remote.last_shell.max_requests_in_flight is None  # serial: no cap


def test_halo_lightcone_file_from_a_remote_catalogue(fake_runs, fake_run_dir, tmp_path):
    """Test that a halo lightcone file written with the units of a remote SOAP catalogue is the same as one on disk."""
    lightcone = lightcone_of(Simulation.from_label(BOX_RES, HALO_SIM), (0, 0, 1), "cube", ang_radius_deg=20.0)
    haloes = lightcone.place_halos_in_shell((0.05, 0.25), halo_format=fake_runs)
    snap_nr = int(haloes["Lightcone/SnapshotNumber"].value[0])

    keep = haloes["Lightcone/SnapshotNumber"].value == snap_nr
    one = snapshot_lightcone.LightconeOutput({k: v[keep] for k, v in haloes.items()}, haloes.metadata)

    local = fake_runs.format(snap_nr=snap_nr)
    # write test halo lightcone files
    snapshot_lightcone.write_halo_lightcone_file(tmp_path / "local.hdf5", one, local, {}, one.metadata)
    snapshot_lightcone.write_halo_lightcone_file(
        tmp_path / "remote.hdf5",
        one,
        os.path.relpath(local, fake_run_dir),
        {},
        one.metadata,
        remote=FakeRemoteDirectory(fake_run_dir),
    )

    with h5py.File(tmp_path / "local.hdf5", "r") as mine, h5py.File(tmp_path / "remote.hdf5", "r") as theirs:
        names = []
        mine.visit(names.append)

        for name in names:
            if isinstance(mine[name], h5py.Dataset):
                np.testing.assert_array_equal(mine[name][...], theirs[name][...])

            assert dict(mine[name].attrs).keys() == dict(theirs[name].attrs).keys(), name


@pytest.mark.parametrize(
    "nr_ranks, remote, max_concurrent_requests, cap",
    [
        pytest.param(3, True, "default", 2, id="remote_3_ranks_default"),
        pytest.param(4, True, 1, 1, id="remote_4_ranks_cap_1"),
        pytest.param(3, False, 1, None, id="on_disk_3_ranks_never_capped"),
    ],
)
def test_mpi_halo_requests_capped(fake_run_dir, fake_runs, tmp_path, nr_ranks, remote, max_concurrent_requests, cap):
    """On several ranks the SOAP requests are capped as the particles' are, and the haloes are those of serial."""
    case = dict(beam_vector=None, ang_radius_deg=None, redshift_range=(0.1, 0.2), orientation_lock="cube")
    entries, placed = run_remote_reads(
        fake_run_dir,
        tmp_path,
        nr_ranks,
        remote=remote,
        delay=0.01 if remote else 0.0,
        max_concurrent_requests=max_concurrent_requests,
        haloes=True,
        soap_chunk_size=500,  # 8 chunks of each catalogue of 4000 haloes
        **case,
    )

    lightcone = lightcone_of(Simulation.from_label(BOX_RES, HALO_SIM), None, "cube")
    expected = halo_rows(lightcone.place_halos_in_shell(case["redshift_range"], halo_format=fake_runs))

    centre = placed["HaloCentre"]

    order = np.lexsort(
        (
            centre[:, 2],
            centre[:, 1],
            centre[:, 0],
            placed["SOAPIndex"],
            placed["HaloCatalogueIndex"],
            placed["SnapshotNumber"],
        )
    )
    rows = tuple(placed[name][order] for name in ("SnapshotNumber", "HaloCatalogueIndex", "SOAPIndex", "HaloCentre"))
    assert_same_haloes(rows, expected)

    most = {entry["max_requests_in_flight"] for entry in entries}
    assert len(most) == 1

    if cap is None:
        assert most.pop() is None
        return

    assert 1 <= most.pop() <= cap

    intervals = [tuple(interval) for entry in entries for interval in entry["particle_requests"]]
    assert intervals and max_in_flight(intervals) <= cap
