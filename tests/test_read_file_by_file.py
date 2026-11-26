"""Test reading snapshot files once each and placing their particles in every tile that uses them."""

import numpy as np
import pytest
import unyt
from lightcone_helpers import assert_same_rows, rows_of, tile_file_pieces
from read_counter import ReadCounter, _merged_length
from test_snapshot_lightcone_placement import (  # noqa: F401 the fixtures, used by name
    HALO_SIM,
    LIGHTCONES,
    MPC_CM,
    NEDGE,
    PTYPE,
    fake_run_dir,
    fake_runs,
    new_lightcone,
)

h5py = pytest.importorskip("h5py")

PROPERTIES = ["ParticleIDs", "Coordinates", "TileIndex"]


def place_tile_by_tile(lightcone, redshift_range):
    """
    Place the particles of a shell one (snapshot, tile, file) piece at a time, independently of the pipeline: each
    piece's cells read with h5py, moved into the tile with Snapshot2Lightcone, and cut to the piece's redshift range.

    Returns a dict of the particles placed: SnapshotNumber, TileIndex, ParticleIDs and Coordinates.
    """
    lightcone.gather_files(PTYPE, redshift_range)
    shell = lightcone.last_shell
    z_range = {(piece.snap_nr, piece.tile): piece.z_range for piece in shell.pieces}
    cut_to_shell = lightcone._SnapshotLightcone__select_particles_in_shell
    to_mpc = unyt.unyt_quantity(MPC_CM, "cm").to_value("Mpc")  # the fake snapshots' length unit
    placed = {"SnapshotNumber": [], "TileIndex": [], "ParticleIDs": [], "Coordinates": []}

    for spec in tile_file_pieces(shell):  # one piece per (snapshot, tile, file), found without the plans
        rows = np.concatenate([np.arange(o, o + n) for o, n in zip(spec.offsets, spec.lengths)]).astype(np.int64)

        with h5py.File(lightcone.simulation.snapshot_file(spec.snap_nr, spec.file_nr), "r") as f:
            coords = f[PTYPE]["Coordinates"][...][rows] * to_mpc
            ids = f[PTYPE]["ParticleIDs"][...][rows]

        moved = lightcone.Snapshot2Lightcone(spec.snap_nr, coords, spec.tile)
        z_min, z_max = z_range[(spec.snap_nr, spec.tile)]
        kept, _ = cut_to_shell(moved, zmin=z_min, zmax=z_max, method="exact", r_tol=0 * unyt.Mpc)

        placed["SnapshotNumber"].append(np.full(len(kept), spec.snap_nr))
        placed["TileIndex"].append(np.tile(spec.tile, (len(kept), 1)))
        placed["ParticleIDs"].append(ids[kept])
        placed["Coordinates"].append(moved[kept])

    return {
        "SnapshotNumber": unyt.unyt_array(np.concatenate(placed["SnapshotNumber"]), "dimensionless"),
        "TileIndex": unyt.unyt_array(np.concatenate(placed["TileIndex"]), "dimensionless"),
        "ParticleIDs": unyt.unyt_array(np.concatenate(placed["ParticleIDs"]), "dimensionless"),
        "Coordinates": unyt.unyt_array(np.concatenate(placed["Coordinates"]), "Mpc"),
    }


@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_same_particles_as_reading_tile_by_tile(
    fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock
):
    """
    Test that reading each file once places the particles the same way as reading each (snapshot, tile, file) on its own
    .
    """
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg)

    file_by_file = lightcone.place_snapshot_particles_in_shell(redshift_range, PROPERTIES, PTYPE)[PTYPE]

    reference = place_tile_by_tile(lightcone, redshift_range)

    def off_the_faces(particles):
        clear = particles["ParticleIDs"].value >= NEDGE
        return {name: values[clear] for name, values in particles.items() if name in reference}

    assert_same_rows(rows_of(off_the_faces(file_by_file)), rows_of(off_the_faces(reference)))
    tile_length = 4 if orientation_lock == "sphere" else 3

    assert file_by_file["TileIndex"].shape == (len(file_by_file["ParticleIDs"]), tile_length)


@pytest.mark.parametrize("orientation_lock", [None, "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg, redshift_range", LIGHTCONES)
def test_one_tile_at_a_time(fake_runs, beam_vector, ang_radius_deg, redshift_range, orientation_lock):
    """
    Check that for every file, the particles placed one tile at a time are exactly the same as those when the particles
    are placed all tiles at once and each tile's particles carry its: TileIndex, and metadata.
    """
    lightcone = new_lightcone(HALO_SIM, beam_vector, orientation_lock, ang_radius_deg=ang_radius_deg)
    nr_files, plans = lightcone.gather_files(PTYPE, redshift_range)
    assert nr_files == len(plans) > 0

    for file_number, plan in enumerate(plans):
        whole = lightcone.place_file_in_shell(file_number, PTYPE, PROPERTIES)
        assert whole.metadata["snapshot_number"] == plan.snap_nr and whole.metadata["file_number"] == plan.file_nr
        assert whole.metadata["tile"] is None

        tiles = sorted({use.tile[:3] for use in plan.uses})
        parts = []

        for tile in tiles:
            one = lightcone.place_file_in_shell(file_number, PTYPE, PROPERTIES, tile=tile)
            assert one.metadata["tile"] == tile
            if one:
                assert np.all(one["TileIndex"].value[:, :3] == tile)
                parts.append(one)

        if not whole or len(whole["ParticleIDs"]) == 0:  # no particles of the file kept in the lightcone
            assert sum(len(part["ParticleIDs"]) for part in parts) == 0
            continue

        joined = {name: np.concatenate([part[name] for part in parts]) for name in whole}
        joined["SnapshotNumber"] = np.concatenate([part["SnapshotNumber"] for part in parts])

        assert_same_rows(rows_of(joined), rows_of(whole))


def test_one_layer_of_a_sphere_tile(fake_runs):
    """Test "sphere" lock, (nx, ny, nz) places every layer of the tile and (nx, ny, nz, layer) one layer."""
    lightcone = new_lightcone(HALO_SIM, None, "sphere")
    _, plans = lightcone.gather_files(PTYPE, (0.1, 0.2))

    # a file used by two layers of one tile
    found = [
        (number, use.tile)
        for number, plan in enumerate(plans)
        for use in plan.uses
        if sum(other.tile[:3] == use.tile[:3] for other in plan.uses) > 1
    ]

    assert found, "no tile crosses a layer boundary in this shell"

    file_number, tile = found[0]
    all_layers = lightcone.place_file_in_shell(file_number, PTYPE, PROPERTIES, tile=tile[:3])
    layers = {int(layer) for layer in all_layers["TileIndex"].value[:, 3]}
    assert len(layers) > 1

    one_layer = lightcone.place_file_in_shell(file_number, PTYPE, PROPERTIES, tile=tile)
    assert len(one_layer["ParticleIDs"]) < len(all_layers["ParticleIDs"])

    np.testing.assert_array_equal(one_layer["TileIndex"].value, np.tile(tile, (len(one_layer["ParticleIDs"]), 1)))


def test_tile_not_in_the_file_or_the_shell(fake_runs):
    """A tile of the shell that the file has no use for gives an empty output; a tile outside the shell, an error."""
    lightcone = new_lightcone(HALO_SIM, (0, 0, 1), "cube", ang_radius_deg=20.0)
    _, plans = lightcone.gather_files(PTYPE, (0.05, 0.25))
    shell_tiles = {piece.tile for piece in lightcone.last_shell.pieces}
    unused = [(number, tile) for number, plan in enumerate(plans) for tile in shell_tiles - {u.tile for u in plan.uses}]
    assert unused
    file_number, tile = unused[0]
    empty = lightcone.place_file_in_shell(file_number, PTYPE, ["ParticleIDs"], tile=tile)
    assert not empty and empty.metadata["tile"] == tile
    with pytest.raises(ValueError, match="not in the shell"):
        lightcone.place_file_in_shell(0, PTYPE, ["ParticleIDs"], tile=(40, 40, 40))


def test_one_tile_reads_only_its_cells(fake_runs):
    """Placing one tile of a file reads that tile's cells, each once, and no others."""
    lightcone = new_lightcone(HALO_SIM, None, "cube")
    _, plans = lightcone.gather_files(PTYPE, (0.1, 0.2))
    file_number, plan = max(enumerate(plans), key=lambda item: len(item[1].uses))
    use = plan.uses[1]
    with ReadCounter() as counter:
        lightcone.place_file_in_shell(file_number, PTYPE, ["ParticleIDs"], tile=use.tile)
    coordinates = [entry for (_, name), entry in counter.reads.items() if name == f"/{PTYPE}/Coordinates"]
    assert len(coordinates) == 1
    expected = _merged_length(
        [(int(o), int(o + n)) for o, n in zip(plan.offsets[use.cells], plan.lengths[use.cells]) if n > 0]
    )
    assert coordinates[0].unique_rows == expected == coordinates[0].rows_read
    assert sum(counter.opens.values()) == 1
