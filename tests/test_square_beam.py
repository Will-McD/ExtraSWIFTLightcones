"""
Test SnapshotSquareBeam makes a square pyramid about the beam's axis, sides along its transverse axes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from swiftlet import snapshot_lightcone
from swiftlet.repeat_check import find_close_cell_repeats
from test_snapshot_lightcone_placement import (  # noqa: F401 the fixtures, used by name
    BOXSIZE,
    HALO_SIM,
    NEDGE,
    NHALOS,
    PTYPE,
    assert_same_particles,
    assert_same_rows,
    fake_run_dir,
    fake_runs,
    mpiexec_command,
    new_lightcone,
    particle_rows,
    place_particles,
    sorted_rows,
)

SIDE = 20.0  # angular side length [deg] of the square beams


def in_square(coords, lightcone, side_deg, slack=0.0, rotation_deg=0.0):
    """
    Points inside the square pyramid of lightcone, from its axis and transverse axes: abs(x.e1) and abs(x.e2) at most
    tan(side / 2) (x.axis), with slack [Mpc] to spare (positive: inside by at least slack).
    """
    tan_half = np.tan(0.5 * np.deg2rad(side_deg))
    angle = np.deg2rad(rotation_deg)
    e1, e2 = lightcone.beam_transverse_e1, lightcone.beam_transverse_e2
    u, v = np.cos(angle) * e1 + np.sin(angle) * e2, np.cos(angle) * e2 - np.sin(angle) * e1
    along = coords @ lightcone.beam_vec
    across = np.maximum(np.abs(coords @ u), np.abs(coords @ v))
    return (along > 0) & ((tan_half * along - across) / np.sqrt(1 + tan_half**2) >= slack)


def square(beam_vector, orientation_lock, side_deg=SIDE, **kwargs):
    """A SnapshotSquareBeam for a fake simulation with haloes."""
    return new_lightcone(HALO_SIM, beam_vector, orientation_lock, ang_side_length_deg=side_deg, **kwargs)


@pytest.mark.parametrize(
    "beam_vector, orientation_lock",
    [((0, 0, 1), None), ((1, 2, 3), "cube"), ((0.1, 0.5, -0.9), "sphere"), ((-1, 0.5, 0.2), None)],
)
def test_square_is_part_of_all_sky(fake_runs, beam_vector, orientation_lock):
    """Test a square beam is exactly the part of the all-sky lightcone inside its pyramid."""
    redshift_range = (0.1, 0.2)
    all_sky = particle_rows(place_particles(new_lightcone(HALO_SIM, None, orientation_lock), redshift_range))
    lightcone = square(beam_vector, orientation_lock)
    beam = particle_rows(place_particles(lightcone, redshift_range))

    inside = in_square(all_sky[2], lightcone, SIDE)
    assert len(beam[0]) > 0
    assert_same_particles(tuple(a[inside] for a in all_sky), beam)


def test_square_between_its_inner_and_outer_circles(fake_runs):
    """
    Test that the square holds every particle of the circular beam inside it (radius side / 2) and is part of the
    circular beam.
    """
    beam_vector, redshift_range = (1, 2, 3), (0.05, 0.3)
    lightcone = square(beam_vector, "cube")
    assert lightcone.ang_radius_deg == pytest.approx(np.degrees(np.arctan(np.sqrt(2) * np.tan(np.deg2rad(SIDE / 2)))))

    in_sq = particle_rows(place_particles(lightcone, redshift_range))
    inner = particle_rows(
        place_particles(new_lightcone(HALO_SIM, beam_vector, "cube", ang_radius_deg=SIDE / 2), redshift_range)
    )
    outer = particle_rows(
        place_particles(
            new_lightcone(HALO_SIM, beam_vector, "cube", ang_radius_deg=lightcone.ang_radius_deg), redshift_range
        )
    )

    def keys(rows):
        return {(int(s), int(i), tuple(np.round(c, 6))) for s, i, c in zip(*rows)}

    assert len(inner[0]) < len(in_sq[0]) < len(outer[0])
    assert keys(inner) <= keys(in_sq) <= keys(outer)
    assert_same_particles(tuple(a[in_square(outer[2], lightcone, SIDE)] for a in outer), in_sq)


@pytest.mark.parametrize("orientation_lock", [None, "sphere"])
def test_square_haloes_placed_as_particles(fake_runs, orientation_lock):
    """
    Check the haloes of a square beam are placed exactly as particles at their centres, and every one is inside the
    square.
    """
    redshift_range = (0.05, 0.3)
    lightcone = square((1, 2, 3), orientation_lock)
    halos = lightcone.place_halos_in_shell(redshift_range, halo_format=fake_runs)
    particles = place_particles(lightcone, redshift_range)
    rows = {}
    for name, data, ids, centre, a in (
        ("haloes", halos, "InputHalos/HaloCatalogueIndex", "Lightcone/HaloCentre", "Lightcone/ExpansionFactor"),
        ("particles", particles, "ParticleIDs", "Coordinates", "ExpansionFactors"),
    ):
        snap = data["Lightcone/SnapshotNumber" if name == "haloes" else "SnapshotNumber"]
        rows[name] = sorted_rows(
            snap.value.astype(int), data[ids].value.astype(int), data[centre].to_value("Mpc"), data[a].value, False
        )
    assert len(rows["haloes"][0]) > 0
    assert_same_rows(rows["haloes"], rows["particles"])
    assert np.all(in_square(rows["haloes"][2], lightcone, SIDE, slack=-1e-9))


def test_numba_and_numpy_pyramids_agree():
    """
    Test that numba and NumPy flag the same points, with and without a buffer, and a buffer keeps every
    point within its distance of a point inside.
    """
    rng = np.random.default_rng(4)
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    u = np.cross(axis, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    tan_half = np.tan(np.deg2rad(15.0))
    coords = rng.uniform(-800.0, 800.0, size=(200_000, 3))

    for buffer in (0.0, 30.0):
        expected = snapshot_lightcone._classify_pyramid(coords, axis, u, v, tan_half, 100.0, 700.0, buffer)
        assert 0 < np.count_nonzero(expected) < len(coords)
        if snapshot_lightcone._HAVE_NUMBA:
            mask = np.empty(len(coords), dtype=np.bool_)
            snapshot_lightcone._classify_pyramid_numba(coords, axis, u, v, tan_half, 100.0, 700.0, buffer, mask)
            np.testing.assert_array_equal(mask, expected)

    inside = coords[snapshot_lightcone._classify_pyramid(coords, axis, u, v, tan_half, 100.0, 700.0, 0.0)]
    step = rng.normal(size=inside.shape)
    step *= rng.uniform(0.0, 30.0, size=(len(inside), 1)) / np.linalg.norm(step, axis=1, keepdims=True)
    assert np.all(snapshot_lightcone._classify_pyramid(inside + step, axis, u, v, tan_half, 100.0, 700.0, 30.0))


def test_square_side_length_checked(fake_runs):
    """The side length is needed, positive and at most MAX_SQUARE_BEAM_SIDE_DEG; so is a beam vector."""
    largest = snapshot_lightcone.MAX_SQUARE_BEAM_SIDE_DEG
    assert largest == pytest.approx(101.5, abs=0.1)
    assert square((0, 0, 1), None, side_deg=largest).ang_radius_deg == pytest.approx(
        snapshot_lightcone.MAX_BEAM_ANG_RADIUS_DEG
    )
    with pytest.raises(TypeError, match="ang_side_length_deg"):
        snapshot_lightcone.SnapshotSquareBeam(square((0, 0, 1), None).simulation, beam_vector=(0, 0, 1))
    with pytest.raises(TypeError, match="beam_vector"):
        snapshot_lightcone.SnapshotSquareBeam(square((0, 0, 1), None).simulation, ang_side_length_deg=SIDE)
    for bad in (0.0, -5.0, float("nan"), largest + 0.1):
        with pytest.raises(ValueError, match="side length|positive"):
            square((0, 0, 1), None, side_deg=bad)


def test_square_metadata_sizes_and_solid_angle(fake_runs):
    """
    A square beam records its side length in its metadata, and its side length, diagonal and solid angle are those of
    the pyramid; the circular beam and the all-sky lightcone have no side length.
    """
    lightcone = square((1, 2, 3), "cube")
    metadata = lightcone.lightcone_metadata((0.1, 0.2))
    assert metadata["square_rotation_deg"] == 0.0
    assert square((1, 2, 3), "cube", square_rotation_deg=30.0).lightcone_metadata()["square_rotation_deg"] == 30.0
    assert metadata["ang_radius_deg"] == pytest.approx(lightcone.ang_radius_deg)
    for other in (new_lightcone(HALO_SIM, (1, 2, 3), "cube", ang_radius_deg=10.0), new_lightcone(HALO_SIM, None, None)):
        assert other.lightcone_metadata((0.1, 0.2))["ang_side_length_deg"] is None
        assert other.lightcone_metadata((0.1, 0.2))["square_rotation_deg"] is None
    assert "square beam" in lightcone._describe_geometry()

    assert lightcone.beam_side_length(500.0, use_comoving_dist=True) == pytest.approx(1000.0 * np.tan(np.deg2rad(10.0)))
    assert lightcone.beam_diameter(500.0, use_comoving_dist=True) == pytest.approx(
        np.sqrt(2) * lightcone.beam_side_length(500.0, use_comoving_dist=True)
    )

    # the solid angle: the fraction of random directions inside the pyramid, and about side^2 for a small square
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(2_000_000, 3))
    fraction = np.mean(in_square(directions, lightcone, SIDE))
    assert lightcone.solid_angle == pytest.approx(4 * np.pi * fraction, rel=0.01)
    assert square((1, 2, 3), "cube", side_deg=1.0).solid_angle == pytest.approx(np.deg2rad(1.0) ** 2, rel=1e-4)


def test_square_number_density(fake_runs):
    """Check that the number of particles in a square beam is the number density times its pyramid's volume."""
    redshift_range = (0.05, 0.4)
    lightcone = square((0, 0, 1), None, side_deg=40.0)
    snap_nr, ids, coords = particle_rows(place_particles(lightcone, redshift_range))
    count = np.count_nonzero(ids >= NEDGE)

    density = (NHALOS - NEDGE) / BOXSIZE**3
    r_lo, r_hi = lightcone.cosmo.comoving_distance(redshift_range).to_value("Mpc")
    expected = density * lightcone.solid_angle / 3 * (r_hi**3 - r_lo**3)
    assert abs(count - expected) < 3 * np.sqrt(expected), (count, expected)


def test_square_file_by_file_is_the_whole_shell(fake_runs):
    """Check that for square beam placing particles one file at a time gives the particles of the whole lightcone."""
    redshift_range = (0.05, 0.3)
    lightcone = square((1, 2, 3), "sphere")
    whole = particle_rows(place_particles(lightcone, redshift_range))
    nr_files, _ = lightcone.gather_files(PTYPE, redshift_range)
    pieces = [lightcone.place_file_in_shell(i, PTYPE, ["ParticleIDs", "Coordinates"]) for i in range(nr_files)]
    pieces = [piece for piece in pieces if piece and len(piece["ParticleIDs"]) > 0]
    joined = {
        name: np.concatenate([piece[name] for piece in pieces])
        for name in ("SnapshotNumber", "ParticleIDs", "Coordinates")
    }
    assert nr_files > 1
    assert_same_particles(particle_rows(joined), whole)


def test_square_mpi_matches_serial(fake_run_dir, fake_runs, tmp_path):
    """Check that the particles of a square beam made with 3 MPI ranks are the same as when done in serial."""
    pytest.importorskip("mpi4py")
    mpiexec = mpiexec_command()
    if mpiexec is None:
        pytest.skip("mpiexec is not available")

    args = dict(
        run_dir=str(fake_run_dir),
        sim_name=HALO_SIM,
        beam_vector=(1, 2, 3),
        ang_radius_deg=None,
        ang_side_length_deg=SIDE,
        redshift_range=(0.05, 0.3),
        orientation_lock="cube",
        redistribute=False,
        output=str(tmp_path / "placed"),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    script = str(Path(__file__).parent / "mpi_place_particles.py")
    run = subprocess.run(
        [mpiexec, "-n", "3", sys.executable, script, json.dumps(args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    ranks = [np.load(tmp_path / f"placed.{rank}.npz") for rank in range(3)]
    assert sum(len(rank["ids"]) > 0 for rank in ranks) > 1, "the particles were not split between the ranks"
    snap_nr, ids, coords = (np.concatenate([rank[name] for rank in ranks]) for name in ("snap_nr", "ids", "coords"))
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], ids, snap_nr))
    serial = particle_rows(place_particles(square((1, 2, 3), "cube"), (0.05, 0.3)))
    assert_same_particles((snap_nr[order], ids[order], coords[order]), serial)


def test_square_close_repeats_between_its_circles(fake_runs):
    """
    The close repeat check of a square beam looks at the cells of the square: as many cell copies as the circular beam
    inside it at least, and as the circular beam through its corners at most.
    """
    redshift_range, beam_vector = (0.05, 0.3), (1, 2, 3)

    def copies(lightcone):
        return find_close_cell_repeats(lightcone, redshift_range, warn=False).nr_cell_copies

    lightcone = square(beam_vector, "cube")
    inner = copies(new_lightcone(HALO_SIM, beam_vector, "cube", ang_radius_deg=SIDE / 2))
    outer = copies(new_lightcone(HALO_SIM, beam_vector, "cube", ang_radius_deg=lightcone.ang_radius_deg))
    assert inner < copies(lightcone) < outer


@pytest.mark.parametrize(
    "beam_vector, orientation_lock, rotation_deg",
    [((1, 2, 3), "cube", 30.0), ((0.1, 0.5, -0.9), None, 135.0), ((0, 0, 1), "sphere", -20.0)],
)
def test_turned_square_is_part_of_all_sky(fake_runs, beam_vector, orientation_lock, rotation_deg):
    """
    Check that a square beam turned about its axis is exactly the part of the all-sky lightcone inside its turned
    pyramid, and differs from the square that isn't turned.
    """
    redshift_range = (0.1, 0.2)
    all_sky = particle_rows(place_particles(new_lightcone(HALO_SIM, None, orientation_lock), redshift_range))
    lightcone = square(beam_vector, orientation_lock, square_rotation_deg=rotation_deg)
    beam = particle_rows(place_particles(lightcone, redshift_range))

    inside = in_square(all_sky[2], lightcone, SIDE, rotation_deg=rotation_deg)
    assert len(beam[0]) > 0
    assert_same_particles(tuple(a[inside] for a in all_sky), beam)
    assert np.any(inside != in_square(all_sky[2], lightcone, SIDE))


def test_square_turned_by_90_is_the_same_square(fake_runs):
    """
    Check that turning a square beam by 90 degrees gives the same square and particles, with its side axes swapped.
    """
    redshift_range = (0.05, 0.3)
    straight, turned = (square((1, 2, 3), "cube", square_rotation_deg=angle) for angle in (0.0, 90.0))
    assert_same_particles(
        particle_rows(place_particles(turned, redshift_range)),
        particle_rows(place_particles(straight, redshift_range)),
    )
    np.testing.assert_allclose(turned.side_axes[0], straight.side_axes[1], atol=1e-15)
    np.testing.assert_allclose(turned.side_axes[1], -straight.side_axes[0], atol=1e-15)


def test_square_rotation_and_rotation_to(fake_runs):
    """
    Check the side axes are the transverse axes turned by square_rotation_deg (orthonormal), and rotation_to gives the
    rotation that puts the first side along a direction.
    """
    lightcone = square((1, 2, 3), "cube")
    np.testing.assert_allclose(lightcone.side_axes[0], lightcone.beam_transverse_e1)
    np.testing.assert_allclose(lightcone.side_axes[1], lightcone.beam_transverse_e2)

    rng = np.random.default_rng(5)
    for beam_vector, direction in zip(rng.normal(size=(6, 3)), rng.normal(size=(6, 3))):
        rotation = snapshot_lightcone.SnapshotSquareBeam.rotation_to(beam_vector, direction)
        turned = square(beam_vector, None, square_rotation_deg=rotation)
        u, v = turned.side_axes
        across = direction - (direction @ turned.beam_vec) * turned.beam_vec
        np.testing.assert_allclose(u, across / np.linalg.norm(across), atol=1e-12)
        np.testing.assert_allclose(np.cross(u, v), turned.beam_vec, atol=1e-12)
        # the same side axes without making the lightcone, as for drawing a planned square beam
        for axis, expected in zip(snapshot_lightcone.SnapshotSquareBeam.side_axes_of(beam_vector, rotation), (u, v)):
            np.testing.assert_allclose(axis, expected, atol=1e-15)
        assert abs(u @ v) < 1e-12 and abs(np.linalg.norm(v) - 1) < 1e-12

    with pytest.raises(ValueError, match="along the beam"):
        snapshot_lightcone.SnapshotSquareBeam.rotation_to((1, 2, 3), (2, 4, 6))

    with pytest.raises(ValueError, match="square_rotation_deg"):
        square((1, 2, 3), None, square_rotation_deg=float("nan"))

    assert "turned 30 deg" in square((1, 2, 3), None, square_rotation_deg=30.0)._describe_geometry()
