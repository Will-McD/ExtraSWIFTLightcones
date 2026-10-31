import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("h5py")
pytest.importorskip("lightcone_io")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/diagrams"))
import snapshot_lattice_diagram as lattice  # noqa: E402
from swiftlet.snapshot_cosmology import SnapshotCosmology  # noqa: E402

"""
Test that SnapshotLightcones works with small L25 and L50 COLIBRE boxes.
For these small boxes, the comoving distance corresponding to one snapshots redshift range is greater than the simulations sidelength 
(close to z = 0 snapshots fill a slice with a radius of 44 - 99 Mpc) therefore every line of sight crosses many tiles of 
the same snapshot. This can potentially create problems with having many-many replications of the same objects or introducing a very large number 
of discontinuities. 

Build a fake simulation of each box where all particles are placed in an regular lattice, check that particles and as haloes placed into an 
all-sky shell and beam, with every orientation lock, is placed exactly once and not skipped. 
"""

# label, box size [Mpc], cells per axis, lattice spacing [Mpc], shell [comoving Mpc]. The lattice spacing is half a
# cell, so a shift by whole cells (the default tables shift by eighths of the box, here one cell) keeps the lattice on
# itself
BOXES = {
    "L25": ("L0025N0376", 25., 8, 1.5625, (5.3, 60.3)),
    "L50": ("L0050N0752", 50., 8, 3.125, (10.3, 120.3)),
}
SNAPSHOTS = range(122, 128)  # COLIBRE z = 0 to 0.05, out to about 210 Mpc
LIGHTCONES = [
    pytest.param(None, None, id="all_sky"),
    pytest.param((0, 0, 1), 15., id="beam_z_axis"),
    pytest.param((1, 2, 3), 15., id="beam_off_axis"),
]


@pytest.fixture(scope="module")
def small_boxes(tmp_path_factory):
    """
    Write a fake lattice simulation of each small box.

    Returns {name: (Simulation, cosmology, SOAP catalogue filename format)}.
    """
    run_dir = tmp_path_factory.mktemp("small_boxes")
    boxes = {}
    for name, (label, boxsize, ncells, spacing, _) in BOXES.items():
        simulation, halo_format = lattice.write_lattice_simulation(run_dir, box_res=label, sim_name="LATTICE",
                                                                   snapshots=SNAPSHOTS, boxsize=boxsize,
                                                                   ncells_axis=ncells, spacing=spacing)
        boxes[name] = (simulation, SnapshotCosmology.from_simulation(simulation), halo_format)
    return boxes


@pytest.mark.parametrize("box", list(BOXES))
@pytest.mark.parametrize("orientation_lock", [None, "cube", "sphere"])
@pytest.mark.parametrize("beam_vector, ang_radius_deg", LIGHTCONES)
def test_small_box_lattice_placed_once(small_boxes, box, orientation_lock, beam_vector, ang_radius_deg):
    """
    Test that every lattice point of the shell is placed exactly once, from many tiles and several snapshots, as
    particles and as haloes, and that the haloes are placed exactly where the particles are.
    """
    label, boxsize, _, spacing, r_range = BOXES[box]
    simulation, cosmo, halo_format = small_boxes[box]
    redshift_range = tuple(lattice.redshift_at_distance(cosmo, np.array(r_range)))
    expected = lattice.lattice_in_shell(*r_range, beam_vector, ang_radius_deg, boxsize=boxsize, spacing=spacing)

    placed = {}
    for kind in ["particles", "haloes"]:
        lightcone = lattice.new_lightcone(simulation, beam_vector, orientation_lock)
        placed[kind] = lattice.place_lattice(lightcone, redshift_range, ang_radius_deg, kind, halo_format)
        missing, extra = lattice.lattice_check(placed[kind][2], expected, boxsize=boxsize, spacing=spacing)
        assert len(missing) == 0 and len(extra) == 0, \
            f"{label} {kind}: {len(missing)} lattice points missing, {len(extra)} repeated or off the lattice"

    particles, haloes = placed["particles"], placed["haloes"]
    assert np.array_equal(particles[0], haloes[0]) and np.array_equal(particles[1], haloes[1])
    np.testing.assert_allclose(particles[2], haloes[2], atol=1e-6)

    # the shell reaches more than two box lengths out, from more than one snapshot, and some snapshots fill tiles
    # one behind another along the line of sight
    snap_nr, _, coords = particles
    tiles = np.floor(coords / boxsize + 0.5).astype(int)
    
    assert len(np.unique(tiles, axis=0)) >= 5 and len(np.unique(snap_nr)) > 1
    
    layers_per_snapshot = [len(np.unique(np.max(np.abs(tiles[snap_nr == s]), axis=1))) for s in np.unique(snap_nr)]
    assert max(layers_per_snapshot) > 1

