#!/bin/env python
import numpy as np
import pytest
import unyt

h5py = pytest.importorskip("h5py")
from astropy.cosmology import w0waCDM

from swiftlet.snapshot_cosmology import SnapshotCosmology, as_astropy_cosmology, find_snapshot_files, neutrino_masses
from swiftlet.simulation_suites import Simulation

"""
Tests swiftlet.cosmology
The cosmology of FLAMINGO and COLIBRE simulations, read from fake SWIFT snapshots with the Cosmology, units and Parameters groups SWIFT writes.
"""

# build mock cosmology params
MPC_CM = 3.08567758149e24
H = 0.681
COSMOLOGY = {"h": H, "Omega_m": 0.306, "Omega_lambda": 0.693922, "Omega_b": 0.0486, "w_0": -1.0, "w_a": 0.0,
             "T_CMB_0 [K]": 2.7255, "Scale-factor": 1.0, "Redshift": 0.0}
# the code's time unit: Mpc / (km/s) as in FLAMINGO, or another unit
TIME_UNITS = {"Mpc/(km/s)": MPC_CM / 1e5, "Gyr": 3.15576e16}


def write_snapshot(path, time_unit_s=TIME_UNITS["Mpc/(km/s)"], parameters=None, cosmology=COSMOLOGY):
    """
    Write a fake SWIFT snapshot with the metadata the cosmology is read from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        constants = f.create_group("PhysicalConstants/CGS")
        constants.attrs["parsec"] = [MPC_CM / 1e6]
        constants.attrs["solar_mass"] = [1.98841e33]
        constants.attrs["newton_G"] = [6.6743e-8]
        group = f.create_group("Cosmology")
        # H0 in the code's own units: 100 h km/s/Mpc times the time unit
        group.attrs["H0 [internal units]"] = [100. * H * 1e5 / MPC_CM * time_unit_s]
        for name, value in cosmology.items():
            group.attrs[name] = np.atleast_1d(value)
        for group_name in ("Units", "InternalCodeUnits"):
            units = f.create_group(group_name)
            units.attrs["Unit length in cgs (U_L)"] = [MPC_CM]
            units.attrs["Unit mass in cgs (U_M)"] = [1.98841e43]
            units.attrs["Unit time in cgs (U_t)"] = [time_unit_s]
            units.attrs["Unit temperature in cgs (U_T)"] = [1.]
            units.attrs["Unit current in cgs (U_I)"] = [1.]
        params = f.create_group("Parameters")
        for name, value in (parameters or {}).items():
            params.attrs[name] = np.bytes_(value)
    return path


@pytest.mark.parametrize("time_unit", TIME_UNITS)
def test_Hubble_const_from_the_code_units(tmp_path, time_unit):
    cosmo = SnapshotCosmology(write_snapshot(tmp_path / "flamingo_0077.hdf5", TIME_UNITS[time_unit]))
    # the same H0 whatever the code's time unit, in the snapshot's megaparsec
    assert cosmo.Hubble_const.to_value("km/s/swift_mpc") == pytest.approx(100. * H, rel=1e-12)
    assert cosmo.COSMO.H0.to_value("km/(Mpc s)") == pytest.approx(100. * H, rel=1e-12)
    # the unit registry is the snapshot's, with its code units, a and h
    assert unyt.unyt_quantity(1., "code_time", registry=cosmo.unit_registry).to_value("s") == TIME_UNITS[time_unit]
    assert unyt.unyt_quantity(1., "h", registry=cosmo.unit_registry).to_value("dimensionless") == pytest.approx(H)


def test_cosmology_parameters(tmp_path):
    cosmo = SnapshotCosmology(write_snapshot(tmp_path / "colibre_0127.hdf5"))
    expected = w0waCDM(H0=100. * H, Om0=0.306, Ode0=0.693922, Ob0=0.0486, w0=-1., wa=0., Tcmb0=2.7255)
    z = np.linspace(0., 5., 51)
    assert np.allclose(cosmo.COSMO.comoving_distance(z).value, expected.comoving_distance(z).value, rtol=1e-10)
    assert np.allclose(cosmo.comoving_distance(z).to_value("Mpc"), expected.comoving_distance(z).value, rtol=1e-10)
    # dt before z = 0.5 is the redshift where the universe is dt younger
    z_earlier = cosmo.redshift_with_time_offset(0.5, 100.)
    assert cosmo.redshift_with_time_offset(0.5, 0.1 * unyt.Gyr) == pytest.approx(z_earlier, rel=1e-8)


def test_same_as_lightcone_io(tmp_path):
    xray_utils = pytest.importorskip("lightcone_io.xray_utils")
    snapshot_dir = tmp_path / "snapshots"
    write_snapshot(snapshot_dir / "flamingo_0077" / "flamingo_0077.hdf5",parameters={"EAGLEAGN:AGN_delta_T_K": "3.162e7"}) # need to include for xray lightcone_io cosmology
    old = xray_utils.Snapshot_Cosmology_For_Lightcone(str(snapshot_dir))
    new = SnapshotCosmology.from_path(str(snapshot_dir))
    z = np.linspace(0.01, 5., 100)
    assert np.allclose(new.COSMO.comoving_distance(z).value, old.COSMO.comoving_distance(z).value, rtol=1e-8)


def test_find_snapshot_files(tmp_path):
    for name in ("flamingo_0076/flamingo_0076.hdf5", "flamingo_0077/flamingo_0077.0.hdf5",
                 "flamingo_0077/flamingo_0077.hdf5", "colibre_0127/colibre_0127.3.hdf5", "notes.txt",
                 "flamingo_77.hdf5"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    found = [path.split("/")[-1] for path in find_snapshot_files(str(tmp_path))]
    # sorted by snapshot number, the lowest redshift last; not the other files
    assert found == ["flamingo_0076.hdf5", "flamingo_0077.0.hdf5", "flamingo_0077.hdf5", "colibre_0127.3.hdf5"]
    with pytest.raises(FileNotFoundError, match="no snapshot files"):
        find_snapshot_files(str(tmp_path / "flamingo_0076"), pattern=r"^none_(\d)(\d{4})$")


@pytest.mark.parametrize("label, sim_name, suite", [("L1000N1800", "HYDRO_FIDUCIAL", "flamingo"),
                                                     ("L0200N1504", "Thermal", "colibre")])
def test_from_simulation(tmp_path, label, sim_name, suite):
    sim = Simulation.from_label(label, sim_name, base_dir_format=str(tmp_path) + "/{box_res}/{sim_name}")
    # the redshift 0 snapshot, named as the suite names its snapshots
    write_snapshot(tmp_path / sim.snapshot_file(sim.z0_snapshot))
    cosmo = SnapshotCosmology.from_simulation(sim)
    assert cosmo.snapshot_filename == sim.snapshot_file(sim.z0_snapshot)
    assert f"{suite}_{sim.z0_snapshot:04d}" in cosmo.snapshot_filename


def test_from_simulation_without_its_z0_snapshot(tmp_path):
    sim = Simulation.from_label("L0200N1504", "Thermal", base_dir_format=str(tmp_path) + "/{box_res}/{sim_name}")
    write_snapshot(tmp_path / sim.snapshot_file(100))
    write_snapshot(tmp_path / sim.snapshot_file(120))
    # the lowest redshift snapshot there is
    assert SnapshotCosmology.from_simulation(sim).snapshot_filename.endswith("colibre_0120.0.hdf5")

def test_volumes_and_shells(tmp_path):
    cosmo = SnapshotCosmology(write_snapshot(tmp_path / "flamingo_0077.hdf5"))
    whole_sky = cosmo.lightcone_volume(100., 200.)
    assert whole_sky.to_value("Mpc**3") == pytest.approx(4. / 3. * np.pi * (200.**3 - 100.**3))
    # a cone of half angle pi is the whole sky, of half angle pi / 2 half of it
    assert cosmo.lightcone_volume(100., 200., half_angle=np.pi) == pytest.approx(whole_sky)
    assert cosmo.lightcone_volume(100., 200., half_angle=np.pi / 2) == pytest.approx(whole_sky / 2)
    # from redshifts, or radii with units
    r = cosmo.comoving_distance([0.1, 0.2])
    assert cosmo.lightcone_volume(0.1, 0.2, use_redshift=True) == pytest.approx(cosmo.lightcone_volume(r[0], r[1]))
    assert cosmo.lightcone_volume(0.1 * unyt.Gpc, 0.2 * unyt.Gpc) == pytest.approx(whole_sky)

    shells = tmp_path / "shells.txt"
    shells.write_text("0.0, 0.05\n0.05, 0.1\n")
    radii = cosmo.shell_comoving_radii(str(shells))
    assert radii.shape == (2, 2) and radii.units == unyt.Mpc
    assert np.allclose(radii.to_value("Mpc"), cosmo.comoving_distance([[0., 0.05], [0.05, 0.1]]).to_value("Mpc"))

def test_not_a_swift_snapshot(tmp_path):
    path = tmp_path / "flamingo_0077.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("Header")
    with pytest.raises(KeyError):
        SnapshotCosmology(path)
    missing_w = {name: value for name, value in COSMOLOGY.items() if name != "w_0"}
    with pytest.raises(KeyError, match="Cosmology has no \\['w_0'\\]"):
        SnapshotCosmology(write_snapshot(tmp_path / "colibre_0127.hdf5", cosmology=missing_w))

def test_as_astropy_cosmology(tmp_path):
    path = write_snapshot(tmp_path / "snapshots" / "colibre_0127" / "colibre_0127.hdf5")
    cosmo = SnapshotCosmology(path)
    # a SnapshotCosmology, an astropy cosmology, or a snapshot file or directory to read one from
    assert as_astropy_cosmology(cosmo) is cosmo.COSMO
    assert as_astropy_cosmology(cosmo.COSMO) is cosmo.COSMO
    for source in (path, str(path), str(tmp_path / "snapshots")):
        assert as_astropy_cosmology(source).H0 == cosmo.COSMO.H0
    with pytest.raises(TypeError, match="must be a SnapshotCosmology"):
        as_astropy_cosmology(68.1)

def test_swiftlet_takes_any_cosmology(tmp_path):
    from swiftlet.snapshot_lightcone import define_redshift_at_comoving_distance_function
    from swiftlet.beam_plotting import BeamPlot
    cosmo = SnapshotCosmology(write_snapshot(tmp_path / "flamingo_0077.hdf5"))
    r = np.linspace(10., 3000., 50)
    expected = define_redshift_at_comoving_distance_function(cosmo.COSMO, 0.001, 1.2)(r)
    for given in (cosmo, cosmo.COSMO, str(tmp_path / "flamingo_0077.hdf5")):
        assert np.allclose(define_redshift_at_comoving_distance_function(given, 0.001, 1.2)(r), expected)
        assert BeamPlot(given, angular_diameter=10., redshift_range=(0.1, 0.5)).cosmology.H0 == cosmo.COSMO.H0


# FLAMINGO-like neutrinos: one massive species of 0.06 eV and 2.0328 massless ones, N_eff = 3.044. SWIFT's Omega_m is
# without the neutrinos, and Omega_lambda makes the universe flat with them
OMEGA_NU = 0.06 / 93.14 / H**2
NEUTRINOS = {"N_eff": 3.044, "N_ur": 2.0328, "M_nu_eV": [0.06], "deg_nu": [1.]}
# (Omega_r, which SWIFT writes too, is only read by swiftsimio, and only when T_CMB_0 is missing)
COSMOLOGY_WITH_NEUTRINOS = dict(COSMOLOGY, Omega_m=0.306 - OMEGA_NU, Omega_r=9.1e-5, **NEUTRINOS)


def swiftsimio_cosmology(path, time_unit_s=TIME_UNITS["Mpc/(km/s)"]):
    """
    swiftsimio's astropy cosmology of a fake snapshot, from its own converter.
    """
    from types import SimpleNamespace
    conversions = pytest.importorskip("swiftsimio.conversions")
    with h5py.File(path, "r") as f:
        cosmology = {name: np.atleast_1d(value) for name, value in f["Cosmology"].attrs.items()}
    return conversions.swift_cosmology_to_astropy(cosmology, SimpleNamespace(time=unyt.unyt_quantity(time_unit_s, "s")))


@pytest.mark.parametrize("neutrinos, masses", [
    ({}, [0., 0., 0.]),                                                         # no neutrino metadata: massless
    (NEUTRINOS, [0.06, 0., 0.]),
    ({"N_eff": 3.044, "N_ur": 0.0328, "M_nu_eV": [0.02, 0.06], "deg_nu": [2., 1.]}, [0.02, 0.02, 0.06]),
    ({"N_eff": 3.044, "N_ur": 3.044, "M_nu_eV": [0.], "deg_nu": [0.]}, [0., 0., 0.]),
])
def test_neutrino_masses(neutrinos, masses):
    assert neutrino_masses(dict(COSMOLOGY, **neutrinos)).tolist() == masses


def test_neutrinos_astropy_cannot_hold():
    with pytest.raises(ValueError, match="only whole numbers"):
        neutrino_masses(dict(NEUTRINOS, deg_nu=[1.5]))
    with pytest.raises(ValueError, match="don't add up|but N_eff"):
        neutrino_masses(dict(NEUTRINOS, N_ur=3.044))


@pytest.mark.parametrize("time_unit", TIME_UNITS)
def test_neutrinos_as_swiftsimio(tmp_path, time_unit):
    path = write_snapshot(tmp_path / "flamingo_0077.hdf5", TIME_UNITS[time_unit], cosmology=COSMOLOGY_WITH_NEUTRINOS)
    cosmo = SnapshotCosmology(path)
    expected = swiftsimio_cosmology(path, TIME_UNITS[time_unit])
    assert cosmo.COSMO.m_nu.to_value("eV").tolist() == expected.m_nu.to_value("eV").tolist() == [0.06, 0., 0.]
    assert cosmo.COSMO.Neff == expected.Neff == 3.044
    z = np.linspace(0., 5., 51)
    assert np.allclose(cosmo.COSMO.comoving_distance(z).value, expected.comoving_distance(z).value, rtol=1e-9)
    # the massive neutrino is matter today, so the universe is flat with it. astropy's Onu0 counts every neutrino,
    # so the massive one is what it has more than with all massless
    assert abs(cosmo.COSMO.Ok0) < 1e-4
    massless = w0waCDM(H0=100. * H, Om0=0.306 - OMEGA_NU, Ode0=0.693922, Ob0=0.0486, w0=-1., wa=0., Tcmb0=2.7255,
                       Neff=3.044, m_nu=[0., 0., 0.])
    assert cosmo.COSMO.Onu0 - massless.Onu0 == pytest.approx(OMEGA_NU, rel=0.02)


def test_neutrinos_change_the_distances(tmp_path):
    with_neutrinos = SnapshotCosmology(write_snapshot(tmp_path / "a" / "flamingo_0077.hdf5", cosmology=COSMOLOGY_WITH_NEUTRINOS))
    # the same metadata without the neutrinos: the 0.0014 of the massive neutrino becomes curvature
    without = SnapshotCosmology(write_snapshot(tmp_path / "b" / "flamingo_0077.hdf5",
                                               cosmology=dict(COSMOLOGY, Omega_m=0.306 - OMEGA_NU)))
    assert without.COSMO.Ok0 == pytest.approx(OMEGA_NU, rel=0.1)
    d_with = with_neutrinos.comoving_distance(1.).to_value("Mpc")
    d_without = without.comoving_distance(1.).to_value("Mpc")
    assert 1e-5 < abs(d_with - d_without) / d_with < 1e-2


