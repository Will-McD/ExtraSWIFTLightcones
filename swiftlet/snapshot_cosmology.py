#!/bin/env python
"""
The cosmology of a SWIFT simulation (FLAMINGO or COLIBRE), read from one of its snapshots, with the snapshot's own
unit registry.
Adapted from the LightconeIO lightcone_io/xray_utils.py

The cosmology is the astropy w0waCDM of the snapshot's Cosmology group: H0, Omega_m, Omega_lambda, Omega_b, w_0,
w_a and T_CMB_0. 
"""
import os
import re
import h5py
import numpy as np
import unyt
from astropy.cosmology import FLRW, w0waCDM, z_at_value

from . import snapshot_units as sw_units

# snapshot files of the simulation suites: <suite>_<4 digits>.hdf5, or the files of a snapshot,
# <suite>_<4 digits>.<file number>.hdf5
SNAPSHOT_FILE_PATTERN = re.compile(r"^(flamingo|colibre)_(\d{4})(\.\d+)?\.hdf5$")


def find_snapshot_files(snapshot_dir, pattern=SNAPSHOT_FILE_PATTERN):
    """
    Returns a list of paths, sorted by snapshot number then file name, so the lowest redshift snapshot is last.

    :param  snapshot_dir:   directory of the snapshots, e.g. "<simulation>/snapshots"
    :type   snapshot_dir:   str
    :param  pattern:        regular expression matching the snapshot file names, with the snapshot number as its
                                second group (by default FLAMINGO and COLIBRE snapshot files)
    :type   pattern:        re.Pattern or str
    """
    pattern = re.compile(pattern)
    found = []
    for root, _, files in os.walk(snapshot_dir):
        for name in files:
            match = pattern.match(name)
            if match:
                found.append((int(match.group(2)), name, os.path.join(root, name)))
    if not found:
        raise FileNotFoundError(f"no snapshot files matching {pattern.pattern!r} in {snapshot_dir}")
    return [path for _, _, path in sorted(found)]


def _attributes(snapshot, group_name):
    """
    Read the attributes of a group of a snapshot file, one-element arrays as numbers and bytes as strings.
    """
    if group_name not in snapshot:
        raise KeyError(f"{snapshot.filename} doesn't have a group called {group_name}")
    values = {}
    for name, value in snapshot[group_name].attrs.items():
        if isinstance(value, bytes): # if byte format then decode to str
            value = value.decode() 
        elif isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0] # if array, it is only 1 sized array. 
            value = value.decode() if isinstance(value, bytes) else value.item() # decode if array item in bytes format
        values[name] = value
    return values


def neutrino_masses(cosmology):
    """
    Convert SWIFT neutrino masses to astropy format. 

    Returns an np.ndarray of masses [eV], one per species (int(N_eff) of them).

    Raises ValueError for neutrinos astropy can't hold. i.e. degeneracies that aren't whole numbers, or massive and
    massless species that don't add up to N_eff.

    :param  cosmology:  Cosmology attributes of the snapshot (see SnapshotCosmology.cosmology)
    :type   cosmology:  dict
    """
    n_eff = float(cosmology.get("N_eff", 3.04)) # try read effective number of neutrino, otherwise fall back to astropy default
    masses = np.atleast_1d(np.asarray(cosmology.get("M_nu_eV", 0.), dtype=float)) # if no mass found, fall back to 0.0
    degeneracies = np.atleast_1d(np.asarray(cosmology.get("deg_nu", 0.), dtype=float)) # repeat for degeneracies
    n_massless = float(np.squeeze(cosmology.get("N_ur", n_eff))) # numb of masses species
    if not np.all(degeneracies == np.round(degeneracies)): 
        raise ValueError(f"astropy can't hold neutrinos of degeneracies {degeneracies} (deg_nu), only whole numbers")
    if int(n_eff) != int(degeneracies.sum()) + int(n_massless):
        raise ValueError(f"astropy can't hold these neutrinos: {int(degeneracies.sum())} massive (deg_nu) and {int(n_massless)} massless (N_ur) species, but N_eff = {n_eff}")
    massive = np.repeat(masses, degeneracies.astype(int)) # store the species of massive neutrinos
    return np.concatenate([massive, np.zeros(int(n_massless))])


class SnapshotCosmology:
    """
    The cosmology from one of a snapshot .hdf5 file or of simulation class object (see swiftlet/simulations_suites.py)

    Attributes:
        snapshot_filename:  the snapshot file read
        COSMO:              astropy w0waCDM cosmology
        cosmology:          the snapshot's Cosmology attributes
        internal_units:     the snapshot's InternalCodeUnits attributes (CGS conversion factors)
        parameters:         the snapshot's Parameters attributes (the run's parameter file), {} if it has none
        unit_registry:      unyt registry of the snapshot's units (snap_* and code_* units, a and h, see
                                snapshot_units.unit_registry_from_metadata)
    """

    def __init__(self, snapshot_filename):
        """
        Read the cosmology of a snapshot file.

        :param  snapshot_filename:  path of a SWIFT snapshot file (a virtual file or one file of a snapshot)
        :type   snapshot_filename:  str
        """
        self.snapshot_filename = str(snapshot_filename)
        with h5py.File(self.snapshot_filename, "r") as snapshot:
            self.unit_registry = sw_units.unit_registry_from_snapshot(snapshot)
            self.cosmology = _attributes(snapshot, "Cosmology")
            self.internal_units = _attributes(snapshot, "InternalCodeUnits")
            self.parameters = _attributes(snapshot, "Parameters") if "Parameters" in snapshot else {}

        required_cosmo_params = ("H0 [internal units]", "Omega_m", "Omega_lambda", "Omega_b", "w_0", "w_a", "T_CMB_0 [K]")
        missing = [name for name in required_cosmo_params if name not in self.cosmology]
        
        if missing:
            raise KeyError(f"{self.snapshot_filename}: Cosmology has no {missing}")

        self.COSMO = w0waCDM(
            H0=self.Hubble_const.to_astropy(),
            Om0=self.cosmology["Omega_m"],  # Omega_m in SWIFT = cold dark matter and baryons, without the massive neutrinos
            Ode0=self.cosmology["Omega_lambda"],
            Ob0=self.cosmology["Omega_b"],
            w0=self.cosmology["w_0"],
            wa=self.cosmology["w_a"],
            Tcmb0=unyt.unyt_quantity(self.cosmology["T_CMB_0 [K]"], "K").to_astropy(),
            Neff=self.cosmology.get("N_eff", 3.04),
            m_nu=unyt.unyt_array(neutrino_masses(self.cosmology), units=unyt.eV).to_astropy(),
        )

    @classmethod
    def from_simulation(cls, simulation):
        """
        The cosmology of a Simulation, from its redshift 0 snapshot, or its lowest redshift snapshot file if that
        isn't there.

        :param  simulation: the simulation
        :type   simulation: swiftlet.simulations.Simulation
        """
        filename = simulation.snapshot_file(simulation.z0_snapshot, 0)
        if os.path.isfile(filename):
            return cls(filename)
        return cls(find_snapshot_files(simulation.snapshot_dir)[-1])

    @classmethod
    def from_path(cls, path, pattern=SNAPSHOT_FILE_PATTERN):
        """
        The cosmology from a snapshot file, or from the lowest redshift snapshot file in a directory.

        :param  path:       a snapshot file, or a directory of snapshots, e.g. "<simulation>/snapshots"
        :type   path:       str
        :param  pattern:    regular expression matching the snapshot file names in a directory, see
                                find_snapshot_files
        :type   pattern:    re.Pattern or str
        """
        if os.path.isdir(path):
            return cls(find_snapshot_files(path, pattern)[-1])
        return cls(path)

    @property
    def Hubble_const(self):
        """
        Hubble constant H0, unyt_quantity [km/s/Mpc], converted from the code's time unit.
        """
        h0 = unyt.unyt_quantity(self.cosmology["H0 [internal units]"], "1/code_time", registry=self.unit_registry)
        return h0.to("km/s/Mpc")

    def parameter(self, name):
        """
        A parameter of the run, from the snapshot's Parameters group, e.g. "EAGLEAGN:AGN_delta_T_K".

        Returns the value as written (a string), or None if the snapshot doesn't have it.

        :param  name:   name of the parameter, "<section>:<parameter>"
        :type   name:   str
        """
        return self.parameters.get(name)

    @property
    def AGN_delta_T_K(self):
        """
        Temperature change of gas heated by AGN feedback, unyt_quantity [K], from the thermal (EAGLEAGN) or jet
        (SPINJETAGN) AGN model of the run, or None if the run has neither.
        """
        for name in ("EAGLEAGN:AGN_delta_T_K", "SPINJETAGN:AGN_delta_T_K"):
            value = self.parameter(name)
            if value is not None:
                return unyt.unyt_quantity(float(value), "K")
        return None

    def comoving_distance(self, z):
        """
        Comoving distance to redshift(s), unyt_array [Mpc].

        :param  z:  redshift(s)
        :type   z:  float or array-like
        """
        return unyt.unyt_array.from_astropy(self.COSMO.comoving_distance(z)).to("Mpc")

    def redshift_with_time_offset(self, z, dt):
        """
        Redshift a time dt before redshift z.

        Returns the redshift as a float.

        :param  z:  redshift
        :type   z:  float
        :param  dt: time before z [Myr], or a unyt quantity of time
        :type   dt: float or unyt.unyt_quantity
        """
        dt = sw_units.apply_expected_units(dt, unyt.Myr)
        earlier = self.age(z) - dt
        return float(z_at_value(self.COSMO.age, earlier.to_astropy()).value)

    def shell_comoving_radii(self, redshift_filename):
        """
        Comoving radii of lightcone shells from a file of their redshifts.

        Returns a unyt_array [Mpc] of shape (number of shells, 2), [inner, outer] for each shell.

        :param  redshift_filename:  path of the shell redshift file
        :type   redshift_filename:  str
        """
        shell_redshifts = np.loadtxt(redshift_filename, delimiter=",", ndmin=2)
        return self.comoving_distance(shell_redshifts)

    def lightcone_volume(self, inner, outer, use_redshift=False, half_angle=None):
        """
        Comoving volume of a lightcone shell between two radii, over the whole sky or in a cone.

        Returns a unyt_quantity (or unyt_array) [Mpc^3].

        :param  inner:          inner comoving radius [Mpc], or redshift if use_redshift is True
        :type   inner:          float, np.ndarray or unyt.unyt_array
        :param  outer:          outer comoving radius [Mpc], or redshift if use_redshift is True
        :type   outer:          float, np.ndarray or unyt.unyt_array
        :param  use_redshift:   If True, inner and outer are redshifts
        :type   use_redshift:   boolean
        :param  half_angle:     half the opening angle of the cone [rad]. If None, the whole sky
        :type   half_angle:     float
        """
        if use_redshift:
            inner, outer = self.comoving_distance(inner), self.comoving_distance(outer)
        else:
            inner = sw_units.apply_expected_units(inner, unyt.Mpc)
            outer = sw_units.apply_expected_units(outer, unyt.Mpc)
        # the solid angle of the sky, or of a cone: 2 pi (1 - cos(half_angle))
        solid_angle = 4. * np.pi if half_angle is None else 2. * np.pi * (1. - np.cos(half_angle))
        return (solid_angle / 3. * (outer**3 - inner**3)).to("Mpc**3")


def as_astropy_cosmology(cosmology):
    """
    Make all cosmologies used consistently transformed too astropy format,
    handle reading cosmology as done in LightconeIO, from a hdf5 group and a Simulation object. 

    Returns an astropy cosmology.

    :param  cosmology:  the cosmology
    :type   cosmology:  SnapshotCosmology, astropy.cosmology.FLRW, str or os.PathLike
    """
    if isinstance(cosmology, FLRW):
        return cosmology
    if isinstance(getattr(cosmology, "COSMO", None), FLRW):
        return cosmology.COSMO
    if isinstance(cosmology, (str, os.PathLike)):
        return SnapshotCosmology.from_path(cosmology).COSMO
    raise TypeError(f"cosmology must be a SnapshotCosmology, an astropy cosmology or the path of a snapshot, not {type(cosmology).__name__}")

