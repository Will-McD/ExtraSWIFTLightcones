#!/bin/env python
import os
import re
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from importlib import resources
from pathlib import Path
from . import snapshot_redshifts

"""
SimulationSuite:    what is the same for every simulation of a suite, FLAMINGO and COLIBRE
Simulation:         one simulation, its suite, box size and resolution label, name and directory   

Note that the base directory of each suite's simulations are checked in this order:
    1. base_dir_format, if given to Simulation.from_label,
    2. the suite's environment variable (SWIFTLET_FLAMINGO_DIR or SWIFTLET_COLIBRE_DIR), the directory holding
        the simulations, each in <box_res>/<sim_name>,
    3. the suite's directory on COSMA.
"""

# the snapshot output redshift tables, see read_output_redshifts
OUTPUT_REDSHIFT_DIR = resources.files(__package__) / "data" / "output_redshifts"

# the columns of an output redshift table
OUTPUT_REDSHIFT_COLUMNS = ("snapshot_number", "redshift")

# make globally cached and callable read redshift tables function
@lru_cache(maxsize=None)
def _read_output_redshift_file(path):
    """
    Read an output redshift file. Cached so each file is read once.
    """
    source = OUTPUT_REDSHIFT_DIR / path if os.path.basename(path) == path else Path(path)
    if path.lower().endswith((".yaml", ".yml")):
        from .table_files import read_table_file
        _, table = read_table_file(source, OUTPUT_REDSHIFT_COLUMNS)
    else:
        with source.open("r") as infile:
            table = np.loadtxt(infile, delimiter=",", comments="#", ndmin=2)
        if table.shape[1] != 2:
            raise ValueError(f"{path}: expected 2 comma separated columns, snapshot_number and redshift, "
                             f"not {table.shape[1]}")
    snapshot_numbers, redshifts = table[:, 0], table[:, 1].astype(float)
    if not np.array_equal(snapshot_numbers, np.arange(len(table))):
        raise ValueError(f"{path}: the snapshot numbers must be 0, 1, 2, ... in order")
    return tuple(redshifts)

# outerlayer of redshift .txt file reader
def read_output_redshifts(path):
    """
    Read the output redshifts of a simulation's snapshots from a text file of comma separated values, one snapshot
    per line, "snapshot_number, redshift", with the snapshot numbers 0, 1, 2, ... in order. 
    Lines starting with '#' are comments.

    Returns an np.ndarray of the redshifts, in order of snapshot number.

    :param  path:   a file name, for the tables shipped with the package (in OUTPUT_REDSHIFT_DIR), or the path of a
                        file of your own
    :type   path:   str or os.PathLike
    """
    return np.array(_read_output_redshift_file(str(path)), dtype=float)


def write_output_redshifts(filename, redshifts, description=None, **header):
    """
    Write the output redshifts of a simulation's snapshots to a YAML file, 
    one row per snapshot [snapshot_number, redshift]. 
    The redshifts are written so they read back exactly.

    :param  filename:       path of the .yaml file to write
    :type   filename:       str or os.PathLike
    :param  redshifts:      redshift of each snapshot, in order of snapshot number 0, 1, 2, ...
    :type   redshifts:      array-like of float
    :param  description:    what the redshifts are, e.g. the simulations they are for
    :type   description:    str
    :param  header:         more to write before the rows, e.g. suite="colibre"
    """
    from .table_files import write_table_file
    redshifts = np.asarray(redshifts, dtype=float)
    rows = np.empty((len(redshifts), 2), dtype=object)
    rows[:, 0], rows[:, 1] = np.arange(len(redshifts)), redshifts
    header = {**({"description": description} if description else {}), **header, "number_of_snapshots": len(redshifts)}
    write_table_file(filename, header, OUTPUT_REDSHIFT_COLUMNS, rows, comment="SWIFTLET output redshifts of the snapshots, see docs/simulation_suites.md")

###################################################################################################
# build FLAMINGO and COLIBRE simulation suites.
# These are essentially objects to contain all the paths to snapshots and relevenat redshifts of each snapshot

@dataclass(frozen=True) # do not allow this to be mutated
class SimulationSuite:
    """
    Class to contain SWIFT simulation suites (i.e. FLAMINGO and COLIBRE) and handle what differs between them. 

    :param  name:                       "flamingo" or "colibre"
    :type   name:                       str
    :param  base_dir_format:            directory of a simulation, with {box_res} and {sim_name} fields
    :type   base_dir_format:            str
    :param  base_dir_env:               environment variable that, if set, replaces the directory holding the simulations
    :type   base_dir_env:               str
    :param  snapshot_format:            snapshot file, relative to the simulation's directory, with {snap_nr} and {file_nr}
                                            fields
    :type   snapshot_format:            str
    :param  output_redshift_tables:     file of the output redshifts of the snapshots (see read_output_redshifts), for
                                            each box size and resolution label; the file under "*" is used for any
                                            label of the suite
    """
    
    # define types of each element
    name: str
    label_pattern: str
    base_dir_format: str
    base_dir_env: str
    snapshot_format: str
    output_redshift_tables: dict = field(compare=False, hash=False, repr=False)

    def recognises(self, boxsize_resolution):
        """
        Check if a box size and resolution label is recognised by the simulation suite.

        :param  boxsize_resolution: box size and resolution label, e.g. "L1000N1800"
        :type   boxsize_resolution: str
        """
        return boxsize_resolution is not None and re.fullmatch(self.label_pattern, str(boxsize_resolution)) is not None

    def output_redshifts(self, boxsize_resolution=None):
        """
        Output redshifts of the snapshots of the suite's simulations of a box size and resolution, in order of
        snapshot number.

        Returns an np.ndarray.

        :param  boxsize_resolution: box size and resolution label, e.g. "L1000N1800". Not needed for a suite with
                                        one table for every label (COLIBRE)
        :type   boxsize_resolution: str
        """
        tables = self.output_redshift_tables
        filename = tables.get(boxsize_resolution, tables.get("*"))
        if filename is None:
            raise ValueError(f"boxsize_resolution {boxsize_resolution!r} not recognised: {self.name.upper()} has output redshifts for {sorted(tables)}")
        return read_output_redshifts(filename)

    def directory_format(self):
        """
        Base directory of a simulation. 
        
        Returns a str of the formatted path to the simulations base directory. 
            Formatted path contains {box_res} and {sim_name} fields: from the suite's environment variable if set, otherwise base_dir_format.
        
        """
        base_dir = os.environ.get(self.base_dir_env)
        if base_dir:
            return os.path.join(base_dir, "{box_res}", "{sim_name}")
        return self.base_dir_format


# define FLAMINGO simulation suite
FLAMINGO = SimulationSuite(
    name="flamingo",
    label_pattern=r"L1000N0900|L1000N1800|L1000N3600|L2800N5040",
    base_dir_format="/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}",
    base_dir_env="SWIFTLET_FLAMINGO_DIR",
    snapshot_format="snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5",
    output_redshift_tables={
        "L1000N0900": "flamingo_L1.yaml",
        "L1000N1800": "flamingo_L1.yaml",
        "L1000N3600": "flamingo_L2p8.yaml",
        "L2800N5040": "flamingo_L2p8.yaml",
        },
    )

# define COLIBRE simulation suite
COLIBRE = SimulationSuite(
    name="colibre",
    # the box size zero padded to 4 digits, e.g. "L0200N1504", or with the mass resolution, e.g. "L200m6", "L200_m6"
    label_pattern=r"L0\d{3}N\d{4}|L\d{3}_?m\d",
    base_dir_format="/cosma8/data/dp004/colibre/Runs/{box_res}/{sim_name}",
    base_dir_env="SWIFTLET_COLIBRE_DIR",
    snapshot_format="snapshots/colibre_{snap_nr:04d}/colibre_{snap_nr:04d}.{file_nr}.hdf5",
    output_redshift_tables={"*": "colibre.yaml"},
)


SUITES = {suite.name: suite for suite in (FLAMINGO, COLIBRE)}


###################################################################################################
# define simulation suite functions

def get_suite(suite):
    """
    Get the SimulationSuite class of a pre-defined swift simulation suite (e.g. "flamingo" or "colibre") from the suites its name. 

    Returns the SimulationSuite object 

    :param  suite:  name of the suite, or a SimulationSuite
    :type   suite:  str or SimulationSuite
    """
    if isinstance(suite, SimulationSuite):
        return suite
    name = str(suite).lower()
    if name not in SUITES:
        raise ValueError(f"simulation_suite_name must be one of {tuple(SUITES)}, not {suite!r}")
    return SUITES[name]


def suite_for_label(boxsize_resolution=None, simulation_suite_name=None):
    """
    The suite is either given as a parameter, or recognised by matching labels with the boxsize_resolution.
    Raises ValueError if the suite can't be found, or if the named suite and the label's disagree.

    :param  boxsize_resolution:     box size and resolution label, e.g. "L1000N1800" (FLAMINGO) or "L0200N1504"
                                        (COLIBRE)
    :type   boxsize_resolution:     str
    :param  simulation_suite_name:  "flamingo" or "colibre" (or a SimulationSuite). If None, found from
                                        boxsize_resolution
    :type   simulation_suite_name:  str or SimulationSuite
    """
    named = None if simulation_suite_name is None else get_suite(simulation_suite_name)
    matching = [suite for suite in SUITES.values() if suite.recognises(boxsize_resolution)]
    from_label = matching[0] if len(matching) == 1 else None

    if named is None:
        if from_label is None:
            patterns = ", ".join(f"{suite.name.upper()}: {suite.label_pattern}" for suite in SUITES.values())
            raise ValueError(f"can't tell which simulation suite boxsize_resolution={boxsize_resolution!r} belongs "
                             f"to ({patterns}). Pass simulation_suite_name='flamingo' or 'colibre'")
        return from_label
    # compared by value (output_redshift_tables aside), so a suite made with dataclasses.replace, e.g. with output
    # redshifts of your own, is taken for the labels of the suite it was made from
    if from_label is not None and from_label != named:
        raise ValueError(f"boxsize_resolution {boxsize_resolution!r} is a {from_label.name.upper()} simulation, but simulation_suite_name is {named.name!r}")
    return named


def output_redshifts(boxsize_resolution, simulation_suite_name=None):
    """
    Output redshifts of the snapshots of a box size and resolution, as read from its snapshot number - redshift tables, 
    for when only the redshifts are needed (otherwise use Simulation.output_redshifts).

    Returns an np.ndarray, in order of snapshot number.

    :param  boxsize_resolution:     box size and resolution label, e.g. "L1000N1800" (FLAMINGO) or "L0200N1504"
                                        (COLIBRE)
    :type   boxsize_resolution:     str
    :param  simulation_suite_name:  "flamingo" or "colibre". If None, found from boxsize_resolution
    :type   simulation_suite_name:  str
    """
    
    return suite_for_label(boxsize_resolution, simulation_suite_name).output_redshifts(boxsize_resolution)


###################################################################################################

@dataclass(frozen=True)
class Simulation:
    """
    For one simulation: its suite name, box size and resolution str, model name and base directory. 
    
    For every simulation used, every path to a snapshot and every snapshot number - redshift conversion comes from this class. 
    This class allows for consistent use of snapshot number - redshift conversions and pathing for a swift simulation suite 

    :param  suite:              the simulation's suite, FLAMINGO or COLIBRE
    :type   suite:              SimulationSuite
    :param  box_res:            box size and resolution label, e.g. "L1000N1800"
    :type   box_res:            str
    :param  name:               name of the simulation, e.g. "HYDRO_FIDUCIAL"
    :type   name:               str
    :param  directory:          directory of the simulation
    :type   directory:          str
    """
    suite: SimulationSuite
    box_res: str
    name: str
    directory: str

    @classmethod
    def from_label(cls, boxsize_resolution, simulation_name, simulation_suite_name=None, base_dir_format=None):
        """
        Get a simulation suite name from its box size and resolution str and name.

        :param  boxsize_resolution: box size and resolution label, e.g. "L1000N1800" (FLAMINGO) or "L0200N1504"
                                        (COLIBRE)
        :type   boxsize_resolution: str
        :param  simulation_name:    name of the simulation, e.g. "HYDRO_FIDUCIAL"
        :type   simulation_name:    str
        :param  simulation_suite:   "flamingo" or "colibre" (or a SimulationSuite). If None, found from
                                        boxsize_resolution
        :type   simulation_suite:   str or SimulationSuite
        :param  base_dir_format:    directory of the simulation, with {box_res} and {sim_name} fields. 
                                        If None, the suite's directory, see SimulationSuite.directory_format
        :type   base_dir_format:    str
        """
        
        #if isinstance(simulation_suite, SimulationSuite):
        #    simulation_suite = simulation_suite.name
        
        # find the suite
        #suite = get_suite(nz.simulation_suite(boxsize_resolution, simulation_suite=simulation_suite))
        suite = suite_for_label(boxsize_resolution, simulation_suite_name)
        # make formatted path to base dir
        directory_format = suite.directory_format() if base_dir_format is None else base_dir_format
        directory = directory_format.format(box_res=boxsize_resolution, sim_name=simulation_name)

        return cls(suite=suite, box_res=boxsize_resolution, name=simulation_name, directory=directory)

    @classmethod
    def from_path(cls, path, simulation_suite_name=None):
        """
        Get a simulation from its directory, <...>/<box_res>/<sim_name>,  e.g. "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL". 
        The suite is found from the box size and resolution label, or from "flamingo" or "colibre" in the path.

        :param  path:                   directory of the simulation
        :type   path:                   str
        :param  simulation_suite_name:  "flamingo" or "colibre" (or a SimulationSuite). If None, found from the path
        :type   simulation_suite_name:  str or SimulationSuite
        """
        directory = os.path.normpath(str(path))
        box_res = os.path.basename(os.path.dirname(directory))
        sim_name = os.path.basename(directory)
        if simulation_suite_name is None:
            in_path = [name for name in SUITES if re.search(rf"(^|/){name}(/|$)", directory.lower())]
            if len(in_path) == 1:
                simulation_suite_name = in_path[0]
        suite = suite_for_label(box_res, simulation_suite_name)
        return cls(suite=suite, box_res=box_res, name=sim_name, directory=directory)


    @property
    def snapshot_format(self):
        """
        Formatted path of the snapshot files, with {snap_nr} and {file_nr} fields.
        """
        return os.path.join(self.directory, self.suite.snapshot_format)

    @property
    def snapshot_dir(self):
        """
        Base directory of the snapshots.
        """
        return os.path.join(self.directory, "snapshots")

    def snapshot_file(self, snap_nr, file_nr=0):
        """
        Path of a snapshot file.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        :param  file_nr:    file number of the snapshot
        :type   file_nr:    int
        """
        return self.snapshot_format.format(snap_nr=int(snap_nr), file_nr=int(file_nr))

    # make sure to cache output redshifts once read
    @cached_property
    def output_redshifts(self):
        """
        Output redshift of each snapshot, in order of snapshot number.
        """
        return self.suite.output_redshifts(self.box_res)
    
    def snapshot_redshifts(self):
        """
        Snapshot numbers and their redshifts.

        Returns a tuple of (snapshot numbers, redshifts).
        """
        return np.arange(len(self.output_redshifts)), self.output_redshifts.copy()

    def snapshot_redshift(self, snap_nr):
        """
        Output redshift of a snapshot (or of an array of snapshots).

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int or np.ndarray
        """
        return self.output_redshifts[snap_nr]

    def snapshot_number(self, redshift):
        """
        Number of the snapshot at a redshift, which must be one of the output redshifts.

        :param  redshift:   redshift of the snapshot
        :type   redshift:   float
        """
        return snapshot_redshifts.snapshot_at_redshift(self.output_redshifts, redshift)

    def snapshot_redshift_range(self, snap_nr):
        """
        Redshift range a snapshot covers.

        Returns a tuple of (minimum redshift, maximum redshift).

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        """
        return snapshot_redshifts.snapshot_redshift_range(self.output_redshifts, snap_nr)

    def snapshots_in_range(self, redshift_range, range_tol=1e-5, strict=False):
        """
        Snapshots whose redshift ranges overlap a redshift range, including ranges that only touch it.

        Returns an np.ndarray of snapshot numbers, increasing (from high to low redshift).

        :param  redshift_range: minimum and maximum redshifts [z_min, z_max]
        :type   redshift_range: sequence of two floats
        """
        return snapshot_redshifts.snapshots_in_range(self.output_redshifts, redshift_range=redshift_range, range_tol=range_tol, strict=strict)

    @property
    def z0_snapshot(self):
        """
        Number of the snapshot at redshift 0.
        """
        return self.snapshot_number(0.)


