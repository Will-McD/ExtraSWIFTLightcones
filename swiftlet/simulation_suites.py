#!/bin/env python
"""
The SWIFT simulation suites (FLAMINGO and COLIBRE) and their simulations: where their files are and the redshifts they
were output at.

Every path to a snapshot or SOAP catalogue and every conversion between snapshot numbers and redshifts comes from a
:class:`~swiftlet.simulation_suites.Simulation`, so they are the same wherever they are used.

- ``SimulationSuite``: track information that is the same for every simulation of a suite
- ``Simulation``: one simulation, with its suite, box size and resolution label, name and directory

The suites are read from YAML files: FLAMINGO's and COLIBRE's are installed with SWIFTLET (in
``swiftlet/data/simulation_suites``), and the files in the directories listed in ``SWIFTLET_SUITES_DIR`` add new suites
or update (and replace) an installed one of the same name. See :func:`~swiftlet.simulation_suites.read_suite_file`.

The base directory of each suite's simulations is taken from

1. the base directory (``base_dir_format``) if given to a :meth:`~swiftlet.simulation_suites.Simulation` object
2. the suite's environment variable (``SWIFTLET_FLAMINGO_DIR`` or ``SWIFTLET_COLIBRE_DIR``), the directory holding the
   simulations, each in ``<box_res>/<sim_name>``
3. the suite's directory on COSMA

For example::

    from swiftlet import Simulation

    simulation = Simulation.from_label("L1000N1800", "HYDRO_FIDUCIAL")  # the suite is found from the label
    simulation.snapshot_file(77, 0)  # path of file 0 of snapshot 77
    simulation.soap_file(77)  # path of the SOAP catalogue of snapshot 77
    simulation.snapshot_redshift(77)  # its output redshift

"""

import os
import re
import numpy as np
import string
import yaml
from dataclasses import dataclass, field, fields, replace
from functools import cached_property, lru_cache
from importlib import resources
from pathlib import Path
from . import snapshot_redshifts


# every swift sim needs, use for redshifts of each snapshot
OUTPUT_LIST_NAME = "output_list.txt"

# universal format of soap catalogues from the simulations root directory
SOAP_FORMAT = "SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"

# snapshot files of a suite that doesn't give its own snapshot_format
SNAPSHOT_FORMAT = "snapshots/snap_{snap_nr:04d}/snap_{snap_nr:04d}.{file_nr}.hdf5"

# the directory of a remote simulation, relative to the remote directory, unless base_dir_format is given
REMOTE_DIR_FORMAT = "{box_res}/{sim_name}"

# directories of suite files (.yaml files), separated by os.pathsep. A .yaml added to this directory adds a new suite or
# replaces the installed one of the same name
SUITES_DIR_ENV = "SWIFTLET_SUITES_DIR"

# the fields each path format may have, and those it must have (the output list is a plain path)
_PATH_FIELDS = {
    "base_dir_format": ({"box_res", "sim_name"}, set()),
    "snapshot_format": ({"snap_nr", "file_nr"}, {"snap_nr"}),
    "soap_format": ({"snap_nr"}, {"snap_nr"}),
}


def _format_fields(path_format):
    """The names of the {fields} of a format string."""
    return {field for _, field, _, _ in string.Formatter().parse(path_format) if field is not None}


def _listed(fields):
    """Field names written as {name}, for error messages."""
    return ", ".join("{" + field + "}" for field in sorted(fields)) or "none"


def _check_path_format(name, path_format):
    """
    Raise ValueError if a path format has fields it may not have or lacks those it needs.

    :param  name:           the field of SimulationSuite, e.g. "soap_format"
    :type   name:           str
    :param  path_format:    the path format, e.g. "SOAP/halo_properties_{snap_nr:04d}.hdf5"
    :type   path_format:    str
    """
    if not isinstance(path_format, (str, os.PathLike)):
        raise TypeError(f"{name} must be a path, not {path_format!r}")

    allowed, needed = _PATH_FIELDS[name]

    try:
        fields = _format_fields(os.fspath(path_format))
    except ValueError as err:
        raise ValueError(f"{name}={path_format!r} is not a valid format: {err}") from None

    problems = []

    if needed - fields:
        problems.append(f"needs {_listed(needed - fields)}")

    if fields - allowed:
        problems.append(f"may not have {_listed(fields - allowed)}")

    if problems:
        raise ValueError(f"{name}={path_format!r} {' and '.join(problems)} (its fields are {_listed(allowed)})")


# make globally cached and callable read redshift tables function
@lru_cache(maxsize=None)
def _read_output_list_file(path):
    """Read an output list once, and cache its redshifts."""
    # if file not found raise FileNotFoundError, redshifts are needed for all lightcones
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no output list at {path}: SWIFTLET reads the output redshifts of a simulation's snapshots from "
            f"{OUTPUT_LIST_NAME}"
        )

    # open file and read lines
    with open(path, "r") as infile:
        return _parse_output_list(list(infile), path)


def _parse_output_list(lines, path):
    """
    Returns the redshifts of an output list, as a tuple, from the lines of its file.

    :param  lines:  the lines of the file
    :type   lines:  list
    :param  path:   path of the file, for the error messages
    :type   path:   str
    """
    lines = [line.strip() for line in lines]

    lines = [line for line in lines if line]

    # if no Redshift header assume the file is incorrect
    if not lines or not lines[0].startswith("#"):
        raise ValueError(f"{path}: an output list starts with a header line, e.g. '# Redshift'")

    # read redshifts from the first column only
    first_column = lines[0].lstrip("#").split(",")[0].strip().lower()

    if first_column not in ("redshift", "scale factor"):
        raise ValueError(
            f"{path}: the first column of an output list must be 'Redshift' or 'Scale Factor', not "
            f"{lines[0].lstrip('#').split(',')[0].strip()!r}"
        )
    try:
        # ignore the second column, assume redshifts are always the first column of the output list
        values = np.array([float(line.split(",")[0]) for line in lines[1:] if not line.startswith("#")])
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None

    redshifts = values if first_column == "redshift" else 1.0 / values - 1.0

    try:
        redshifts = snapshot_redshifts._as_redshifts(redshifts)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None

    return tuple(redshifts)


def _as_pairs(name, mapping):
    """Returns a dict of names as a sorted tuple of (str, str) pairs, so a suite can be hashed."""
    items = mapping.items() if isinstance(mapping, dict) else mapping
    try:
        pairs = tuple(sorted((str(key), str(value)) for key, value in items))
    except (TypeError, ValueError):
        raise TypeError(f"{name} must map names to names, not {mapping!r}") from None
    if len({value for _, value in pairs}) != len(pairs):
        raise ValueError(f"{name} gives two labels the same other name: {dict(pairs)}")
    return pairs


# outerlayer of redshift .txt file reader
def read_output_list(path):
    """
    Read the output redshifts of a simulation's snapshots from its SWIFT output list.

    The output list is a text file with a header line and one output per line, in order of snapshot number. Any further
    columns are ignored. Each file is read once.

    Raises FileNotFoundError if there is no file at the path, and ValueError if it is not an output list, or the
    redshifts don't decrease with snapshot number.

    Returns a the redshifts, in order of snapshot number.

    :param  path:   path of the output list, e.g. <simulation directory>/output_list.txt
    :type   path:   str or os.PathLike
    """
    return np.array(_read_output_list_file(os.path.abspath(os.fspath(path))), dtype=float)


def write_output_list(path, redshifts, select_output=None):
    r"""

    Write the output redshifts of a simulation's snapshots as a SWIFT output list.

    :param  path:           path of the file to write, e.g. <simulation directory>/output_list.txt
    :type   path:           str or os.PathLike
    :param  redshifts:      redshift of each snapshot, decreasing, in order of snapshot number
    :type   redshifts:      array-like of float
    :param  select_output:  the output selection of each snapshot (or "Snipshot"). If None, no second column
    :type   select_output:  sequence of str
    """
    redshifts = snapshot_redshifts._as_redshifts(redshifts)

    if select_output is not None and len(select_output) != len(redshifts):
        raise ValueError("select_output needs one entry for each redshift")

    os.makedirs(os.path.dirname(os.path.abspath(os.fspath(path))), exist_ok=True)

    with open(path, "w") as outfile:
        if select_output is None:
            outfile.write("# Redshift\n")
            outfile.writelines(f"{z:f}\n" for z in redshifts)
        else:
            outfile.write("# Redshift, Select Output\n")
            outfile.writelines(f"{z:f}, {selection}\n" for z, selection in zip(redshifts, select_output))

    _read_output_list_file.cache_clear()


###################################################################################################
# build FLAMINGO and COLIBRE simulation suites.
# These are essentially objects to contain all the paths to snapshots and relevenat redshifts of each snapshot


@dataclass(frozen=True)  # do not allow this to be mutated
class SimulationSuite:
    """
    For a SWIFT simulation suite (FLAMINGO and COLIBRE) track what is common for all simulations in the suite:
    the directories and path formats for the snapshots, halo catalogues and output redshifts.

    :ivar name:                     name of the suite, lower case, e.g. "flamingo" or "colibre"
    :vartype name:                  str
    :ivar label_pattern:            regular expression matching the box size and resolution labels of the suite's
                                    simulations, e.g. "L1000N1800"
    :vartype label_pattern:         str
    :ivar base_dir_format:          directory of a simulation, with {box_res} and {sim_name} fields
    :vartype base_dir_format:       str
    :ivar base_dir_env:             environment variable that, if set, replaces the directory holding the simulations.
                                    None for none
    :vartype base_dir_env:          str
    :ivar snapshot_format:          snapshot file, relative to the simulation's directory, with {snap_nr} and {file_nr}
                                    fields (default to SNAPSHOT_FORMAT)
    :vartype snapshot_format:       str
    :ivar output_list:              the SWIFT output list of a simulation, relative to its directory (or an absolute
                                    path), read for the output redshifts of its snapshots (see read_output_list)
    :vartype output_list:           str
    :ivar soap_format:              SOAP halo catalogue, relative to the simulation's directory, with a {snap_nr} field
                                    (default to SOAP_FORMAT)
    :vartype soap_format:           str
    :ivar description:              what the suite is, for people reading its file
    :vartype description:           str
    :ivar box_res_aliases:          other names of the box size and resolution labels, e.g. on an hdfstream server, as
                                    (label, other name) pairs, e.g. (("L1000N1800", "L1_m9"), ...); given as a dict.
                                    A simulation is found by either name, and always has the label
    :vartype box_res_aliases:       tuple
    :ivar alternative_snapshot_formats: other layouts of the snapshot files, tried in order when the suite's
                                        snapshot_format doesn't find them (see Simulation.from_label)
    :vartype alternative_snapshot_formats: tuple
    """

    # define types of each element
    name: str
    label_pattern: str
    base_dir_format: str
    base_dir_env: str = None
    snapshot_format: str = SNAPSHOT_FORMAT
    output_list: str = OUTPUT_LIST_NAME
    soap_format: str = SOAP_FORMAT
    description: str = ""
    box_res_aliases: tuple = ()
    alternative_snapshot_formats: tuple = ()

    def __post_init__(self):
        """Check the name, label pattern and path formats, so a mistake shows when the suite is made."""
        # check that name is correctly formatted
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z0-9_\-]+", self.name):
            raise ValueError(f"a suite's name is lower case letters, digits, '_' and '-', not {self.name!r}")

        # check label is re pattern with fields
        if not isinstance(self.label_pattern, str):
            raise TypeError(f"label_pattern must be a regular expression, not {self.label_pattern!r}")
        try:
            re.compile(self.label_pattern)
        except re.error as error:
            raise ValueError(f"label_pattern={self.label_pattern!r} is not a regular expression: {error}") from None

        # check base directory is given and a str
        if self.base_dir_env is not None and not isinstance(self.base_dir_env, str):
            raise TypeError(
                f"base_dir_env must be the name of an environment variable or None, not {self.base_dir_env!r}"
            )

        # check that file has a description str
        if not isinstance(self.description, str):
            raise TypeError(f"description must be a str, not {self.description!r}")

        for name in _PATH_FIELDS:
            _check_path_format(name, getattr(self, name))

        # a plain path, never formatted
        if not isinstance(self.output_list, (str, os.PathLike)):
            raise TypeError(f"output_list must be a path, not {self.output_list!r}")

        # the other names of the labels, and the other snapshot layouts: as tuples, so the suite can be hashed
        object.__setattr__(self, "box_res_aliases", _as_pairs("box_res_aliases", self.box_res_aliases or ()))
        formats = self.alternative_snapshot_formats or ()
        formats = (formats,) if isinstance(formats, (str, os.PathLike)) else formats
        object.__setattr__(self, "alternative_snapshot_formats", tuple(os.fspath(path) for path in formats))
        for path_format in self.alternative_snapshot_formats:
            _check_path_format("snapshot_format", path_format)

    def with_paths(self, snapshot_format=None, soap_format=None, output_list=None):
        """
        Returns a SimulationSuite, taken for the labels of this suite.

        :param  snapshot_format:    snapshot files, with {snap_nr} and {file_nr} fields,
                                    e.g. "snaps/run_{snap_nr:03d}.{file_nr}.hdf5"
        :type   snapshot_format:    str
        :param  soap_format:        SOAP halo catalogues, with a {snap_nr} field
        :type   soap_format:        str
        :param  output_list:        the SWIFT output list, e.g. "config/outputs.txt"
        :type   output_list:        str
        """
        paths = {"snapshot_format": snapshot_format, "soap_format": soap_format, "output_list": output_list}
        changes = {name: os.fspath(path) for name, path in paths.items() if path is not None}
        return replace(self, **changes) if changes else self

    def recognises(self, boxsize_resolution):
        """
        Check if a box size and resolution label is recognised by the simulation suite.

        :param  boxsize_resolution: box size and resolution label, e.g. "L1000N1800"
        :type   boxsize_resolution: str
        """
        return self.label_of(boxsize_resolution) is not None

    def label_of(self, boxsize_resolution):
        """
        Returns the box size and resolution label of a label or another name of one (e.g. "L1000N1800" or "L1_m9" ->
        "L1000N1800"), or None if it is neither.

        :param  boxsize_resolution: box size and resolution label, or another name of one
        :type   boxsize_resolution: str
        """
        if boxsize_resolution is None:
            return None

        label = str(boxsize_resolution)

        if re.fullmatch(self.label_pattern, label) is not None:
            return label

        return {alias: name for name, alias in self.box_res_aliases}.get(label)

    def box_res_names(self, boxsize_resolution):
        """
        Returns the names of a box: the label, then its alias if it has one (e.g. ["L1000N1800", "L1_m9"]).

        :param  boxsize_resolution: box size and resolution label, or another name of one
        :type   boxsize_resolution: str
        """
        label = self.label_of(boxsize_resolution) or str(boxsize_resolution)
        alias = dict(self.box_res_aliases).get(label)
        return [label] if alias is None else [label, alias]

    def directory_format(self):
        """
        Base directory of a simulation.

        Returns a str of the formatted path to the simulations base directory.
        Formatted path contains {box_res} and {sim_name} fields: from the suite's environment variable if set,
        otherwise base_dir_format.

        """
        if self.base_dir_env is None:
            base_dir = None
        else:
            base_dir = os.environ.get(self.base_dir_env)

        if base_dir:
            return os.path.join(base_dir, "{box_res}", "{sim_name}")
        return self.base_dir_format


# the fields of a suite file, the fields of SimulationSuite; name and label_pattern and base_dir_format are needed
_SUITE_FIELDS = tuple(field.name for field in fields(SimulationSuite))
_NEEDED_SUITE_FIELDS = ("label_pattern", "base_dir_format")


def read_suite_file(path):
    r"""
    Read a simulation suite from a YAML file.

    The file holds the fields of SimulationSuite, e.g.::

        name: 'mysuite'
        description: 'my simulations'
        label_pattern: 'L\d{4}N\d{4}_my'
        base_dir_format: '/data/mysuite/{box_res}/{sim_name}'
        base_dir_env: 'SWIFTLET_MYSUITE_DIR'
        snapshot_format: 'snapshots/snap_{snap_nr:04d}/snap_{snap_nr:04d}.{file_nr}.hdf5'
        soap_format: 'SOAP/halo_properties_{snap_nr:04d}.hdf5'
        output_list: 'output_list.txt'

    ``label_pattern`` and ``base_dir_format`` are needed; ``name`` is the file's name without ``.yaml``.

    Returns a SimulationSuite.

    Raises ValueError, naming the file, if the file isn't a suite (unknown or missing fields, or a field that is wrong).

    :param  path:   path of the suite file, e.g. "my_new_suite.yaml"
    :type   path:   str or os.PathLike
    """
    path = os.fspath(path)  # convert path to str

    # read .yaml sim suite file
    with open(path, "r") as infile:
        try:
            values = yaml.safe_load(infile)
        except yaml.YAMLError as error:
            raise ValueError(f"{path}: not a YAML file: {error}") from None

    # raise error if files values cannot be used as a dict
    if not isinstance(values, dict):
        raise ValueError(f"{path}: a suite file holds the fields of a suite ({', '.join(_SUITE_FIELDS)})")

    # raise error if values are not a recognised field
    unknown = sorted(set(values) - set(_SUITE_FIELDS))
    if unknown:
        raise ValueError(f"{path}: unknown fields {unknown}, a suite has {', '.join(_SUITE_FIELDS)}")

    # identify the missing fields and give as an error
    missing = [name for name in _NEEDED_SUITE_FIELDS if values.get(name) is None]
    if missing:
        raise ValueError(f"{path}: a suite needs {', '.join(missing)}")

    # place known values into dict and then make into a SimulationSuite
    values.setdefault("name", Path(path).stem)
    values = {name: value for name, value in values.items() if value is not None}

    try:
        return SimulationSuite(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from None


def write_suite_file(suite, path):
    """
    Write a simulation suite as a YAML file, which read_suite_file reads back as the same suite.

    To change an installed suite: edit the .yaml files in the SWIFTLET_SUITES_DIR directory or write a new YAML file to
    the same directory.

    :param  suite:  the suite, e.g. FLAMINGO
    :type   suite:  SimulationSuite
    :param  path:   path of the file to write, e.g. "suites/flamingo.yaml"
    :type   path:   str or os.PathLike
    """

    def quoted(value):
        # a single quoted YAML string: only ' is escaped (as ''), so regular expressions and {fields} are as written
        return "null" if value is None else "'" + str(value).replace("'", "''") + "'"

    lines = [
        "# SWIFTLET simulation suite, see docs/user_guide/new_simulations_and_suites.rst",
        "",
    ]

    for name in _SUITE_FIELDS:
        value = getattr(suite, name)

        if name == "box_res_aliases":
            lines.append(f"{name}:" + ("" if value else " {}"))
            lines += [f"  {quoted(label)}: {quoted(alias)}" for label, alias in value]

        elif name == "alternative_snapshot_formats":
            lines.append(f"{name}:" + ("" if value else " []"))
            lines += [f"  - {quoted(path_format)}" for path_format in value]

        else:
            lines.append(f"{name}: {quoted(value)}")

    # ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(os.fspath(path))), exist_ok=True)

    # write YAML suite file
    with open(path, "w") as outfile:
        outfile.write("\n".join(lines) + "\n")


def suite_directories():
    """
    The directories the suites are read from: the installed suites first, then those of SWIFTLET_SUITES_DIR in order.

    Returns a list of paths.
    """
    directories = [str(resources.files("swiftlet") / "data" / "simulation_suites")]
    directories += [d for d in os.environ.get(SUITES_DIR_ENV, "").split(os.pathsep) if d]
    return directories


def load_suites(directories=None):
    """
    Read the suites of every suite file (``*.yaml``) in the given directories.

    A suite file in a directory of SWIFTLET_SUITES_DIR replaces the installed suite of its name.

    Returns a dict {name: SimulationSuite}.

    Raises ValueError if a directory has two suites of the same name, or isn't a directory.

    :param  directories:    directories of suite files. If None, suite_directories()
    :type   directories:    list
    """
    suites = {}
    for directory in suite_directories() if directories is None else directories:
        if not os.path.isdir(directory):
            raise ValueError(f"{SUITES_DIR_ENV}: {directory} is not a directory of simulation suite files")
        in_directory = {}
        for path in sorted(Path(directory).glob("*.yaml")):
            suite = read_suite_file(path)
            if suite.name in in_directory:
                raise ValueError(f"{in_directory[suite.name]} and {path} are both the suite {suite.name!r}")
            in_directory[suite.name] = path
            suites[suite.name] = suite
    return suites


def add_suite(suite):
    """
    Add a simulation suite to the suites SWIFTLET knows, so its simulations are found from their labels.

    It replaces a suite of the same name. For this Python session only: to keep a suite, write it with
    write_suite_file into a directory of SWIFTLET_SUITES_DIR.

    Returns the SimulationSuite added.

    :param  suite:  the suite, or the path of a suite file
    :type   suite:  SimulationSuite or str or os.PathLike
    """
    if not isinstance(suite, SimulationSuite):
        suite = read_suite_file(suite)
    SUITES[suite.name] = suite
    return suite


###################################################################################################
# load known suites from swiftlet/data/simulation_suites

# the suites SWIFTLET knows are installed (FLAMINGO and COLIBRE), and those in SWIFTLET_SUITES_DIR
SUITES = load_suites()

# FLAMINGO and COLIBRE, as installed or replaced by a suite file of the same name
FLAMINGO = SUITES["flamingo"]
COLIBRE = SUITES["colibre"]

###################################################################################################
# define simulation suite functions


def get_suite(suite):
    """
    Returns the SimulationSuite of a suite SWIFTLET knows (e.g. "flamingo" or "colibre"), see SUITES.

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
    Identify the simulation suite from a simultations label.

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
            raise ValueError(
                f"can't tell which simulation suite boxsize_resolution={boxsize_resolution!r} belongs to ({patterns}). "
                f"Pass simulation_suite_name, one of {tuple(SUITES)}"
            )
        return from_label

    if from_label is not None and from_label.name != named.name:
        raise ValueError(
            (
                f"boxsize_resolution {boxsize_resolution!r} is a {from_label.name.upper()} simulation, "
                f"but simulation_suite_name is {named.name!r}"
            )
        )
    return named


def suite_for_label_or_none(boxsize_resolution):
    """
    The simulation suite recognising a box size and resolution label, or None if no suite (or more than one) does.

    :param  boxsize_resolution: box size and resolution label, e.g. "L1000N1800"
    :type   boxsize_resolution: str
    """
    matching = [suite for suite in SUITES.values() if suite.recognises(boxsize_resolution)]
    return matching[0] if len(matching) == 1 else None


def output_redshifts(simulation):
    """
    Returns the output redshifts of a SWIFT simulation's snapshots, in order of snapshot number.

    :param  simulation: the simulation, or the path of its output list (e.g. <simulation directory>/output_list.txt)
    :type   simulation: Simulation or str or os.PathLike
    """
    # check if sim object
    if isinstance(simulation, Simulation):
        return simulation.output_redshifts

    path = os.fspath(simulation)

    # fail if output list not found
    if not os.path.exists(path) and suite_for_label_or_none(path) is not None:
        raise ValueError(
            f"output_redshifts takes a Simulation or the path of an output list, not the label {path!r}: the output "
            f"redshifts are read from each simulation's {OUTPUT_LIST_NAME}, e.g. "
            f"output_redshifts(Simulation.from_label({path!r}, simulation_name))"
        )
    return read_output_list(path)


###################################################################################################


@dataclass(frozen=True)
class Simulation:
    """
    Define one simulation within a SimulationSuite and track its essential information.

    This class informs us about what simulation is being used (model name, box size and resolution label), what suite it
    belongs to ('flamingo' or 'colibre'), the redshift that each snapshot was output at and the paths to both the
    snapshots and halo catalogues.

    :ivar suite:        the simulation's suite, FLAMINGO or COLIBRE
    :vartype suite:     SimulationSuite
    :ivar box_res:      box size and resolution label, e.g. "L1000N1800"
    :vartype box_res:   str
    :ivar name:         name of the simulation, e.g. "HYDRO_FIDUCIAL"
    :vartype name:      str
    :ivar directory:    directory of the simulation (on disk, or relative to remote dir)
    :vartype directory: str
    :ivar remote:       the remote directory (an hdfstream.RemoteDirectory) the simulation's files are read from, with
                        directory and the snapshot paths relative to it. None for a simulation on disk
    :vartype remote:    hdfstream.RemoteDirectory
    """

    suite: SimulationSuite
    box_res: str
    name: str
    directory: str
    remote: object = field(default=None, compare=False, repr=False)

    @classmethod
    def from_label(
        cls,
        boxsize_resolution,
        simulation_name,
        simulation_suite_name=None,
        base_dir_format=None,
        snapshot_format=None,
        soap_format=None,
        output_list=None,
        remote=None,
        find_directory=None,
    ):
        """
        Make the Simulation object of a box size and resolution label and a simulation name.

        For a simulation stored remotely, give remote (a hdfstream.RemoteDirectory) and base_dir_format, the
        simulation's directory relative to it, e.g. "FLAMINGO/{box_res}/{sim_name}".

        Returns a Simulation.

        :param  boxsize_resolution:     box size and resolution label, e.g. "L1000N1800" (FLAMINGO) or "L0200N1504"
                                        (COLIBRE)
        :type   boxsize_resolution:     str
        :param  simulation_name:        name of the simulation, e.g. "HYDRO_FIDUCIAL"
        :type   simulation_name:        str
        :param  simulation_suite_name:  "flamingo" or "colibre" (or a SimulationSuite). If None, found from
                                        boxsize_resolution
        :type   simulation_suite_name:  str or SimulationSuite
        :param  base_dir_format:        directory of the simulation, with {box_res} and {sim_name} fields. If None, the
                                        suite's directory, see SimulationSuite.directory_format
        :type   base_dir_format:        str
        :param  snapshot_format:        snapshot files, relative to the simulation's directory (or absolute), with
                                        {snap_nr} and {file_nr} fields. If None, the suite's
        :type   snapshot_format:        str
        :param  soap_format:            SOAP halo catalogues, relative to the simulation's directory (or absolute), with
                                        a {snap_nr} field. If None, the suite's
        :type   soap_format:            str
        :param  output_list:            the SWIFT output list, relative to the simulation's directory (or absolute). If
                                        None, the suite's (output_list.txt)
        :type   output_list:            str
        :param  remote:                 the remote directory to read the simulation's files from, e.g.
                                        hdfstream.open(server, "/"). If None, the files are on disk
        :type   remote:                 hdfstream.RemoteDirectory
        :param  find_directory:         look for the simulation's directory with each name of the box, and for its
                                        snapshot layout, as above. If None, only for a remote simulation
        :type   find_directory:         bool
        """
        # find the suite, and the box's label if another name of it was given
        suite = suite_for_label(boxsize_resolution, simulation_suite_name)

        given_box_res = str(boxsize_resolution)
        boxsize_resolution = suite.label_of(boxsize_resolution) or given_box_res

        suite = suite.with_paths(snapshot_format, soap_format, output_list)

        # make formatted path to base dir
        if base_dir_format is not None:
            _check_path_format("base_dir_format", base_dir_format)

        if base_dir_format is not None:
            directory_format = base_dir_format

        else:
            # the boxes are in the remote directory
            directory_format = REMOTE_DIR_FORMAT if remote is not None else suite.directory_format()

        # the directory, with each name of the box (the label first, then the name given, then any other)
        names = suite.box_res_names(boxsize_resolution)
        names = list(dict.fromkeys([names[0], given_box_res] + names[1:]))

        directories = list(
            dict.fromkeys(directory_format.format(box_res=name, sim_name=simulation_name) for name in names)
        )

        find = remote is not None if find_directory is None else bool(find_directory)
        directory = directories[0]

        if find:
            from .hdf5stream_reader import directory_exists

            directory = next((path for path in directories if directory_exists(path, remote)), directory)

        simulation = cls(
            suite=suite, box_res=boxsize_resolution, name=simulation_name, directory=directory, remote=remote
        )

        if find and snapshot_format is None and suite.alternative_snapshot_formats:
            simulation = simulation._with_snapshot_layout()

        return simulation

    def _with_snapshot_layout(self):
        """Returns the simulation with the snapshot_format that finds its snapshot files."""
        from .hdf5stream_reader import file_exists

        try:
            snap_nr = self.z0_snapshot

        except (FileNotFoundError, ValueError):
            return self

        for path_format in (self.suite.snapshot_format,) + self.suite.alternative_snapshot_formats:
            path = os.path.join(self.directory, path_format.format(snap_nr=snap_nr, file_nr=0))

            if file_exists(path, self.remote):
                return (
                    self if path_format == self.suite.snapshot_format else self.with_paths(snapshot_format=path_format)
                )

        return self

    @classmethod
    def from_path(cls, path, simulation_suite_name=None, snapshot_format=None, soap_format=None, output_list=None):
        """
        Returns the paths for a simulation.

        Get a simulation from its directory, <...>/<box_res>/<sim_name>,
        e.g. "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL".
        The suite is found from the box size and resolution label, or from "flamingo" or "colibre" in the path.

        :param  path:                   directory of the simulation
        :type   path:                   str
        :param  simulation_suite_name:  "flamingo" or "colibre" (or a SimulationSuite). If None, found from the path
        :type   simulation_suite_name:  str or SimulationSuite
        :param  snapshot_format:        snapshot files, relative to the simulation's directory (or absolute), with
                                        {snap_nr} and {file_nr} fields. If None, the suite's
        :type   snapshot_format:        str
        :param  soap_format:            SOAP halo catalogues, relative to the simulation's directory (or absolute), with
                                        a {snap_nr} field. If None, the suite's
        :type   soap_format:            str
        :param  output_list:            the SWIFT output list, relative to the simulation's directory (or absolute). If
                                        None, the suite's (output_list.txt)
        :type   output_list:            str
        """
        directory = os.path.normpath(str(path))
        box_res = os.path.basename(os.path.dirname(directory))
        sim_name = os.path.basename(directory)

        if simulation_suite_name is None:
            in_path = [name for name in SUITES if re.search(rf"(^|/){name}(/|$)", directory.lower())]
            if len(in_path) == 1:
                simulation_suite_name = in_path[0]

        suite = suite_for_label(box_res, simulation_suite_name).with_paths(snapshot_format, soap_format, output_list)

        return cls(suite=suite, box_res=suite.label_of(box_res) or box_res, name=sim_name, directory=directory)

    def with_paths(self, directory=None, snapshot_format=None, soap_format=None, output_list=None, remote=None):
        """
        A copy of the simulation in another directory, or with other snapshot, SOAP catalogue or output list paths,
        or read from a remote directory.

        Keepp the paths that are not None, and the remote directory if remote is None.
        The paths are relative to the simulation's directory (or absolute).

        Returns a Simulation.

        :param  directory:          directory of the simulation
        :type   directory:          str or os.PathLike
        :param  snapshot_format:    snapshot files, with {snap_nr} and {file_nr} fields
        :type   snapshot_format:    str
        :param  soap_format:        SOAP halo catalogues, with a {snap_nr} field
        :type   soap_format:        str
        :param  output_list:        the SWIFT output list
        :type   output_list:        str
        :param  remote:             the remote directory to read the simulation's files from (a
                                    hdfstream.RemoteDirectory), with directory relative to it
        :type   remote:             hdfstream.RemoteDirectory
        """
        return replace(
            self,
            suite=self.suite.with_paths(snapshot_format, soap_format, output_list),
            directory=self.directory if directory is None else os.fspath(directory),
            remote=self.remote if remote is None else remote,
        )

    @property
    def snapshot_format(self):
        """Formatted path of the snapshot files, with {snap_nr} and {file_nr} fields."""
        return os.path.join(self.directory, self.suite.snapshot_format)

    @property
    def snapshot_format_whole(self):
        """
        Formatted path of the whole snapshot file, with a {snap_nr} field.

        It is snapshot_format without its file number, e.g. .../flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.hdf5: the
        virtual file that combines a snapshot's files.
        """
        return re.sub(r"\.\{file_nr[^}]*\}", "", self.snapshot_format)

    @property
    def snapshot_dir(self):
        """
        Returns the base directory of the snapshots.

        It is the directory of snapshot_format up to its first {field},
        e.g. <directory>/snapshots for .../snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5

        """
        parts = self.snapshot_format.split(os.sep)
        fixed = next(i for i, part in enumerate(parts) if _format_fields(part) or i == len(parts) - 1)
        return os.sep.join(parts[:fixed]) or os.sep

    def snapshot_file(self, snap_nr, file_nr=0):
        """
        Path of a snapshot file.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        :param  file_nr:    file number of the snapshot
        :type   file_nr:    int
        """
        return self.snapshot_format.format(snap_nr=int(snap_nr), file_nr=int(file_nr))

    @property
    def soap_format(self):
        """Formatted path of the SOAP halo catalogues, with a {snap_nr} field, e.g. for gather_halo_catalogues."""
        return os.path.join(self.directory, self.suite.soap_format)

    @property
    def soap_dir(self):
        """Directory of the SOAP halo catalogues."""
        return os.path.dirname(self.soap_format)

    def soap_file(self, snap_nr):
        """
        Path of the SOAP halo catalogue.

        :param  snap_nr:    snapshot number
        :type   snap_nr:    int
        """
        return self.soap_format.format(snap_nr=int(snap_nr))

    @property
    def output_list(self):
        """Path of the simulation's SWIFT output list, in its directory unless the suite gives an absolute path."""
        return os.path.join(self.directory, self.suite.output_list)

    # make sure to cache output redshifts once read
    @cached_property
    def output_redshifts(self):
        """Output redshift of each snapshot (and snipshot), in order of snapshot number, from the output list."""
        if self.remote is None:
            return read_output_list(self.output_list)

        from .hdf5stream_reader import read_text

        try:
            lines = read_text(self.output_list, self.remote)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"no output list at {self.output_list} (remote): SWIFTLET reads the output redshifts "
                f"from {OUTPUT_LIST_NAME}"
            ) from None

        return np.array(_parse_output_list(lines, self.output_list), dtype=float)

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
        :type   snap_nr:    int or numpy.ndarray
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
        Returns the snapshot numbers, from high to low redshift,of snapshots whose redshift ranges overlap a redshift
        range, including ranges that only touch it.

        :param  redshift_range: minimum and maximum redshifts [z_min, z_max]
        :type   redshift_range: sequence of two floats
        :param  range_tol:      a snapshot edge within range_tol of the range counts as touching it
        :type   range_tol:      float
        :param  strict:         If True, leave out the snapshots whose ranges only touch the range (overlap it by no
                                    more than range_tol)
        :type   strict:         bool
        """
        return snapshot_redshifts.snapshots_in_range(
            self.output_redshifts, redshift_range=redshift_range, range_tol=range_tol, strict=strict
        )

    @property
    def z0_snapshot(self):
        """Number of the snapshot at redshift 0."""
        return self.snapshot_number(0.0)
