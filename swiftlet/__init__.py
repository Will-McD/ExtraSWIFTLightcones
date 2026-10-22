"""
SWIFTLET (SWIFT Lightcone Extension Tools): tools to build new lightcones from SWIFT snapshots, post-process and
visualise SWIFT lightcones.

The main classes and functions can be imported from the package:

    from swiftlet import SnapshotBeam, BeamProjection
    snapshot_lightcone:                 SnapshotLightcone, SnapshotBeam, SnapshotAllSky, ORIENTATION_LOCK_OPTIONS,
    snapshot_cosmology:                 SnapshotCosmology
    simulation_suites:                  Simulation, SimulationSuite, FLAMINGO, COLIBRE, suite_for_label, output_redshifts,
                                        read_output_redshifts
    snapshot_redshifts:                 redshift_edges, snapshots_in_range
    snapshot_orientation:               write_orientation_tables, read_orientation_tables, generate_orientation_tables
    lightcone_projections:              BeamProjection
    beam_plotting:                      BeamPlot
    healpix_map_utils:                  write_rotated_lightcone_chunks, sum_maps
    mask_haloes:                        write_binary_masks
    flamingo_lightcones:                flamingo_shell_redshift_file
    snapshot_units:                     apply_expected_units
    config:                             configure, download_shell_redshifts
    threads:                            set_max_threads, max_threads, thread_limit

Each is imported when first used, so importing the package doesn't need MPI, matplotlib or lightcone_io
until something needing them is used. The submodules can also be used as attributes, e.g. swiftlet.config.
"""
import importlib

# Use setuptools_scm to get version from git tags
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("swiftlet-lightcones")
except PackageNotFoundError:
    __version__ = "unknown"

# module each public class or function is defined in
_PUBLIC_NAMES = {
    # building lightcones from snapshots
    "SnapshotLightcone": "snapshot_lightcone",
    "SnapshotBeam": "snapshot_lightcone",
    "SnapshotAllSky": "snapshot_lightcone",
    "ORIENTATION_LOCK_OPTIONS": "snapshot_lightcone",
    "write_lightcone_metadata": "snapshot_lightcone",
    "read_lightcone_metadata": "snapshot_lightcone",
    # the cosmology of a simulation given by the snapshot
    "SnapshotCosmology": "snapshot_cosmology",
    # the simulations: paths and snapshot redshifts of FLAMINGO and COLIBRE
    "Simulation": "simulation_suites",
    "SimulationSuite": "simulation_suites",
    "FLAMINGO": "simulation_suites",
    "COLIBRE": "simulation_suites",
    "suite_for_label": "simulation_suites",
    "output_redshifts": "simulation_suites",
    "read_output_redshifts": "simulation_suites",
    # orientation tables of the box tiles
    "write_orientation_tables": "snapshot_orientation",
    "read_orientation_tables": "snapshot_orientation",
    "generate_orientation_tables": "snapshot_orientation",
    # projecting and plotting beams
    "BeamProjection": "lightcone_projections",
    "BeamPlot": "beam_plotting",
    # HEALPix maps and halo masks
    "write_rotated_lightcone_chunks": "healpix_map_utils",
    "sum_maps": "healpix_map_utils",
    "write_binary_masks": "mask_haloes",
    # snapshot and shell redshifts
    "flamingo_shell_redshift_file": "flamingo_lightcones",
    "redshift_edges": "snapshot_redshifts",
    "snapshots_in_range": "snapshot_redshifts",
    # units
    "apply_expected_units": "snapshot_units",
    # setting up the environment
    "configure": "config",
    "download_shell_redshifts": "config",
    # max numb of threads for numba's parallel functions and healpy's alm transforms
    "set_max_threads": "threads",
    "max_threads": "threads",
    "thread_limit": "threads",
}

_SUBMODULES = (
    "beam_plotting",
    "config",
    "snapshot_cosmology",
    "flamingo_lightcones",
    "healpix_map_utils",
    "lightcone_projections",
    "mask_haloes",
    "property_to_field_names",
    "snapshot_lightcone",
    "simulation_suites",
    "snapshot_orientation",
    "snapshot_units",
    "snapshot_redshifts",
    "threads",
)

__all__ = list(_PUBLIC_NAMES)


def __getattr__(name):
    """
    Import the public classes, functions and submodules of the package when first used, so that modules
    which don't need MPI (e.g. swiftlet.config) can be imported without it.

    :param  name:   name of the attribute
    :type   name:   str
    """
    if name in _PUBLIC_NAMES:
        module = importlib.import_module(f".{_PUBLIC_NAMES[name]}", __name__)
        value = getattr(module, name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # store it, so it is only looked up once
    globals()[name] = value
    return value


def __dir__():
    """
    Names in the package, including the ones imported when first used.
    """
    return sorted(set(globals()) | set(_PUBLIC_NAMES) | set(_SUBMODULES))
