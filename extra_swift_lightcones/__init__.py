"""
ExtraSWIFTLightcones: tools to build new lightcones from SWIFT snapshots, and to post-process and
visualise SWIFT lightcones.

The main classes and functions can be imported from the package:

    from extra_swift_lightcones import SnapshotBeam, BeamProjection

    snapshot_lightcone:                 SnapshotLightcone, SnapshotBeam, SnapshotAllSky, ORIENTATION_LOCK_OPTIONS
    lightcone_projections:              BeamProjection
    beam_plotting:                      BeamPlot
    healpix_map_utils:                  write_rotated_lightcone_chunks, sum_maps
    mask_haloes:                        write_binary_masks
    swift_snapshot_redshift_conversion: flamingo_shell_redshift_file, snapshot_number_redshifts, snapshot_redshift_range
    snapshot_units:                     apply_expected_units
    config:                             configure, download_shell_redshifts

Each is imported when first used, so importing the package doesn't need MPI, matplotlib or lightcone_io
until something needing them is used. The submodules can also be used as attributes, e.g. extra_swift_lightcones.config.
"""
import importlib

# Use setuptools_scm to get version from git tags
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("ExtraSWIFTLightcones")
except PackageNotFoundError:
    __version__ = "unknown"

# module each public class or function is defined in
_PUBLIC_NAMES = {
    # building lightcones from snapshots
    "SnapshotLightcone": "snapshot_lightcone",
    "SnapshotBeam": "snapshot_lightcone",
    "SnapshotAllSky": "snapshot_lightcone",
    "ORIENTATION_LOCK_OPTIONS": "snapshot_lightcone",
    # projecting and plotting beams
    "BeamProjection": "lightcone_projections",
    "BeamPlot": "beam_plotting",
    # HEALPix maps and halo masks
    "write_rotated_lightcone_chunks": "healpix_map_utils",
    "sum_maps": "healpix_map_utils",
    "write_binary_masks": "mask_haloes",
    # snapshot and shell redshifts
    "flamingo_shell_redshift_file": "swift_snapshot_redshift_conversion",
    "snapshot_number_redshifts": "swift_snapshot_redshift_conversion",
    "snapshot_redshift_range": "swift_snapshot_redshift_conversion",
    # units
    "apply_expected_units": "snapshot_units",
    # setting up the environment
    "configure": "config",
    "download_shell_redshifts": "config",
}

_SUBMODULES = (
    "beam_plotting",
    "config",
    "healpix_map_utils",
    "lightcone_projections",
    "mask_haloes",
    "property_to_field_names",
    "snapshot_lightcone",
    "snapshot_orientation",
    "snapshot_units",
    "swift_snapshot_redshift_conversion",
)

__all__ = list(_PUBLIC_NAMES)


def __getattr__(name):
    """
    Import the public classes, functions and submodules of the package when first used, so that modules
    which don't need MPI (e.g. extra_swift_lightcones.config) can be imported without it.

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
