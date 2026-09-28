__all__ = ["SnapshotLightcone", "SnapshotBeam", "SnapshotAllSky"]

# Use setuptools_scm to get version from git tags
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("ExtraSWIFTLightcones")
except PackageNotFoundError:
    __version__ = "unknown"

# Classes for building lightcones from snapshots
from .snapshot_lightcone import SnapshotLightcone, SnapshotBeam, SnapshotAllSky
