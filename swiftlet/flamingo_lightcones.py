#!/bin/env python
import os
import re
from .config import SHELL_REDSHIFT_FILES, default_redshift_dir, download_shell_redshifts, environment_data_dir


def flamingo_shell_redshift_file(box, download=True):
    """
    Returns the absolute path to the FLAMINGO shell redshifts .txt file for the 1000 Mpc ("L1") or 
    2800 Mpc ("L2p8") box sidelength simulations.

    Checks, in order:
    
    1. the L1_REDSHIFTS_FILENAME / L2P8_REDSHIFTS_FILENAME environment variable
    2. <environment>/share/swiftlet/redshifts/<file>, where swiftlet-configure downloads the files to
    3. ~/.cache/swiftlet/redshifts/<file>, used when the environment can't be written to
    4. <repo>/data/redshifts/<file> (only when called from inside the package, e.g. an editable install)
    
    If none exist and download is True, the files are downloaded (see config.download_shell_redshifts).
    Raises FileNotFoundError if the file can't be found or downloaded.

    :param  box:        simulation box, "L1" or "L2p8"
    :type   box:        str
    :param  download:   If True, download the files when they can't be found
    :type   download:   bool
    """
    if box not in SHELL_REDSHIFT_FILES:
        raise ValueError(f"box must be one of {list(SHELL_REDSHIFT_FILES)}, got {box!r}")
    env_name, filename, _ = SHELL_REDSHIFT_FILES[box]
    
    candidates = [
        os.environ.get(env_name),
        os.path.join(environment_data_dir(), "redshifts", filename),
        os.path.join(os.path.expanduser("~"), ".cache", "swiftlet", "redshifts", filename),
    ]
    
    # work around, _file__ only exists when this code lives in a .py file
    module_file = globals().get("__file__")
    if module_file is not None:
        package_dir = os.path.dirname(os.path.abspath(module_file))
        candidates.append(os.path.join(package_dir, "..", "data", "redshifts", filename))
    
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    
    download_error = ""
    if download:
        try:
            return download_shell_redshifts(verbose=False)[env_name]
        except OSError as error:
            download_error = f"Downloading them to {default_redshift_dir()} failed: {error}."
    raise FileNotFoundError(
        f"FLAMINGO shell redshifts for {box} not found (looked in: {[p for p in candidates if p]}).{download_error} Run swiftlet-configure on a machine with internet access, or set {env_name} to the path of the file"
    )


def flamingo_box_resolution(path):

    """
    Find the box size / resolution label of a FLAMINGO simulation from a path to the simulation's data.

    Returns the label, e.g. "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/data/products.." -> "L1000N1800".
    
    Raises ValueError if none (or more than one different one) is found.

    :param  path:   path to the simulation data
    :type   path:   str
    """

    flamingo_box_res_pattern = re.compile(r"(?:^|/)(L(\d+)N(\d+))(?=/|$)")

    matches = {m.group(1) for m in flamingo_box_res_pattern.finditer(str(path))}
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one FLAMINGO box/resolution label (e.g. L1000N1800) in {path!r}, found {sorted(matches) or 'none'}")
    return matches.pop()

