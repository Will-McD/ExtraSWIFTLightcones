#!/bin/env python
import argparse
import os
import sys
import urllib.request

"""
Set up the data files SWIFTLET needs in the Python environment it is installed in.

After installing the package with pip, run: swiftlet-configure

To download the FLAMINGO lightcone shell redshift files into <environment>/share/swiftlet/redshifts
and export their paths (L1_REDSHIFTS_FILENAME and L2P8_REDSHIFTS_FILENAME) whenever the environment is activated.
"""

# FLAMINGO lightcone shell redshift files: environment variable, file name and download URL
FLAMINGO_DATA_URL = "https://dataweb.cosma.dur.ac.uk:8443/hdfstream/download/FLAMINGO"
SHELL_REDSHIFT_FILES = {
    "L1":   ("L1_REDSHIFTS_FILENAME",   "L1_shell_redshifts_z3.txt",   f"{FLAMINGO_DATA_URL}/L1_m9/L1_m9/shell_redshifts_z3.txt"),
    "L2p8": ("L2P8_REDSHIFTS_FILENAME", "L2p8_shell_redshifts_z5.txt", f"{FLAMINGO_DATA_URL}/L2p8_m9/L2p8_m9/shell_redshifts.txt"),
}

# marks the lines this module writes into an activate script, so they can be replaced when run again
ACTIVATE_MARKER = "# added by swiftlet-configure"
# markers of earlier names of this command, whose lines are replaced too
OLD_ACTIVATE_MARKERS = ("# added by extra_swift_lightcones-configure",)
# activate.d / deactivate.d script name of a conda environment under the earlier package name, removed when run again
OLD_CONDA_SCRIPT = "extra_swift_lightcones.sh"

def environment_data_dir():
    """
    Directory for the data files inside the Python environment the package is installed in.

    Returns <sys.prefix>/share/swiftlet.
    """
    return os.path.join(sys.prefix, "share", "swiftlet")


def default_redshift_dir():
    """
    Directory the shell redshift files are downloaded to by default: inside the Python environment
    when it can be written to, otherwise ~/.cache/swiftlet/redshifts.

    Returns the directory.
    """
    env_dir = os.path.join(environment_data_dir(), "redshifts")
    existing = env_dir
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    if os.access(existing, os.W_OK):
        return env_dir
    return os.path.join(os.path.expanduser("~"), ".cache", "swiftlet", "redshifts")


def download_file(url, target, timeout=60):
    """
    Download a text file, checking a web page (e.g. a login or error page) was not returned instead.
    The file is written to <target>.part first and only moved to target once complete.

    :param  url:        URL of the file
    :type   url:        str
    :param  target:     path to save the file to
    :type   target:     str
    :param  timeout:    timeout of the request [s]
    :type   timeout:    float
    """
    tmp = target + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read()
    except OSError as error:
        raise OSError(f"could not download {url}: {error}") from error
    if len(content) == 0 or b"<html" in content[:1000].lower():
        raise OSError(f"{url} did not return a text file")
    with open(tmp, "wb") as outfile:
        outfile.write(content)
    os.replace(tmp, target)


def download_shell_redshifts(dest_dir=None, force=False, verbose=True):
    """
    Download the FLAMINGO lightcone shell redshift files.

    Returns a dict of {environment variable name: path of the file}.

    :param  dest_dir:   directory to download the files to. If None, see default_redshift_dir
    :type   dest_dir:   str
    :param  force:      If True, download the files again even if they already exist
    :type   force:      boolean
    :param  verbose:    If True, print what is downloaded
    :type   verbose:    boolean
    """
    dest_dir = os.path.abspath(dest_dir if dest_dir is not None else default_redshift_dir())
    os.makedirs(dest_dir, exist_ok=True)

    paths = {}
    for env_name, filename, url in SHELL_REDSHIFT_FILES.values():
        target = os.path.join(dest_dir, filename)
        if force or not os.path.isfile(target) or os.path.getsize(target) == 0:
            if verbose:
                print(f"Downloading {url}")
            download_file(url, target)
            if verbose:
                print(f"Saved {target}")
        elif verbose:
            print(f"Already downloaded: {target}")
        paths[env_name] = target
    return paths


def activate_scripts():
    """
    Scripts run when the Python environment is activated: bin/activate for a virtual environment (venv),
    and etc/conda/activate.d / deactivate.d scripts for a conda environment.

    Returns a tuple of (environment type, activate script, deactivate script). The deactivate script is
    None for a venv, and the environment type is None when not in a venv or conda environment.
    """
    if os.path.isdir(os.path.join(sys.prefix, "conda-meta")):
        return ("conda",
                os.path.join(sys.prefix, "etc", "conda", "activate.d", "swiftlet.sh"),
                os.path.join(sys.prefix, "etc", "conda", "deactivate.d", "swiftlet.sh"))
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix) and os.path.isfile(os.path.join(sys.prefix, "bin", "activate")):
        return "venv", os.path.join(sys.prefix, "bin", "activate"), None
    return None, None, None


def _replace_marked_lines(filename, new_lines):
    """
    Remove the lines added by an earlier run (see ACTIVATE_MARKER and OLD_ACTIVATE_MARKERS) from a file and append new ones.

    :param  filename:   file to update
    :type   filename:   str
    :param  new_lines:  lines to add, without the marker
    :type   new_lines:  list of str
    """
    lines = []
    if os.path.isfile(filename):
        with open(filename) as infile:
            lines = [line for line in infile.read().splitlines() if not line.endswith((ACTIVATE_MARKER,) + OLD_ACTIVATE_MARKERS)]
    lines += [f"{line}  {ACTIVATE_MARKER}" for line in new_lines]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as outfile:
        outfile.write("\n".join(lines) + "\n")


def export_paths_on_activate(paths, verbose=True):
    """
    Export environment variables whenever the Python environment is activated, by adding them
    to its activate script(s). Running this again replaces the values added before.

    Returns True if the variables were added, False if not in a venv or conda environment.

    :param  paths:      {environment variable name: value}
    :type   paths:      dict
    :param  verbose:    If True, print where the variables were added
    :type   verbose:    boolean
    """
    env_type, activate, deactivate = activate_scripts()
    if env_type is None:
        if verbose:
            print("Not in a virtual or conda environment, add these to your shell or job scripts instead:")
            for name, value in paths.items():
                print(f'  export {name}="{value}"')
        return False

    _replace_marked_lines(activate, [f'export {name}="{value}"' for name, value in paths.items()])
    if deactivate is not None:
        _replace_marked_lines(deactivate, [f"unset {name}" for name in paths])
    if env_type == "conda":
        # scripts written under the earlier package name
        for script in (activate, deactivate):
            old_script = os.path.join(os.path.dirname(script), OLD_CONDA_SCRIPT)
            if os.path.isfile(old_script):
                os.remove(old_script)
    if verbose:
        print(f"{', '.join(paths)} set in {activate}, re-activate the environment to use them")
    return True


def configure(dest_dir=None, force=False, update_activate=True, verbose=True):
    """
    Download the FLAMINGO lightcone shell redshift files and export their paths when the environment is activated.

    Returns a dict of {environment variable name: path of the file}.

    :param  dest_dir:           directory to download the files to. If None, see default_redshift_dir
    :type   dest_dir:           str
    :param  force:              If True, download the files again even if they already exist
    :type   force:              boolean
    :param  update_activate:    If True, add the paths to the environment's activate script(s)
    :type   update_activate:    boolean
    :param  verbose:            If True, print what is done
    :type   verbose:            boolean
    """
    paths = download_shell_redshifts(dest_dir=dest_dir, force=force, verbose=verbose)
    if update_activate:
        export_paths_on_activate(paths, verbose=verbose)
    return paths


def cli_configure(argv=None):
    """
    Command line entry point, installed as swiftlet-configure.

    :param  argv:   command line arguments. If None, use sys.argv
    :type   argv:   list of str
    """
    parser = argparse.ArgumentParser(
        description="Download the FLAMINGO lightcone shell redshift files into the Python environment SWIFTLET is installed in, and export their paths when it is activated")
    parser.add_argument("--dest_dir", type=str, default=None,
                        help=f"directory to download the files to (default: {default_redshift_dir()})")
    parser.add_argument("--force", action="store_true", help="download the files again even if they already exist")
    parser.add_argument("--no_activate", action="store_true", help="do not add the paths to the environment's activate script")
    args = parser.parse_args(argv)
    try:
        configure(dest_dir=args.dest_dir, force=args.force, update_activate=not args.no_activate)
    except OSError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    cli_configure()