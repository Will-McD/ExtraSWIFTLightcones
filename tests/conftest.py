#!/bin/env python
"""
pytest settings.
With --figures-dir DIR, tests that can make figures output said figures to given directory. 
"""
import os
from pathlib import Path
import pytest

# set max num of threads for testing to 8. 
# this limit should be sufficent for testing and not beyond the limits of any machine. 
TEST_MAX_THREADS = 8
def pytest_configure(config):
    """
    set max threads for the tests
    """
    os.environ["SWIFTLET_MAX_THREADS"] = str(TEST_MAX_THREADS)

def pytest_addoption(parser):
    parser.addoption("--figures-dir", default=None, help="directory to save figures of the tests to, e.g. of the lattice placed in the lightcone. No figures are made if not given")
@pytest.fixture
def figures_dir(request):
    """
    Directory to save ./tests/... figures to. Only made if needed, i.e. --figures-dir was given.
    """
    directory = request.config.getoption("--figures-dir")
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
