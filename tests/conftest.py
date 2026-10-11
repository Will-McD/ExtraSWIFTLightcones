#!/bin/env python
"""
pytest settings.
With --figures-dir DIR, tests that can make figures output said figures to given directory. 
"""

from pathlib import Path
import pytest

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
