#!/bin/env python
"""
pytest settings.
With --figures-dir DIR, tests that can make figures output said figures to given directory.
"""

import os
from pathlib import Path
import pytest

# set max num of threads for testing to 8.
# this limit should be sufficient for testing and not beyond the limits of any machine.
TEST_MAX_THREADS = 8


def pytest_configure(config):
    """
    set max threads for the tests, and register the slow marker
    """
    os.environ["SWIFTLET_MAX_THREADS"] = str(TEST_MAX_THREADS)
    config.addinivalue_line("markers", "slow: takes long and only matters when what it checks changes (--run-slow)")


def pytest_collection_modifyitems(config, items):
    """
    Skip the tests marked slow, unless --run-slow is given.
    """
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow: give --run-slow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def pytest_addoption(parser):
    parser.addoption(
        "--figures-dir",
        default=None,
        help="directory to save figures of the tests to. No figures are made if not given",
    )
    parser.addoption("--run-slow", action="store_true", help="also run the tests marked slow")


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
