#!/bin/env python
"""
Tests for swiftlet.threads:
    - test for both sets of numba and healpy alm functions that rely on threads set
    - maximum number of threads is set by SWIFTLET_MAX_THREADS 
    - abides by default 30
    - max number is never more than the cores the process may use.
"""
import os

import numpy as np
import pytest

from swiftlet import threads

# SWIFTLET_MAX_THREADS as conftest.py set it for the tests, before the fixture below removes it in each test here
MAX_THREADS_IN_TESTS = os.environ.get(threads.MAX_THREADS_ENV)


@pytest.fixture(autouse=True)
def no_max_threads_set(monkeypatch):
    """
    Start each test without SWIFTLET_MAX_THREADS set, and restore it after.
    """
    monkeypatch.delenv(threads.MAX_THREADS_ENV, raising=False)


def test_default_maximum_is_30():
    assert threads.DEFAULT_MAX_THREADS == 30
    assert threads.max_threads() == 30


def test_maximum_from_the_environment(monkeypatch):
    monkeypatch.setenv(threads.MAX_THREADS_ENV, "12")
    assert threads.max_threads() == 12
    monkeypatch.setenv(threads.MAX_THREADS_ENV, " ")
    assert threads.max_threads() == 30


@pytest.mark.parametrize("value", ["0", "-2", "2.5", "many"])
def test_invalid_maximum_is_rejected(monkeypatch, value):
    monkeypatch.setenv(threads.MAX_THREADS_ENV, value)
    with pytest.raises(ValueError, match=threads.MAX_THREADS_ENV):
        threads.max_threads()


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "x"])
def test_set_max_threads_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        threads.set_max_threads(value)


def test_set_max_threads():
    threads.set_max_threads(7)
    assert os.environ[threads.MAX_THREADS_ENV] == "7"
    assert threads.max_threads() == 7


@pytest.mark.parametrize("value", ["all", "ALL", " all "])
def test_all_means_no_fixed_maximum(monkeypatch, value):
    monkeypatch.setenv(threads.MAX_THREADS_ENV, value)
    assert threads.max_threads() is None
    monkeypatch.setattr(threads, "available_cores", lambda: 64)
    assert threads.thread_limit() == 64  # every core the process is allowed, more than the default of 30


def test_set_max_threads_to_all():
    threads.set_max_threads("all")
    assert os.environ[threads.MAX_THREADS_ENV] == "all"
    threads.set_max_threads(None)
    assert os.environ[threads.MAX_THREADS_ENV] == "all"
    assert threads.thread_limit() == threads.available_cores()


def test_limit_is_at_most_the_available_cores(monkeypatch):
    monkeypatch.setattr(threads, "available_cores", lambda: 8)
    assert threads.thread_limit() == 8  # the default maximum of 30 is more than the cores
    threads.set_max_threads(2)
    assert threads.thread_limit() == 2


def test_numba_threads_limited(monkeypatch):
    numba = pytest.importorskip("numba")
    threads.set_max_threads(1)
    assert threads.limit_numba_threads() == 1
    assert numba.get_num_threads() == 1
    threads.set_max_threads(10**6)
    assert threads.limit_numba_threads() == min(threads.available_cores(), numba.config.NUMBA_NUM_THREADS)
    assert numba.get_num_threads() == threads.limit_numba_threads()


def test_reorientation_uses_the_limit():
    """
    Re-orienting coordinates with numba sets numba's threads to the limit first.
    """
    numba = pytest.importorskip("numba")
    from swiftlet import snapshot_orientation
    threads.set_max_threads(1)
    coords = np.random.default_rng(0).uniform(0, 10, (100, 3))
    snapshot_orientation.rotate_coords_cartesian(coords, (90, 0, 0), periodic_shift=(0, 0, 0), sidelengths=10., degrees=True)
    assert numba.get_num_threads() == 1


def test_healpy_threads_limited_in_the_block():
    """
    Within limit_openmp_threads, healpy's OpenMP threads are limited to the limit, and restored after.
    """
    hp = pytest.importorskip("healpy")
    threadpoolctl = pytest.importorskip("threadpoolctl")
    hp.map2alm(np.zeros(hp.nside2npix(4)), lmax=4)  # load healpy's OpenMP library
    openmp = lambda: [pool["num_threads"] for pool in threadpoolctl.threadpool_info() if pool["user_api"] == "openmp"]
    before = openmp()
    if not before:
        pytest.skip("no OpenMP library loaded by healpy")
    threads.set_max_threads(1)
    with threads.limit_openmp_threads():
        assert openmp() == [1] * len(before)
        hp.map2alm(np.zeros(hp.nside2npix(4)), lmax=4)
    assert openmp() == before


def test_tests_use_at_most_8_threads():
    """
    conftest.py sets the maximum to 8 threads for all the tests.
    """
    assert MAX_THREADS_IN_TESTS == "8"