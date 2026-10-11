#!/bin/env python
import os
from contextlib import contextmanager

"""
Maximum number of threads used by the threaded code:
    - numba parallel functions (re-orienting the snapshot box tiles, and swiftsimio's parallel projections)
    - healpy alm spherical harmonic transforms (map2alm, rotate_alm and alm2map)

SWIFTLET_MAX_THREADS = the max number of threads. If None then default to 30, otherwise when set 
never allow it to be more than the number of cores the process is allowed to run on (e.g. the cores given to a job or an MPI rank). 
With SWIFTLET_MAX_THREADS=all there is no fixed maximum, and each process uses all the cores it is allowed. Change it with

    swiftlet-configure --max_threads N      # N or all, saved in the environment's activate script
    export SWIFTLET_MAX_THREADS=N           # for one shell or job script
    swiftlet.set_max_threads(N)             # while running, in Python
"""

MAX_THREADS_ENV = "SWIFTLET_MAX_THREADS"
DEFAULT_MAX_THREADS = 30
# value of SWIFTLET_MAX_THREADS for no fixed maximum: use all the cores the process is allowed to run on
ALL_THREADS = "all"


def _check_number_of_threads(value, source):
    """
    Check a maximum number of threads is a whole number of at least 1, or "all" (or None) for no fixed maximum.

    Returns the number of threads, otherwise None if no fixed maximum.

    :param  value:  maximum number of threads
    :type   value:  int, str or None
    :param  source: where the value came from, for the error message
    :type   source: str
    """
    if value is None or str(value).strip().lower() == ALL_THREADS:
        return None
    try:
        number = int(str(value).strip())  # also rejects e.g. 2.5 and "2.5"
    except ValueError:
        number = 0
    if isinstance(value, bool) or number < 1:
        raise ValueError(f"{source} must be a whole number of threads of at least 1, or {ALL_THREADS!r}, got {value!r}")
    return number


def max_threads():
    """
    Returns the maximum number of threads
        - SWIFTLET_MAX_THREADS:     if a value is set. 
        - DEFAULT_MAX_THREADS:      if not value set 
        - None:                     if 'all' and no fixed maximum
    """
    value = os.environ.get(MAX_THREADS_ENV, "")
    if value.strip() == "":
        return DEFAULT_MAX_THREADS
    return _check_number_of_threads(value, MAX_THREADS_ENV)


def set_max_threads(number):
    """
    Set SWIFTLET_MAX_THREADS, the maximum number of threads to use. 

    :param  number: maximum number of threads, at least 1, or "all" (or None) to use all the cores the process is allowed
    :type   number: int, str or None
    """
    number = _check_number_of_threads(number, "the maximum number of threads")
    os.environ[MAX_THREADS_ENV] = ALL_THREADS if number is None else str(number)

def available_cores():
    """
    Returns the number of cores this process is allowed to run on (e.g. those given to a job or an MPI rank).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on macOS
        return max(1, os.cpu_count() or 1)

def thread_limit():
    """
    Returns the number of threads to use: the maximum, but no more than the cores this process may run on, or all
    of those cores if SWIFTLET_MAX_THREADS is "all".
    """

    maximum = max_threads()
    return available_cores() if maximum is None else max(1, min(maximum, available_cores()))


def limit_numba_threads():
    """
    Set the number of threads numba's parallel functions use to thread_limit(), and no more than numba started
    with (NUMBA_NUM_THREADS). Called before running a numba parallel function, in the thread that runs it.

    Returns the number of threads numba uses.
    """
    import numba
    number = min(thread_limit(), numba.config.NUMBA_NUM_THREADS)
    if numba.get_num_threads() != number:
        numba.set_num_threads(number)
    return number


@contextmanager
def limit_openmp_threads():
    """
    Within the block, limit the thread pools to thread_limit() and restore them after. 
    Uses threadpoolctl. 
    Without it the threads are not limited, and OMP_NUM_THREADS can be set instead.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=thread_limit()):
        yield