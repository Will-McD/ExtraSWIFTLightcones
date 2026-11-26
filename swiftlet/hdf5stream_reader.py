#!/bin/env python
"""
Read snapshot files and SOAP catalogues on disk or remotely.

Every snapshot file is opened with :func:`open_file` and every slice of particle data is read with
:func:`read_slices`, so the pipeline reads files on disk (with h5py) and on an hdfstream server (a
``hdfstream.RemoteDirectory``) in the same way.

.. code-block:: python

    with open_file(path, remote) as snapshot:
        dataset = snapshot["PartType1/Coordinates"]
        coordinates = read_slices(dataset, starts=[0, 1000], lengths=[10, 20])

A remote dataset is read with one ``request_slices`` call for all the slices.
:func:`read_slices` works for read simulations and SOAP catalogues that are disk and remote.

When reading remote datasets :class:`RequestCap` holds the number of read requests across the MPI ranks to a maximum of
2 - 4 simultaneous requests.
"""

import os
import random
import time
from contextlib import contextmanager, nullcontext

import h5py
import numpy as np


def open_file(path, remote=None):
    """
    Open a snapshot file for reading, on disk or remotely: an h5py.File, or the hdfstream remote file ``remote[path]``.

    Both are context managers: ``with open_file(path, remote) as snapshot: ...``.

    Raises FileNotFoundError if there is no remote file at the path.

    :param  path:   path of the file: on disk, or relative to the remote directory
    :type   path:   str
    :param  remote: the remote directory the path is in (an hdfstream.RemoteDirectory). If None, the file is on disk
    :type   remote: hdfstream.RemoteDirectory
    """
    if remote is None:
        return h5py.File(path, "r")
    try:
        return remote[str(path)]
    except KeyError:
        raise FileNotFoundError(f"no remote file {path}") from None


def file_exists(path, remote=None):
    """
    Returns True if there is a file at the path, on disk or in the remote directory.

    A remote directory's listing is requested once and kept, so checking many files of one directory costs one request.

    :param  path:   path of the file: on disk, or relative to the remote directory
    :type   path:   str
    :param  remote: the remote directory the path is in. If None, on disk
    :type   remote: hdfstream.RemoteDirectory
    """
    if remote is None:
        return os.path.isfile(path)

    directory, name = os.path.split(str(path))

    try:
        listing = remote[directory] if directory else remote
        files = listing.files

    except (KeyError, AttributeError):
        return False

    return name in files


def directory_exists(path, remote=None):
    """
    Returns True if there is a directory at the path, on disk or in the remote directory.

    :param  path:   path of the directory: on disk, or relative to the remote directory ("" for the remote directory)
    :type   path:   str
    :param  remote: the remote directory the path is in. If None, on disk
    :type   remote: hdfstream.RemoteDirectory
    """
    if remote is None:
        return os.path.isdir(path)
    path = str(path).strip("/")
    if path in ("", "."):
        return True
    try:
        found = remote[path]
    except KeyError:
        return False

    return hasattr(found, "files")  # a directory lists its files, a file doesn't


def read_text(path, remote=None):
    """
    Returns the lines of a text file (e.g. a SWIFT output list), on disk or in the remote directory.

    Raises FileNotFoundError if there is no file at the path.

    :param  path:   path of the file: on disk, or relative to the remote directory
    :type   path:   str
    :param  remote: the remote directory the path is in. If None, on disk
    :type   remote: hdfstream.RemoteDirectory
    """
    if remote is None:
        with open(path, "r") as infile:
            return infile.read().splitlines()
    if not file_exists(path, remote):
        raise FileNotFoundError(f"no remote file {path}")
    with remote[str(path)].open("r") as infile:
        return infile.read().splitlines()


def check_slices(starts, lengths):
    """
    Check slices start in ascending order and have no overlap.

    Returns (starts, lengths) of the slices with rows, as int64 arrays.

    Raises a ValueError naming the broken rule.

    :param  starts:     first row of each slice
    :type   starts:     sequence of int
    :param  lengths:    number of rows of each slice
    :type   lengths:    sequence of int
    """
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if starts.shape != lengths.shape:
        raise ValueError(f"{len(starts)} starts for {len(lengths)} lengths")
    if np.any(lengths < 0) or np.any(starts < 0):
        raise ValueError("slices need starts and lengths of at least 0")
    keep = lengths > 0
    starts, lengths = starts[keep], lengths[keep]
    if np.any(starts[1:] < starts[:-1]):
        raise ValueError("slices must be in ascending order of their starts")
    if np.any(starts[1:] < starts[:-1] + lengths[:-1]):
        raise ValueError("slices must not overlap: merge neighbouring ranges first (e.g. merge_cells)")
    return starts, lengths


def read_slices(dataset, starts, lengths, dtype=None, cap=None):
    """
    Returns the rows [start, start + length) of a dataset for each slice given, joined in order along the first axis.

    A remote dataset is read with one request for all the slices and a local dataset is read per slice into one array.
    Slices with no assigned rows are skipped.

    :param  dataset:    the dataset to read, along its first axis: an h5py.Dataset or a remote dataset
    :type   dataset:    h5py.Dataset
    :param  starts:     first row of each slice, in ascending order
    :type   starts:     numpy.ndarray
    :param  lengths:    number of rows of each slice
    :type   lengths:    numpy.ndarray
    :param  dtype:      dtype of the array returned. If None, the dataset's
    :type   dtype:      numpy.dtype
    :param  cap:        held for the request, so at most its number of requests are in flight across MPI ranks. If
                        None, no cap
    :type   cap:        RequestCap
    """
    starts, lengths = check_slices(starts, lengths)
    dtype = np.dtype(dtype or dataset.dtype)
    values = np.empty((int(np.sum(lengths)),) + tuple(dataset.shape[1:]), dtype=dtype)

    if len(starts) == 0:
        return values

    with cap.slot() if cap is not None else nullcontext():
        # remote: one request for every slice
        if hasattr(dataset, "request_slices"):
            slices = [np.s_[int(start) : int(start + length)] for start, length in zip(starts, lengths)]
            if dtype == np.dtype(dataset.dtype):
                dataset.request_slices(slices, dest=values)
            else:
                values[...] = dataset.request_slices(slices)
        else:  # on disk: one read of the union of the slices
            file_space = dataset.id.get_space()
            trailing = tuple(int(n) for n in file_space.get_simple_extent_dims()[1:])
            file_space.select_none()
            for start, length in zip(starts, lengths):
                file_space.select_hyperslab(
                    (int(start),) + (0,) * len(trailing), (int(length),) + trailing, op=h5py.h5s.SELECT_OR
                )
            dataset.id.read(h5py.h5s.create_simple(values.shape), file_space, values)
    return values


@contextmanager
def _no_slot():
    yield


class RequestCap:
    """
    At most max_requests read requests at once across the MPI ranks.

    An MPI window on rank 0 holding two integers (the ongoing requests, the most request allowed).
    Before a request a rank takes a slot (adds 1, if that keeps the count below the limit; otherwise it
    waits and tries again), and gives it back after finishing the request or on an error. The number of ranks placing
    the particles is not capped.

    Making and freeing the cap is done collectively, i.e. every rank takes part.

    :ivar max_requests: the most requests in flight at once
    :vartype max_requests: int
    :ivar requests:     number of requests this rank made through the cap
    :vartype requests:  int
    :ivar wait:         seconds this rank waited for slots
    :vartype wait:      float

    :param  comm:           MPI communicator
    :type   comm:           mpi4py.MPI.Comm
    :param  max_requests:   the most requests in flight at once, at least 1
    :type   max_requests:   int
    """

    def __init__(self, comm, max_requests):  # noqa: D107
        from mpi4py import MPI

        if int(max_requests) < 1:
            raise ValueError(f"max_concurrent_requests must be at least 1, not {max_requests}")

        self._MPI = MPI
        self.comm = comm
        self._rank = comm.Get_rank()

        self.max_requests = int(max_requests)
        self.requests = 0
        self.wait = 0.0

        itemsize = MPI.INT64_T.Get_size()

        # index 0: current requests
        # index 1: highest number of simultaneous requests so far.
        self._win = MPI.Win.Allocate(2 * itemsize if self._rank == 0 else 0, itemsize, comm=comm)

        if self._rank == 0:
            self._win.Lock(0)  # lock window to rank 0
            self._win.Put(self._buffer([0, 0]), 0, (0, 2, MPI.INT64_T))  # start with index 0 and 1 = 0 , 0
            self._win.Unlock(0)  # release the lock

        # all ranks must reach this point
        comm.Barrier()

        self._win.Lock_all()  # allow this rank to perform operations on the window.
        self._random = random.Random(self._rank)  # ranks retry at different times to avoid clashing with each other

    def _buffer(self, values):
        """An MPI buffer of 64-bit integers, e.g. [array, MPI.INT64_T]."""
        return [np.asarray(values, dtype=np.int64), self._MPI.INT64_T]

    def _add(self, value, disp=0):
        """Add value to the integer in the window and returns the value before."""
        old = np.zeros(1, dtype=np.int64)
        # make sure rank operations are not simultaneous.
        self._win.Fetch_and_op(self._buffer([value]), [old, self._MPI.INT64_T], 0, disp, self._MPI.SUM)
        self._win.Flush(0)  # always complete at rank 0

        return int(old[0])

    @contextmanager
    def slot(self):
        """Hold a slot for one request and wait until fewer than max_requests are ongoing."""
        waited, delay = time.perf_counter(), 0.001

        # try to get a slot
        while True:
            in_flight = self._add(1) + 1  # new counter value
            if in_flight <= self.max_requests:
                break
            self._add(-1)

            # wait and then try again
            time.sleep(delay * (1 + 0.5 * self._random.random()))
            delay = min(2 * delay, 0.05)  # increase delay for each successful attempt

        # record waiting stats
        self.wait += time.perf_counter() - waited
        self.requests += 1

        # update current requests
        self._win.Accumulate(self._buffer([in_flight]), 0, (1, 1, self._MPI.INT64_T), self._MPI.MAX)
        self._win.Flush(0)

        # make the actual request from inside a with statement
        try:
            yield

        # make sure the slot is always released
        finally:
            self._add(-1)

    def most_in_flight(self):
        """Returns the most requests at once so far, across the ranks."""
        result = np.zeros(1, dtype=np.int64)
        self._win.Fetch_and_op(self._buffer([0]), [result, self._MPI.INT64_T], 0, 1, self._MPI.NO_OP)
        self._win.Flush(0)
        return int(result[0])

    def free(self):
        """Free the window, collectively; the cap can't be used after."""
        if self._win is not None:
            self._win.Unlock_all()
            self._win.Free()
            self._win = None
