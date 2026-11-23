#!/bin/env python
"""
Count the ranks reading remote files and the expected number of particles added to the lightcone per file read.

Include an option to make fake remote files of snapshots to test against reading local fake snapshots.
"""

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import h5py
import numpy as np


def _rows_of_key(key, nr_rows):
    """
    Returns the rows a dataset read selects along its first axis, as a list of (start, stop) ranges.

    :param  key:        the index given to the dataset, e.g. a slice, an int, an array of ints, Ellipsis or a tuple
    :type   key:        object
    :param  nr_rows:    length of the dataset's first axis (1 for a scalar dataset)
    :type   nr_rows:    int
    """
    key = key if isinstance(key, tuple) else (key,)
    first = key[0] if key else Ellipsis
    if first is Ellipsis or isinstance(first, (str, bytes)):  # the whole first axis (or fields of every row)
        return [(0, nr_rows)] if nr_rows > 0 else []
    if isinstance(first, slice):
        start, stop, step = first.indices(nr_rows)
        if step == 1:
            return [(start, stop)] if stop > start else []
        return _runs(np.arange(start, stop, step))
    if isinstance(first, (int, np.integer)):
        row = int(first) + (nr_rows if first < 0 else 0)
        return [(row, row + 1)]
    rows = np.asarray(first)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    rows = rows.astype(np.int64).ravel()
    return _runs(np.where(rows < 0, rows + nr_rows, rows))


def _runs(rows):
    """
    Returns sorted rows as ranges of consecutive rows [(start, stop), ...], each row in a range as often as it is given.

    :param  rows:   row numbers
    :type   rows:   numpy.ndarray
    """
    if len(rows) == 0:
        return []
    rows = np.sort(rows)
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    return [(int(run[0]), int(run[-1]) + 1) for run in np.split(rows, breaks)]


def _merged_length(ranges):
    """
    Returns the number of distinct rows covered by row ranges [(start, stop), ...].

    :param  ranges: row ranges, overlapping or not
    :type   ranges: list
    """
    total, end = 0, None
    for start, stop in sorted(ranges):
        if end is None or start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


@dataclass
class DatasetReads:
    """
    The reads of one dataset of one file.

    :ivar requests:     number of reads (calls) of the dataset
    :vartype requests:  int
    :ivar ranges:       row ranges read along the first axis, [(start, stop), ...], one or more per read
    :vartype ranges:    list
    """

    requests: int = 0
    ranges: list = field(default_factory=list)

    @property
    def rows_read(self):
        """Rows read, counting a row each time it is read."""
        return sum(stop - start for start, stop in self.ranges)

    @property
    def unique_rows(self):
        """Distinct rows read."""
        return _merged_length(self.ranges)

    @property
    def rows_read_again(self):
        """Rows read beyond the first time they were read."""
        return self.rows_read - self.unique_rows


def _is_filename(name):
    return isinstance(name, (str, bytes, os.PathLike))


class ReadCounter:
    """
    Record every HDF5 file opened for reading and every dataset read, while active (as a context manager).

    h5py.File and h5py.Dataset are patched for the duration only, so every caller is counted, SWIFTLET's and any
    other. Files opened to write, or opened before the counter started, are not counted.

    .. code-block:: python

        with ReadCounter() as counter:
            lightcone.place_snapshot_particles_in_shell((0.1, 0.2), ["Coordinates"], "PartType1")
        print(counter.format_table(select=ReadCounter.particle_datasets))

    :ivar opens:    number of times each file (real path) was opened for reading
    :vartype opens: collections.Counter
    :ivar reads:    the reads of each dataset, {(file, dataset name): DatasetReads}
    :vartype reads: dict
    """

    def __init__(self):  # noqa: D107
        self.opens = Counter()
        self.reads = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        self._originals = None

    # ---- which datasets ----

    @staticmethod
    def particle_datasets(filename, dataset):
        """Select the particle datasets of snapshots, e.g. /PartType1/Coordinates."""
        return dataset.startswith("/PartType")

    @staticmethod
    def cell_datasets(filename, dataset):
        """Select the cell metadata of snapshots, e.g. /Cells/OffsetsInFile/PartType1."""
        return dataset.startswith("/Cells/")

    # ---- patching ----

    def __enter__(self):  # noqa: D105
        counter = self
        file_init = h5py.File.__init__
        getitem = h5py.Dataset.__getitem__
        read_direct = h5py.Dataset.read_direct
        self._originals = (file_init, getitem, read_direct)

        def counted_file_init(file, name, *args, **kwargs):
            file_init(file, name, *args, **kwargs)
            # h5py also makes File objects from open file ids (e.g. dataset.file): those are not opens
            if _is_filename(name) and file.mode == "r":
                counter._record_open(file.filename)

        def counted_getitem(dataset, key, *args, **kwargs):
            if not getattr(counter._local, "busy", False):
                counter._record_read(dataset, key)
            counter._local.busy = True
            try:
                return getitem(dataset, key, *args, **kwargs)
            finally:
                counter._local.busy = False

        def counted_read_direct(dataset, dest, source_sel=None, dest_sel=None):
            if not getattr(counter._local, "busy", False):
                counter._record_read(dataset, Ellipsis if source_sel is None else source_sel)
            counter._local.busy = True
            try:
                return read_direct(dataset, dest, source_sel, dest_sel)
            finally:
                counter._local.busy = False

        h5py.File.__init__ = counted_file_init
        h5py.Dataset.__getitem__ = counted_getitem
        h5py.Dataset.read_direct = counted_read_direct
        return self

    def __exit__(self, *exc):  # noqa: D105
        h5py.File.__init__, h5py.Dataset.__getitem__, h5py.Dataset.read_direct = self._originals
        self._originals = None
        return False

    def _record_open(self, filename):
        with self._lock:
            self.opens[os.path.realpath(filename)] += 1

    def _record_read(self, dataset, key):
        self._local.busy = True  # dataset.file and dataset.shape must not count as reads or opens
        try:
            filename = os.path.realpath(dataset.file.filename)
            nr_rows = dataset.shape[0] if dataset.shape else 1
            ranges = _rows_of_key(key, nr_rows)
        finally:
            self._local.busy = False
        with self._lock:
            entry = self.reads.setdefault((filename, dataset.name), DatasetReads())
            entry.requests += 1
            entry.ranges.extend(ranges)

    # ---- reports ----

    def rows(self, select=None):
        """
        Returns one dict per (file, dataset) read: file, dataset, opens (of its file), requests, rows_read,
        unique_rows and rows_read_again, sorted by file then dataset.

        :param  select: select(file, dataset) -> bool, the datasets to include, e.g. ReadCounter.particle_datasets.
                        If None, all
        :type   select: callable
        """
        rows = []
        for (filename, dataset), entry in sorted(self.reads.items()):
            if select is not None and not select(filename, dataset):
                continue
            rows.append(
                {
                    "file": filename,
                    "dataset": dataset,
                    "opens": self.opens[filename],
                    "requests": entry.requests,
                    "rows_read": entry.rows_read,
                    "unique_rows": entry.unique_rows,
                    "rows_read_again": entry.rows_read_again,
                }
            )
        return rows

    def totals(self, select=None):
        """
        Returns the totals over the datasets selected: files (read from), opens (of those files), requests,
        rows_read, unique_rows, rows_read_again and reread_factor (rows_read / unique_rows, 1 if nothing is read
        twice).

        :param  select: the datasets to include, see rows. If None, all
        :type   select: callable
        """
        rows = self.rows(select)
        files = sorted({row["file"] for row in rows})
        totals = {
            "files": len(files),
            "opens": sum(self.opens[name] for name in files),
            "requests": sum(row["requests"] for row in rows),
            "rows_read": sum(row["rows_read"] for row in rows),
            "unique_rows": sum(row["unique_rows"] for row in rows),
            "rows_read_again": sum(row["rows_read_again"] for row in rows),
        }
        totals["reread_factor"] = totals["rows_read"] / totals["unique_rows"] if totals["unique_rows"] else 1.0
        return totals

    def format_table(self, select=None, root=None):
        """
        Returns the reads as a markdown table, one row per (file, dataset), and a row of totals.

        :param  select: the datasets to include, see rows. If None, all
        :type   select: callable
        :param  root:   directory the file names are shown relative to. If None, the file names only
        :type   root:   str or pathlib.Path
        """

        def name(filename):
            return os.path.relpath(filename, root) if root is not None else os.path.basename(filename)

        lines = [
            "| file | dataset | opens | requests | rows read | unique rows | rows read again |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in self.rows(select):
            lines.append(
                f"| {name(row['file'])} | {row['dataset']} | {row['opens']} | {row['requests']} | {row['rows_read']} "
                f"| {row['unique_rows']} | {row['rows_read_again']} |"
            )
        t = self.totals(select)
        lines.append(
            f"| **{t['files']} files** | | **{t['opens']}** | **{t['requests']}** | **{t['rows_read']}** "
            f"| **{t['unique_rows']}** | **{t['rows_read_again']}** (x{t['reread_factor']:.2f}) |"
        )
        return "\n".join(lines)


# A stand-in for hdfstream's remote files, to use later


def check_request_slices(slices, ndim):
    """
    Check the slices of one request_slices call against the rules of the hdfstream service.

    The rules: the starts in the first dimension ascend, the slices don't overlap, the indexes in the other dimensions
    are the same for every slice, and slices are joined along the first dimension only (a step of 1).

    Returns a list of (start, stop) for the first dimension, in order. Raises a ValueError naming the broken rule.

    :param  slices: slice objects, e.g. [np.s_[10:20, :], np.s_[50:60, :]]
    :type   slices: sequence
    :param  ndim:   number of dimensions of the dataset
    :type   ndim:   int
    """
    ranges, others = [], None
    for i, key in enumerate(slices):
        key = key if isinstance(key, tuple) else (key,)
        if not key or not isinstance(key[0], slice):
            raise ValueError(f"slice {i}: the first dimension must be a slice, got {key!r}")
        first, rest = key[0], tuple(key[1:])
        if first.start is None or first.stop is None:
            raise ValueError(f"slice {i}: give the start and stop of the first dimension, got {first!r}")
        if first.step not in (None, 1):
            raise ValueError(f"slice {i}: slices are joined along the first dimension only (step 1), got {first!r}")
        if len(key) > ndim:
            raise ValueError(f"slice {i}: {len(key)} indexes for a dataset of {ndim} dimensions")
        if others is None:
            others = rest
        elif rest != others:
            raise ValueError(f"slice {i}: indexes in the other dimensions must not differ, {rest!r} != {others!r}")
        if ranges and first.start < ranges[-1][0]:
            raise ValueError(f"slice {i}: starts must be in ascending order, {first.start} after {ranges[-1][0]}")
        if ranges and first.start < ranges[-1][1]:
            raise ValueError(f"slice {i}: slices must not overlap, {first.start} is inside {ranges[-1]}")
        ranges.append((int(first.start), int(first.stop)))
    return ranges


@dataclass
class RemoteRequest:
    """
    One request to the fake service.

    :ivar file:         path of the file, relative to the directory
    :vartype file:      str
    :ivar dataset:      name of the dataset
    :vartype dataset:   str
    :ivar nr_slices:    number of slices in the request (1 for an ordinary index)
    :vartype nr_slices: int
    :ivar rows:         number of rows returned
    :vartype rows:      int
    :ivar start:        when the request started (time.time(), comparable between processes on one machine)
    :vartype start:     float
    :ivar end:          when it returned
    :vartype end:       float
    """

    file: str
    dataset: str
    nr_slices: int
    rows: int
    start: float = 0.0
    end: float = 0.0


def max_in_flight(intervals):
    """
    Returns the most requests at once, i.e. the largest number of [start, end) intervals overlapping at any time.

    A request starting exactly when another ends is not counted as overlapping it.

    :param  intervals:  (start, end) of each request, e.g. of RemoteRequests (from any number of processes or ranks)
    :type   intervals:  iterable of tuple
    """
    intervals = list(intervals)  # read twice below, so not a generator
    events = sorted([(end, -1) for _, end in intervals] + [(start, 1) for start, _ in intervals])
    in_flight, most = 0, 0
    for _, change in events:  # at equal times an end (-1) sorts before a start (+1)
        in_flight += change
        most = max(most, in_flight)
    return most


class RequestLog:
    """
    What a FakeRemoteDirectory was asked for: the files opened and every data request.

    :ivar opens:        number of times each file (path relative to the directory) was opened
    :vartype opens:     collections.Counter
    :ivar requests:     every data request, in order, as RemoteRequest
    :vartype requests:  list
    """

    def __init__(self):
        self.opens = Counter()
        self.requests = []
        self._lock = threading.Lock()

    def add(self, request):
        """Record a request."""
        with self._lock:
            self.requests.append(request)

    def requests_of(self, file=None, dataset=None):
        """Returns the requests of one file and/or dataset."""
        return [
            r for r in self.requests if (file is None or r.file == file) and (dataset is None or r.dataset == dataset)
        ]

    def totals(self):
        """
        Returns the totals of the log: files (opened), opens, requests, slices, rows and max_in_flight (the most
        requests at once).
        """
        return {
            "files": len(self.opens),
            "opens": sum(self.opens.values()),
            "requests": len(self.requests),
            "slices": sum(r.nr_slices for r in self.requests),
            "rows": sum(r.rows for r in self.requests),
            "max_in_flight": max_in_flight((r.start, r.end) for r in self.requests),
        }


class FakeRemoteDataset:
    """
    A dataset of a FakeRemoteFile: shape, dtype and attrs as h5py's, reads by index or with request_slices, each read
    one request.
    """

    def __init__(self, dataset, relpath, log, delay=0.0):
        self._dataset, self._relpath, self._log, self._delay = dataset, relpath, log, delay
        self.name = dataset.name
        self.shape = dataset.shape
        self.dtype = dataset.dtype
        self.ndim = dataset.ndim
        self.attrs = dict(dataset.attrs)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        start = time.time()
        values = self._dataset[key]
        time.sleep(self._delay)
        rows = int(np.shape(values)[0]) if np.ndim(values) else 1
        self._log.add(RemoteRequest(self._relpath, self.name, 1, rows, start, time.time()))
        return values

    def request_slices(self, slices):
        """
        Returns the slices, joined along the first dimension, in one request.

        :param  slices: slice objects following the rules of check_request_slices
        :type   slices: sequence
        """
        ranges = check_request_slices(slices, self.ndim)
        start = time.time()
        parts = [self._dataset[key] for key in slices]
        values = np.concatenate(parts) if parts else self._dataset[0:0]
        time.sleep(self._delay)
        rows = sum(b - a for a, b in ranges)
        self._log.add(RemoteRequest(self._relpath, self.name, len(ranges), rows, start, time.time()))
        return values


class FakeRemoteGroup:
    """A group of a FakeRemoteFile: members by name, attrs and membership as h5py's."""

    def __init__(self, group, relpath, log, delay=0.0):  # noqa: D107
        self._group, self._relpath, self._log, self._delay = group, relpath, log, delay
        self.name = group.name
        self.attrs = dict(group.attrs)

    def __contains__(self, name):  # noqa: D105
        return name in self._group

    def keys(self):
        """Names of the members."""
        return self._group.keys()

    def __getitem__(self, name):  # noqa: D105
        member = self._group[name]
        if isinstance(member, h5py.Dataset):
            return FakeRemoteDataset(member, self._relpath, self._log, self._delay)
        return FakeRemoteGroup(member, self._relpath, self._log, self._delay)


class FakeRemoteFile(FakeRemoteGroup):
    """A file of a FakeRemoteDirectory; a context manager, closed with close()."""

    def __init__(self, path, relpath, log, delay=0.0):
        self._file = h5py.File(path, "r")
        super().__init__(self._file, relpath, log, delay)
        self.filename = relpath
        log.opens[relpath] += 1

    def close(self):
        """Close the file."""
        self._file.close()

    def __enter__(self):  # noqa: D105
        return self

    def __exit__(self, *exc):  # noqa: D105
        self.close()
        return False


class FakeRemoteDirectory:
    """
    A stand-in for hdfstream.RemoteDirectory over a local directory: remote_dir[path] opens a file, and every file
    opened and data request made is recorded in remote_dir.log (a RequestLog).

    :param  root:   the local directory the paths are relative to
    :type   root:   str or pathlib.Path
    """

    def __init__(self, root, delay=0.0):  # noqa: D107
        self.root = Path(root)
        self.delay = delay
        self.log = RequestLog()

    def __getitem__(self, path):  # noqa: D105
        relpath = str(Path(path).relative_to(self.root)) if Path(path).is_absolute() else str(path)
        return FakeRemoteFile(self.root / relpath, relpath, self.log, self.delay)
