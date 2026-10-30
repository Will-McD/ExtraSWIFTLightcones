#!/bin/env python
import argparse
import time
from pathlib import Path
import numpy as np
from swiftlet import snapshot_orientation as so

"""
Make the orientation tables installed with SWIFTLET, as YAML files in swiftlet/data/orientation_tables:

    default_v1.yaml:    the first default tables, 79 rows with the cell shifts in whole cells (for 32 cells per axis).
                        They were hard-coded in SWIFTLET before default_v2, and are copied here (DEFAULT_V1_ROWS), so
                        lightcones made with them can be made again.
    default_v2.yaml:    the default tables: no two rows make a SWIFT cell touch its own copy across the face between
                        neighbouring tiles, for any orientation_lock and orientation_seed (see docs/orientation_tables.md
                        and docs/repeated_objects.md).

The cell shifts of default_v2 are in eighths of the box (0, +-1/8, +-1/4, +-3/8 and 1/2), so the same table is
clash-free for every number of cells per axis that is a multiple of 8 and at least 16: the 16, 32 and 64 of the
FLAMINGO and COLIBRE simulations. The rows are chosen from every distinct orientation (48 turns and reflections, times
8^3 shifts = 24576) by snapshot_orientation.generate_clash_free_tables; seed 21 gives the most even tables of seeds 4
to 63: 5184 rows, 108 for each of the 48 turns and reflections, with the shifts along each axis close to even.

Both are checked against their fingerprints (the fingerprint depends only on the tables). 
"""

# the fingerprints of the installed tables, see snapshot_orientation.orientation_tables_fingerprint
FINGERPRINTS = {"default_v1": "dd43da3e96d8add4", "default_v2": "7006e905808e256f"}

# the rows of default_v1: quarter turns, reflections and cell shifts (in eighths of the box) along x, y and z
DEFAULT_V1_ROWS = (
    ( 0,  0,  0,  1,  1,  1,  0,  0,  0), ( 0,  1, -1, -1,  1, -1,  1,  0,  4), (-1, -1, -2, -1,  1, -1,  2, -2,  2),
    ( 1,  0, -1,  1,  1,  1,  1,  0,  1), (-1,  0,  1, -1,  1, -1, -2,  2, -2), ( 1, -2,  0, -1,  1, -1,  4, -2,  4),
    (-1,  2,  1, -1,  1,  1,  1,  1,  4), ( 1,  1,  2,  1,  1,  1, -1,  0, -1), (-2,  2,  1, -1,  1, -1,  2, -1, -1),
    ( 2,  2, -2, -1,  1,  1,  0, -2,  0), (-2,  1, -2,  1,  1, -1, -2,  2,  1), ( 0, -2, -1, -1,  1,  1,  4, -2, -2),
    (-2, -2,  2,  1,  1,  1,  4,  4,  1), (-2,  0, -1,  1,  1, -1,  0,  0, -1), ( 1,  1,  2, -1, -1, -1,  4,  4,  4),
    (-1,  2,  2, -1, -1,  1,  0,  1, -2), ( 1,  0,  1, -1, -1,  1,  1,  4, -1), ( 1,  0,  2, -1,  1,  1,  2,  0,  1),
    (-1,  2,  2, -1,  1,  1,  4, -2,  4), ( 1,  0, -2,  1,  1,  1,  4, -2,  0), ( 1, -1,  2,  1, -1, -1,  4,  4, -1),
    ( 0,  1, -1,  1,  1,  1, -2,  2,  0), ( 1,  1,  0,  1,  1, -1,  0,  2,  4), (-1, -1,  2,  1, -1,  1,  4,  0,  2),
    ( 1, -2, -2,  1, -1, -1, -2,  4,  0), ( 0, -2, -2,  1, -1,  1,  4,  4,  2), ( 1,  0, -2,  1, -1, -1,  1,  4,  4),
    ( 0,  2,  0,  1,  1,  1,  0,  0,  0), ( 1,  2, -2, -1, -1,  1, -2,  1,  0), (-1,  1,  0, -1,  1, -1,  4,  2,  4),
    (-2, -2, -1,  1,  1,  1, -1,  4, -1), ( 0,  0,  1, -1,  1,  1,  1,  4, -1), (-2, -1, -2, -1, -1, -1,  4,  4,  2),
    ( 2, -2,  2,  1,  1, -1,  4,  4,  4), ( 0,  1, -2,  1,  1,  1,  4, -2,  1), ( 2, -1, -1, -1, -1,  1, -2, -1,  4),
    (-2, -1, -2, -1, -1,  1, -2,  0,  1), (-2,  1,  2,  1, -1, -1,  0,  4,  4), ( 0, -2,  2,  1, -1, -1, -1,  1,  1),
    (-1, -2, -1,  1,  1, -1,  1,  1,  1), ( 0, -2, -2, -1, -1, -1,  0,  2,  0), ( 0,  0,  0, -1,  1,  1,  2,  0,  4),
    ( 2,  0, -2,  1, -1, -1, -1,  4, -1), (-2, -2,  1,  1,  1, -1,  4, -1,  1), (-2,  1, -2,  1, -1, -1, -2,  1,  0),
    ( 1, -1,  2, -1, -1,  1,  0, -1,  1), (-2,  1, -1, -1, -1,  1,  0,  4,  1), (-2, -1,  1, -1,  1,  1,  2,  4,  4),
    (-2, -2,  0, -1,  1, -1,  1, -1,  4), (-1,  2,  2,  1, -1,  1,  4,  2,  0), ( 0, -2,  2, -1, -1,  1,  0,  0, -1),
    (-1, -2, -2,  1,  1, -1,  4, -1,  4), ( 0,  0,  0, -1, -1, -1,  2,  4, -2), ( 2, -2,  0, -1, -1,  1, -1, -1,  1),
    ( 2,  2, -1, -1, -1, -1,  1, -2, -2), ( 2,  2,  2,  1, -1,  1,  0,  0, -1), ( 2, -2,  2, -1, -1, -1,  0,  4,  0),
    ( 2,  1,  2, -1,  1, -1,  1,  1, -2), ( 2, -1,  1, -1, -1,  1,  4,  0,  4), ( 0,  2, -1,  1,  1, -1, -2,  4,  4),
    (-1,  1,  1, -1,  1, -1,  1, -1,  4), ( 2,  0,  0, -1,  1,  1,  0, -1, -2), ( 1,  2,  0, -1,  1,  1, -1,  1,  2),
    (-2,  1,  1,  1, -1,  1,  2, -2,  4), (-2, -1,  1, -1,  1, -1,  0,  0,  4), (-2,  0, -1, -1,  1,  1,  2,  1, -1),
    (-1,  1, -1,  1, -1, -1, -1,  1, -1), (-2, -2, -1, -1,  1, -1,  4, -2, -1), (-2,  2,  2,  1,  1,  1,  0,  2,  4),
    ( 0,  0,  0, -1, -1,  1,  4,  1,  0), ( 1,  2,  1, -1,  1, -1,  2,  0,  2), ( 2,  0,  2,  1, -1, -1,  0,  2,  0),
    (-1, -2,  0, -1, -1, -1,  1, -2,  0), ( 2, -1, -1, -1, -1,  1, -1,  4, -1), ( 0, -2, -2,  1,  1,  1,  4, -1,  2),
    (-2, -2,  0,  1,  1, -1,  1,  2,  4), (-1, -1, -2,  1,  1, -1,  1,  1, -1), (-2, -2,  0, -1, -1, -1, -2,  0,  4),
    (-1,  0,  0,  1, -1,  1, -1,  4,  2),
)


def default_v1_tables():
    """
    The first default tables, from DEFAULT_V1_ROWS.

    Returns the tables as a dict, see snapshot_orientation.check_orientation_tables.
    """
    rows = np.array(DEFAULT_V1_ROWS)
    tables = {name: rows[:, 3 * i:3 * i + 3] for i, name in enumerate(so.ORIENTATION_TABLE_NAMES)}
    return so.check_orientation_tables(tables)


def most_even_seed(seeds):
    """
    The seed of generate_clash_free_tables giving the most rows for every turn and reflection, from those given.

    Returns (seed, rows for each turn and reflection).

    :param  seeds:  seeds to try
    :type   seeds:  iterable of int
    """
    best_seed, best_rows = None, -1
    for seed in seeds:
        tables = so.generate_clash_free_tables(seed=seed)
        rows_each = len(tables["QuarterTurns"]) // 48
        print(f"seed {seed}: {len(tables['QuarterTurns'])} rows, {rows_each} for each turn and reflection", flush=True)
        if rows_each > best_rows:
            best_seed, best_rows = seed, rows_each
    return best_seed, best_rows


def write_tables(output_dir, name, tables, attributes):
    """
    Write installed tables to output_dir/<name>.yaml, checking their fingerprint.

    Returns the path written.
    """
    fingerprint = so.orientation_tables_fingerprint(tables)
    if fingerprint != FINGERPRINTS[name]:
        if name == "default_v1":
            raise SystemExit(f"default_v1 has fingerprint {fingerprint}, not {FINGERPRINTS[name]}: not written")
        
        print(f"warning: {name} has fingerprint {fingerprint}, not that of the installed {name}, {FINGERPRINTS[name]}: these are new tables")
    
    path = Path(output_dir) / f"{name}.yaml"
    so.write_orientation_tables(path, tables=tables, attributes={"name": name, **attributes})
    
    if so.orientation_tables_fingerprint(so.read_orientation_tables(path)) != fingerprint:
        raise SystemExit(f"{path} does not read back as the tables written")
    
    print(f"written {path}: {len(tables['QuarterTurns'])} rows, fingerprint {fingerprint}", flush=True)
    
    return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make the orientation tables installed with SWIFTLET")
    parser.add_argument("--output_dir", default="swiftlet/data/orientation_tables", help="directory to write to")
    parser.add_argument("--tables", nargs="+", default=["default_v1", "default_v2"],
                        choices=["default_v1", "default_v2"], help="which tables to make")
    parser.add_argument("--seed", type=int, default=21, help="seed of generate_clash_free_tables for default_v2")
    parser.add_argument("--search_seeds", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                        help="instead of --seed, use the most even seed from FIRST to LAST - 1 (slow)")
    parser.add_argument("--check", type=int, nargs="*", default=[16, 32],
                        help="numbers of cells per axis to check default_v2 is clash-free for (64 takes minutes)")
    args = parser.parse_args()

    if "default_v1" in args.tables:
        write_tables(args.output_dir, "default_v1", default_v1_tables(), 
        {
            "version": 1,
            "made_by": "tools/make_installed_orientation_tables.py, from the hard-coded default tables of SWIFTLET before default_v2",
            "valid_nr_cells_axis": "any; the shifts are in cells, chosen for 32 cells per axis",
        })

    if "default_v2" in args.tables:
        seed = args.seed
        if args.search_seeds is not None:
            seed, _ = most_even_seed(range(*args.search_seeds))
            print(f"most even: seed {seed}")
        start = time.time()
        tables = so.generate_clash_free_tables(seed=seed)
        print(f"default_v2: {len(tables['QuarterTurns'])} rows in {time.time() - start:.0f} s", flush=True)
        for nr_cells_axis in args.check:
            clashes = so.tables_are_clash_free(tables, nr_cells_axis)
            print(f"{nr_cells_axis} cells per axis: {clashes} clashing pairs of rows", flush=True)
            if clashes:
                raise SystemExit("default_v2 is not clash-free, not written")
        write_tables(args.output_dir, "default_v2", tables, {
            "version": 2,
            "made_by": "tools/make_installed_orientation_tables.py, snapshot_orientation.generate_clash_free_tables",
            "generator_seed": seed,
            "clash_free": True,
            "clash_free_for_nr_cells_axis": [16, 32, 64],
            "valid_nr_cells_axis": "multiples of 8; clash-free for 16 and more (checked for 16, 32 and 64)",
        })


