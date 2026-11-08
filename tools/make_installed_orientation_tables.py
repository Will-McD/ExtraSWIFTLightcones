#!/bin/env python
"""
Make the orientation tables installed with SWIFTLET, as YAML files in ``swiftlet/data/orientation_tables``.

``default_v2.yaml``
    The default table for SnapshotLightcone. No two rows make a SWIFT cell clash with its own replica across the edge of
    neighbouring tiles, for any orientation_lock and orientation_seed
    (see docs/orientation_tables.md and docs/repeated_objects.md). The cell shifts of default_v2 are in eighths of the
    box (0, +-1/8, +-1/4, +-3/8 and 1/2), and the rows are chosen from every distinct orientation
    (48 turns and reflections, times 8^3 shifts = 24576), resulting in 5184 clash-free combinations, 108 for each of
    the 48 turns and reflections, with the shifts along each axis close to even.
``no_reorientation.yaml``
    One row that leaves the box unchanged: no quarter turns, reflections of +1 and no cell shifts. Every tile of a
    lightcone made with it is an exact periodic replica of the snapshot box, e.g. to compare with a re-oriented
    lightcone, or to check where each particle comes from.
``cube_quarter_spacing.yaml``
    For orientation_lock="cube". Every two rows keep the copies of each cell in neighbouring tiles (across a face, an
    edge or a corner) at least a quarter of the box apart, so with "cube" no cell is closer than L/4 to its own copy in
    the next layer, for any orientation_seed (it holds with no lock too, but not with "sphere"). Its 31 rows are copied
    here as CUBE_QUARTER_SPACING_ROWS.
``sphere_quarter_spacing.yaml``
    For orientation_lock="sphere". With "sphere" two layers meet inside the tiles crossing a layer sphere, so the copies
    of a cell from two rows can be in the same tile as well as in neighbouring ones. Every two rows keep them at least a
    quarter of the box apart in both, measured across the periodic box (the same tile and its 26 neighbours). So no cell
    centre is closer than L/4 to its own copy in the next layer, for any orientation_seed. Its 44 rows are copied here
    as SPHERE_QUARTER_SPACING_ROWS.

Every table is checked against its unique fingerprint.
"""

import argparse
import time
from pathlib import Path
import numpy as np
from swiftlet import snapshot_orientation as so

# the fingerprints of the installed tables, see snapshot_orientation.orientation_tables_fingerprint
FINGERPRINTS = {
    "default_v2": "7006e905808e256f",
    "no_reorientation": "a849af1c8e441133",
    "cube_quarter_spacing": "1162d17084cffd9a",
    "sphere_quarter_spacing": "9b3bb6051251e66a",
}

# the one row of no_reorientation: no quarter turns, no reflections (+1) and no cell shifts along x, y and z
NO_REORIENTATION_ROWS = ((0, 0, 0, 1, 1, 1, 0, 0, 0),)


# cube_quarter_spacing: quarter turns, reflections and cell shifts along x, y and z
# any two rows keep the copies of a cell in neighbouring tiles at least L/4 apart
CUBE_QUARTER_SPACING_ROWS = (
    (0, 0, 0, 1, 1, 1, 0, 0, 0),
    (0, 0, 1, -1, 1, 1, 4, 4, -2),
    (0, 0, 0, -1, 1, 1, -2, 0, 0),
    (0, 0, 1, 1, -1, 1, -1, 0, 2),
    (0, 0, 1, -1, -1, 1, -2, -3, -2),
    (0, 0, 1, 1, 1, 1, -1, 4, 2),
    (0, 0, 0, 1, -1, 1, 0, 2, 0),
    (0, 0, 0, -1, 1, 1, -1, 1, 4),
    (0, 0, 0, -1, -1, 1, 3, -3, 4),
    (0, 0, 0, -1, -1, 1, 2, 0, 4),
    (0, 0, 0, -1, 1, 1, -1, -2, 4),
    (0, 0, 0, -1, -1, 1, 3, 4, 0),
    (0, 0, 1, -1, -1, 1, 1, -3, -2),
    (0, 0, 1, -1, -1, 1, 3, 0, 2),
    (0, 0, 0, 1, -1, 1, 2, 3, 4),
    (0, 0, 1, 1, -1, 1, -3, 2, 2),
    (0, 0, 0, 1, -1, 1, 4, -2, 0),
    (0, 0, 1, -1, -1, 1, -1, 2, -2),
    (0, 0, 1, -1, -1, 1, 0, -1, -2),
    (0, 0, 1, -1, 1, 1, 3, 4, 2),
    (0, 0, 0, -1, -1, 1, -2, 4, 0),
    (0, 0, 1, -1, -1, 1, 2, 1, -2),
    (0, 0, 1, -1, 1, 1, 1, -2, 2),
    (0, 0, 0, 1, -1, 1, 0, -2, 0),
    (0, 0, 0, 1, -1, 1, 2, 2, 0),
    (0, 0, 1, 1, 1, 1, 1, 2, 2),
    (0, 0, 1, -1, 1, 1, 4, 0, -2),
    (0, 0, 0, 1, -1, 1, 4, 2, 0),
    (0, 0, 0, 1, 1, 1, -2, 3, 4),
    (0, 0, 0, -1, -1, 1, -2, 2, 0),
    (0, 0, 0, -1, -1, 1, 1, -3, 4),
    (0, 0, 0, 1, -1, 1, 4, 3, 4),
    (0, 0, 1, -1, -1, 1, -2, -1, -2),
    (0, 0, 0, -1, -1, 1, 4, 1, 4),
    (0, 0, 1, -1, -1, 1, 4, -2, -2),
    (0, 0, 1, -1, -1, 1, 2, 3, -2),
    (0, 0, 1, 1, -1, 1, -3, 0, 2),
    (0, 0, 1, 1, -1, 1, -1, 2, 2),
    (0, 0, 0, -1, -1, 1, 2, -2, 0),
    (0, 0, 0, -1, -1, 1, 4, -1, 4),
    (0, 0, 1, -1, -1, 1, 4, 2, -2),
    (0, 0, 1, 1, 1, 1, 1, 4, 2),
    (0, 0, 0, -1, -1, 1, -2, -2, 0),
    (0, 0, 1, -1, -1, 1, 2, -1, -2),
    (0, 0, 1, 1, 1, 1, -1, -2, 2),
    (0, 0, 1, -1, 1, 1, 3, -2, 2),
    (0, 0, 0, 1, -1, 1, 0, 4, 0),
    (0, 0, 1, 1, 1, 1, -3, 4, 2),
    (0, 0, 1, 1, 1, 1, -3, -2, 2),
)
# sphere_quarter_spacing: quarter turns, reflections and cell shifts along x, y and z
# any two rows keep the copies of a cell at least L/4 apart, in the same or neighbouring tiles
SPHERE_QUARTER_SPACING_ROWS = (
    (0, 0, 0, 1, 1, 1, 0, 0, 0),
    (1, 0, 0, 1, 1, 1, 4, 3, 1),
    (1, 0, 0, 1, 1, 1, 4, -2, 2),
    (1, 0, 0, 1, -1, 1, 1, -2, 2),
    (1, 0, 0, 1, -1, -1, 2, 4, 0),
    (0, 0, 0, 1, -1, -1, -2, 2, -2),
    (1, 0, 0, 1, -1, 1, 2, -2, -3),
    (0, 0, 0, 1, 1, -1, -2, -3, 0),
    (1, 0, 0, 1, 1, 1, 4, -2, 4),
    (1, 0, 0, 1, 1, 1, 4, 4, -3),
    (1, 0, 0, 1, -1, -1, 2, 0, -3),
    (1, 0, 0, 1, 1, 1, 4, -3, 0),
    (1, 0, 0, 1, -1, -1, 2, 0, -1),
    (0, 0, 0, 1, -1, -1, -2, -1, 3),
    (0, 0, 0, 1, -1, -1, -2, 4, 4),
    (1, 0, 0, 1, -1, -1, 2, 4, -2),
    (0, 0, 0, 1, -1, -1, -2, -1, -3),
    (1, 0, 0, 1, 1, 1, 4, 2, -3),
    (1, 0, 0, 1, -1, 1, 0, 0, -3),
    (1, 0, 0, 1, -1, 1, 1, 2, -1),
    (1, 0, 0, 1, 1, 1, 4, 2, 3),
    (0, 0, 0, 1, 1, -1, -2, -1, 1),
    (1, 0, 0, 1, -1, 1, 2, 2, 4),
    (1, 0, 0, 1, -1, -1, 2, 0, 1),
    (0, 0, 0, 1, -1, -1, -2, 1, 3),
    (1, 0, 0, 1, 1, -1, 4, 0, 0),
    (1, 0, 0, 1, 1, -1, 4, 0, -2),
    (1, 0, 0, 1, 1, -1, 4, 0, 2),
    (0, 0, 0, 1, 1, 1, 0, -3, -3),
    (0, 0, 0, 1, 1, 1, 0, 3, 4),
    (0, 0, 0, 1, 1, 1, 0, 1, 2),
    (0, 0, 0, 1, 1, -1, -2, 3, 1),
    (1, 0, 0, 1, -1, -1, 2, 4, 3),
    (0, 0, 0, 1, 1, -1, -2, 1, 0),
    (1, 0, 0, 1, -1, 1, 2, -2, 0),
    (1, 0, 0, 1, 1, 1, 4, 3, -1),
    (0, 0, 0, 1, 1, 1, 0, -2, -1),
    (1, 0, 0, 1, -1, 1, 2, 2, 1),
    (0, 0, 0, 1, -1, -1, -2, -3, -2),
    (1, 0, 0, 1, -1, 1, 0, 4, 1),
    (1, 0, 0, 1, 1, 1, 4, 4, 3),
    (1, 0, 0, 1, 1, 1, 4, -2, -2),
    (1, 0, 0, 1, 1, -1, 4, 0, 4),
    (1, 0, 0, 1, -1, -1, 2, 0, 3),
)


def tables_from_rows(rows):
    """
    Orientation tables from rows of quarter turns, reflections and cell shifts along x, y and z.

    Returns the tables as a dict, see snapshot_orientation.check_orientation_tables.

    :param  rows:   the rows, each of 9 whole numbers
    :type   rows:   tuple
    """
    rows = np.array(rows)
    tables = {name: rows[:, 3 * i : 3 * i + 3] for i, name in enumerate(so.ORIENTATION_TABLE_NAMES)}
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
        if name in ("no_reorientation", "cube_quarter_spacing", "sphere_quarter_spacing"):
            raise SystemExit(f"{name} has fingerprint {fingerprint}, not {FINGERPRINTS[name]}: not written")
        print(f"warning: {name} has fingerprint {fingerprint}. Doesn't match the installed table, {FINGERPRINTS[name]}")

    path = Path(output_dir) / f"{name}.yaml"
    so.write_orientation_tables(path, tables=tables, attributes={"name": name, **attributes})

    if so.orientation_tables_fingerprint(so.read_orientation_tables(path)) != fingerprint:
        raise SystemExit(f"{path} does not read back as the tables written")

    print(f"written {path}: {len(tables['QuarterTurns'])} rows, fingerprint {fingerprint}", flush=True)

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make the orientation tables installed with SWIFTLET")
    parser.add_argument("--output_dir", default="swiftlet/data/orientation_tables", help="directory to write to")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=["default_v2", "no_reorientation", "cube_quarter_spacing", "sphere_quarter_spacing"],
        choices=["default_v2", "no_reorientation", "cube_quarter_spacing", "sphere_quarter_spacing"],
        help="which tables to make",
    )
    parser.add_argument("--seed", type=int, default=21, help="seed of generate_clash_free_tables for default_v2")
    parser.add_argument(
        "--search_seeds",
        type=int,
        nargs=2,
        default=None,
        metavar=("FIRST", "LAST"),
        help="instead of --seed, use the most even seed from FIRST to LAST - 1 (slow)",
    )
    parser.add_argument(
        "--check",
        type=int,
        nargs="*",
        default=[16, 32],
        help="numbers of cells per axis to check default_v2 is clash-free for (64 takes minutes)",
    )
    args = parser.parse_args()

    if "no_reorientation" in args.tables:
        write_tables(
            args.output_dir,
            "no_reorientation",
            tables_from_rows(NO_REORIENTATION_ROWS),
            {
                "version": 1,
                "made_by": "tools/make_installed_orientation_tables.py: one row that leaves the box unchanged",
                "valid_nr_cells_axis": "any",
            },
        )

    if "cube_quarter_spacing" in args.tables:
        write_tables(
            args.output_dir,
            "cube_quarter_spacing",
            tables_from_rows(CUBE_QUARTER_SPACING_ROWS),
            {
                "version": 2,
                "made_by": (
                    "tools/make_installed_orientation_tables.py: quarter turns, reflections and shifts, every two rows "
                    "keeping the copies of a cell in neighbouring tiles at least L/4 apart"
                ),
                "for_orientation_lock": "cube",
                "min_copy_separation": "0.25 L, between the centres of the copies of a cell in neighbouring tiles",
                "valid_nr_cells_axis": "multiples of 8 (checked for 16, 32 and 64)",
            },
        )

    if "sphere_quarter_spacing" in args.tables:
        write_tables(
            args.output_dir,
            "sphere_quarter_spacing",
            tables_from_rows(SPHERE_QUARTER_SPACING_ROWS),
            {
                "version": 1,
                "made_by": (
                    "tools/make_installed_orientation_tables.py: quarter turns, reflections and shifts, every two rows "
                    "keeping the copies of a cell at least L/4 apart, in the same tile or a neighbouring one"
                ),
                "for_orientation_lock": "sphere (also cube and none)",
                "min_copy_separation": (
                    "0.25 L, between the centres of the copies of a cell in the same tile or neighbouring tiles"
                ),
                "valid_nr_cells_axis": "multiples of 8 (checked for 16, 32 and 64)",
            },
        )

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
        write_tables(
            args.output_dir,
            "default_v2",
            tables,
            {
                "version": 2,
                "made_by": (
                    "tools/make_installed_orientation_tables.py, snapshot_orientation.generate_clash_free_tables"
                ),
                "generator_seed": seed,
                "clash_free": True,
                "clash_free_for_nr_cells_axis": [16, 32, 64],
                "valid_nr_cells_axis": "multiples of 8; clash-free for 16 and more (checked for 16, 32 and 64)",
            },
        )
