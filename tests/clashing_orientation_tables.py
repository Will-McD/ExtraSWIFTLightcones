"""
Orientation tables that make cells clash, for the tests that need clashing cells.

These are the first default tables of SWIFTLET (installed as default_v1 until it was removed): 79 rows of quarter
turns, reflections and cell shifts (in eighths of the box: 0, +-1, +-2 and 4) along x, y and z, chosen before the
clash-free tables, so some pairs of rows make a cell clash with its own copy in a neighbouring tile.
"""

from swiftlet import snapshot_orientation as so

# the fingerprint of these tables, see snapshot_orientation.orientation_tables_fingerprint
CLASHING_FINGERPRINT = "dd43da3e96d8add4"

# quarter turns, reflections and cell shifts along x, y and z
CLASHING_ROWS = (
    (0, 0, 0, 1, 1, 1, 0, 0, 0),
    (0, 1, -1, -1, 1, -1, 1, 0, 4),
    (-1, -1, -2, -1, 1, -1, 2, -2, 2),
    (1, 0, -1, 1, 1, 1, 1, 0, 1),
    (-1, 0, 1, -1, 1, -1, -2, 2, -2),
    (1, -2, 0, -1, 1, -1, 4, -2, 4),
    (-1, 2, 1, -1, 1, 1, 1, 1, 4),
    (1, 1, 2, 1, 1, 1, -1, 0, -1),
    (-2, 2, 1, -1, 1, -1, 2, -1, -1),
    (2, 2, -2, -1, 1, 1, 0, -2, 0),
    (-2, 1, -2, 1, 1, -1, -2, 2, 1),
    (0, -2, -1, -1, 1, 1, 4, -2, -2),
    (-2, -2, 2, 1, 1, 1, 4, 4, 1),
    (-2, 0, -1, 1, 1, -1, 0, 0, -1),
    (1, 1, 2, -1, -1, -1, 4, 4, 4),
    (-1, 2, 2, -1, -1, 1, 0, 1, -2),
    (1, 0, 1, -1, -1, 1, 1, 4, -1),
    (1, 0, 2, -1, 1, 1, 2, 0, 1),
    (-1, 2, 2, -1, 1, 1, 4, -2, 4),
    (1, 0, -2, 1, 1, 1, 4, -2, 0),
    (1, -1, 2, 1, -1, -1, 4, 4, -1),
    (0, 1, -1, 1, 1, 1, -2, 2, 0),
    (1, 1, 0, 1, 1, -1, 0, 2, 4),
    (-1, -1, 2, 1, -1, 1, 4, 0, 2),
    (1, -2, -2, 1, -1, -1, -2, 4, 0),
    (0, -2, -2, 1, -1, 1, 4, 4, 2),
    (1, 0, -2, 1, -1, -1, 1, 4, 4),
    (0, 2, 0, 1, 1, 1, 0, 0, 0),
    (1, 2, -2, -1, -1, 1, -2, 1, 0),
    (-1, 1, 0, -1, 1, -1, 4, 2, 4),
    (-2, -2, -1, 1, 1, 1, -1, 4, -1),
    (0, 0, 1, -1, 1, 1, 1, 4, -1),
    (-2, -1, -2, -1, -1, -1, 4, 4, 2),
    (2, -2, 2, 1, 1, -1, 4, 4, 4),
    (0, 1, -2, 1, 1, 1, 4, -2, 1),
    (2, -1, -1, -1, -1, 1, -2, -1, 4),
    (-2, -1, -2, -1, -1, 1, -2, 0, 1),
    (-2, 1, 2, 1, -1, -1, 0, 4, 4),
    (0, -2, 2, 1, -1, -1, -1, 1, 1),
    (-1, -2, -1, 1, 1, -1, 1, 1, 1),
    (0, -2, -2, -1, -1, -1, 0, 2, 0),
    (0, 0, 0, -1, 1, 1, 2, 0, 4),
    (2, 0, -2, 1, -1, -1, -1, 4, -1),
    (-2, -2, 1, 1, 1, -1, 4, -1, 1),
    (-2, 1, -2, 1, -1, -1, -2, 1, 0),
    (1, -1, 2, -1, -1, 1, 0, -1, 1),
    (-2, 1, -1, -1, -1, 1, 0, 4, 1),
    (-2, -1, 1, -1, 1, 1, 2, 4, 4),
    (-2, -2, 0, -1, 1, -1, 1, -1, 4),
    (-1, 2, 2, 1, -1, 1, 4, 2, 0),
    (0, -2, 2, -1, -1, 1, 0, 0, -1),
    (-1, -2, -2, 1, 1, -1, 4, -1, 4),
    (0, 0, 0, -1, -1, -1, 2, 4, -2),
    (2, -2, 0, -1, -1, 1, -1, -1, 1),
    (2, 2, -1, -1, -1, -1, 1, -2, -2),
    (2, 2, 2, 1, -1, 1, 0, 0, -1),
    (2, -2, 2, -1, -1, -1, 0, 4, 0),
    (2, 1, 2, -1, 1, -1, 1, 1, -2),
    (2, -1, 1, -1, -1, 1, 4, 0, 4),
    (0, 2, -1, 1, 1, -1, -2, 4, 4),
    (-1, 1, 1, -1, 1, -1, 1, -1, 4),
    (2, 0, 0, -1, 1, 1, 0, -1, -2),
    (1, 2, 0, -1, 1, 1, -1, 1, 2),
    (-2, 1, 1, 1, -1, 1, 2, -2, 4),
    (-2, -1, 1, -1, 1, -1, 0, 0, 4),
    (-2, 0, -1, -1, 1, 1, 2, 1, -1),
    (-1, 1, -1, 1, -1, -1, -1, 1, -1),
    (-2, -2, -1, -1, 1, -1, 4, -2, -1),
    (-2, 2, 2, 1, 1, 1, 0, 2, 4),
    (0, 0, 0, -1, -1, 1, 4, 1, 0),
    (1, 2, 1, -1, 1, -1, 2, 0, 2),
    (2, 0, 2, 1, -1, -1, 0, 2, 0),
    (-1, -2, 0, -1, -1, -1, 1, -2, 0),
    (2, -1, -1, -1, -1, 1, -1, 4, -1),
    (0, -2, -2, 1, 1, 1, 4, -1, 2),
    (-2, -2, 0, 1, 1, -1, 1, 2, 4),
    (-1, -1, -2, 1, 1, -1, 1, 1, -1),
    (-2, -2, 0, -1, -1, -1, -2, 0, 4),
    (-1, 0, 0, 1, -1, 1, -1, 4, 2),
)


def clashing_tables():
    """
    The clashing orientation tables.

    Returns the tables as a dict, see snapshot_orientation.check_orientation_tables.
    """
    return so.check_orientation_tables(
        {name: [row[3 * i : 3 * i + 3] for row in CLASHING_ROWS] for i, name in enumerate(so.ORIENTATION_TABLE_NAMES)}
    )


def write_clashing_tables(directory):
    """
    Write the clashing orientation tables to a YAML file in directory, for a lightcone's orientation_table.

    Returns the path of the file, a str.

    :param  directory:  directory to write the file to
    :type   directory:  pathlib.Path
    """
    path = directory / "clashing_orientation_tables.yaml"
    so.write_orientation_tables(path, tables=clashing_tables())
    return str(path)
