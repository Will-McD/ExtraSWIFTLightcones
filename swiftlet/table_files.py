#!/bin/env python
import re
import numpy as np

"""
YAML files of tables: the orientation tables and the snapshot output redshifts installed with SWIFTLET (and any of
your own). Each file is a mapping of what describes the table, then "columns", the name of each column, and "rows",
one list of values per line:

    # comments
    name: default_v2
    description: ...
    columns: [QuarterTurns_x, QuarterTurns_y, ...]
    rows:
      - [ 0,  0,  0,  1,  1,  1,  0,  0,  0]
      - [ 1,  0,  0,  1, -1,  1,  0,  1, -1]

The files are plain YAML, read with yaml.safe_load by any YAML reader. read_table_file reads the rows written one per
line, as write_table_file writes them, without the (slow, pure Python) YAML parser; any other layout is read with it.
"""

# a row of the table on a line of its own: "- [v, v, ...]", with an optional comment
_ROW_LINE = re.compile(r"^\s+-\s*\[([^\]#]*)\]\s*(#.*)?$")


def _yaml_loader():
    import yaml
    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _parse_rows(lines):
    """
    The rows written one per line, as a list of lists of str, or None if any line is not a row (or a comment).
    """
    rows = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ROW_LINE.match(line)
        if match is None:
            return None
        rows.append([value.strip() for value in match.group(1).split(",")])
    return rows


def _number(text):
    """
    A value of a row as YAML reads it: an int, or a float.
    """
    try:
        return int(text)
    except ValueError:
        return float(text)


def read_table_file(source, columns, kind="a table"):

    """
    Read a YAML table file and check its columns.

    Returns (header, rows): header is a dict of everything but "columns" and "rows", rows an np.ndarray of shape
    (number of rows, number of columns), of int64 if every value is a whole number written as one, otherwise float.

    :param  source:     path of the file, or an importlib.resources Traversable
    :type   source:     str, os.PathLike or Traversable
    :param  columns:    the names the file's columns must have, in order
    :type   columns:    sequence of str
    :param  kind:       what the file should be, for the error printout if the file is not either a 
                            redshift or orientation table
    :type   kind:       str
    """
    import yaml
    from pathlib import Path
    source = Path(source) if isinstance(source, str) else source
    try:
        text = source.read_text()
    except UnicodeDecodeError:
        raise ValueError(f"{source}: not {kind} file, it is not a text (YAML) file") from None
    try:
        return _read_table_text(source, text, columns, kind)
    except yaml.YAMLError as error:
        raise ValueError(f"{source}: not {kind} file, it is not valid YAML: {error}") from None


def _read_table_text(source, text, columns, kind):
    
    """
    Read the text of a YAML table file, see read_table_file.
    """
    import yaml
    
    # the rows one per line after a last "rows:" key, read without the YAML parser
    lines = text.splitlines()
    rows_at = [i for i, line in enumerate(lines) if re.match(r"^rows\s*:\s*(#.*)?$", line)]
    rows = _parse_rows(lines[rows_at[-1] + 1:]) if len(rows_at) == 1 else None
    
    if rows is not None:
        header = yaml.load("\n".join(lines[:rows_at[-1]]), Loader=_yaml_loader()) or {}
        try:
            rows = [[_number(value) for value in row] for row in rows]
        except ValueError:
            raise ValueError(f"{source}: the rows must hold only numbers") from None
    
    else:
        header = yaml.load(text, Loader=_yaml_loader())
        if not isinstance(header, dict) or "rows" not in header:
            raise ValueError(f"{source}: not {kind} file, it has no 'rows'")
        rows = header.pop("rows")
    
    if not isinstance(header, dict):
        raise ValueError(f"{source}: not {kind} file, it must be a mapping with 'columns' and 'rows'")
    
    header = dict(header)
    if list(header.pop("columns", [])) != list(columns):
        raise ValueError(f"{source}: the columns must be {list(columns)}")
    
    if not isinstance(rows, list) or not all(isinstance(row, list) and len(row) == len(columns) for row in rows):
        raise ValueError(f"{source}: every row must be a list of {len(columns)} values")
    
    values = [value for row in rows for value in row]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        raise ValueError(f"{source}: the rows must hold only numbers")
    
    dtype = np.int64 if all(isinstance(value, int) for value in values) else float
    
    return header, np.array(rows, dtype=dtype).reshape(len(rows), len(columns))


def _float_text(value):
    """
    A float as repr writes it, which reads back exactly, with a decimal point so YAML 1.1 (PyYAML) reads it as a float
    (e.g. 1.0e-05, not 1e-05).
    """
    text = repr(float(value))
    if not np.isfinite(value):
        raise ValueError(f"the rows must be finite, not {text}")
    mantissa, _, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    return mantissa + (f"e{exponent}" if exponent else "")


def _yaml_value(value):
    """
    A value of the header as plain Python, which yaml.safe_dump can write.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    return value


def write_table_file(filename, header, columns, rows, comment=None):
    """
    Write a YAML table file: the comment, the header, the columns, then the rows one per line. Floats are written
    with repr, so they read back exactly.

    :param  filename:   path of the file to write
    :type   filename:   str or os.PathLike
    :param  header:     what describes the table, written before the columns
    :type   header:     dict
    :param  columns:    the name of each column
    :type   columns:    sequence of str
    :param  rows:       the rows: a column of integers is written as whole numbers, any other as floats
    :type   rows:       array-like, shape (number of rows, number of columns)
    :param  comment:    lines written at the top of the file, as YAML comments
    :type   comment:    str
    """
    import yaml
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != len(columns):
        raise ValueError(f"the rows must have shape (N, {len(columns)}), not {rows.shape}")

    # a column of integers is written as whole numbers, any other as floats
    def column_text(column):
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in column):
            return [str(int(value)) for value in column]
        return [_float_text(value) for value in column]

    # every value of a column padded to the same width, so the columns line up
    text_columns = [column_text(rows[:, i]) for i in range(rows.shape[1])]
    widths = [max((len(text) for text in column), default=1) for column in text_columns]
    cells = [[column[i].rjust(width) for column, width in zip(text_columns, widths)] for i in range(len(rows))]
    with open(filename, "w") as outfile:
        for line in (comment or "").splitlines():
            outfile.write(f"# {line}".rstrip() + "\n")
        # one "key: value" per line, lists written [a, b, ...]
        for name, value in {**header, "columns": list(columns)}.items():
            text = yaml.safe_dump(_yaml_value(value), default_flow_style=True, width=float("inf"))
            text = text.removesuffix("\n").removesuffix("\n...").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
                raise ValueError(f"the names of the header must be letters, digits and _, not {name!r}")
            outfile.write(f"{name}: {text}\n")
        outfile.write("rows:\n")
        for row in cells:
            outfile.write("  - [" + ", ".join(row) + "]\n")

