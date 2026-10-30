#!/bin/env python
import functools
import numpy as np
from .threads import limit_numba_threads

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

"""
The orientation of the box tiles of a lightcone built from the snapshots: the rotations (quarter turns),
reflections and periodic cell shifts that make each tile a new copy of the initial snapshot box, the tables they are
chosen from (default, or written to and read from a file), and the rotation of coordinates with them.
"""

##############################################################
# FUNCTIONS FOR RE-ORIENTATION OF CELLS, PARTICLES AND HALOES

def _build_rotation_matrix(rot_angles, order="xyz", degrees=False):
    """
    Build 3x3 rotation matrix.

    Returns the rotation matrix as a (3, 3) np.ndarray.

    :param  rot_angles: (angle_x, angle_y, angle_z), radians unless degrees=True
    :type   rot_angles: array-like, shape (3,)
    :param  order:      the order of axes that coordinates are rotated about, a permutation of "x", "y", "z"
    :type   order:      str
    :param  degrees:    If True, then angles are given in degrees
    :type   degrees:    boolean
    """

    rot_angles = np.asarray(rot_angles, dtype=np.float64)

    if rot_angles.shape != (3,):
        raise ValueError("rot_angles must have shape (3,)")

    if set(order.lower()) != {"x", "y", "z"} or len(order) != 3:
        raise ValueError("order must be a permutation of 'x', 'y', 'z'")

    if degrees:
        rot_angles = np.deg2rad(rot_angles)

    angle_x, angle_y, angle_z = rot_angles

    def rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    matrices = {"x": rot_x(angle_x), "y": rot_y(angle_y), "z": rot_z(angle_z)}

    R = np.eye(3)
    for axis in order.lower():
        R = matrices[axis] @ R
    
    # IMPORTANT FOR 1/4 TURNS TO PREVENT ERRORS DUE TO ROUNDING 
    # make the elements of rotations by multiples of 90 degrees exactly 0 or +-1. as 
    # cos and sin functions leave ~1e-16, which can move points on the faces of the box across its periodic boundaries
    R_round = np.round(R)
    R = np.where(np.abs(R - R_round) < 1e-12, R_round, R)
    
    return R.astype(np.float64)


if _HAVE_NUMBA:
    
    # simple rotation, no periodic shift of coordinates
    @njit(parallel=True, fastmath=True, cache=True)
    def _apply_rotation_numba(coords, R, out):
        """
        Rotate coordinates by the rotation matrix, writing into out.

        :param  coords: coordinates, shape (N, 3)
        :type   coords: np.ndarray
        :param  R:      rotation matrix, shape (3, 3)
        :type   R:      np.ndarray
        :param  out:    output buffer, shape (N, 3)
        :type   out:    np.ndarray
        """
        n = coords.shape[0]
        r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
        r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
        r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
        for i in prange(n):
            x = coords[i, 0]
            y = coords[i, 1]
            z = coords[i, 2]
            out[i, 0] = r00 * x + r01 * y + r02 * z
            out[i, 1] = r10 * x + r11 * y + r12 * z
            out[i, 2] = r20 * x + r21 * y + r22 * z
        return out

    # rotation that includes a periodic shift of coordinates
    # 1. apply periodic shift, i.e. new coordinates
    # 2. rotate shifted coordinates
    @njit(parallel=True, fastmath=True, cache=True)
    def _apply_rotation_periodic_numba(coords, R, shift, L, out):
        """
        Periodically shift coordinates, then rotate them about the box centre, writing into out.

        :param  coords: coordinates, shape (N, 3)
        :type   coords: np.ndarray
        :param  R:      rotation matrix, shape (3, 3)
        :type   R:      np.ndarray
        :param  shift:  shift along x, y, z
        :type   shift:  np.ndarray
        :param  L:      side lengths of the box along x, y, z
        :type   L:      np.ndarray
        :param  out:    output buffer, shape (N, 3)
        :type   out:    np.ndarray
        """
        n = coords.shape[0]
        r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
        r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
        r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
        sx, sy, sz = shift[0], shift[1], shift[2]
        Lx, Ly, Lz = L[0], L[1], L[2]
        cx, cy, cz = Lx / 2.0, Ly / 2.0, Lz / 2.0
        for i in prange(n):
            x = ((coords[i, 0] + sx) % Lx) - cx
            y = ((coords[i, 1] + sy) % Ly) - cy
            z = ((coords[i, 2] + sz) % Lz) - cz

            rx = r00 * x + r01 * y + r02 * z + cx
            ry = r10 * x + r11 * y + r12 * z + cy
            rz = r20 * x + r21 * y + r22 * z + cz

            # only warp the points outside [0, L]. 
            # A point reflected or rotated onto the tiles face stays there,
            # rather than being moved to the opposite face at 0. 
            out[i, 0] = rx % Lx if (rx < 0.0 or rx > Lx) else rx
            out[i, 1] = ry % Ly if (ry < 0.0 or ry > Ly) else ry
            out[i, 2] = rz % Lz if (rz < 0.0 or rz > Lz) else rz
        return out

    # inverse of _apply_rotation_periodic_numba, recovers the original
    # coordinates 
    @njit(parallel=True, fastmath=True, cache=True)
    def _apply_rotation_periodic_numba_inverse(coords, R, shift, L, out):
        """
        Inverse of _apply_rotation_periodic_numba, writing the original coordinates into out.

        :param  coords: rotated coordinates, shape (N, 3)
        :type   coords: np.ndarray
        :param  R:      rotation matrix used in the forward rotation, shape (3, 3)
        :type   R:      np.ndarray
        :param  shift:  shift along x, y, z used in the forward rotation
        :type   shift:  np.ndarray
        :param  L:      side lengths of the box along x, y, z
        :type   L:      np.ndarray
        :param  out:    output buffer, shape (N, 3)
        :type   out:    np.ndarray
        """
        n = coords.shape[0]
        r00, r01, r02 = R[0, 0], R[1, 0], R[2, 0]
        r10, r11, r12 = R[0, 1], R[1, 1], R[2, 1]
        r20, r21, r22 = R[0, 2], R[1, 2], R[2, 2]
        sx, sy, sz = shift[0], shift[1], shift[2]
        Lx, Ly, Lz = L[0], L[1], L[2]
        cx, cy, cz = Lx / 2.0, Ly / 2.0, Lz / 2.0
        for i in prange(n):
            vx = coords[i, 0] - cx
            vy = coords[i, 1] - cy
            vz = coords[i, 2] - cz

            ux = r00 * vx + r01 * vy + r02 * vz
            uy = r10 * vx + r11 * vy + r12 * vz
            uz = r20 * vx + r21 * vy + r22 * vz

            out[i, 0] = (ux + cx - sx) % Lx
            out[i, 1] = (uy + cy - sy) % Ly
            out[i, 2] = (uz + cz - sz) % Lz
        return out

def rotate_coords_cartesian(coords, rot_angles, 
                   reflections=None, periodic_shift=None, sidelengths=None,
                   order="xyz", degrees=False, 
                   out=None, inplace=False,
                   invert=False
                   ):

    """
    Rotate Cartesian coordinates about the x, y, z axes. 
    Uses a numba-compiled, parallelized loop when numba is available and falls back to NumPy matmul otherwise.

    Returns the rotated coordinates, shape (N, 3).

    :param  coords:         coordinates, shape (N, 3)
    :type   coords:         np.ndarray
    :param  rot_angles:     (angle_x, angle_y, angle_z), radians unless degrees=True
    :type   rot_angles:     array-like, shape (3,)
    :param  reflections:    per-axis reflection signs (+1/-1). Pass None for no reflection
    :type   reflections:    array-like, shape (3,)
    :param  periodic_shift: shift along x, y, z. Must be given with sidelengths
    :type   periodic_shift: array-like of int, shape (3,)
    :param  sidelengths:    side length of the snapshot box (scalar for cubic, or (3,))
    :type   sidelengths:    float or array-like, shape (3,)
    :param  order:          rotation composition order, a permutation of "x", "y", "z"
    :type   order:          str
    :param  degrees:        If True, then angles are given in degrees
    :type   degrees:        boolean
    :param  out:            preallocated output buffer, shape (N, 3) (avoids reallocation on repeated calls)
    :type   out:            np.ndarray
    :param  inplace:        If True, overwrite coords in place instead of allocating new memory
    :type   inplace:        boolean
    :param  invert:         If True, apply the inverse of the rotation and periodic shift. Requires periodic_shift
    :type   invert:         boolean
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if (periodic_shift is None) != (sidelengths is None):
        raise ValueError("periodic_shift and L must be given together")
    

    # build rotation matrix from rotation angles
    R = _build_rotation_matrix(rot_angles, order=order, degrees=degrees)
    
    # include refelcitons about axis 
    if reflections is not None: # allow for reflecting each axis
        reflections = np.asarray(reflections, dtype=np.float64)
        if reflections.shape != (3,):
            raise ValueError("reflections must have shape (3,)")
        # R @ diag(reflections) == scaling each column of R by the
        # matching sign -- cheap broadcast, no diag matrix needed.
        R = R * reflections[np.newaxis, :]

    # create output array if not given. 
    if inplace:
        target = coords
    elif out is not None:
        if out.shape != coords.shape:
            raise ValueError("out must have the same shape as coords")
        target = out
    else:
        target = np.empty_like(coords)

    if _HAVE_NUMBA:
        limit_numba_threads() # set max threads for numba functions to SWIFTLET_MAX_THREADS

    if periodic_shift is not None:
        # include shift across periodic bounds
        periodic_shift = np.asarray(periodic_shift, dtype=np.float64)
        sidelengths = np.asarray(sidelengths, dtype=np.float64)
        # sanity checks on shift and sidelengths parameters
        if periodic_shift.shape != (3,):
            raise ValueError("periodic_shift must have shape (3,)")
        if sidelengths.ndim == 0:
            sidelengths = np.full(3, float(sidelengths))
        elif sidelengths.shape != (3,):
            raise ValueError("L must be a scalar or have shape (3,)")

        if invert:
            if _HAVE_NUMBA:
                _apply_rotation_periodic_numba_inverse(coords, R, periodic_shift, sidelengths, target)
            else:
                # inverse of the branch below: v = R.T @ (coords - centre),
                # then undo the shift and re-wrap into [0, sidelengths)
                centre = sidelengths / 2.0
                np.matmul(coords - centre, R, out=target)
                target += centre
                target -= periodic_shift
                np.mod(target, sidelengths, out=target)
        else:
            if _HAVE_NUMBA: 
                # use optimised rotation with periodic shifts
                _apply_rotation_periodic_numba(coords, R, periodic_shift, sidelengths, target)
            else:
                # regular rotation with periodic shifts
                centre = sidelengths / 2.0
                centred = np.mod(coords + periodic_shift, sidelengths) - centre
                np.matmul(centred, R.T, out=target)
                target += centre
                #np.mod(target, sidelengths, out=target)
                
                # wrap only the points outside [0, L]. 
                outside = (target < 0.) | (target > sidelengths)
                target[outside] = np.mod(target, sidelengths)[outside]

    else:
        if invert:
            raise NotImplementedError("rotate_coords_cartesian: invert=True is not supported without periodic_shift")
        # perform rotations without shift across periodic boundaries
        if _HAVE_NUMBA: 
            # optimised rotation 
            _apply_rotation_numba(coords, R, target) 
        else:
            # regular rotation
            np.matmul(coords, R.T, out=target)

    return target

###############################################################################
# GLOBAL VARIABLES FOR DEFAULT ORIENTATION TABLES + CELL PERIODIC SHIFT UNITS 

ORIENTATION_TABLE_NAMES = ("QuarterTurns", "Reflections", "CellShifts")

# the columns of an orientation table file in YAML: each table along x, y and z
ORIENTATION_TABLE_COLUMNS = tuple(f"{name}_{axis}" for name in ORIENTATION_TABLE_NAMES for axis in "xyz")

# the units the cell shifts of a table can be in: whole cells, or eighths of the box (converted to cells with the
# number of cells per axis of the snapshots, which must be a multiple of 8)
CELL_SHIFT_UNIT =  "eighths"
# every shift in eighths of the box: 0, +-1/8, +-1/4, +-3/8 and 1/2 (= -1/2)
EIGHTH_SHIFT_VALUES = (0, 1, -1, 2, -2, 3, -3, 4)

# the orientation tables installed with SWIFTLET, by name, as YAML files in swiftlet/data/orientation_tables
#   default_v1: the first default tables, 79 rows, not clash-free; kept for tests and examples of close repeats
#   default_v2: 5184 rows with no two rows making a cell touch its own copy in neighbouring tiles, shifts in eighths
#                   of the box 

INSTALLED_ORIENTATION_TABLES = ("default_v1", "default_v2")
DEFAULT_ORIENTATION_TABLE = "default_v2"

##################################################################
# FUNCTIONS GENERATING, READING AND CHECKING ORIENTATION TABLES

def installed_orientation_table_file(name=DEFAULT_ORIENTATION_TABLE):
    """
    The path of an orientation table file installed with SWIFTLET.

    Returns the path, a str.

    :param  name:   name of the tables, one of INSTALLED_ORIENTATION_TABLES ("default" is DEFAULT_ORIENTATION_TABLE)
    :type   name:   str
    """
    from importlib import resources
    name = DEFAULT_ORIENTATION_TABLE if name == "default" else name
    if name not in INSTALLED_ORIENTATION_TABLES:
        raise ValueError(f"no orientation tables called {name!r} are installed, only {INSTALLED_ORIENTATION_TABLES}")
    return str(resources.files("swiftlet") / "data" / "orientation_tables" / f"{name}.yaml")

@functools.lru_cache(maxsize=None)
def _installed_orientation_tables(name):
    """
    Read installed orientation tables. Cached so each file is read once.
    """
    return _read_orientation_tables_yaml(installed_orientation_table_file(name))

def default_orientation_tables(name=DEFAULT_ORIENTATION_TABLE):
    """
    The orientation table reverted to when no orientation table is given, these tables are installed with SWIFTLET.
    By default the current default tables (DEFAULT_ORIENTATION_TABLE); "default_v2" gives the first default tables.

    Returns the tables as a dict, see check_orientation_tables.

    :param  name:   name of the installed tables, one of INSTALLED_ORIENTATION_TABLES
    :type   name:   str
    """
    name = DEFAULT_ORIENTATION_TABLE if name == "default" else name
    installed_orientation_table_file(name)  # checks the name
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in _installed_orientation_tables(name).items()}

def cell_shifts_in_cells(cell_shift, nr_cells_axis):
    """
    Convert the tables cell shift from multiples of 1/8 to number of cells. 

    Returns the shift in cells, np.ndarray of int64, shape (3,).

    :param  cell_shift:         shift along x, y and z, in eighths of the box
    :type   cell_shift:         array-like of int, shape (3,)
    :param  nr_cells_axis:      number of cells along each axis of the snapshot, a multiple of 8


    :type   nr_cells_axis:      int or array-like of int, shape (3,)
    """
    cell_shift = np.asarray(cell_shift, dtype=np.int64)

    nr_cells_axis = np.broadcast_to(np.asarray(nr_cells_axis, dtype=np.int64), (3,))
    if np.any(nr_cells_axis % 8 != 0) or np.any(nr_cells_axis < 8):
        raise ValueError(f"the cell shifts are n/8, so snapshot cell count must be multiple of 8")
    return cell_shift * (nr_cells_axis // 8)

def check_orientation_tables(tables):
    """
    Check orientation tables. 
    Three tables with the same number of rows (at least 1) of 3 whole numbers to build new snapshot tiles:

        - the quarter turns about the x, y and z axes
        - reflections of +1 or -1 per axis
        - periodic shifts in eighths of the box per axis and in EIGHTH_SHIFT_VALUES

    The tables can also hold "cell_shift_unit", which must be "eighths".
    
    Returns the tables as a dict. 

    :param  tables: {"QuarterTurns", "Reflections", "CellShifts"}
    :type   tables: dict
    """
    missing = [name for name in ORIENTATION_TABLE_NAMES if name not in tables]
    if missing:
        raise ValueError(f"orientation tables {missing} missing, the tables are {ORIENTATION_TABLE_NAMES}")
    checked = {}
    for name in ORIENTATION_TABLE_NAMES:
        values = np.asarray(tables[name])
    
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1:
            raise ValueError(f"orientation table {name} must have shape (N, 3) with N >= 1, not {values.shape}")
    
        if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
            raise ValueError(f"orientation table {name} must hold whole numbers")
        checked[name] = values.astype(np.int64)
    
    if len({values.shape[0] for values in checked.values()}) != 1:
        raise ValueError("the orientation tables must have the same number of rows, "
                         + ", ".join(f"{name}: {checked[name].shape[0]}" for name in ORIENTATION_TABLE_NAMES))
    
    if not np.all(np.isin(checked["Reflections"], (-1, 1))):
        raise ValueError("orientation table Reflections must hold only +1 and -1")
    
    _check_cell_shift_unit(tables.get("cell_shift_unit", CELL_SHIFT_UNIT))
    
    if not np.all(np.isin(checked["CellShifts"], EIGHTH_SHIFT_VALUES)):
        raise ValueError(f"CellShifts must be a multiple of 1/8: {EIGHTH_SHIFT_VALUES}")

    return checked

def _check_cell_shift_unit(unit):
    unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    if unit != CELL_SHIFT_UNIT:
        raise ValueError(f"incorrect units, the cell shifts must be a multiple of 1/8 (cell_shift_unit {CELL_SHIFT_UNIT!r})")

def orientation_tables_fingerprint(tables):
    """
    Fingerprint of the contents of orientation tables, the same for the same tables wherever they come from.

    Returns the first 16 hexadecimal digits of the tables hashlib SHA-256 hash.

    :param  tables: orientation tables
    :type   tables: dict
    """
    import hashlib
    tables = check_orientation_tables(tables)
    digest = hashlib.sha256()
    for name in ORIENTATION_TABLE_NAMES:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tables[name], dtype="<i8").tobytes())
    
    # the unit of the shifts, eighths of the box
    digest.update(f"cell_shift_unit={CELL_SHIFT_UNIT}".encode())

    return digest.hexdigest()[:16]

def generate_orientation_tables(seed=0, n_rows=None, cell_shift_values=EIGHTH_SHIFT_VALUES):

    """
    Orientation tables holding every distinct orientation: each of the 48 ways of turning and reflecting the box
    (the signed permutations of the axes) with each combination of the cell shifts along x, y and z, e.g.
    48 x 7^3 = 16464 rows for the default shifts. 
    The first row leaves the box unchanged, so the observer's box tile
    (0, 0, 0) is not re-oriented with orientation_seed=0, and the other rows are shuffled.

    Returns the tables as a dict, see check_orientation_tables.

    :param  seed:               seed of the shuffle of the rows, the same seed gives the same tables
    :type   seed:               int
    :param  n_rows:             number of rows to keep (the first, unchanged, and n_rows - 1 others). If None, all
    :type   n_rows:             int
    :param  cell_shift_values:  shifts each axis can take, including 0, in 1/8 of the box
    :type   cell_shift_values:  sequence of int
    """

    import itertools
    
    cell_shift_values = [int(value) for value in cell_shift_values]

    if 0 not in cell_shift_values or len(set(cell_shift_values)) != len(cell_shift_values):
        raise ValueError("cell_shift_values must include 0 and have no repeats")

    # one (quarter turns, reflections) for each of the 48 distinct matrices, the unchanged box first
    turns_and_reflections = {}
    for turns in itertools.product((0, 1, 2, -1), repeat=3):
        for reflections in itertools.product((1, -1), repeat=3):
            # build rot matrix of quater turns
            matrix = _build_rotation_matrix(np.array(turns) * 90., degrees=True) * np.array(reflections)[np.newaxis, :]
            turns_and_reflections.setdefault(tuple(np.rint(matrix).astype(int).ravel()), (turns, reflections))
    
    assert len(turns_and_reflections) == 48 # ensure all 48 combinations have been sampled
    
    orientations = list(turns_and_reflections.values())
    shifts = list(itertools.product(cell_shift_values, repeat=3))

    # add rows to table with a given quater turn, relfection and periodic shift. 
    rows = [(turns, reflections, shift) for turns, reflections in orientations for shift in shifts]
    unchanged = ((0, 0, 0), (1, 1, 1), (0, 0, 0))
    rows.remove(unchanged)
    order = np.random.default_rng(seed).permutation(len(rows))
    rows = [unchanged] + [rows[i] for i in order]
    
    if n_rows is not None:
        if not 1 <= int(n_rows) <= len(rows):
            raise ValueError(f"n_rows must be between 1 and {len(rows)}, not {n_rows}")
        rows = rows[:int(n_rows)]
    
    tables = {name: np.array([row[i] for row in rows]) for i, name in enumerate(ORIENTATION_TABLE_NAMES)}
    
    return check_orientation_tables(tables)

def write_orientation_tables(filename, tables=None, seed=0, n_rows=None, cell_shift_values=EIGHTH_SHIFT_VALUES, attributes=None):
    """
    Write orientation tables to a YAML file. The tables are given, or generated with every distinct orientation. 
    The .hdf5 file holds the datasets 'QuarterTurns', 'Reflections' and 'CellShifts', and the attributes describing them.

    Returns the fingerprint of the tables.

    :param  filename:           path of the file to write, .yaml or .yml
    :type   filename:           str
    :param  tables:             tables to write, see check_orientation_tables. If None, generate them
    :type   tables:             dict
    :param  seed:               seed of generate_orientation_tables, if the tables are generated
    :type   seed:               int
    :param  n_rows:             number of rows of generate_orientation_tables, if the tables are generated
    :type   n_rows:             int
    :param  cell_shift_values:  cell shifts of generate_orientation_tables, if the tables are generated
    :type   cell_shift_values:  sequence of int
    """

    from .table_files import write_table_file
    _check_yaml_file_name(filename)

    generated = tables is None
    if generated:
        tables = generate_orientation_tables(seed=seed, n_rows=n_rows, cell_shift_values=cell_shift_values)
    tables = check_orientation_tables(tables)
    fingerprint = orientation_tables_fingerprint(tables)

    header = dict(attributes or {})
    header.update({
        "description": ("orientation tables of the box tiles of a snapshot lightcone: tile reads row (tile hash % "
                        "number of rows)"),
        "columns_description": ("QuarterTurns: number of quarter turns (90 degrees) about the x, y and z axes; "
                                "Reflections: reflection (+1 or -1) of the x, y and z axes; CellShifts: periodic "
                                "shift of the box along x, y and z, in eighths of the box"),
        "cell_shift_unit": CELL_SHIFT_UNIT,
        "number_of_rows": len(tables["QuarterTurns"]),
        "fingerprint": fingerprint,
        "generated": generated,
    })
    if generated:
        header["seed"] = int(seed)
        header["cell_shift_values"] = [int(value) for value in cell_shift_values]
    rows = np.column_stack([tables[name] for name in ORIENTATION_TABLE_NAMES])
    write_table_file(filename, header, ORIENTATION_TABLE_COLUMNS, rows, comment="SWIFTLET orientation tables, see docs/orientation_tables.md")

    return fingerprint

def _check_yaml_file_name(filename):
    if not str(filename).lower().endswith((".yaml", ".yml")):
        raise ValueError(f"orientation table files are YAML, so their name must end in .yaml (or .yml), not {filename}")

def _read_orientation_tables_yaml(filename):
    """
    Read and check orientation tables from a YAML file written by write_orientation_tables. If the file holds a
    fingerprint, it must be that of the tables read, so a table edited by hand is not taken for the one it was.
    """
    from .table_files import read_table_file
    header, rows = read_table_file(filename, ORIENTATION_TABLE_COLUMNS, kind="an orientation table")
    tables = {name: rows[:, 3 * i:3 * i + 3] for i, name in enumerate(ORIENTATION_TABLE_NAMES)}
    _check_cell_shift_unit(header.get("cell_shift_unit", CELL_SHIFT_UNIT))
    tables = check_orientation_tables(tables)
    stored = header.get("fingerprint")
    if stored is not None and str(stored) != orientation_tables_fingerprint(tables):
        raise ValueError(f"{filename}: the tables have fingerprint {orientation_tables_fingerprint(tables)}, not the "
                         f"{stored} written in the file; they have been changed since it was written (remove the "
                         f"fingerprint line if that was on purpose)")
    return tables

def read_orientation_tables(filename):
    """
    Read and check orientation tables written by write_orientation_tables.
    Table must be a file ending in .yaml or .yml

    Returns the tables as a dict. 

    :param  filename:   path of the HDF5 file
    :type   filename:   str
    """

    if str(filename) == "default" or str(filename) in INSTALLED_ORIENTATION_TABLES:
        return default_orientation_tables(str(filename))
    _check_yaml_file_name(filename)
    return _read_orientation_tables_yaml(filename)

##################################################################
# FUNCTIONS FOR COURSE CHECK OF REPLICATED CELLS USING ONLY TABLES

class NeighbourTileReport:
    """
    The result of check_neighbouring_tiles.

    nr_neighbour_pairs:     number of pairs of tiles sharing a face that were checked
    nr_reoriented_pairs:    number of those pairs whose two tiles have different orientations
    nr_touching_pairs:      number of pairs of tiles in which some cell touches its own copy across the shared face
    nr_touching_cells:      number of cells touching their own copy, over every pair of tiles
    touching:               every cell touching its own copy: a dict of arrays "tile_a" and "tile_b" (N, 3), the
                                two tiles, and "cell", the cell's index, (i * n + j) * n + k for the cell (i, j, k)
                                of a box of n cells per axis, as SWIFT numbers them
    shell:                  the shell [Mpc] the touching cells were kept in, or None for whole tiles
    """

    def __init__(self, nr_neighbour_pairs, nr_reoriented_pairs, touching, shell=None):
        self.nr_neighbour_pairs = nr_neighbour_pairs
        self.nr_reoriented_pairs = nr_reoriented_pairs
        self.touching = touching
        self.shell = shell
        self.nr_touching_cells = len(touching["cell"])
        pairs = np.column_stack([touching["tile_a"], touching["tile_b"]])
        self.nr_touching_pairs = len(np.unique(pairs, axis=0)) if len(pairs) else 0

    def summary(self):
        """
        Returns a message describing the touching cells found.
        """
        #where = "" if self.shell is None else f" in the shell {self.shell[0]:.0f}-{self.shell[1]:.0f} Mpc"
        summary_str= f"{self.nr_touching_cells} cells touching their own replica across the faces of {self.nr_touching_pairs} pairs of neighbouring tiles"
        summary_str+=f"({self.nr_reoriented_pairs} / {self.nr_neighbour_pairs} pairs of neighbouring tiles are oriented differently)."
        return (summary_str)


    def __repr__(self):
        return f"NeighbourTileReport({self.summary()})"


def check_neighbouring_tiles(tables, nr_cells_axis, orientation_lock=None, orientation_seed=0, extent=3, warn=True, boxsize=None, shell=None):
    """
    Check orientation tables for cells that touch their own copy in a neighbouring tile as a 
    cell sharing a face with its own copy puts the same structures side by side.

    Only neighbouring tiles that are oriented differently can lead to replicas of the same cell have touching, 
    e.g., with orientation_lock="cube" pairs of cells can touch only be touching on the faces between layers. 
    
    This method is quicker and coarser than swiftlet.repeat_check.find_close_cell_repeats, which builds the actual lightcone shell 
    of a simulation and also finds copies close along a line of sight.
    
    Due to only doing a course check orientation_lock="sphere" cannot be used with this method as it requires a 
    more compelte rebuild of the lightcone. To test with Sphere use swiftlet.repeat_check.find_close_cell_repeats 

    Returns a NeighbourTileReport.

    :param  tables:             orientation tables, as a dict (see check_orientation_tables) or the path of a table
                                    file (see write_orientation_tables)
    :type   tables:             dict or str
    :param  nr_cells_axis:      number of SWIFT cells along each axis of the snapshots, e.g. 32 for FLAMINGO
    :type   nr_cells_axis:      int
    :param  orientation_lock:   None (or "none") or "cube". "sphere" splits tiles between layers by distance, which
                                    needs the lightcone: use find_close_cell_repeats
    :type   orientation_lock:   str
    :param  orientation_seed:   seed of the tile orientations
    :type   orientation_seed:   int
    :param  extent:             check the tiles (nx, ny, nz) with |nx|, |ny|, |nz| <= extent
    :type   extent:             int
    :param  warn:               if True, warn (swiftlet.repeat_check.CloseRepeatWarning) if cells touch their copy
    :type   warn:               boolean
    :param  shell:              (minimum, maximum) comoving distance [Mpc] of a lightcone shell. If given, only
                                    touching copies whose cells overlap the shell are kept
    :type   shell:              tuple
    """

    import warnings
    from .snapshot_lightcone import TILE_HASH_PRIMES, TILE_HASH_SEED_PRIME

    if not isinstance(tables, dict):
        tables = read_orientation_tables(tables)

    tables = check_orientation_tables(tables)

    if orientation_lock == "none":
        orientation_lock = None
    
    if orientation_lock not in (None, "cube"):
        raise ValueError(f"check_neighbouring_tiles takes orientation_lock None or 'cube', not {orientation_lock!r}")
    
    n = int(nr_cells_axis)
    n_rows = len(tables["QuarterTurns"])

    if shell is not None:
        if boxsize is None:
            raise ValueError("check_neighbouring_tiles needs the boxsize with a shell")
        cell_size = float(boxsize) / n
        half_diagonal = 0.5 * np.sqrt(3.) * cell_size
    
    p1, p2, p3 = TILE_HASH_PRIMES

    def row_of(tile):
        # the row each tile takes, as SnapshotLightcone._snapshot_reorientation chooses it
        nx, ny, nz = tile if orientation_lock is None else (0, 0, max(abs(t) for t in tile))
        return (nx * p1 + ny * p2 + nz * p3 + orientation_seed * TILE_HASH_SEED_PRIME) % n_rows

    def orientation(row):
        return tuple(int(x) for name in ORIENTATION_TABLE_NAMES for x in tables[name][row])

    # the rows the tiles take, and the pairs of neighbouring tiles
    tiles = np.indices((2 * extent + 1,) * 3).reshape(3, -1).T - extent
    tile_rows = {tuple(int(t) for t in tile): row_of(tuple(int(t) for t in tile)) for tile in tiles}

    # only cells on the shared face can touch: the cells on each face of the box after each row used moves it, as
    # keys (cell, u, v), see _face_cell_keys. A cell of tile a touches its copy in the next tile along an axis if
    # its key is on the + face of a's row and on the - face of the next tile's row
    rows_used = np.unique(list(tile_rows.values()))
    position_of_row = {int(row): i for i, row in enumerate(rows_used)}
    plus, minus = _face_cell_keys({name: tables[name][rows_used] for name in ORIENTATION_TABLE_NAMES}, n)
    touching_keys = {}

    def touching_on_face(row_a, row_b, axis):
        # keys of the cells touching their copy between a tile of row_a and the next tile along axis, of row_b
        key = (row_a, row_b, axis)
        if key not in touching_keys:
            touching_keys[key] = np.intersect1d(plus[axis, position_of_row[row_a]], minus[axis, position_of_row[row_b]], assume_unique=True)
        return touching_keys[key]

    # gather all tiles and then iterate through all possible pairs of neighbouring tiles
    tile_a, tile_b, cells = [], [], []
    nr_neighbour_pairs = nr_reoriented_pairs = 0
    #tiles = np.indices((2 * extent + 1,) * 3).reshape(3, -1).T - extent
    for tile in tiles:
        for axis in range(3):
            neighbour = tile.copy()
            neighbour[axis] += 1
            if neighbour[axis] > extent:
                continue
            nr_neighbour_pairs += 1
            row_a, row_b = tile_rows[tuple(int(t) for t in tile)], tile_rows[tuple(int(t) for t in neighbour)]
            
            if orientation(row_a) == orientation(row_b):
                continue
            
            nr_reoriented_pairs += 1

            keys = touching_on_face(int(row_a), int(row_b), axis)
            touching = keys // (n * n)
            
            if len(touching) and shell is not None:
                # only copies whose cells overlap the shell: distances [Mpc] from the observer, at the centre of the
                # observer's tile. The touching cell is on the + face of tile a and the - face of its neighbour
                
                u, v = [a for a in range(3) if a != axis]
                
                face_position = np.zeros((len(keys), 3))
                face_position[:, u] = (keys // n) % n
                face_position[:, v] = keys % n
                
                keep = np.ones(len(touching), dtype=bool)
                
                for face, origin in ((n - 1, tile), (0, neighbour)):
                    position = face_position.copy()
                    position[:, axis] = face
                    
                    r = np.linalg.norm(position + 0.5 - 0.5 * n + n * origin, axis=1) * cell_size
                    
                    keep &= (r + half_diagonal >= shell[0]) & (r - half_diagonal <= shell[1])
                    
                
                touching = touching[keep]
            
            tile_a.append(np.tile(tile, (len(touching), 1)))
            tile_b.append(np.tile(neighbour, (len(touching), 1)))
            cells.append(touching)
    
    def joined(arrays, shape):
        return np.concatenate(arrays).astype(np.int64) if arrays else np.zeros(shape, dtype=np.int64)

    # reporting touching cells from table alone 
    report = NeighbourTileReport(
        nr_neighbour_pairs, 
        nr_reoriented_pairs, 
        {"tile_a": joined(tile_a, (0, 3)), "tile_b": joined(tile_b, (0, 3)), "cell": joined(cells, (0,))}, 
        shell=shell
        )

    if warn and report.nr_touching_cells > 0:
        from .repeat_check import CloseRepeatWarning
        warnings.warn(f"orientation tables (orientation_lock={orientation_lock!r}, orientation_seed={orientation_seed}, {n} cells per axis): {report.summary()}", CloseRepeatWarning, stacklevel=2)
    
    return report

# tables in which no two rows make a cell touch its own copy

def _face_cell_keys(tables, nr_cells_axis):
    """
    The cells on each face of the box after each row of the tables moves it, as sorted keys (cell, u, v): the cell's
    index and its position on the face. A cell of a tile touches its own copy in the next tile along an axis if the
    key is on the + face of the first tile's row and the - face of the second's.

    Returns (plus, minus), each np.ndarray of int64, shape (3, number of rows, nr_cells_axis**2): the keys on the +
    and - face along each axis.

    :param  tables:         orientation tables, see check_orientation_tables
    :type   tables:         dict
    :param  nr_cells_axis:  number of cells along each axis
    :type   nr_cells_axis:  int
    """
    tables = check_orientation_tables(tables)
    n = int(nr_cells_axis)
    n_rows = len(tables["QuarterTurns"])
    cell_centres = np.indices((n,) * 3).reshape(3, -1).T + 0.5
    cell_index = np.arange(n ** 3)
    plus = np.zeros((3, n_rows, n * n), dtype=np.int64)
    minus = np.zeros((3, n_rows, n * n), dtype=np.int64)
    for row in range(n_rows):
        shift = cell_shifts_in_cells(tables["CellShifts"][row], n)
        # the cell each cell is moved to, (i, j, k) from 0 to n - 1
        moved = np.rint(rotate_coords_cartesian(
            cell_centres, tables["QuarterTurns"][row] * 90., reflections=tables["Reflections"][row],
            periodic_shift=np.mod(shift, n).astype(np.float64), sidelengths=float(n), order="xyz",
            degrees=True) - 0.5).astype(np.int64)
        for axis in range(3):
            u, v = [a for a in range(3) if a != axis]
            for face, keys in ((n - 1, plus), (0, minus)):
                on_face = moved[:, axis] == face
                keys[axis, row] = np.sort((cell_index[on_face] * n + moved[on_face, u]) * n + moved[on_face, v])
    return plus, minus


class _ClashIndex:
    """
    Which rows of orientation tables clash: put the same cell on both sides of the face between neighbouring tiles.
    Built once from the face keys (see _face_cell_keys); neighbours(row) gives the rows clashing with a row, along
    any axis and either way round.
    """

    def __init__(self, tables, nr_cells_axis):
        self.plus, self.minus = _face_cell_keys(tables, nr_cells_axis)
        n_rows, n_keys = self.plus.shape[1:]
        self.index = []
        rows = np.repeat(np.arange(n_rows), n_keys)
        for axis in range(3):
            for faces in (self.minus[axis], self.plus[axis]):
                keys = faces.ravel()
                order = np.argsort(keys, kind="stable")
                self.index.append((keys[order], rows[order]))

    def neighbours(self, row):
        """
        The rows clashing with row: row's + face against their - face, or their + face against row's - face.

        Returns np.ndarray of int, sorted and unique.
        """
        found = []
        for axis in range(3):
            for keys, (index_keys, index_rows) in ((self.plus[axis, row], self.index[2 * axis]), (self.minus[axis, row], self.index[2 * axis + 1])):
                first = np.searchsorted(index_keys, keys, "left")
                last = np.searchsorted(index_keys, keys, "right")
                for a, b in zip(first[last > first], last[last > first]):
                    found.append(index_rows[a:b])
        return np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.int64)


##################################################################
# FUNCTIONS TO GENERATE TABLES WITHOUT TOUCHING REPLICA CELLS

def generate_clash_free_tables(seed=21, nr_cells_axis=16, cell_shift_values=EIGHTH_SHIFT_VALUES, shuffle_seed=0):
    """
    Orientation tables in which no two rows clash: no cell touches its own copy across the face between two
    neighbouring tiles, whichever two rows the tiles take, for any orientation_lock and orientation_seed. The cell
    shifts are in eighths of the box, so the clashes are the same for every number of cells per axis that is a
    multiple of 8 and at least 16 (checked for 16, 32 and 64), and one table serves every such simulation.

    Every candidate is one of the distinct orientations (48 turns and reflections, times the shifts along each axis,
    generate_orientation_tables). Starting from the unchanged box, the rows are chosen in rounds, one for each of the
    48 turns and reflections per round, at random, skipping rows that clash with one already chosen; then every turn
    and reflection is cut down to the same number of rows, choosing the rows that keep the shifts along each axis
    even. The first row is the unchanged box, and the others are shuffled. The same arguments give the same tables.
    This is how the installed default_v2 tables were made (seed 21, the most even of seeds 4 to 63; see
    tools/make_default_orientation_tables.py).

    Returns the tables as a dict, see check_orientation_tables.

    :param  seed:               seed of the random order in which the rows are chosen
    :type   seed:               int
    :param  nr_cells_axis:      number of cells per axis the clashes are found for (16 gives those of 32 and 64 too)
    :type   nr_cells_axis:      int
    :param  cell_shift_values:  shifts each axis can take, in eighths of the box
    :type   cell_shift_values:  sequence of int
    :param  shuffle_seed:       seed of the shuffle of the rows after the first, and of the choice between equally
                                    even rows
    :type   shuffle_seed:       int
    """
    candidates = generate_orientation_tables(seed=0, cell_shift_values=cell_shift_values)

    clashes = _ClashIndex(candidates, nr_cells_axis)
    n_candidates = len(candidates["QuarterTurns"])
    # the turns and reflections of each candidate, numbered
    combination = np.unique(np.column_stack([candidates["QuarterTurns"], candidates["Reflections"]]), axis=0,
                            return_inverse=True)[1].ravel()
    nr_combinations = combination.max() + 1
    unchanged = 0  # the first candidate is the unchanged box

    # choose rows in rounds, one for each turn and reflection, skipping rows that clash with a chosen row
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(np.flatnonzero(combination == c))) for c in range(nr_combinations)]
    blocked = np.zeros(n_candidates, dtype=bool)
    chosen = [unchanged]
    blocked[clashes.neighbours(unchanged)] = True
    blocked[unchanged] = True
    progress = True
    while progress:
        progress = False
        for c in rng.permutation(nr_combinations):
            pool = pools[c]
            while pool and blocked[pool[-1]]:
                pool.pop()
            if pool:
                row = pool.pop()
                chosen.append(row)
                blocked[clashes.neighbours(row)] = True
                blocked[row] = True
                progress = True

    # the same number of rows for every turn and reflection, choosing those that keep the shifts of each axis even
    per_combination = np.bincount(combination[chosen], minlength=nr_combinations).min()
    shift_values = np.asarray(cell_shift_values)
    rng = np.random.default_rng(shuffle_seed)
    shift_counts = np.zeros((3, len(shift_values)), dtype=int)

    def shift_columns(row):
        return [int(np.flatnonzero(shift_values == s)[0]) for s in candidates["CellShifts"][row]]

    kept = [unchanged]
    shift_counts[np.arange(3), shift_columns(unchanged)] += 1
    for c in range(nr_combinations):
        pool = [row for row in chosen if combination[row] == c and row != unchanged]
        for _ in range(per_combination - (1 if combination[unchanged] == c else 0)):
            cost = np.array([sum(shift_counts[axis, column] for axis, column in enumerate(shift_columns(row)))
                             for row in pool])
            row = pool.pop(int(np.argmin(cost + 0.1 * rng.random(len(cost)))))
            kept.append(row)
            shift_counts[np.arange(3), shift_columns(row)] += 1

    # the unchanged box first, the other rows shuffled
    kept = [kept[0]] + [kept[i] for i in 1 + rng.permutation(len(kept) - 1)]
    tables = {name: candidates[name][kept] for name in ORIENTATION_TABLE_NAMES}

    return check_orientation_tables(tables)


def tables_are_clash_free(tables, nr_cells_axis=16):
    """
    Check that no two rows of orientation tables (nor a row with itself) make a cell touch its own copy across the
    face between neighbouring tiles.

    Returns the number of clashing (row, row) pairs, 0 if the tables are clash-free.

    :param  tables:         orientation tables, see check_orientation_tables
    :type   tables:         dict
    :param  nr_cells_axis:  number of cells along each axis
    :type   nr_cells_axis:  int
    """
    clashes = _ClashIndex(tables, nr_cells_axis)
    return int(sum(len(clashes.neighbours(row)) for row in range(len(check_orientation_tables(tables)["QuarterTurns"]))))

