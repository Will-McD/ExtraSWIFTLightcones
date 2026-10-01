#!/bin/env python
import numpy as np

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

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

            out[i, 0] = rx % Lx
            out[i, 1] = ry % Ly
            out[i, 2] = rz % Lz
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
                np.mod(target, sidelengths, out=target)

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


def random_angles(N, low=-180.0, high=180.0, seed=None):
    """
    Generate an array of random angles [deg] with shape (N, 3).

    Returns the angles as an np.ndarray, shape (N, 3).

    :param  N:      number of rows
    :type   N:      int
    :param  low:    inclusive lower bound for the uniform distribution
    :type   low:    float
    :param  high:   exclusive upper bound for the uniform distribution
    :type   high:   float
    :param  seed:   seed for reproducibility. If None, uses fresh entropy each call
    :type   seed:   int
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(N, 3))


def random_quater_turns(N, seed=None):
    """
    Generate random quarter turns (multiples of 90 deg, or pi/2) and reflections with shape (N, 3).

    Returns a tuple of (quarter turns in the range -2 to 2, reflection signs +1/-1), each an np.ndarray, shape (N, 3).

    :param  N:      number of rows
    :type   N:      int
    :param  seed:   seed for reproducibility. If None, uses fresh entropy each call
    :type   seed:   int
    """
    rng = np.random.default_rng(seed)
    quater_turns =rng.choice([-2, -1, 0, 1, 2], size=(N, 3))
    reflections = rng.choice([1, -1], size=quater_turns.shape)
    return quater_turns, reflections


def random_cell_shift_values(N, nr_cells_per_axis, seed=None):
    """
    Generate an array of random periodic shifts, in number of cells, with shape (N, 3).

    Returns the shifts as an np.ndarray, shape (N, 3).

    :param  N:                  number of rows
    :type   N:                  int
    :param  nr_cells_per_axis:  number of cells per axis, sets the allowed shift sizes
    :type   nr_cells_per_axis:  int
    :param  seed:               seed for reproducibility. If None, uses fresh entropy each call
    :type   seed:               int
    """
    rng = np.random.default_rng(seed)
    if nr_cells_per_axis==32:
        shift_sizes=[-16,-8,-4,0,4,8,16]
    elif nr_cells_per_axis==64:
        shift_sizes=[-32,-16,-8,0,8,16,32]
    else:
        shift_sizes=[-16,-8,-4,-2,0,2,4,8,16]
    shifts =rng.choice(shift_sizes, size=(N, 3))
    return shifts


SnapshotBeamAngles_quaters=np.array([
    [0,0,0],
    [0,1,-1],
    [-1, -1, -2],
    [1, 0, -1],
    [-1, 0, 1],
    [1, -2, 0],
    [-1, 2, 1],
    [1, 1, 2],
    [-2, 2, 1],
    [2, 2, -2],
    [-2, 1, -2],
    [0, -2, -1],
    [-2, -2, 2],
    [-2, 0, -1],
    [1, 1, 2],
    [-1, 2, 2],
    [1, 0, 1],
    [1, 0, 2],
    [-1, 2, 2],
    [1, 0, -2],
    [1, -1, 2],
    [0, 1, -1],
    [1, 1, 0],
    [-1, -1, 2],
    [1, -2, -2],
    [0, -2, -2],
    [1, 0, -2],
    [0, 2, 0],
    [1, 2, -2],
    [-1, 1, 0],
    [-2, -2, -1],
    [0, 0, 1],
    [-2, -1, -2],
    [2, -2, 2],
    [0, 1, -2],
    [2, -1, -1],
    [-2, -1, -2],
    [-2, 1, 2],
    [0, -2, 2],
    [-1, -2, -1],
    [0, -2, -2],
    [0, 0, 0],
    [2, 0, -2],
    [-2, -2, 1],
    [-2, 1, -2],
    [1, -1, 2],
    [-2, 1, -1],
    [-2, -1, 1],
    [-2, -2, 0],
    [-1, 2, 2],
    [0, -2, 2],
    [-1, -2, -2],
    [0, 0, 0],
    [2, -2, 0],
    [2, 2, -1],
    [2, 2, 2],
    [2, -2, 2],
    [2, 1, 2],
    [2, -1, 1],
    [0, 2, -1],
    [-1, 1, 1],
    [2, 0, 0],
    [1, 2, 0],
    [-2, 1, 1],
    [-2, -1, 1],
    [-2, 0, -1],
    [-1, 1, -1],
    [-2, -2, -1],
    [-2, 2, 2],
    [0, 0, 0],
    [1, 2, 1],
    [2, 0, 2],
    [-1, -2, 0],
    [2, -1, -1],
    [0, -2, -2],
    [-2, -2, 0],
    [-1, -1, -2],
    [-2, -2, 0],
    [-1, 0, 0],
    ], dtype=int)

SnapshotBeamAngles_reflections=np.array([
    [1,1,1],
    [-1, 1, -1],
    [-1, 1, -1],
    [1, 1, 1],
    [-1, 1, -1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, 1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, 1, -1],
    [-1, 1, 1],
    [1, 1, 1],
    [1, 1, -1],
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1],
    [-1, 1, 1],
    [1, 1, 1],
    [1, -1, -1],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, -1],
    [1, 1, -1],
    [1, 1, 1],
    [-1, -1, 1],
    [-1, -1, 1],
    [1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, -1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1],
    [-1, 1, -1],
    [1, -1, 1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
    [1, -1, 1],
    [-1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
    [-1, -1, 1],
    [1, 1, 1],
    [1, 1, -1],
    [1, 1, -1],
    [-1, -1, -1],
    [1, -1, 1],
    ], dtype=int)

SnapshotBeamAngles_shifts=np.array([
    [0,0,0],
    [4, 0, -16],
    [8, -8, 8],
    [4, 0, 4],
    [-8, 8, -8],
    [16, -8, 16],
    [4, 4, -16],
    [-4, 0, -4],
    [8, -4, -4],
    [0, -8, 0],
    [-8, 8, 4],
    [16, -8, -8],
    [16, 16, 4],
    [0, 0, -4],
    [16, 16, 16],
    [0, 4, -8],
    [4, 16, -4],
    [8, 0, 4],
    [16, -8, 16],
    [16, -8, 0],
    [-16, -16, -4],
    [-8, 8, 0],
    [0, 8, 16],
    [-16, 0, 8],
    [-8, 16, 0],
    [-16, 16, 8],
    [4, -16, -16],
    [0, 0, 0],
    [-8, 4, 0],
    [16, 8, -16],
    [-4, -16, -4],
    [4, -16, -4],
    [-16, -16, 8],
    [16, -16, 16],
    [16, -8, 4],
    [-8, -4, 16],
    [-8, 0, 4],
    [0, 16, 16],
    [-4, 4, 4],
    [4, 4, 4],
    [0, 8, 0],
    [8, 0, 16],
    [-4, -16, -4],
    [16, -4, 4],
    [-8, 4, 0],
    [0, -4, 4],
    [0, 16, 4],
    [8, 16, 16],
    [4, -4, -16],
    [16, 8, 0],
    [0, 0, -4],
    [16, -4, -16],
    [8, 16, -8],
    [-4, -4, 4],
    [4, -8, -8],
    [0, 0, -4],
    [0, 16, 0],
    [4, 4, -8],
    [-16, 0, 16],
    [-8, -16, 16],
    [4, -4, -16],
    [0, -4, -8],
    [-4, 4, 8],
    [8, -8, 16],
    [0, 0, 16],
    [8, 4, -4],
    [-4, 4, -4],
    [-16, -8, -4],
    [0, 8, 16],
    [-16, 4, 0],
    [8, 0, 8],
    [0, 8, 0],
    [4, -8, 0],
    [-4, -16, -4],
    [16, -4, 8],
    [4, 8, 16],
    [4, 4, -4],
    [-8, 0, 16],
    [-4, -16, 8],
    ], dtype=int)

SnapshotBeamAngles_deg=np.array([
    [-140.23698051937959, -158.84607032410833, 4.580713219059334],
    [-157.53283511086946, 19.02508453339945, 129.19217339545673],
    [141.14312740788262, -103.65708970489749, -28.300494661105347],
    [154.8513342681544, -59.54122353338134, -150.26798725466364],
    [-73.29659170698685, -62.48235301737675, -19.278233044875606],
    [103.36485049023048, -27.691085860806737, 69.68135152959138],
    [49.81157558321311, -179.47067699383365, -143.6905453818346],
    [91.15921894170265, -172.38186286690708, 135.75640830083125],
    [-58.67084352042218, 170.12545482643617, -21.73558025536758],
    [155.88883275867505, -162.87083373991663, -113.42330581056817],
    [-7.835477531915615, -50.88178985196052, -38.67219428561435],
    [-129.3691383119941, 3.881758273709096, 16.83077587063508],
    [-104.70982471506731, 87.45096343806478, 12.227009703506013],
    [107.03685203914677, -154.70965601547198, 29.024391795254786],
    [94.01096193103109, 117.6974800097367, -99.46784495625751],
    [94.85335668159559, 164.16565708660346, -97.18108594208573],
    [-95.00647259378573, -68.17831343914808, -178.65703770807633],
    [-109.19592846508958, -39.68608133736069, -138.34891713589465],
    [178.05267609771, 13.462283405450563, -18.385515042742128],
    [-59.57968167085379, -107.17610915157675, 33.770758063822],
    [45.0665093986419, -52.995838126355736, -150.57764139778808],
    [-19.089495594313178, -26.591061828211735, 5.661829480774145],
    [139.02980446714736, 54.08001773383046, 3.45401894659733],
    [132.8378696688248, -18.543915336049224, 141.0501275925714],
    [-62.09629620480597, 73.34900774735345, -39.08405968016939],
    [-99.3228697477531, -15.855715276203313, -153.18681856714161],
    [-92.36536624675203, -4.072980952496039, -5.645062071786668],
    [-101.46646618960003, -63.48826304198661, 35.19287192952069],
    [82.49298706419052, 108.16471301828443, 146.2349323896675],
    [72.13755655709355, 143.71940126523623, 137.04225353925915],
    [137.71227576806615, -29.192496566291908, 115.05852172640022],
    [31.56099474291085, 11.238883537744982, 108.28984117199042],
    [-70.39010070783202, -9.492524819898819, 154.8607501461808],
    [-12.793895523139412, -164.71110494949525, 123.23983819647816],
    [57.606996303250924, 19.421209003183293, 118.72418607579749],
    [-58.635823361073435, -99.26076464951717, 26.581706686029065],
    [-70.36526018487994, -18.30942998337335, -83.42761162215959],
    [75.8850443165253, 120.4868040543247, 149.4333512255576],
    [50.39014010020634, 61.15642083968524, -162.01267000845334],
    [39.64610984022633, 29.915771920457132, 171.6998199904154],
    [63.15355565668315, -46.44193964103118, 4.014366242067268],
    [55.90973492561264, -66.03335412470398, -163.45400384150295],
    [-64.20090035362303, -26.164173723950256, -87.30612733043017],
    [-42.408469827959266, -172.02629752247063, 100.15292796845642],
    [39.42793666005733, 127.71006222699322, -95.52028947057958],
    [179.78236167540717, -11.684113300554145, 46.92425767800236],
    [-111.72800837372925, 115.39345013254814, 77.30159169377879],
    [54.22216374085218, -13.3136429743532, -127.88705906462323],
    [-70.54699862275845, 90.52969627301172, -28.365962233968077],
    [-156.69355686691605, -1.856275003087319, 21.20324975072836],
    [60.31445529115649, -148.8927286855201, 129.76105538876595],
    [-116.83211457977228, 20.683747234283175, -120.56226692598521],
    [178.58531605798896, -39.59821402764959, 112.11233244819795],
    [-82.50606124018702, -165.38116351574757, -47.34271262282769],
    [127.38156548308962, 139.54226362279104, 111.77649903673819],
    [133.1868406073554, 80.83423774700066, -64.21091505378149],
    [62.44899402761047, -34.99483733099754, 62.79137058887511],
    [-176.28664258676213, 171.02832981089472, -144.98867609075927],
    [-165.48989970689314, -35.25310859405096, -136.86419832721572],
    [41.431522158762476, -164.57970183789692, -20.180114114878506],
    [14.619515979539472, 18.359312225273072, -17.47045261087092],
    [-98.25695885948662, 165.59408633334328, 11.917426295826601],
    [25.51224471388963, -85.32061378386757, 103.0436647211784],
    [-49.03633690012387, -128.198226234206, 82.34106305157013],
    [168.46084012156138, -53.01880013211782, 10.459208522753784],
    [-110.22999002274406, -157.14908286883315, -74.85873788513787],
    [-11.115109163122725, -153.03467663429512, 81.14825458294479],
    [-137.04746908867364, 94.42520027900855, 139.83800021840227],
    [-27.093899312890443, 123.19594559689898, -11.040380349015294],
    [-93.52292181466237, 119.5070404345924, 53.01100234895466],
    [133.78875864230042, 26.175313304958138, 133.6521921635105],
    [13.043831805216314, -47.5234936541236, -140.18708431359843],
    [103.32040348918673, -122.0453144509957, 31.667297483678766],
    [-148.3140296421019, -104.61238581219035, 152.23422045785992],
    [2.819546451477919, 158.40178917576986, 75.3029631351063],
    [-77.91007782053991, 129.23892045040975, 79.34034159582029],
    [-22.808518021854923, 3.9368165717869488, -12.47821132179675],
    [-79.01619384401654, -51.705253761865436, -144.82715907671647],
    ], dtype=float)

