"""
Grid construction and matrix preallocation utilities

Mirrors TC_GUI.m:
    Build_grid           (L2094-L2120) ring-zone grid without overlap/leakage
    matrix_preallocating (L5141-L5152) preallocate 10 result vectors
"""

from typing import Tuple, List
import numpy as np


def build_grid(lon: float, lat: float,
               extend_x1: float, extend_y1: float,
               extend_x2: float, extend_y2: float,
               resolution_x2: float, resolution_y2: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a ring-zone grid avoiding overlap and leakage between adjacent zones.

    Corresponds to TC_GUI.m L2094-L2120 (Build_grid).

    Generates a grid covering [lon-extend_x2, lon+extend_x2-res_x2] x
    [lat-extend_y2, lat+extend_y2-res_y2] at the given resolution, then
    removes the inner rectangle [lon-extend_x1, lon+extend_x1] x
    [lat-extend_y1, lat+extend_y1] so that only the ring remains. Finally
    cell centers are shifted by +resolution/2 in both directions.

    MATLAB:
        lon21 = (lon-extend_x2):resolution_x2:(lon+extend_x2-resolution_x2);
        lat21 = (lat-extend_y2):resolution_y2:(lat+extend_y2-resolution_y2);
        [lon21,lat21] = meshgrid(lon21,lat21);
        [m n] = size(lat21);
        n1 = round((extend_y2-extend_y1)/resolution_y2)+1;   % first inner row (1-based)
        n2 = m - round((extend_y2-extend_y1)/resolution_y2); % last inner row (1-based)
        n3 = round((extend_x2-extend_x1)/resolution_x2)+1;   % first inner col (1-based)
        n4 = n - round((extend_x2-extend_x1)/resolution_x2); % last inner col (1-based)
        r = n1:n2; c = n3:n4;
        [c r] = meshgrid(c,r);
        nn = r(:) + m*(c(:)-1);          % column-major linear indices of inner cells
        lon21(nn) = []; lat21(nn) = [];  % delete inner cells, flatten column-major
        lon21 = lon21 + resolution_x2/2;
        lat21 = lat21 + resolution_y2/2;

    The flattened output is in MATLAB column-major (Fortran) order. We
    replicate this exactly so downstream summation order matches MATLAB.

    Args:
        lon, lat: Center of the zone.
        extend_x1, extend_y1: Inner half-extent (excluded).
        extend_x2, extend_y2: Outer half-extent (included).
        resolution_x2, resolution_y2: Grid spacing in lon/lat [deg].

    Returns:
        (lon21, lat21): Flattened arrays of ring-zone cell centers,
        in Fortran (column-major) order matching MATLAB.
    """
    # 1D coordinate vectors (cell lower-left corners)
    # MATLAB a:step:b includes b if reachable. Add a half-step tolerance to
    # guarantee the last point is included despite floating-point error.
    lon_1d = np.arange(lon - extend_x2,
                       lon + extend_x2 - resolution_x2 + resolution_x2 * 0.5,
                       resolution_x2)
    lat_1d = np.arange(lat - extend_y2,
                       lat + extend_y2 - resolution_y2 + resolution_y2 * 0.5,
                       resolution_y2)

    # meshgrid: lon varies along columns (axis 1), lat along rows (axis 0)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    m, n = lat2d.shape  # m = len(lat_1d), n = len(lon_1d)

    # Inner rectangle row/col bounds (convert MATLAB 1-based to 0-based)
    dy = int(round((extend_y2 - extend_y1) / resolution_y2))
    dx = int(round((extend_x2 - extend_x1) / resolution_x2))
    n1 = dy + 1          # 1-based first inner row
    n2 = m - dy          # 1-based last inner row
    n3 = dx + 1          # 1-based first inner col
    n4 = n - dx          # 1-based last inner col

    # Build a boolean mask of the inner rectangle (to be removed)
    mask = np.ones((m, n), dtype=bool)
    # 0-based slice: rows [n1-1 .. n2-1], cols [n3-1 .. n4-1]
    mask[n1 - 1:n2, n3 - 1:n4] = False

    # Flatten in Fortran (column-major) order to match MATLAB's
    # lon21(nn)=[] which flattens column-major after deletion.
    lon21 = lon2d.flatten(order='F')[mask.flatten(order='F')]
    lat21 = lat2d.flatten(order='F')[mask.flatten(order='F')]

    # Shift to cell centers
    lon21 = lon21 + resolution_x2 / 2.0
    lat21 = lat21 + resolution_y2 / 2.0

    return lon21, lat21


def build_grid_layout(lon: float, lat: float,
                      extend_x1: float, extend_y1: float,
                      extend_x2: float, extend_y2: float,
                      resolution_x2: float, resolution_y2: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same ring-zone grid as build_grid(), but returned as a tensor layout.

    build_grid() flattens the ring to 1-D arrays; sampling an interpolant on
    those 3.7e6 scattered points costs ~12 s per field because FITPACK has to
    binary-search the knot vector per point. The identical ring is a regular
    tensor grid, so the caller can evaluate the interpolant with bispev
    (grid=True) in ~0.05 s and then apply `mask` - numerically bit-identical,
    since both paths use the same spline coefficients and the same sample
    coordinates.

    Args:
        (same as build_grid)

    Returns:
        (lon_1d, lat_1d, mask):
            lon_1d, lat_1d: 1-D cell-center axes of the full outer square.
            mask: boolean array of shape (len(lat_1d), len(lon_1d)), True for
                ring cells. Flattening a (nlat, nlon) value array with
                ``order='F'`` and indexing by ``mask.flatten(order='F')``
                reproduces build_grid()'s ordering exactly.
    """
    lon_1d = np.arange(lon - extend_x2,
                       lon + extend_x2 - resolution_x2 + resolution_x2 * 0.5,
                       resolution_x2)
    lat_1d = np.arange(lat - extend_y2,
                       lat + extend_y2 - resolution_y2 + resolution_y2 * 0.5,
                       resolution_y2)
    lon_1d = lon_1d + resolution_x2 / 2.0
    lat_1d = lat_1d + resolution_y2 / 2.0

    m, n = len(lat_1d), len(lon_1d)
    dy = int(round((extend_y2 - extend_y1) / resolution_y2))
    dx = int(round((extend_x2 - extend_x1) / resolution_x2))

    mask = np.ones((m, n), dtype=bool)
    if dy > 0 and dx > 0:
        mask[dy:m - dy, dx:n - dx] = False
    return lon_1d, lat_1d, mask


def matrix_preallocating(m: int, n: int) -> List[np.ndarray]:
    """
    Preallocate 10 zero-filled result vectors of shape (m, n).

    Corresponds to TC_GUI.m L5141-L5152 (matrix_preallocating).

    MATLAB:
        function [var1..var10] = matrix_preallocating(m,n)
            var1=zeros(m,n); ... var10=zeros(m,n);

    Used for V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz.

    Args:
        m, n: Result matrix dimensions (typically m=1, n=n_points).

    Returns:
        List of 10 zero arrays, each of shape (m, n).
    """
    return [np.zeros((m, n)) for _ in range(10)]
