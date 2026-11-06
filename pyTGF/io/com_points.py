"""
Computation points file I/O

Mirrors MATLAB logic in TC_GUI.m L786-L810:
    fid = fopen(ComPoints,'r');
    [data, cnt] = fread(fid, inf, 'double','native');
    fclose(fid);
    nrow=cnt/4;
    data=reshape(data,nrow,4);
    com_lat=data(:,2);
    com_lon=data(:,3);
    com_h=data(:,4);
    num=data(end,1);
    com_h(com_h<0)=1;   % negative heights corrected with positive constant height
    H_com=com_h;
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ComPointData:
    """Container for computation points"""

    ids: np.ndarray      # Point IDs (column 1)
    lats: np.ndarray     # Latitudes [deg]  (column 2)
    lons: np.ndarray     # Longitudes [deg] (column 3)
    heights: np.ndarray  # Ellipsoidal heights [m] (column 4)

    @property
    def n_points(self) -> int:
        return len(self.lats)

    @property
    def num(self) -> float:
        """MATLAB: num = data(end,1) — the ID of the last point."""
        return float(self.ids[-1])

    def as_array(self) -> np.ndarray:
        """Return as (n, 4) array [id, lat, lon, h] matching MATLAB layout."""
        return np.column_stack([self.ids, self.lats, self.lons, self.heights])


def read_com_points(filename: str, correct_negative: bool = True) -> ComPointData:
    """
    Read binary computation points file.

    Corresponds to TC_GUI.m L786-L810.

    Binary format: sequence of native-endian doubles, laid out as
    rows of [N, lat, lon, H] (4 doubles per point).

    MATLAB:
        [data, cnt] = fread(fid, inf, 'double','native');
        nrow = cnt/4;
        data = reshape(data, nrow, 4);   % column-major -> (nrow, 4)
        com_lat = data(:,2); com_lon = data(:,3); com_h = data(:,4);
        num = data(end,1);
        com_h(com_h<0) = 1;

    The file stores data in MATLAB column-major order after reshape:
    [id1,id2,...,idN, lat1,lat2,...,latN, lon1,...,lonN, h1,...,hN]
    so numpy must reshape with Fortran (column-major) order to match
    MATLAB's reshape(data, nrow, 4).

    Args:
        filename: Path to binary file.
        correct_negative: If True, negative heights set to 1 (mirrors L807).

    Returns:
        ComPointData with arrays of lat/lon/h.
    """
    data = np.fromfile(filename, dtype=np.float64)
    cnt = data.size

    if cnt % 4 != 0:
        raise IOError(
            f"Computation points file '{filename}': count {cnt} is not "
            f"a multiple of 4 (expected [N, lat, lon, H] per point)"
        )

    nrow = cnt // 4
    if nrow == 0:
        raise IOError(f"Computation points file '{filename}' contains no points")
    # MATLAB reshape(data, nrow, 4) fills column-major (Fortran order):
    # column 1 = ids, column 2 = lats, column 3 = lons, column 4 = heights.
    data = np.reshape(data, (nrow, 4), order='F')

    ids = data[:, 0]
    lats = data[:, 1]
    lons = data[:, 2]
    heights = data[:, 3].copy()  # copy so we can mutate

    bad_lat = ~np.isfinite(lats) | (np.abs(lats) > 90.0)
    bad_lon = ~np.isfinite(lons) | (np.abs(lons) > 360.0)
    if bad_lat.any() or bad_lon.any():
        raise IOError(
            f"Computation points file '{filename}': "
            f"{int(bad_lat.sum())} invalid latitude(s) and "
            f"{int(bad_lon.sum())} invalid longitude(s)"
        )

    # MATLAB: com_h(com_h<0) = 1
    if correct_negative:
        heights[heights < 0] = 1.0

    return ComPointData(ids=ids, lats=lats, lons=lons, heights=heights)


def write_com_points(filename: str,
                     lats: np.ndarray,
                     lons: np.ndarray,
                     heights: np.ndarray,
                     ids: Optional[np.ndarray] = None) -> None:
    """
    Write computation points to the TGF binary format.

    Corresponds to the inverse of read_com_points and to MATLAB:
        com = [N' com_lat(:) com_lon(:) com_h(:)];
        fwrite(fid, com, 'double', 'native');

    The file is laid out in MATLAB column-major order after reshape:
    [id1,id2,...,idN, lat1,lat2,...,latN, lon1,...,lonN, h1,...,hN]
    so that MATLAB reshape(data, nrow, 4) fills columns correctly.

    Args:
        filename: Output path.
        lats: 1D array of latitudes [deg].
        lons: 1D array of longitudes [deg].
        heights: 1D array of ellipsoidal heights [m].
        ids: Optional 1D array of point IDs. If None, uses 1..N.
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    heights = np.asarray(heights, dtype=np.float64).ravel()

    n = lats.size
    if lons.size != n or heights.size != n:
        raise ValueError(
            f"Size mismatch: lats={lats.size}, lons={lons.size}, "
            f"heights={heights.size}"
        )

    if ids is None:
        ids = np.arange(1, n + 1, dtype=np.float64)
    else:
        ids = np.asarray(ids, dtype=np.float64).ravel()
        if ids.size != n:
            raise ValueError(f"ids size {ids.size} != {n}")

    # Stack as (n, 4) then flatten in Fortran (column-major) order so that
    # MATLAB reshape(data, n, 4) reproduces the same [id, lat, lon, h] columns.
    com = np.column_stack([ids, lats, lons, heights])
    com.ravel(order="F").tofile(filename)
