"""
Binary DEM file I/O

Mirrors MATLAB read_DEM.m (L2072-L2092 in TC_GUI.m):
    [minlat, maxlat, minlon, maxlon, reslat, reslon, elev] = read_DEM(fileName)

Binary format (native-endian doubles):
    Header (6 doubles): minlat, maxlat, minlon, maxlon, reslat, reslon
    Body   (numRows * numCols doubles): elevation, stored row by row
        numCols = ((maxlon - minlon) / reslon) + 1
        numRows = (cnt - 6) / numCols
    MATLAB reshape: reshape(elev, round(numCols), round(numRows))'
    => row i of the result is the i-th DEM row (latitude band).
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class DEMData:
    """Container for a DEM grid loaded from binary file"""

    minlat: float
    maxlat: float
    minlon: float
    maxlon: float
    reslat: float
    reslon: float
    z: np.ndarray  # shape (n_lat, n_lon)

    @property
    def n_lat(self) -> int:
        return self.z.shape[0]

    @property
    def n_lon(self) -> int:
        return self.z.shape[1]

    @property
    def lats(self) -> np.ndarray:
        # MATLAB: y1 = minlat:reslat:maxlat+1e-7
        return np.arange(self.minlat, self.maxlat + 1e-7, self.reslat)

    @property
    def lons(self) -> np.ndarray:
        # MATLAB: x1 = minlon:reslon:maxlon+1e-7
        return np.arange(self.minlon, self.maxlon + 1e-7, self.reslon)

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon_grid, lat_grid) like MATLAB meshgrid(x1, y1)

        Result shapes: (n_lat, n_lon), lat varies along rows (axis 0),
        lon varies along columns (axis 1).
        """
        return np.meshgrid(self.lons, self.lats)


def read_dem(filename: str) -> DEMData:
    """
    Read a binary DEM file.

    Corresponds to MATLAB read_DEM (TC_GUI.m L2072-L2092).

    MATLAB:
        fid = fopen(fileName,'r');
        [data, cnt] = fread(fid, inf, 'double','native');
        fclose(fid);
        minlat=data(1); maxlat=data(2); minlon=data(3); maxlon=data(4);
        reslat=data(5); reslon=data(6);
        numCols=((maxlon-minlon)/reslon)+1;
        numRows = (cnt-6) / numCols;
        elev=data(7:end);
        elev = reshape(elev,round(numCols),round(numRows))';

    The reshape+transpose in MATLAB is equivalent to numpy's row-major
    reshape to (numRows, numCols), since the file stores elevation row by
    row (each row spans longitudes at a fixed latitude).

    Args:
        filename: Path to binary DEM file.

    Returns:
        DEMData with grid metadata and elevation matrix.

    Raises:
        FileNotFoundError, IOError
    """
    # fread(fid, inf, 'double', 'native') -> native-endian float64
    data = np.fromfile(filename, dtype=np.float64)
    cnt = data.size

    if cnt < 6:
        raise IOError(f"DEM file '{filename}' too small: only {cnt} values")

    minlat = float(data[0])
    maxlat = float(data[1])
    minlon = float(data[2])
    maxlon = float(data[3])
    reslat = float(data[4])
    reslon = float(data[5])

    if not (np.isfinite(reslat) and np.isfinite(reslon)) or reslat <= 0 or reslon <= 0:
        raise IOError(
            f"DEM file '{filename}': invalid resolution "
            f"(reslat={reslat}, reslon={reslon}); both must be positive"
        )
    if maxlon <= minlon or maxlat <= minlat:
        raise IOError(
            f"DEM file '{filename}': invalid extent "
            f"(min={minlat},{minlon} max={maxlat},{maxlon})"
        )

    # number of columns and rows (mirrors MATLAB)
    numCols = int(round(((maxlon - minlon) / reslon) + 1))
    if numCols <= 0:
        raise IOError(f"DEM file '{filename}': computed {numCols} columns")
    numRows = int(round((cnt - 6) / numCols))
    if numRows <= 0:
        raise IOError(f"DEM file '{filename}': computed {numRows} rows")

    elev = data[6:]
    if elev.size != numRows * numCols:
        raise IOError(
            f"DEM file '{filename}': elevation count {elev.size} != "
            f"numRows*numCols ({numRows}*{numCols}={numRows*numCols})"
        )

    # MATLAB: reshape(elev, numCols, numRows)'  (column-major fill + transpose)
    # Equivalent numpy: reshape to (numRows, numCols) in C order (row-major),
    # since the file stores row by row.
    z = np.reshape(elev, (numRows, numCols))

    return DEMData(minlat, maxlat, minlon, maxlon, reslat, reslon, z)


def write_dem(filename: str, dem: DEMData) -> None:
    """
    Write a DEMData object back to binary format.

    Corresponds to MATLAB Write_DEM.m (inverse of read_DEM).

    Args:
        filename: Output path.
        dem: DEMData instance.
    """
    header = np.array(
        [dem.minlat, dem.maxlat, dem.minlon, dem.maxlon,
         dem.reslat, dem.reslon],
        dtype=np.float64,
    )
    # Flatten row by row (matches read order)
    body = dem.z.reshape(-1).astype(np.float64)
    np.concatenate([header, body]).tofile(filename)
