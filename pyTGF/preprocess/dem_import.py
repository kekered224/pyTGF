"""
DEM import utilities

Read DEM grids from common geospatial formats (GeoTIFF, ESRI ASCII,
NumPy npz) into pyTGF's DEMData container, and write them to the
binary format expected by the TGF engine.

The binary format (see io/dem_io.py) is:
    [minlat, maxlat, minlon, maxlon, reslat, reslon, z(:)]  (float64, row-major)
"""

from typing import Tuple, Optional
import numpy as np

from ..io.dem_io import DEMData, write_dem


def _geotransform_to_bounds(gt: tuple, n_rows: int, n_cols: int
                            ) -> Tuple[float, float, float, float,
                                       float, float]:
    """
    Convert a GDAL geotransform to (minlat, maxlat, minlon, maxlon,
    reslat, reslon) in degrees.

    GDAL geotransform convention:
        gt[0] = top-left x (lon)
        gt[1] = pixel width  (reslon, east-facing positive)
        gt[2] = row rotation (usually 0)
        gt[3] = top-left y (lat)
        gt[4] = column rotation (usually 0)
        gt[5] = pixel height (reslat, north-facing NEGATIVE)

    Returns degrees with reslat forced positive.
    """
    reslon = abs(gt[1])
    reslat = abs(gt[5])
    # GDAL gives outer pixel edges; DEMData stores node-center coordinates.
    minlon = gt[0] + reslon / 2.0
    maxlon = gt[0] + reslon * (n_cols - 0.5)
    maxlat = gt[3] - reslat / 2.0
    minlat = gt[3] - reslat * (n_rows - 0.5)
    return minlat, maxlat, minlon, maxlon, reslat, reslon


def read_tif(filename: str, band: int = 1) -> DEMData:
    """
    Read a GeoTIFF DEM into a DEMData object.

    Args:
        filename: Path to the .tif / .tiff file.
        band: Raster band index (1-based).

    Returns:
        DEMData with grid metadata derived from the geotransform.

    Raises:
        ImportError: if rasterio is not installed.
        RuntimeError: if the raster has no geotransform.
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required to read GeoTIFF files. "
            "Install with: pip install rasterio  (or conda install rasterio)"
        ) from e

    with rasterio.open(filename) as src:
        if not src.transform:
            raise RuntimeError(
                f"GeoTIFF '{filename}' has no geotransform / CRS metadata."
            )
        gt = src.transform.to_gdal()
        z = src.read(band).astype(np.float64)
        nodata = src.nodata

    n_rows, n_cols = z.shape
    minlat, maxlat, minlon, maxlon, reslat, reslon = _geotransform_to_bounds(
        gt, n_rows, n_cols
    )

    # Replace nodata with 0 (sea level) to match MATLAB convention
    if nodata is not None:
        z[z == nodata] = 0.0

    # rasterio reads row 0 = top (maxlat). DEMData expects row 0 = minlat,
    # so flip vertically to match the MATLAB/TC_GUI convention.
    z = np.flipud(z)

    return DEMData(minlat, maxlat, minlon, maxlon, reslat, reslon, z)


def read_ascii_grid(filename: str) -> DEMData:
    """
    Read an ESRI ASCII grid (.asc / .grd) into a DEMData object.

    Header keys recognized: ncols, nrows, xllcorner, yllcorner,
    xllcenter, yllcenter, cellsize, nodata_value.

    Returns:
        DEMData with row 0 = minlat (bottom row of the grid).
    """
    header = {}
    data_lines = []
    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Header lines have the form "KEY VALUE"
            parts = stripped.split()
            key = parts[0].lower()
            if key in ("ncols", "nrows", "xllcorner", "yllcorner",
                       "xllcenter", "yllcenter", "cellsize", "nodata_value"):
                header[key] = float(parts[1])
            else:
                # Once we hit a non-header line, treat the rest as data
                data_lines.append(stripped)
                data_lines.extend(line.strip() for line in f if line.strip())
                break

    n_cols = int(header["ncols"])
    n_rows = int(header["nrows"])
    cellsize = header["cellsize"]

    # DEMData stores node-center coordinates. ESRI "corner" keywords give the
    # outer lower-left corner, while "center" keywords already give the
    # lower-left cell center.
    if "xllcorner" in header:
        minlon = header["xllcorner"] + cellsize / 2.0
        minlat = header["yllcorner"] + cellsize / 2.0
    else:
        minlon = header["xllcenter"]
        minlat = header["yllcenter"]

    maxlon = minlon + cellsize * (n_cols - 1)
    maxlat = minlat + cellsize * (n_rows - 1)

    z = np.loadtxt("\n".join(data_lines).splitlines(), dtype=np.float64)
    z = z.reshape(n_rows, n_cols)

    if "nodata_value" in header:
        z[z == header["nodata_value"]] = 0.0

    # ESRI ASCII stores row 0 = top (maxlat). Flip to match DEMData convention.
    z = np.flipud(z)

    return DEMData(minlat, maxlat, minlon, maxlon, cellsize, cellsize, z)


def read_npz_grid(filename: str,
                  z_key: str = "z",
                  minlat_key: str = "minlat",
                  maxlat_key: str = "maxlat",
                  minlon_key: str = "minlon",
                  maxlon_key: str = "maxlon",
                  reslat_key: str = "reslat",
                  reslon_key: str = "reslon") -> DEMData:
    """
    Read a DEM grid stored in a NumPy .npz archive.

    All six metadata fields and the z array must be present as keys
    in the archive. Row 0 of z must correspond to minlat.
    """
    with np.load(filename) as npz:
        z = np.asarray(npz[z_key], dtype=np.float64)
        return DEMData(
            minlat=float(npz[minlat_key]),
            maxlat=float(npz[maxlat_key]),
            minlon=float(npz[minlon_key]),
            maxlon=float(npz[maxlon_key]),
            reslat=float(npz[reslat_key]),
            reslon=float(npz[reslon_key]),
            z=z,
        )


def dem_to_bin(dem: DEMData, filename: str) -> None:
    """
    Write a DEMData object to the TGF binary format.

    Thin wrapper around io.dem_io.write_dem for API symmetry with
    compoints_to_bin.
    """
    write_dem(filename, dem)


def crop_dem(dem: DEMData,
             minlat: float, maxlat: float,
             minlon: float, maxlon: float) -> DEMData:
    """
    Crop a DEMData grid to the given lat/lon bounds (inclusive).

    Returns a new DEMData with the same resolution.
    """
    lats = dem.lats
    lons = dem.lons
    i0 = np.searchsorted(lats, minlat, side="left")
    i1 = np.searchsorted(lats, maxlat, side="right")
    j0 = np.searchsorted(lons, minlon, side="left")
    j1 = np.searchsorted(lons, maxlon, side="right")

    z_crop = dem.z[i0:i1, j0:j1]
    new_minlat = lats[i0] if i0 < len(lats) else dem.minlat
    new_maxlat = lats[i1 - 1] if i1 > 0 else dem.maxlat
    new_minlon = lons[j0] if j0 < len(lons) else dem.minlon
    new_maxlon = lons[j1 - 1] if j1 > 0 else dem.maxlon

    return DEMData(new_minlat, new_maxlat, new_minlon, new_maxlon,
                   dem.reslat, dem.reslon, z_crop)
