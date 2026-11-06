"""
Read DEM data from the SIS tile database.

This is a Python port of MATLAB dem2017YM_MERIT_R1.m. It reads seamless
DEM/REF mosaics from the SIS tile database (5°×5° or 10°×10° tiles
organized in 30°×30° region subdirectories).

Supported DEMs:
  - MERIT2017_R1 : 3" DEM, 5° tiles, int16 big-endian (.MER files)
  - MERIT_2160   : 30" REF, 10° tiles, float32 (.bin files)
"""

import os
import numpy as np
from scipy.io import loadmat

from pyTGF.io.dem_io import DEMData


# ---------------------------------------------------------------------------
# Configuration for each DEM type in the SIS database
# ---------------------------------------------------------------------------

_DEM_CONFIG = {
    'MERIT2017_R1': {
        'subdir': 'MERIT2017_R1',      # directory under DEMs_Earth/
        'index_file': 'MERIT2017.mat',  # MATLAB index file
        'tilesize': 5,                  # degrees per tile
        'resolution': 3.0 / 3600.0,     # degrees
        'tileelems': 6000,              # elements per tile side
        'dtype': np.int16,              # storage type
        'endian': '>',                  # big-endian (ieee-be)
        'shift_lat': -1.5 / 3600.0,     # georeferencing correction
        'shift_lon': -1.5 / 3600.0,
        'seaheightfl': -20000,          # sea-height flag value
    },
    'MERIT_2160': {
        'subdir': 'MERIT_2160',
        'index_file': 'MERIT2160.mat',
        'tilesize': 10,
        'resolution': 30.0 / 3600.0,
        'tileelems': 1200,
        'dtype': np.float32,            # 'single' in MATLAB
        'endian': '',                   # native byte order
        'shift_lat': 0.0,
        'shift_lon': 0.0,
        'seaheightfl': -20000,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latlon2filename(lon: float, lat: float) -> str:
    """
    Generate a region subdirectory name from (lon, lat).

    Port of MATLAB latlon2filename.m.
    Example: (107, 37) -> 'N37E107', (-107, 0) -> 'N00W107'
    """
    if lat >= 0:
        name = 'N'
    else:
        name = 'S'
    if abs(lat) < 10:
        name += '0'
    name += str(abs(int(lat)))

    if lon >= 0:
        name += 'E'
    else:
        name += 'W'
    if abs(lon) < 100:
        name += '0'
    if abs(lon) < 10:
        name += '0'
    name += str(abs(int(lon)))
    return name


def _parse_fchar(fchar) -> list:
    """
    Parse the 'fchar' variable from a MATLAB .mat index file into a list
    of tile filenames (strings).

    fchar can be:
      - a numpy string array (e.g. dtype '<U11')  -> directly ravel
      - a cell array of strings  -> numpy object array
      - a 2-D char array         -> numpy 2-D uint8 array
    """
    if not isinstance(fchar, np.ndarray):
        return [str(fchar)]

    # numpy string dtype (e.g. '<U11') — most common with scipy.io.loadmat
    if fchar.dtype.kind == 'U':
        return [str(s) for s in fchar.ravel()]

    # numpy bytes dtype (e.g. 'S11')
    if fchar.dtype.kind == 'S':
        return [s.decode('utf-8', errors='replace').strip() for s in fchar.ravel()]

    # cell array of strings
    if fchar.dtype == object:
        result = []
        for item in fchar.ravel():
            if isinstance(item, np.ndarray):
                if item.dtype.kind in ('u', 'i'):
                    result.append(''.join(chr(c) for c in item.ravel()))
                elif item.dtype.kind in ('U', 'S'):
                    result.append(str(item.ravel()[0]) if item.size else '')
                else:
                    result.append(str(item.ravel()[0]) if item.size else '')
            else:
                result.append(str(item))
        return result

    # 2-D char array (each row is a filename, space-padded)
    if fchar.dtype.kind in ('u', 'i'):
        result = []
        for row in fchar:
            s = ''.join(chr(c) for c in row if c != 0)
            result.append(s.strip())
        return result

    return [str(fchar)]


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_merit_dem(
    dem_name: str,
    minlon: float,
    maxlon: float,
    minlat: float,
    maxlat: float,
    sis_data_path: str,
    sea: float = 0.0,
    facX: float = 1.0,
    facY: float = 1.0,
    method: str = 'linear',
) -> DEMData:
    """
    Read a DEM from the SIS tile database.

    Python port of dem2017YM_MERIT_R1.m.

    Parameters
    ----------
    dem_name : str
        DEM type identifier: 'MERIT2017_R1' or 'MERIT_2160'.
    minlon, maxlon, minlat, maxlat : float
        Bounding box in degrees.
    sis_data_path : str
        Path to the DEMs_Earth directory (containing MERIT2017_R1/,
        MERIT_2160/, etc.).
    sea : float, optional
        Value to use for ocean cells (default 0.0).
    facX, facY : float, optional
        Interpolation factors (1.0 = no interpolation, pass through).
    method : str, optional
        Interpolation method when facX/facY != 1 (default 'linear').

    Returns
    -------
    DEMData
        DEM with row 0 = minlat, column 0 = minlon.
    """
    if dem_name not in _DEM_CONFIG:
        raise ValueError(
            f"Unknown DEM type: '{dem_name}'. "
            f"Supported: {list(_DEM_CONFIG.keys())}"
        )
    cfg = _DEM_CONFIG[dem_name]

    tilesize = cfg['tilesize']
    resolution = cfg['resolution']
    tileelems = cfg['tileelems']
    dtype = cfg['dtype']
    endian = cfg['endian']
    shift_lat = cfg['shift_lat']
    shift_lon = cfg['shift_lon']
    seaheightfl = cfg['seaheightfl']

    # ------------------------------------------------------------------
    # 1. Determine interpolation flag and tolerance area
    # ------------------------------------------------------------------
    if facX == 1.0 and facY == 1.0:
        tolarea = 0.0
    else:
        tolarea = 0.1

    # ------------------------------------------------------------------
    # 2. Extend selection area
    # ------------------------------------------------------------------
    minlon2 = minlon - tolarea
    maxlon2 = maxlon + tolarea
    minlat2 = minlat - tolarea
    maxlat2 = maxlat + tolarea

    # ------------------------------------------------------------------
    # 3. Load index file and find required tiles
    # ------------------------------------------------------------------
    index_path = os.path.join(sis_data_path, cfg['subdir'], cfg['index_file'])
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    mat = loadmat(index_path, squeeze_me=True)
    fchar_raw = mat['fchar']
    flat = np.atleast_1d(mat['flat']).ravel().astype(np.float64)
    flon = np.atleast_1d(mat['flon']).ravel().astype(np.float64)
    filenames = _parse_fchar(fchar_raw)

    # Tile lower-left corners (on the tilesize grid)
    Sall_minlon = np.floor(minlon2 / tilesize) * tilesize
    Sall_maxlon = np.floor(maxlon2 / tilesize) * tilesize
    Sall_minlat = np.floor(minlat2 / tilesize) * tilesize
    Sall_maxlat = np.floor(maxlat2 / tilesize) * tilesize

    lon_vec = np.arange(Sall_minlon, Sall_maxlon + tilesize, tilesize)
    lat_vec = np.arange(Sall_minlat, Sall_maxlat + tilesize, tilesize)

    # ------------------------------------------------------------------
    # 4. Build target matrix geometry (cell-centred registration)
    # ------------------------------------------------------------------
    fac = 3600 * 100  # 0.01 arc-sec units
    r = int(round(resolution * fac))
    r2 = r // 2

    Amin_lat = int(np.floor(minlat2 * fac / r) * r + r2)
    Amax_lat = int(np.floor(maxlat2 * fac / r) * r - r2)
    Amin_lon = int(np.floor(minlon2 * fac / r) * r + r2)
    Amax_lon = int(np.floor(maxlon2 * fac / r) * r - r2)

    Avec_lat = np.arange(Amin_lat, Amax_lat + r, r, dtype=np.int64)
    Avec_lon = np.arange(Amin_lon, Amax_lon + r, r, dtype=np.int64)
    An1 = len(Avec_lat)
    Am1 = len(Avec_lon)

    # Initialize target matrix with seaheightfl
    A = np.full((An1, Am1), float(seaheightfl), dtype=np.float64)

    # ------------------------------------------------------------------
    # 5. Fill the matrix tile by tile
    # ------------------------------------------------------------------
    tilesize_sd = 30  # region subdirectory size in degrees

    for lat0 in lat_vec:
        for lon0 in lon_vec:
            # Find this tile in the index
            mask = (flat == lat0) & (flon == lon0)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue  # tile not in database (ocean / not available)

            tilename = filenames[idx[0]]

            # Tile bounds
            slatmin = float(lat0)
            slatmax = slatmin + tilesize
            slonmin = float(lon0)
            slonmax = slonmin + tilesize

            # Source vector (cell-centred)
            Smin_lat = int(slatmin * fac + r2)
            Smax_lat = int(slatmax * fac - r2)
            Smin_lon = int(slonmin * fac + r2)
            Smax_lon = int(slonmax * fac - r2)

            Svec_lat = np.arange(Smin_lat, Smax_lat + r, r, dtype=np.int64)
            Svec_lon = np.arange(Smin_lon, Smax_lon + r, r, dtype=np.int64)

            # Intersect source and target vectors
            a_lat = np.intersect1d(Svec_lat, Avec_lat, assume_unique=True)
            a_lon = np.intersect1d(Svec_lon, Avec_lon, assume_unique=True)

            if len(a_lat) == 0 or len(a_lon) == 0:
                continue  # no overlap

            # Target indices (0-based)
            Aix_lat_min = int(np.searchsorted(Avec_lat, a_lat[0]))
            Aix_lat_max = int(np.searchsorted(Avec_lat, a_lat[-1]))
            Aix_lon_min = int(np.searchsorted(Avec_lon, a_lon[0]))
            Aix_lon_max = int(np.searchsorted(Avec_lon, a_lon[-1]))

            # Source indices (0-based)
            Six_lat_min = int(np.searchsorted(Svec_lat, a_lat[0]))
            Six_lat_max = int(np.searchsorted(Svec_lat, a_lat[-1]))
            Six_lon_min = int(np.searchsorted(Svec_lon, a_lon[0]))
            Six_lon_max = int(np.searchsorted(Svec_lon, a_lon[-1]))

            # Region subdirectory (30°×30°)
            region_minlon = np.floor(slonmin / tilesize_sd) * tilesize_sd
            region_minlat = np.floor(slatmin / tilesize_sd) * tilesize_sd
            # Fix date-line bug (mirrors MATLAB)
            if region_minlon == 180:
                region_minlon -= 360
            if region_minlon < -180:
                region_minlon += 360

            region_dirname = _latlon2filename(region_minlon, region_minlat)
            tile_path = os.path.join(
                sis_data_path, cfg['subdir'], region_dirname, tilename
            )

            if not os.path.exists(tile_path):
                print(f"  Warning: tile file not found: {tile_path}")
                continue

            # Read tile data
            te = tileelems
            count = te * te
            # Build numpy dtype with explicit endianness.
            # np.dtype(dtype).str already has a byte-order prefix ('<i2'),
            # so we replace it rather than concatenate.
            base_str = np.dtype(dtype).str
            if endian:
                # Replace the leading byte-order char with the requested one
                np_dtype = np.dtype(endian + base_str[1:])
            else:
                np_dtype = np.dtype(base_str)
            with open(tile_path, 'rb') as fid:
                S = np.fromfile(fid, dtype=np_dtype, count=count)

            if S.size < count:
                print(f"  Warning: short read for {tilename}: "
                      f"{S.size}/{count}")
                continue

            # Reshape: MATLAB reshape(S,te,te)' == Python reshape(te,te)
            # (row-major), then flipud (flipdim(S,1))
            S = S.reshape((te, te))
            # Convert to float64 first, then flip. Use float64 to avoid
            # overflow/invalid-cast warnings when raw int16 has nodata.
            S = S.astype(np.float64)
            S = np.flipud(S)

            # Copy data block (inclusive on both ends)
            A[Aix_lat_min:Aix_lat_max + 1,
              Aix_lon_min:Aix_lon_max + 1] = \
                S[Six_lat_min:Six_lat_max + 1,
                  Six_lon_min:Six_lon_max + 1]

    # ------------------------------------------------------------------
    # 6. Replace sea heights / nodata values
    # ------------------------------------------------------------------
    # MATLAB: A(find(A<=seaheightfl+1))=sea;
    # The raw tiles also contain NaN and huge float32 nodata fill values
    # (e.g. 3.4e38, -2e36, 999999, 28843) that are not caught by the
    # seaheightfl test. Replace all of them with sea so downstream
    # interpolation works cleanly.
    A[~np.isfinite(A)] = sea
    A[A <= seaheightfl + 1] = sea
    # Catch absurdly large positive nodata values.
    # Real elevations on Earth stay within [-11000, 9000] m; anything
    # above 10000 m in a DEM/REF product is a fill value.
    A[A > 10000] = sea

    # ------------------------------------------------------------------
    # 7. Build coordinate vectors (apply shift correction)
    # ------------------------------------------------------------------
    Avec_lon_deg = Avec_lon / fac + shift_lon
    Avec_lat_deg = Avec_lat / fac + shift_lat

    # ------------------------------------------------------------------
    # 8. Interpolate if fac != 1, otherwise pass through
    # ------------------------------------------------------------------
    if facX != 1.0 or facY != 1.0:
        from scipy.interpolate import RegularGridInterpolator

        res_targetX = resolution * facX
        res_targetY = resolution * facY

        B_veclon = np.arange(minlon, maxlon + res_targetX * 0.5, res_targetX)
        B_veclat = np.arange(minlat, maxlat + res_targetY * 0.5, res_targetY)

        interp = RegularGridInterpolator(
            (Avec_lat_deg, Avec_lon_deg), A,
            method=method, bounds_error=False, fill_value=sea,
        )
        lon_grid, lat_grid = np.meshgrid(B_veclon, B_veclat)
        Z = interp((lat_grid, lon_grid))

        return DEMData(
            minlat=float(B_veclat[0]),
            maxlat=float(B_veclat[-1]),
            minlon=float(B_veclon[0]),
            maxlon=float(B_veclon[-1]),
            reslat=res_targetY,
            reslon=res_targetX,
            z=Z,
        )
    else:
        return DEMData(
            minlat=float(Avec_lat_deg[0]),
            maxlat=float(Avec_lat_deg[-1]),
            minlon=float(Avec_lon_deg[0]),
            maxlon=float(Avec_lon_deg[-1]),
            reslat=resolution,
            reslon=resolution,
            z=A,
        )
