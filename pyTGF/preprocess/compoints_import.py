"""
Computation-point import utilities

Read computation-point lists from ASCII / CSV files into pyTGF's
ComPointData container, and write them to the binary format expected
by the TGF engine.

Supported ASCII formats (one point per line):
    lon lat h          (3 columns)
    lat lon h          (3 columns, lat_first=True)
    id lon lat h       (4 columns)
    id lat lon h       (4 columns, lat_first=True)

CSV files with a header row are auto-detected by column name
(lon/long, lat, h/height/elev).
"""

from typing import Optional, Tuple
import numpy as np

from ..io.com_points import ComPointData, write_com_points


def _parse_ascii_columns(filename: str, lat_first: bool
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    Optional[np.ndarray]]:
    """
    Parse a whitespace-separated ASCII file into (lats, lons, heights, ids).

    Auto-detects 3-column (lon lat h) or 4-column (id lon lat h) layout.
    If lat_first=True, assumes (lat lon h) or (id lat lon h) order.
    """
    data = np.loadtxt(filename, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]
    if n_cols == 3:
        if lat_first:
            lats, lons, heights = data[:, 0], data[:, 1], data[:, 2]
        else:
            lons, lats, heights = data[:, 0], data[:, 1], data[:, 2]
        ids = None
    elif n_cols == 4:
        ids = data[:, 0]
        if lat_first:
            lats, lons, heights = data[:, 1], data[:, 2], data[:, 3]
        else:
            lons, lats, heights = data[:, 1], data[:, 2], data[:, 3]
    else:
        raise ValueError(
            f"Expected 3 or 4 columns in '{filename}', got {n_cols}. "
            f"Supported layouts: [lon lat h], [id lon lat h]."
        )

    return lats, lons, heights, ids


def _parse_csv_with_header(filename: str
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      Optional[np.ndarray]]:
    """
    Parse a CSV file with a header row, auto-detecting lon/lat/h columns.

    Recognized column names (case-insensitive):
        lon: 'lon', 'long', 'longitude'
        lat: 'lat', 'latitude'
        h:   'h', 'height', 'elev', 'elevation', 'height_ell'
        id:  'id', 'no', 'point', 'point_id'
    """
    import csv

    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lower = [h.strip().lower() for h in header]

        name_map = {
            "lon": ("lon", "long", "longitude"),
            "lat": ("lat", "latitude"),
            "h": ("h", "height", "elev", "elevation", "height_ell"),
            "id": ("id", "no", "point", "point_id"),
        }

        col_idx = {}
        for field, aliases in name_map.items():
            for i, name in enumerate(header_lower):
                if name in aliases:
                    col_idx[field] = i
                    break

        if "lon" not in col_idx or "lat" not in col_idx or "h" not in col_idx:
            raise ValueError(
                f"CSV header {header} missing required columns. "
                f"Need at least lon/lat/h (aliases: "
                f"lon={name_map['lon']}, lat={name_map['lat']}, "
                f"h={name_map['h']})."
            )

        rows = []
        for row in reader:
            rows.append([float(v) for v in row])
    data = np.array(rows, dtype=np.float64)

    lons = data[:, col_idx["lon"]]
    lats = data[:, col_idx["lat"]]
    heights = data[:, col_idx["h"]]
    ids = data[:, col_idx["id"]] if "id" in col_idx else None
    return lats, lons, heights, ids


def read_compoints_ascii(filename: str,
                         lat_first: bool = False,
                         has_header: Optional[bool] = None
                         ) -> ComPointData:
    """
    Read computation points from an ASCII or CSV file.

    Args:
        filename: Path to the input file.
        lat_first: Column order flag for headerless files.
            False (default) -> [lon lat h] or [id lon lat h]
            True            -> [lat lon h] or [id lat lon h]
        has_header: Force header detection.
            None -> auto-detect (CSV with .csv extension or first row
                    containing non-numeric tokens).
            True/False -> force the specified behavior.

    Returns:
        ComPointData with ids/lats/lons/heights arrays.

    Notes:
        Negative heights are NOT corrected here (read_com_points does that
        when loading the binary file). Use write_com_points to persist.
    """
    # Auto-detect CSV header
    if has_header is None:
        lower_name = filename.lower()
        if lower_name.endswith(".csv"):
            has_header = True
        else:
            with open(filename, "r") as f:
                first_line = f.readline().strip()
            # If first token is non-numeric, treat as header
            try:
                float(first_line.split()[0])
                has_header = False
            except (ValueError, IndexError):
                has_header = True

    if has_header:
        lats, lons, heights, ids = _parse_csv_with_header(filename)
    else:
        lats, lons, heights, ids = _parse_ascii_columns(filename, lat_first)

    if ids is None:
        ids = np.arange(1, len(lats) + 1, dtype=np.float64)

    return ComPointData(ids=ids, lats=lats, lons=lons, heights=heights)


def compoints_to_bin(compoints: ComPointData, filename: str) -> None:
    """
    Write a ComPointData object to the TGF binary format.

    Thin wrapper around io.com_points.write_com_points for API symmetry
    with dem_to_bin.
    """
    write_com_points(filename,
                     lats=compoints.lats,
                     lons=compoints.lons,
                     heights=compoints.heights,
                     ids=compoints.ids)
