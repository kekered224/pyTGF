"""
GeoTIFF mosaic and conversion utilities

Read multiple GeoTIFF DEM files, mosaic them into a single grid,
crop to a specified extent with buffer, and write to pyTGF binary format.

Uses rasterio for reading and merging GeoTIFFs.

Key functions:
    scan_tif_extent   — quick metadata-only scan of TIF bounds
    mosaic_tifs       — read + merge + return DEMData
    mosaic_to_bin     — full pipeline: merge → crop → write .bin
"""

from typing import List, Dict, Optional, Tuple
import os
import numpy as np

from ..io.dem_io import DEMData, write_dem


def _check_rasterio():
    """Import rasterio or raise a helpful error."""
    try:
        import rasterio
        return rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required for GeoTIFF processing. "
            "Install with: pip install rasterio"
        ) from e


def scan_tif_extent(tif_paths: List[str]) -> Dict:
    """
    Scan GeoTIFF files and return their combined geographic extent.

    Only reads file metadata (geotransform), not pixel data — fast.

    Args:
        tif_paths: List of absolute or relative paths to .tif files.

    Returns:
        {
            "bounds": { minlat, maxlat, minlon, maxlon },
            "resolution": { reslat, reslon },
            "tiles": [ { file, bounds, shape }, ... ],
            "total_tiles": int,
        }
    """
    rasterio = _check_rasterio()

    if not tif_paths:
        raise ValueError("No TIF files provided")

    tiles = []
    global_minlat = float("inf")
    global_maxlat = float("-inf")
    global_minlon = float("inf")
    global_maxlon = float("-inf")
    best_reslat = float("inf")
    best_reslon = float("inf")

    for path in tif_paths:
        with rasterio.open(path) as src:
            b = src.bounds  # (left, bottom, right, top) = (minlon, minlat, maxlon, maxlat)
            res_lon = abs(src.transform.a)  # pixel width
            res_lat = abs(src.transform.e)  # pixel height (absolute)

            tile_info = {
                "file": os.path.basename(path),
                "bounds": {
                    "minlat": b.bottom,
                    "maxlat": b.top,
                    "minlon": b.left,
                    "maxlon": b.right,
                },
                "shape": [src.height, src.width],
                "resolution": {"reslat": res_lat, "reslon": res_lon},
            }
            tiles.append(tile_info)

            global_minlat = min(global_minlat, b.bottom)
            global_maxlat = max(global_maxlat, b.top)
            global_minlon = min(global_minlon, b.left)
            global_maxlon = max(global_maxlon, b.right)
            best_reslat = min(best_reslat, res_lat)
            best_reslon = min(best_reslon, res_lon)

    return {
        "bounds": {
            "minlat": global_minlat,
            "maxlat": global_maxlat,
            "minlon": global_minlon,
            "maxlon": global_maxlon,
        },
        "resolution": {
            "reslat": best_reslat,
            "reslon": best_reslon,
        },
        "tiles": tiles,
        "total_tiles": len(tif_paths),
    }


def mosaic_tifs(tif_paths: List[str],
                crop_bounds: Optional[Dict] = None,
                buffer: float = 0.0,
                target_reslat: Optional[float] = None,
                target_reslon: Optional[float] = None) -> DEMData:
    """
    Read and mosaic multiple GeoTIFF files into a single DEMData.

    Uses rasterio.merge to handle overlapping regions (last-writer-wins)
    and nodata gaps (filled with 0.0 = sea level).

    Args:
        tif_paths:     List of paths to .tif files.
        crop_bounds:   Optional dict { minlat, maxlat, minlon, maxlon }
                       to crop the mosaic to. If None, use full extent.
        buffer:        Extra degrees to add around crop_bounds on all sides.
        target_reslat: Target latitude resolution (None = finest available).
        target_reslon: Target longitude resolution (None = finest available).

    Returns:
        DEMData with row 0 = minlat (south), matching pyTGF convention.
    """
    rasterio = _check_rasterio()
    from rasterio.merge import merge

    if not tif_paths:
        raise ValueError("No TIF files provided")

    # Open all source datasets
    src_datasets = [rasterio.open(p) for p in tif_paths]

    try:
        # Determine the finest resolution across all tiles
        if target_reslat is None or target_reslon is None:
            reslats = [abs(ds.transform.e) for ds in src_datasets]
            reslons = [abs(ds.transform.a) for ds in src_datasets]
            if target_reslat is None:
                target_reslat = min(reslats)
            if target_reslon is None:
                target_reslon = min(reslons)

        # Build merge bounds if crop_bounds specified
        merge_bounds = None
        if crop_bounds is not None:
            minlat = crop_bounds.get("minlat", -90) - buffer
            maxlat = crop_bounds.get("maxlat", 90) + buffer
            minlon = crop_bounds.get("minlon", -180) - buffer
            maxlon = crop_bounds.get("maxlon", 180) + buffer
            merge_bounds = (minlon, minlat, maxlon, maxlat)

        # Build target transform for the desired resolution
        from rasterio.transform import from_bounds
        if merge_bounds:
            west, south, east, north = merge_bounds
        else:
            # Use union of all tile bounds
            west = min(ds.bounds.left for ds in src_datasets)
            south = min(ds.bounds.bottom for ds in src_datasets)
            east = max(ds.bounds.right for ds in src_datasets)
            north = max(ds.bounds.top for ds in src_datasets)

        # Calculate output dimensions
        ncols = max(1, int(round((east - west) / target_reslon)))
        nrows = max(1, int(round((north - south) / target_reslat)))

        # Limit output size to prevent memory issues (max ~100M pixels)
        max_pixels = 100_000_000
        if nrows * ncols > max_pixels:
            scale = (max_pixels / (nrows * ncols)) ** 0.5
            nrows = max(1, int(nrows * scale))
            ncols = max(1, int(ncols * scale))
            target_reslat = (north - south) / nrows
            target_reslon = (east - west) / ncols

        out_transform = from_bounds(west, south, east, north, ncols, nrows)

        # Perform the merge
        merged, merged_transform = merge(
            src_datasets,
            bounds=merge_bounds,
            res=(target_reslon, target_reslat),
            nodata=0.0,
        )

        # merged shape: (bands, height, width), take first band
        z = merged[0].astype(np.float64)

        # Replace nodata / NaN with 0.0 (sea level)
        z[np.isnan(z)] = 0.0

        # Derive bounds from the actual merged transform
        actual_west = merged_transform.c
        actual_north = merged_transform.f
        actual_reslon = abs(merged_transform.a)
        actual_reslat = abs(merged_transform.e)
        # The transform gives outer pixel edges; convert to node centers.
        actual_east = actual_west + actual_reslon * (z.shape[1] - 0.5)
        actual_south = actual_north - actual_reslat * (z.shape[0] - 0.5)
        actual_west = actual_west + actual_reslon / 2.0
        actual_north = actual_north - actual_reslat / 2.0

        # rasterio row 0 = top (north), DEMData expects row 0 = bottom (south)
        z = np.flipud(z)

        return DEMData(
            minlat=actual_south,
            maxlat=actual_north,
            minlon=actual_west,
            maxlon=actual_east,
            reslat=actual_reslat,
            reslon=actual_reslon,
            z=z,
        )

    finally:
        for ds in src_datasets:
            ds.close()


def mosaic_to_bin(tif_paths: List[str],
                  output_path: str,
                  crop_bounds: Optional[Dict] = None,
                  buffer: float = 0.0,
                  target_reslat: Optional[float] = None,
                  target_reslon: Optional[float] = None,
                  progress_callback=None) -> Dict:
    """
    Full pipeline: mosaic TIFs → crop → write binary .bin file.

    Args:
        tif_paths:         List of paths to .tif files.
        output_path:       Where to write the .bin file.
        crop_bounds:       Optional crop bounds dict.
        buffer:            Buffer degrees around crop bounds.
        target_reslat:     Target lat resolution (None = finest).
        target_reslon:     Target lon resolution (None = finest).
        progress_callback: Optional callable(current, total, phase).

    Returns:
        { "output_path": str, "shape": [nrows, ncols],
          "bounds": { minlat, maxlat, minlon, maxlon },
          "resolution": { reslat, reslon } }
    """
    if progress_callback:
        progress_callback(0, 3, "reading TIF files")

    dem = mosaic_tifs(
        tif_paths,
        crop_bounds=crop_bounds,
        buffer=buffer,
        target_reslat=target_reslat,
        target_reslon=target_reslon,
    )

    if progress_callback:
        progress_callback(2, 3, "writing binary file")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_dem(output_path, dem)

    if progress_callback:
        progress_callback(3, 3, "complete")

    return {
        "output_path": output_path,
        "shape": [dem.n_lat, dem.n_lon],
        "bounds": {
            "minlat": dem.minlat,
            "maxlat": dem.maxlat,
            "minlon": dem.minlon,
            "maxlon": dem.maxlon,
        },
        "resolution": {
            "reslat": dem.reslat,
            "reslon": dem.reslon,
        },
    }
