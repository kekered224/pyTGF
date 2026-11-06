"""
Preprocessing submodule for pyTGF

Tools to convert external DEM/computation-point data into the binary
formats expected by the TGF calculation engine:

    - sis_dem:          read MERIT DEM/REF from the SIS tile database
                        (port of MATLAB dem2017YM_MERIT_R1.m)
    - dem_import:       read GeoTIFF / ESRI ASCII / npz grids into DEMData
    - tif_mosaic:       mosaic multiple GeoTIFFs and convert to binary DEM
    - downsample:       box-mean block reduction (port of MATLAB boxmean.m)
    - compoints_import: read ASCII/CSV computation-point lists into ComPointData
"""

from .sis_dem import read_merit_dem
from .dem_import import (
    read_tif, read_ascii_grid, read_npz_grid, dem_to_bin, crop_dem,
)
from .tif_mosaic import scan_tif_extent, mosaic_tifs, mosaic_to_bin
from .downsample import boxmean, boxmean_dem
from .compoints_import import read_compoints_ascii, compoints_to_bin

__all__ = [
    "read_merit_dem",
    "read_tif", "read_ascii_grid", "read_npz_grid", "dem_to_bin", "crop_dem",
    "scan_tif_extent", "mosaic_tifs", "mosaic_to_bin",
    "boxmean", "boxmean_dem",
    "read_compoints_ascii", "compoints_to_bin",
]
