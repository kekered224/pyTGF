"""
I/O submodule for pyTGF

Handles binary DEM files, computation points, and result/report writing.
Mirrors MATLAB read_DEM.m and the fprintf/save logic in TC_GUI.m.
"""

from .dem_io import read_dem, write_dem, DEMData
from .com_points import read_com_points, write_com_points, ComPointData
from .output import write_report_header, write_report_dem_info, write_result

__all__ = [
    "read_dem", "write_dem", "DEMData",
    "read_com_points", "write_com_points", "ComPointData",
    "write_report_header", "write_report_dem_info", "write_result",
]
