"""
pyTGF - Python implementation of TGF (Terrain Gravity Field) software

A Python reimplementation of the MATLAB TGF software for calculating
full-scale topographic and RTM gravitational fields.

Based on: TC_GUI.m from TGF software by M. Yang, C. Hirt, R. Pail (2020)
"TGF: A New MATLAB-based Software for Terrain-related Gravity Field
Calculations", Remote Sensing, 12(7), 1063. https://doi.org/10.3390/rs12071063
"""

__version__ = "1.0.0"
__author__ = "pyTGF contributors"

from .config import TGFConfig, PhysicalConstants
from .core.tc_gui import TC_GUI

__all__ = ["TGFConfig", "PhysicalConstants", "TC_GUI"]
