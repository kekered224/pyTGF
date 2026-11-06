"""
pyTGF - Python implementation of TGF (Terrain Gravity Field) software

A Python reimplementation of the MATLAB TGF software for calculating
full-scale topographic and RTM gravitational fields.

Based on: TC_GUI.m from TGF software by M. Yang, C. Hirt, R. Pail (2019)
"""

__version__ = "1.0.0"
__author__ = "pyTGF contributors"

from .config import TGFConfig, PhysicalConstants
from .core.tc_gui import TC_GUI

__all__ = ["TGFConfig", "PhysicalConstants", "TC_GUI"]
