"""
Core engine submodule

Contains the main TC_GUI entry point and the RUN-branch workflow
that orchestrates zone-based integration of all mass elements.
"""

from .tc_gui import TC_GUI
from .gravity_engine import GravityEngine
from .zone_strategy import ZoneStrategy, ZoneExtents

__all__ = ["TC_GUI", "GravityEngine", "ZoneStrategy", "ZoneExtents"]
