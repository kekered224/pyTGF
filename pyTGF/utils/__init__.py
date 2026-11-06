"""
Utility functions submodule

Coordinate transforms, grid building, normal gravity, prism helpers.
"""

from .coordinate import global2local, g_prism2point, ggt_prism2point
from .grid import build_grid, matrix_preallocating
from .normal_gravity import normal_gravity_generic
from .prism_helpers import gz3term, g6term

__all__ = [
    "global2local", "g_prism2point", "ggt_prism2point",
    "build_grid", "matrix_preallocating",
    "normal_gravity_generic",
    "gz3term", "g6term",
]
