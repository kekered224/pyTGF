"""
Mass elements submodule

Four mass element types used in different integration zones:
    - Polyhedron : Zone 1 (nearest, most precise)  - Tsoulis (2012)
    - Prism      : Zone 2 (near, precise)          - Nagy et al. (2000/2002)
    - Tesseroid  : Zone 3 (medium, efficient)      - Grombein et al. (2013)
    - Point mass : Zone 4/5 (far, most efficient)

Each element has 5 variants selected by itype:
    _g    : potential only               (itype=0)
    _gx   : 1st derivatives              (itype=1)
    _gxx  : 2nd derivatives              (itype=2)
    _g_gx : potential + 1st derivatives  (itype=4/103/104)
    _full : all 10 components            (itype=10)
"""

from .polyhedron import (
    polyhedron_full, polyhedron_g, polyhedron_g_gx,
    polyhedron_gx, polyhedron_gxx,
)
from .prism import (
    prism, prism_g, prism_g_gx, prism_gx, prism_gxx,
)
from .prism_numba import prism_numba
from .tesseroid import (
    tesseroid, tesseroid_g, tesseroid_g_gx, tesseroid_gx, tesseroid_gxx,
)
from .tesseroid_numba import (
    tess_kernel_10, tess_kernel_g, tess_kernel_gx,
    tess_kernel_g_gx, tess_kernel_gxx,
)
from .point_mass import (
    point_g, point_gx, point_g_gx, point_gxx, point_mass,
)
from .point_mass_numba import (
    pm_kernel_10, pm_kernel_g, pm_kernel_gx, pm_kernel_g_gx, pm_kernel_gxx,
)
from .tess2prism import tess2prism

__all__ = [
    # Polyhedron
    "polyhedron_full", "polyhedron_g", "polyhedron_g_gx",
    "polyhedron_gx", "polyhedron_gxx",
    # Prism
    "prism", "prism_g", "prism_g_gx", "prism_gx", "prism_gxx",
    "prism_numba",
    # Tesseroid
    "tesseroid", "tesseroid_g", "tesseroid_g_gx", "tesseroid_gx", "tesseroid_gxx",
    "tess_kernel_10", "tess_kernel_g", "tess_kernel_gx",
    "tess_kernel_g_gx", "tess_kernel_gxx",
    # Point mass
    "point_g", "point_gx", "point_g_gx", "point_gxx", "point_mass",
    "pm_kernel_10", "pm_kernel_g", "pm_kernel_gx",
    "pm_kernel_g_gx", "pm_kernel_gxx",
    # Conversion
    "tess2prism",
]
