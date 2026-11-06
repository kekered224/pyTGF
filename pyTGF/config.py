"""
Configuration and physical constants for pyTGF

Mirrors the parameter definitions in TC_GUI.m header (L1-L117)
and the parameter reading logic in the RUN branch (L677-L775).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


class PhysicalConstants:
    """Physical constants used throughout TGF calculations"""

    # Gravitational constant [m^3 g^-1 s^-2] (MATLAB cgs-style value)
    G = 6.67384e-3

    # Pi (matching MATLAB value used in TC_GUI.m)
    PI = 3.141592654

    # GRS80 / WGS84 ellipsoid parameters
    A = 6378137.0                       # Semi-major axis [m]
    E2 = 0.00673949677548               # Second eccentricity squared
    B = 6356752.3141                    # Semi-minor axis [m]
    GAMMA_A = 9.7803267715              # Normal gravity at equator [m/s^2]
    GAMMA_B = 9.8321863685              # Normal gravity at pole  [m/s^2]
    M = 0.00344978600308                # Geodetic parameter m

    # Default crustal density [g/cm^3]
    DEFAULT_DENSITY = 2.67

    # Earth mean radius [m]
    EARTH_RADIUS = 6371000.0

    @classmethod
    def d2r(cls) -> float:
        """Degree-to-radian conversion factor"""
        return cls.PI / 180.0


@dataclass
class TGFConfig:
    """
    All parameters required for a TGF calculation run.

    Corresponds to TC_GUI.m function signature (L118):
        TC_GUI(vstpar, ComPoints, DetailedDEM, DetailedREF, MassDensity,
               TessDEM, TessREF, TessDensity, CoarseDEM, CoarseREF,
               CoarseDensity, GlobalDEM, GlobalREF, outname,
               ikind, itype, idensity, flag_earth, rzones, e)
    """

    # --- Run mode ---
    vstpar: str = "RUN"        # 'RUN' for batch calculation

    # --- Input file paths ---
    com_points: str = ""                   # Computation points (binary)
    detailed_dem: str = ""                 # Detailed DEM (binary)
    detailed_ref: str = ""                 # Detailed reference surface
    mass_density: str = ""                 # Detailed density map
    tess_dem: str = ""                     # Tesseroid-zone DEM
    tess_ref: str = ""                     # Tesseroid-zone reference
    tess_density: str = ""                 # Tesseroid-zone density
    coarse_dem: str = ""                   # Coarse DEM (far zone)
    coarse_ref: str = ""                   # Coarse reference
    coarse_density: str = ""               # Coarse density
    global_dem: str = ""                   # Global DEM (only ikind=1)
    global_ref: str = ""                   # Global reference (only ikind=1)

    # --- Output ---
    outname: str = "tgf_result.dat"        # Output result file

    # --- Calculation flags ---
    ikind: int = 2                         # 1=topographic, 2=RTM
    itype: int = 104                       # 0/1/2/4/10/103/104 (see manual)
    idensity: int = 0                      # 0=constant, 1=density map
    flag_earth: int = 0                    # 0=spherical, 1=ellipsoidal

    # --- Integration radii [deg] ---
    rzones: List[float] = field(default_factory=lambda: [0.0, 0.03, 0.3, 2.0])
    # r1: polyhedron, r2: prism, r3: tesseroid, r4: point-mass

    # --- Density ---
    e: float = 2.67                        # Constant density [g/cm^3]

    # --- Runtime options ---
    verbose: bool = True
    report_path: Optional[str] = None      # Auto-derived from outname if None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters (mirrors L752-L770)"""
        if self.ikind not in (1, 2):
            raise ValueError(f"ikind must be 1 or 2, got {self.ikind}")
        if self.itype not in (0, 1, 2, 4, 10, 103, 104):
            raise ValueError(f"Invalid itype: {self.itype}")
        if self.idensity not in (0, 1):
            raise ValueError(f"idensity must be 0 or 1, got {self.idensity}")
        if self.flag_earth not in (0, 1):
            raise ValueError(f"flag_earth must be 0 or 1, got {self.flag_earth}")
        if len(self.rzones) != 4:
            raise ValueError(f"rzones must have 4 elements, got {len(self.rzones)}")

        r_values = [float(v) for v in self.rzones]
        if any(not np.isfinite(v) or v < 0 for v in r_values):
            raise ValueError(f"rzones must be finite and non-negative, got {self.rzones}")

        # Enforce r1 <= r2 <= r3 <= r4 (mirrors L818-L830)
        r1, r2, r3, r4 = r_values
        if r1 > r2:
            r2 = r1
        if r2 > r3:
            r3 = r2
        if r3 > r4:
            r4 = r3
        self.rzones[:] = [r1, r2, r3, r4]

        density = float(self.e)
        if not np.isfinite(density) or not (0.0 < density <= 10.0):
            raise ValueError(
                f"density must be a positive value in (0, 10] g/cm^3, got {self.e}")
        self.e = density

        if self.idensity == 1 and not (self.mass_density and self.coarse_density and self.tess_density):
            raise ValueError("Density maps required when idensity=1")

        # Derive report path
        if self.report_path is None:
            base = self.outname
            if "." in base:
                base = base[:base.rfind(".")]
            self.report_path = base + "_report.txt"

    # ---- Convenience descriptors ----
    @property
    def is_topographic(self) -> bool:
        return self.ikind == 1

    @property
    def is_rtm(self) -> bool:
        return self.ikind == 2

    @property
    def is_ellipsoidal(self) -> bool:
        return self.flag_earth == 1

    @property
    def output_columns(self) -> List[str]:
        """Column names produced for current itype (mirrors L1228/L1330/L1620)"""
        base = ["lon", "lat", "H"]
        if self.itype == 0:
            return base + ["N"]
        if self.itype == 1:
            return base + ["xi", "eta", "dg"]
        if self.itype == 2:
            return base + ["Vxx", "Vxy", "Vxz", "Vyy", "Vyz", "Vzz"]
        if self.itype in (4, 103, 104):
            return base + ["N", "xi", "eta", "dg"]
        if self.itype == 10:
            return base + ["N", "xi", "eta", "dg",
                           "Vxx", "Vxy", "Vxz", "Vyy", "Vyz", "Vzz"]
        raise ValueError(f"Unknown itype {self.itype}")
