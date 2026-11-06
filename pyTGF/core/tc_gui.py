"""
TC_GUI entry point

Mirrors TC_GUI.m L118-L2035.

Provides a single function TC_GUI that accepts the same parameter list as
the MATLAB original, supporting both:
    - Batch mode (vstpar='RUN'): direct calculation with all parameters
    - GUI mode (vstpar omitted): launches interactive interface (future)

In batch mode, this is a thin wrapper around GravityEngine.run().
"""

from typing import Optional, List, Dict, Any
import numpy as np

from ..config import TGFConfig
from .gravity_engine import GravityEngine


def TC_GUI(vstpar: str = "RUN",
           ComPoints: str = "",
           DetailedDEM: str = "",
           DetailedREF: str = "",
           MassDensity: str = "",
           TessDEM: str = "",
           TessREF: str = "",
           TessDensity: str = "",
           CoarseDEM: str = "",
           CoarseREF: str = "",
           CoarseDensity: str = "",
           GlobalDEM: str = "",
           GlobalREF: str = "",
           outname: str = "tgf_result.dat",
           ikind: int = 2,
           itype: int = 104,
           idensity: int = 0,
           flag_earth: int = 0,
           rzones: Optional[List[float]] = None,
           e: float = 2.67) -> Dict[str, Any]:
    """
    Main entry point matching MATLAB TC_GUI.m signature (L118).

    Args:
        vstpar: Run mode. 'RUN' for batch calculation.
        ComPoints: Path to computation points binary file.
        DetailedDEM: Path to detailed DEM binary file.
        DetailedREF: Path to detailed reference surface (RTM only).
        MassDensity: Path to detailed density map (idensity=1 only).
        TessDEM, TessREF, TessDensity: Tesseroid zone files.
        CoarseDEM, CoarseREF, CoarseDensity: Coarse zone files.
        GlobalDEM, GlobalREF: Global DEM (ikind=1 only).
        outname: Output result file path.
        ikind: 1=topographic, 2=RTM.
        itype: 0/1/2/4/10/103/104 (see TGFConfig.output_columns).
        idensity: 0=constant density, 1=density map.
        flag_earth: 0=spherical, 1=ellipsoidal.
        rzones: [r1, r2, r3, r4] integration radii [deg].
        e: Constant density [g/cm^3].

    Returns:
        dict with keys:
            'result': np.ndarray of shape (n_points, n_columns)
            'report_path': str
            'output_path': str
            'config': TGFConfig
            'elapsed_hours': float
    """
    if rzones is None:
        rzones = [0.0, 0.03, 0.3, 2.0]

    # Support being called with a single TGFConfig argument (convenience
    # for callers that build the config object themselves).
    if isinstance(vstpar, TGFConfig):
        config = vstpar
        vstpar = config.vstpar
    else:
        # Build configuration
        config = TGFConfig(
            vstpar=vstpar,
            com_points=ComPoints,
            detailed_dem=DetailedDEM,
            detailed_ref=DetailedREF,
            mass_density=MassDensity,
            tess_dem=TessDEM,
            tess_ref=TessREF,
            tess_density=TessDensity,
            coarse_dem=CoarseDEM,
            coarse_ref=CoarseREF,
            coarse_density=CoarseDensity,
            global_dem=GlobalDEM,
            global_ref=GlobalREF,
            outname=outname,
            ikind=ikind,
            itype=itype,
            idensity=idensity,
            flag_earth=flag_earth,
            rzones=list(rzones),
            e=e,
        )

    if vstpar != "RUN":
        raise NotImplementedError(f"vstpar='{vstpar}' not yet supported. "
                                  f"Only 'RUN' batch mode is implemented.")

    # Delegate to engine
    engine = GravityEngine(config)
    return engine.run_from_files()
