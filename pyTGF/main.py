"""
pyTGF command-line entry point

Usage:
    python -m pyTGF.main --com-points COM.bin --detailed-dem DDEM.bin ...
    python -m pyTGF.main --config config.json

Or as a module:
    from pyTGF import TC_GUI
    result = TC_GUI('RUN', ComPoints='...', DetailedDEM='...', ...)
"""

import argparse
import json
import sys
from typing import Optional, List

from . import TC_GUI
from .config import TGFConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments mirroring TC_GUI.m parameter list."""
    p = argparse.ArgumentParser(
        prog="pyTGF",
        description="Python Terrain Gravity Field calculation software",
    )
    p.add_argument("--com-points", required=True, help="Computation points binary file")
    p.add_argument("--detailed-dem", required=True, help="Detailed DEM binary file")
    p.add_argument("--detailed-ref", default="", help="Detailed reference surface (RTM)")
    p.add_argument("--mass-density", default="", help="Detailed density map")
    p.add_argument("--tess-dem", default="", help="Tesseroid zone DEM")
    p.add_argument("--tess-ref", default="", help="Tesseroid zone reference")
    p.add_argument("--tess-density", default="", help="Tesseroid zone density")
    p.add_argument("--coarse-dem", default="", help="Coarse DEM (far zone)")
    p.add_argument("--coarse-ref", default="", help="Coarse reference")
    p.add_argument("--coarse-density", default="", help="Coarse density")
    p.add_argument("--global-dem", default="", help="Global DEM (topographic only)")
    p.add_argument("--global-ref", default="", help="Global reference (topographic only)")
    p.add_argument("--outname", default="tgf_result.dat", help="Output file path")
    p.add_argument("--ikind", type=int, default=2, choices=[1, 2],
                   help="1=topographic, 2=RTM")
    p.add_argument("--itype", type=int, default=104,
                   choices=[0, 1, 2, 4, 10, 103, 104],
                   help="Output functional type")
    p.add_argument("--idensity", type=int, default=0, choices=[0, 1],
                   help="0=constant density, 1=density map")
    p.add_argument("--flag-earth", type=int, default=0, choices=[0, 1],
                   help="0=spherical, 1=ellipsoidal")
    p.add_argument("--rzones", type=float, nargs=4, default=[0.0, 0.03, 0.3, 2.0],
                   metavar=("R1", "R2", "R3", "R4"),
                   help="Integration radii [deg] for zones 1-4")
    p.add_argument("--density", type=float, default=2.67,
                   help="Constant density [g/cm^3]")
    p.add_argument("--config", default=None,
                   help="JSON config file (overrides CLI args)")
    return p.parse_args(argv)


def load_config_from_json(path: str) -> dict:
    """Load configuration parameters from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for command-line execution."""
    args = parse_args(argv)

    if args.config:
        cfg_dict = load_config_from_json(args.config)
        # JSON config takes precedence; CLI args fill gaps
        kwargs = vars(args)
        kwargs.pop("config")
        kwargs.update({k: v for k, v in cfg_dict.items() if k in kwargs})
    else:
        kwargs = vars(args)
        kwargs.pop("config")

    result = TC_GUI(
        vstpar="RUN",
        ComPoints=kwargs["com_points"],
        DetailedDEM=kwargs["detailed_dem"],
        DetailedREF=kwargs["detailed_ref"],
        MassDensity=kwargs["mass_density"],
        TessDEM=kwargs["tess_dem"],
        TessREF=kwargs["tess_ref"],
        TessDensity=kwargs["tess_density"],
        CoarseDEM=kwargs["coarse_dem"],
        CoarseREF=kwargs["coarse_ref"],
        CoarseDensity=kwargs["coarse_density"],
        GlobalDEM=kwargs["global_dem"],
        GlobalREF=kwargs["global_ref"],
        outname=kwargs["outname"],
        ikind=kwargs["ikind"],
        itype=kwargs["itype"],
        idensity=kwargs["idensity"],
        flag_earth=kwargs["flag_earth"],
        rzones=kwargs["rzones"],
        e=kwargs["density"],
    )

    print(f"\nCalculation complete.")
    print(f"  Result file: {result['output_path']}")
    print(f"  Report file: {result['report_path']}")
    print(f"  Elapsed:     {result['elapsed_hours']:.4f} hours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
