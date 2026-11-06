"""
Tesseroid-to-prism conversion

Mirrors TC_GUI.m L4153-L4181 (tess2prism).

Transforms a spherical tesseroid into a mass-equivalent rectangular prism
in local Cartesian coordinates, preserving total mass.
"""

from typing import Tuple
import numpy as np

from ..config import PhysicalConstants


def tess2prism(x1: np.ndarray, x2: np.ndarray,
               y1: np.ndarray, y2: np.ndarray,
               d1: np.ndarray, d2: np.ndarray,
               e: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Transform a tesseroid into a mass-equivalent rectangular prism.

    Corresponds to TC_GUI.m L4153-L4181 (tess2prism).

    The prism is centered at the tesseroid center in local Cartesian
    coordinates. Density is adjusted to preserve total mass when the
    prism volume differs from the tesseroid volume.

    MATLAB:
        r1=d1; r2=d2; r0 = 0.5*(r1 + r2);
        dx = r0*d2r*(y2-y1);
        dy = r0*cos(d2r*0.5*(y1+y2))*d2r*(x2-x1);
        prism_x1=-0.5*dx; prism_x2=-prism_x1;
        prism_y1=-0.5*dy; prism_y2=-prism_y1;
        prism_z1=r1; prism_z1(1:end)=(r2-r1)/2;   % overwritten
        prism_z2=(r1-r2)/2;
        tess_volume = d2r*(x2-x1)*(r1^3-r2^3)*(sin(d2r*y2)-sin(d2r*y1))/3;
        prism_volume = (prism_x2-prism_x1)*(prism_y2-prism_y1)*(prism_z2-prism_z1);
        prism_e = e * tess_volume / prism_volume;
        prism_e(prism_volume==0) = e(prism_volume==0);
        prism_lon = 0.5*(x1+x2);
        prism_lat = 0.5*(y1+y2);
        prism_r = r0;

    Note: MATLAB's `prism_z1=r1; prism_z1(1:end)=(r2-r1)/2;` first assigns
    r1 then immediately overwrites with (r2-r1)/2. The final value is
    (r2-r1)/2. We replicate this exactly.

    Args:
        x1, x2: Tesseroid longitude bounds [deg].
        y1, y2: Tesseroid latitude bounds [deg].
        d1, d2: Tesseroid radial bounds (upper, lower) [m].
        e: Density [g/cm^3].

    Returns:
        (prism_x1, prism_x2, prism_y1, prism_y2,
         prism_z1, prism_z2, prism_lon, prism_lat, prism_r, prism_e):
            Prism bounds in local Cartesian [m], center coordinates
            [deg, deg, m], and adjusted density [g/cm^3].
    """
    d2r = PhysicalConstants.d2r()

    r1 = d1
    r2 = d2
    r0 = 0.5 * (r1 + r2)

    dx = r0 * d2r * (y2 - y1)
    dy = r0 * np.cos(d2r * 0.5 * (y1 + y2)) * d2r * (x2 - x1)

    prism_x1 = -0.5 * dx
    prism_x2 = -prism_x1
    prism_y1 = -0.5 * dy
    prism_y2 = -prism_y1

    # MATLAB: prism_z1=r1; prism_z1(1:end)=(r2-r1)/2;  -> final = (r2-r1)/2
    prism_z1 = (r2 - r1) / 2.0
    prism_z2 = (r1 - r2) / 2.0

    # Volume-equivalent density adjustment
    tess_volume = d2r * (x2 - x1) * (r1 ** 3 - r2 ** 3) * \
                  (np.sin(d2r * y2) - np.sin(d2r * y1)) / 3.0
    prism_volume = (prism_x2 - prism_x1) * (prism_y2 - prism_y1) * \
                   (prism_z2 - prism_z1)

    # Avoid divide-by-zero; MATLAB: prism_e(n)=e(n) where prism_volume==0
    with np.errstate(divide='ignore', invalid='ignore'):
        prism_e = e * tess_volume / prism_volume
    # Where prism_volume == 0, fall back to original density
    zero_mask = (prism_volume == 0)
    if np.any(zero_mask):
        prism_e = np.where(zero_mask, e, prism_e)
    # Replace any remaining NaN/Inf
    prism_e = np.where(np.isfinite(prism_e), prism_e, e)

    prism_lon = 0.5 * (x1 + x2)
    prism_lat = 0.5 * (y1 + y2)
    prism_r = r0

    return (prism_x1, prism_x2, prism_y1, prism_y2,
            prism_z1, prism_z2, prism_lon, prism_lat, prism_r, prism_e)
