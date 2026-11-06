"""
Prism mass element (Zone 2, near, precise)

Mirrors TC_GUI.m:
    Prism        (L3277-L3567)  all 10 components
    Prism_g      (L3598-L3674)  potential only, itype=0
    Prism_g_gx   (L3677-L3845)  potential + 1st derivs, itype=4/103/104
    Prism_gx     (L3847-L3983)  1st derivatives, itype=1
    Prism_gxx    (L3985-L4151)  2nd derivatives, itype=2

Reference: D. Nagy, G. Papp, J. Benedek (2000/2002) J. Geodesy 74/76.

Each prism is constructed from a spherical tesseroid via tess2prism
(volume-equivalent rectangular prism in local Cartesian coordinates).

Sign convention notes (matching MATLAB):
    g  = -sum(c*res)*1e-5           (potential, m^2/s^2)
    gz =  c*value                   (z-derivative in prism frame)
    gy = -c*value                   (y-derivative in prism frame)
    gx = -c*value                   (x-derivative in prism frame)
    gxx = (c*res)*1e4
    gxy = -c*res*1e4
    gxz =  c*res*1e4
    gyz =  c*res*1e4
    Then rotated to station frame via g_prism2point / ggt_prism2point.
"""

from typing import Tuple, Dict
import numpy as np

from ..config import PhysicalConstants
from ..utils.coordinate import global2local, g_prism2point, ggt_prism2point
from ..utils.prism_helpers import gz3term, g6term
from .tess2prism import tess2prism


def _prism_geometry(x1: np.ndarray, x2: np.ndarray,
                    y1: np.ndarray, y2: np.ndarray,
                    d1: np.ndarray, d2: np.ndarray,
                    lon: float, lat: float, r0: float,
                    e: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build all geometric quantities shared by the 5 prism variants.

    Corresponds to the common preamble in TC_GUI.m Prism/Prism_g/Prism_gx/
    Prism_g_gx/Prism_gxx (tess2prism + global2local + singular-point
    detection + r1..r8).

    Returns a dict with keys:
        c, xpos1, xpos2, ypos1, ypos2, zpos1, zpos2,
        r1, r2, r3, r4, r5, r6, r7, r8,
        nx1, nx2, ny1, ny2, nz1, nz2,
        prism_lon, prism_lat, prism_r
    """
    # build prism from tesseroid
    (prism_x1, prism_x2, prism_y1, prism_y2,
     prism_z1, prism_z2, prism_lon, prism_lat, prism_r, prism_e) = \
        tess2prism(x1, x2, y1, y2, d1, d2, e)

    G = PhysicalConstants.G
    c = G * prism_e  # multiplication of gravitational constant and density

    # coordinates of calculation point in prism-based Cartesian frame
    x, y, z = global2local(lon, lat, r0, prism_lon, prism_lat, prism_r)

    # vertex coordinates relative to station
    xpos2 = (prism_x2 - x)
    xpos1 = (prism_x1 - x)
    ypos2 = (prism_y2 - y)
    ypos1 = (prism_y1 - y)
    # MATLAB: zpos2=(prism_z1-z); zpos1=(prism_z2-z);  (note the swap!)
    zpos2 = (prism_z1 - z)
    zpos1 = (prism_z2 - z)

    # detecting irregular points (abs <= 0.05)
    nx1 = np.flatnonzero(np.abs(xpos1) <= 0.05)
    nx2 = np.flatnonzero(np.abs(xpos2) <= 0.05)
    ny1 = np.flatnonzero(np.abs(ypos1) <= 0.05)
    ny2 = np.flatnonzero(np.abs(ypos2) <= 0.05)
    nz1 = np.flatnonzero(np.abs(zpos1) <= 0.05)
    nz2 = np.flatnonzero(np.abs(zpos2) <= 0.05)

    # distances from station to 8 vertices
    r1 = np.sqrt(xpos1 ** 2 + ypos1 ** 2 + zpos1 ** 2)
    r2 = np.sqrt(xpos2 ** 2 + ypos1 ** 2 + zpos1 ** 2)
    r3 = np.sqrt(xpos1 ** 2 + ypos2 ** 2 + zpos1 ** 2)
    r4 = np.sqrt(xpos2 ** 2 + ypos2 ** 2 + zpos1 ** 2)
    r5 = np.sqrt(xpos1 ** 2 + ypos1 ** 2 + zpos2 ** 2)
    r6 = np.sqrt(xpos2 ** 2 + ypos1 ** 2 + zpos2 ** 2)
    r7 = np.sqrt(xpos1 ** 2 + ypos2 ** 2 + zpos2 ** 2)
    r8 = np.sqrt(xpos2 ** 2 + ypos2 ** 2 + zpos2 ** 2)

    return dict(
        c=c,
        xpos1=xpos1, xpos2=xpos2,
        ypos1=ypos1, ypos2=ypos2,
        zpos1=zpos1, zpos2=zpos2,
        r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7, r8=r8,
        nx1=nx1, nx2=nx2, ny1=ny1, ny2=ny2, nz1=nz1, nz2=nz2,
        prism_lon=prism_lon, prism_lat=prism_lat, prism_r=prism_r,
    )


def _compute_g(geo: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute potential g (per-prism, before summation).

    Corresponds to TC_GUI.m g section in Prism/Prism_g/Prism_g_gx.
    Returns: c*res (caller applies -sum(...)*1e-5).
    """
    c = geo['c']
    xpos1, xpos2 = geo['xpos1'], geo['xpos2']
    ypos1, ypos2 = geo['ypos1'], geo['ypos2']
    zpos1, zpos2 = geo['zpos1'], geo['zpos2']
    nx1, nx2, ny1, ny2, nz1, nz2 = \
        geo['nx1'], geo['nx2'], geo['ny1'], geo['ny2'], geo['nz1'], geo['nz2']
    r1, r2, r3, r4, r5, r6, r7, r8 = \
        geo['r1'], geo['r2'], geo['r3'], geo['r4'], geo['r5'], geo['r6'], geo['r7'], geo['r8']

    res = np.zeros_like(c, dtype=np.float64)

    # 8 vertices with alternating signs: + - - + - + + -
    t1, t2, t3, t4, t5, t6 = g6term(r1, xpos1, ypos1, zpos1, nx1, ny1, nz1)
    res = res + (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r2, xpos2, ypos1, zpos1, nx2, ny1, nz1)
    res = res - (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r3, xpos1, ypos2, zpos1, nx1, ny2, nz1)
    res = res - (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r4, xpos2, ypos2, zpos1, nx2, ny2, nz1)
    res = res + (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r5, xpos1, ypos1, zpos2, nx1, ny1, nz2)
    res = res - (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r6, xpos2, ypos1, zpos2, nx2, ny1, nz2)
    res = res + (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r7, xpos1, ypos2, zpos2, nx1, ny2, nz2)
    res = res + (t1 + t2 + t3 + t4 + t5 + t6)
    t1, t2, t3, t4, t5, t6 = g6term(r8, xpos2, ypos2, zpos2, nx2, ny2, nz2)
    res = res - (t1 + t2 + t3 + t4 + t5 + t6)

    return c * res


def _compute_gz(geo: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute gz (per-prism, before rotation/summation).

    Corresponds to TC_GUI.m gz section in Prism/Prism_g_gx/Prism_gx.
    Returns: c*value (caller uses directly).
    """
    c = geo['c']
    xpos1, xpos2 = geo['xpos1'], geo['xpos2']
    ypos1, ypos2 = geo['ypos1'], geo['ypos2']
    zpos1, zpos2 = geo['zpos1'], geo['zpos2']
    nx1, nx2, ny1, ny2, nz1, nz2 = \
        geo['nx1'], geo['nx2'], geo['ny1'], geo['ny2'], geo['nz1'], geo['nz2']
    r1, r2, r3, r4, r5, r6, r7, r8 = \
        geo['r1'], geo['r2'], geo['r3'], geo['r4'], geo['r5'], geo['r6'], geo['r7'], geo['r8']

    with np.errstate(divide='ignore', invalid='ignore'):
        s1 = xpos1 * np.log((ypos1 + r1) * (ypos2 + r7) / ((ypos2 + r3) * (ypos1 + r5)))
        s2 = xpos2 * np.log((ypos1 + r2) * (ypos2 + r8) / ((ypos2 + r4) * (ypos1 + r6)))
    s1 = np.where(np.isfinite(s1), s1, 0.0)
    s2 = np.where(np.isfinite(s2), s2, 0.0)
    s1.flat[nx1] = 0.0
    s2.flat[nx2] = 0.0
    s = s1 - s2

    with np.errstate(divide='ignore', invalid='ignore'):
        s1b = ypos1 * np.log((xpos1 + r1) * (xpos2 + r6) / ((xpos2 + r2) * (xpos1 + r5)))
        s2b = ypos2 * np.log((xpos1 + r3) * (xpos2 + r8) / ((xpos2 + r4) * (xpos1 + r7)))
    s1b = np.where(np.isfinite(s1b), s1b, 0.0)
    s2b = np.where(np.isfinite(s2b), s2b, 0.0)
    s1b.flat[ny1] = 0.0
    s2b.flat[ny2] = 0.0

    value = s + s1b - s2b

    # gz3term contributions: + - - + - + + -
    value = value + gz3term(r1, xpos1, ypos1, zpos1, nz1)
    value = value - gz3term(r2, xpos2, ypos1, zpos1, nz1)
    value = value - gz3term(r3, xpos1, ypos2, zpos1, nz1)
    value = value + gz3term(r4, xpos2, ypos2, zpos1, nz1)
    value = value - gz3term(r5, xpos1, ypos1, zpos2, nz2)
    value = value + gz3term(r6, xpos2, ypos1, zpos2, nz2)
    value = value + gz3term(r7, xpos1, ypos2, zpos2, nz2)
    value = value - gz3term(r8, xpos2, ypos2, zpos2, nz2)

    return c * value


def _compute_gy(geo: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute gy (per-prism, before rotation/summation).

    Corresponds to TC_GUI.m gy section in Prism/Prism_g_gx/Prism_gx.
    Returns: -c*value (note the leading minus).
    """
    c = geo['c']
    xpos1, xpos2 = geo['xpos1'], geo['xpos2']
    ypos1, ypos2 = geo['ypos1'], geo['ypos2']
    zpos1, zpos2 = geo['zpos1'], geo['zpos2']
    nx1, nx2, ny1, ny2, nz1, nz2 = \
        geo['nx1'], geo['nx2'], geo['ny1'], geo['ny2'], geo['nz1'], geo['nz2']
    r1, r2, r3, r4, r5, r6, r7, r8 = \
        geo['r1'], geo['r2'], geo['r3'], geo['r4'], geo['r5'], geo['r6'], geo['r7'], geo['r8']

    with np.errstate(divide='ignore', invalid='ignore'):
        s1 = xpos1 * np.log((zpos1 + r1) * (zpos2 + r7) / ((zpos1 + r3) * (zpos2 + r5)))
        s2 = xpos2 * np.log((zpos1 + r2) * (zpos2 + r8) / ((zpos1 + r4) * (zpos2 + r6)))
    s1 = np.where(np.isfinite(s1), s1, 0.0)
    s2 = np.where(np.isfinite(s2), s2, 0.0)
    s1.flat[nx1] = 0.0
    s2.flat[nx2] = 0.0
    s = s1 - s2

    with np.errstate(divide='ignore', invalid='ignore'):
        s1b = zpos1 * np.log((xpos1 + r1) * (xpos2 + r4) / ((xpos2 + r2) * (xpos1 + r3)))
        s2b = zpos2 * np.log((xpos1 + r5) * (xpos2 + r8) / ((xpos2 + r6) * (xpos1 + r7)))
    s1b = np.where(np.isfinite(s1b), s1b, 0.0)
    s2b = np.where(np.isfinite(s2b), s2b, 0.0)
    s1b.flat[nz1] = 0.0
    s2b.flat[nz2] = 0.0

    value = s + s1b - s2b

    # gz3term with swapped args: (r, xpos1, zpos1, ypos1, ny1) etc.
    value = value + gz3term(r1, xpos1, zpos1, ypos1, ny1)
    value = value - gz3term(r2, xpos2, zpos1, ypos1, ny1)
    value = value - gz3term(r3, xpos1, zpos1, ypos2, ny2)
    value = value + gz3term(r4, xpos2, zpos1, ypos2, ny2)
    value = value - gz3term(r5, xpos1, zpos2, ypos1, ny1)
    value = value + gz3term(r6, xpos2, zpos2, ypos1, ny1)
    value = value + gz3term(r7, xpos1, zpos2, ypos2, ny2)
    value = value - gz3term(r8, xpos2, zpos2, ypos2, ny2)

    return -c * value


def _compute_gx(geo: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute gx (per-prism, before rotation/summation).

    Corresponds to TC_GUI.m gx section in Prism/Prism_g_gx/Prism_gx.
    Returns: -c*value (note the leading minus).
    """
    c = geo['c']
    xpos1, xpos2 = geo['xpos1'], geo['xpos2']
    ypos1, ypos2 = geo['ypos1'], geo['ypos2']
    zpos1, zpos2 = geo['zpos1'], geo['zpos2']
    nx1, nx2, ny1, ny2, nz1, nz2 = \
        geo['nx1'], geo['nx2'], geo['ny1'], geo['ny2'], geo['nz1'], geo['nz2']
    r1, r2, r3, r4, r5, r6, r7, r8 = \
        geo['r1'], geo['r2'], geo['r3'], geo['r4'], geo['r5'], geo['r6'], geo['r7'], geo['r8']

    with np.errstate(divide='ignore', invalid='ignore'):
        s1 = ypos1 * np.log((zpos1 + r1) * (zpos2 + r6) / ((zpos1 + r2) * (zpos2 + r5)))
        s2 = ypos2 * np.log((zpos1 + r3) * (zpos2 + r8) / ((zpos1 + r4) * (zpos2 + r7)))
    s1 = np.where(np.isfinite(s1), s1, 0.0)
    s2 = np.where(np.isfinite(s2), s2, 0.0)
    s1.flat[ny1] = 0.0
    s2.flat[ny2] = 0.0
    s = s1 - s2

    with np.errstate(divide='ignore', invalid='ignore'):
        s1b = zpos1 * np.log((ypos1 + r1) * (ypos2 + r4) / ((ypos1 + r2) * (ypos2 + r3)))
        s2b = zpos2 * np.log((ypos1 + r5) * (ypos2 + r8) / ((ypos2 + r7) * (ypos1 + r6)))
    s1b = np.where(np.isfinite(s1b), s1b, 0.0)
    s2b = np.where(np.isfinite(s2b), s2b, 0.0)
    s1b.flat[nz1] = 0.0
    s2b.flat[nz2] = 0.0

    value = s + s1b - s2b

    # gz3term with swapped args: (r, ypos1, zpos1, xpos1, nx1) etc.
    value = value + gz3term(r1, ypos1, zpos1, xpos1, nx1)
    value = value - gz3term(r2, ypos1, zpos1, xpos2, nx2)
    value = value - gz3term(r3, ypos2, zpos1, xpos1, nx1)
    value = value + gz3term(r4, ypos2, zpos1, xpos2, nx2)
    value = value - gz3term(r5, ypos1, zpos2, xpos1, nx1)
    value = value + gz3term(r6, ypos1, zpos2, xpos2, nx2)
    value = value + gz3term(r7, ypos2, zpos2, xpos1, nx1)
    value = value - gz3term(r8, ypos2, zpos2, xpos2, nx2)

    return -c * value


def _compute_gxx(geo: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Compute 2nd derivatives (per-prism, before rotation/summation).

    Corresponds to TC_GUI.m Vxx/Vyy/Vzz/Vxy/Vxz/Vyz sections in Prism/Prism_gxx.
    Returns: (gxx, gxy, gxz, gyy, gyz, gzz) each already scaled by 1e4.
    """
    c = geo['c']
    xpos1, xpos2 = geo['xpos1'], geo['xpos2']
    ypos1, ypos2 = geo['ypos1'], geo['ypos2']
    zpos1, zpos2 = geo['zpos1'], geo['zpos2']
    nx1, nx2, ny1, ny2, nz1, nz2 = \
        geo['nx1'], geo['nx2'], geo['ny1'], geo['ny2'], geo['nz1'], geo['nz2']
    r1, r2, r3, r4, r5, r6, r7, r8 = \
        geo['r1'], geo['r2'], geo['r3'], geo['r4'], geo['r5'], geo['r6'], geo['r7'], geo['r8']

    # ---- Vxx ----
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos1) * (ypos1) / ((xpos1) * r1))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx1] = 0.0
    res = vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos1) * (ypos1) / ((xpos2) * r2))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx2] = 0.0
    res = res - vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos1) * (ypos2) / ((xpos1) * r3))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx1] = 0.0
    res = res - vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos1) * (ypos2) / ((xpos2) * r4))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx2] = 0.0
    res = res + vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos2) * (ypos1) / ((xpos1) * r5))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx1] = 0.0
    res = res - vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos2) * (ypos1) / ((xpos2) * r6))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx2] = 0.0
    res = res + vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos2) * (ypos2) / ((xpos1) * r7))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx1] = 0.0
    res = res + vxx
    with np.errstate(divide='ignore', invalid='ignore'):
        vxx = np.arctan((zpos2) * (ypos2) / ((xpos2) * r8))
    vxx = np.where(np.isfinite(vxx), vxx, 0.0)
    vxx.flat[nx2] = 0.0
    res = res - vxx
    gxx = (c * res) * 1e4

    # ---- Vyy ----
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos1) * (xpos1) / ((ypos1) * r1))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny1] = 0.0
    res = vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos1) * (xpos2) / ((ypos1) * r2))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny1] = 0.0
    res = res - vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos1) * (xpos1) / ((ypos2) * r3))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny2] = 0.0
    res = res - vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos1) * (xpos2) / ((ypos2) * r4))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny2] = 0.0
    res = res + vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos2) * (xpos1) / ((ypos1) * r5))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny1] = 0.0
    res = res - vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos2) * (xpos2) / ((ypos1) * r6))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny1] = 0.0
    res = res + vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos2) * (xpos1) / ((ypos2) * r7))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny2] = 0.0
    res = res + vyy
    with np.errstate(divide='ignore', invalid='ignore'):
        vyy = np.arctan((zpos2) * (xpos2) / ((ypos2) * r8))
    vyy = np.where(np.isfinite(vyy), vyy, 0.0)
    vyy.flat[ny2] = 0.0
    res = res - vyy
    gyy = (c * res) * 1e4

    # ---- Vzz ----
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos1) * (ypos1) / ((zpos1) * r1))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz1] = 0.0
    res = vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos2) * (ypos1) / ((zpos1) * r2))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz1] = 0.0
    res = res - vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos1) * (ypos2) / ((zpos1) * r3))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz1] = 0.0
    res = res - vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos2) * (ypos2) / ((zpos1) * r4))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz1] = 0.0
    res = res + vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos1) * (ypos1) / ((zpos2) * r5))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz2] = 0.0
    res = res - vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos2) * (ypos1) / ((zpos2) * r6))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz2] = 0.0
    res = res + vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos1) * (ypos2) / ((zpos2) * r7))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz2] = 0.0
    res = res + vzz
    with np.errstate(divide='ignore', invalid='ignore'):
        vzz = np.arctan((xpos2) * (ypos2) / ((zpos2) * r8))
    vzz = np.where(np.isfinite(vzz), vzz, 0.0)
    vzz.flat[nz2] = 0.0
    res = res - vzz
    gzz = (c * res) * 1e4

    # ---- Vxy ----
    s1 = (r1 + zpos1) * (r4 + zpos1) * (r6 + zpos2) * (r7 + zpos2)
    s2 = (r2 + zpos1) * (r3 + zpos1) * (r5 + zpos2) * (r8 + zpos2)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(s1 / s2)
    n = np.flatnonzero((np.abs(s2) <= 0.001) | (np.abs(s1) <= 0.001))
    res = np.where(np.isfinite(res), res, 0.0)
    if n.size > 0:
        res.flat[n] = 0.0
    gxy = -c * res * 1e4

    # ---- Vxz ----
    s1 = (r1 + ypos1) * (r4 + ypos2) * (r6 + ypos1) * (r7 + ypos2)
    s2 = (r2 + ypos1) * (r3 + ypos2) * (r5 + ypos1) * (r8 + ypos2)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(s1 / s2)
    n = np.flatnonzero((np.abs(s2) <= 0.001) | (np.abs(s1) <= 0.001))
    res = np.where(np.isfinite(res), res, 0.0)
    if n.size > 0:
        res.flat[n] = 0.0
    gxz = c * res * 1e4

    # ---- Vyz ----
    s1 = (r1 + xpos1) * (r4 + xpos2) * (r6 + xpos2) * (r7 + xpos1)
    s2 = (r2 + xpos2) * (r3 + xpos1) * (r5 + xpos1) * (r8 + xpos2)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(s1 / s2)
    n = np.flatnonzero((np.abs(s2) <= 0.001) | (np.abs(s1) <= 0.001))
    res = np.where(np.isfinite(res), res, 0.0)
    if n.size > 0:
        res.flat[n] = 0.0
    gyz = c * res * 1e4

    return gxx, gxy, gxz, gyy, gyz, gzz


# =============================================================================
#  Public API (5 functions)
# =============================================================================

def prism(x1: np.ndarray, x2: np.ndarray,
          y1: np.ndarray, y2: np.ndarray,
          d1: np.ndarray, d2: np.ndarray,
          lon: float, lat: float, r0: float,
          e: np.ndarray,
          resolution_x: float, resolution_y: float
          ) -> Tuple[np.ndarray, ...]:
    """
    Compute all 10 gravity field components for an array of prisms.

    Corresponds to TC_GUI.m L3277-L3567 (Prism).

    Args:
        x1, x2: Prism longitude bounds [deg].
        y1, y2: Prism latitude bounds [deg].
        d1, d2: Prism radial bounds (upper, lower) [m].
        lon, lat, r0: Computation point (station) coordinates [deg, deg, m].
        e: Density for each prism [g/cm^3].
        resolution_x, resolution_y: Prism resolution [deg].

    Returns:
        (g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz): 10 scalars (summed
        contributions of all input prisms).
        Units: g in m²/s², Vx/Vy/Vz in mGal, Vxx..Vzz in E.
    """
    geo = _prism_geometry(x1, x2, y1, y2, d1, d2, lon, lat, r0, e)

    # potential
    g = -np.sum(_compute_g(geo)) * 1e-5

    # 1st derivatives (prism frame)
    gx = _compute_gx(geo)
    gy = _compute_gy(geo)
    gz = _compute_gz(geo)

    # rotate to station frame
    temp_Vx, temp_Vy, temp_Vz = g_prism2point(
        gx, gy, gz, lon, lat, geo['prism_lon'], geo['prism_lat'])
    Vx = np.sum(temp_Vx)
    Vy = np.sum(temp_Vy)
    Vz = np.sum(temp_Vz)

    # 2nd derivatives (prism frame)
    gxx, gxy, gxz, gyy, gyz, gzz = _compute_gxx(geo)

    # rotate to station frame
    temp_Vxx, temp_Vxy, temp_Vxz, temp_Vyy, temp_Vyz, temp_Vzz = \
        ggt_prism2point(gxx, gxy, gxz, gyy, gyz, gzz,
                        lon, lat, geo['prism_lon'], geo['prism_lat'])
    Vxx = np.sum(temp_Vxx)
    Vxy = np.sum(temp_Vxy)
    Vxz = np.sum(temp_Vxz)
    Vyy = np.sum(temp_Vyy)
    Vyz = np.sum(temp_Vyz)
    Vzz = np.sum(temp_Vzz)

    return g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz


def prism_g(x1: np.ndarray, x2: np.ndarray,
            y1: np.ndarray, y2: np.ndarray,
            d1: np.ndarray, d2: np.ndarray,
            lon: float, lat: float, r0: float,
            e: np.ndarray,
            resolution_x: float, resolution_y: float) -> np.ndarray:
    """
    Compute potential only for an array of prisms (itype=0).

    Corresponds to TC_GUI.m L3598-L3674 (Prism_g).

    Note: Prism_g computes r inside the loop (one vertex at a time), unlike
    Prism which precomputes r1..r8. The numerical result is identical.
    """
    geo = _prism_geometry(x1, x2, y1, y2, d1, d2, lon, lat, r0, e)
    g = -np.sum(_compute_g(geo)) * 1e-5
    return (g,)


def prism_g_gx(x1: np.ndarray, x2: np.ndarray,
               y1: np.ndarray, y2: np.ndarray,
               d1: np.ndarray, d2: np.ndarray,
               lon: float, lat: float, r0: float,
               e: np.ndarray,
               resolution_x: float, resolution_y: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute potential + 1st derivatives for an array of prisms (itype=4/103/104).

    Corresponds to TC_GUI.m L3677-L3845 (Prism_g_gx).

    Returns:
        (g, Vx, Vy, Vz): 4 scalars.
    """
    geo = _prism_geometry(x1, x2, y1, y2, d1, d2, lon, lat, r0, e)

    g = -np.sum(_compute_g(geo)) * 1e-5

    gx = _compute_gx(geo)
    gy = _compute_gy(geo)
    gz = _compute_gz(geo)

    temp_Vx, temp_Vy, temp_Vz = g_prism2point(
        gx, gy, gz, lon, lat, geo['prism_lon'], geo['prism_lat'])
    Vx = np.sum(temp_Vx)
    Vy = np.sum(temp_Vy)
    Vz = np.sum(temp_Vz)

    return g, Vx, Vy, Vz


def prism_gx(x1: np.ndarray, x2: np.ndarray,
             y1: np.ndarray, y2: np.ndarray,
             d1: np.ndarray, d2: np.ndarray,
             lon: float, lat: float, r0: float,
             e: np.ndarray,
             resolution_x: float, resolution_y: float
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1st derivatives only for an array of prisms (itype=1).

    Corresponds to TC_GUI.m L3847-L3983 (Prism_gx).

    Returns:
        (Vx, Vy, Vz): 3 scalars.
    """
    geo = _prism_geometry(x1, x2, y1, y2, d1, d2, lon, lat, r0, e)

    gx = _compute_gx(geo)
    gy = _compute_gy(geo)
    gz = _compute_gz(geo)

    temp_Vx, temp_Vy, temp_Vz = g_prism2point(
        gx, gy, gz, lon, lat, geo['prism_lon'], geo['prism_lat'])
    Vx = np.sum(temp_Vx)
    Vy = np.sum(temp_Vy)
    Vz = np.sum(temp_Vz)

    return Vx, Vy, Vz


def prism_gxx(x1: np.ndarray, x2: np.ndarray,
              y1: np.ndarray, y2: np.ndarray,
              d1: np.ndarray, d2: np.ndarray,
              lon: float, lat: float, r0: float,
              e: np.ndarray,
              resolution_x: float, resolution_y: float
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2nd derivatives only for an array of prisms (itype=2).

    Corresponds to TC_GUI.m L3985-L4151 (Prism_gxx).

    Returns:
        (Vxx, Vxy, Vxz, Vyy, Vyz, Vzz): 6 scalars.
    """
    geo = _prism_geometry(x1, x2, y1, y2, d1, d2, lon, lat, r0, e)

    gxx, gxy, gxz, gyy, gyz, gzz = _compute_gxx(geo)

    temp_Vxx, temp_Vxy, temp_Vxz, temp_Vyy, temp_Vyz, temp_Vzz = \
        ggt_prism2point(gxx, gxy, gxz, gyy, gyz, gzz,
                        lon, lat, geo['prism_lon'], geo['prism_lat'])
    Vxx = np.sum(temp_Vxx)
    Vxy = np.sum(temp_Vxy)
    Vxz = np.sum(temp_Vxz)
    Vyy = np.sum(temp_Vyy)
    Vyz = np.sum(temp_Vyz)
    Vzz = np.sum(temp_Vzz)

    return Vxx, Vxy, Vxz, Vyy, Vyz, Vzz
