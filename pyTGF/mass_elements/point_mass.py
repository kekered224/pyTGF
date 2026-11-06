"""
Point mass element (Zone 4/5, far, most efficient)

Mirrors TC_GUI.m:
    Point_g     (L2260-L2281)  potential only, itype=0
    Point_gx    (L2283-L2314)  1st derivatives, itype=1
    Point_g_gx  (L2316-L2353)  potential + 1st derivs, itype=4/103/104
    Point_gxx   (L2355-L2397)  2nd derivatives, itype=2
    Point_mass  (L2399-L2458)  all 10 components, itype=10

Each tesseroid is reduced to an equivalent point mass at its center with:
    mass = e * d2r * (xx2-xx1) * (r1^3 - r2^3) * (sin(d2r*yy2) - sin(d2r*yy1)) / 3

CRITICAL: MATLAB calls global2local with DIFFERENT argument orders in
Point_g vs. the other four functions:
    Point_g     : global2local(lon, lat, r0, point_lon, point_lat, point_r)
                  (station -> prism frame; dz sign flipped in gx/gz formulas)
    Point_gx/g_gx/gxx/mass :
                  global2local(point_lon, point_lat, point_r, lon, lat, r0)
                  (prism -> station frame)
We replicate this exactly, including the sign compensations in the
formulas for gz/gxz/gzz.
"""

from typing import Tuple
import numpy as np

from ..config import PhysicalConstants
from ..utils.coordinate import global2local


def _equivalent_mass(e: np.ndarray,
                     xx1: np.ndarray, xx2: np.ndarray,
                     yy1: np.ndarray, yy2: np.ndarray,
                     r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """
    Tesseroid-equivalent point mass.

    MATLAB:
        mass = d2r*e.*(xx2-xx1).*(r1.^3-r2.^3).*(sin(d2r*yy2)-sin(d2r*yy1))/3;
    """
    d2r = PhysicalConstants.d2r()
    return (d2r * e * (xx2 - xx1) * (r1 ** 3 - r2 ** 3)
            * (np.sin(d2r * yy2) - np.sin(d2r * yy1)) / 3.0)


def point_g(xx1: np.ndarray, xx2: np.ndarray,
            yy1: np.ndarray, yy2: np.ndarray,
            r1: np.ndarray, r2: np.ndarray,
            lon: float, lat: float, r0: float,
            e: np.ndarray) -> np.ndarray:
    """
    Compute potential from an array of point masses (itype=0).

    Corresponds to TC_GUI.m L2260-L2281 (Point_g).

    Note: uses global2local(lon,lat,r0, point_lon,point_lat,point_r)
    (station -> prism frame).

    Returns:
        g: Potential [m²/s²] (summed contribution, scaled by 1e-5).
    """
    G = PhysicalConstants.G
    d2r = PhysicalConstants.d2r()

    point_r = (r1 + r2) / 2.0
    point_lon = (xx1 + xx2) / 2.0
    point_lat = (yy1 + yy2) / 2.0

    mass = _equivalent_mass(e, xx1, xx2, yy1, yy2, r1, r2)

    # Fix: use same convention as point_gx, point_g_gx, point_gxx, point_mass
    # global2local(point, station) — transforms point into station's local frame
    dx, dy, dz = global2local(point_lon, point_lat, point_r,
                              np.array([lon]), np.array([lat]), np.array([r0]))
    l = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)

    c = G * mass
    res = c * l
    g = -np.sum(res) * 1e-5
    return (g,)


def point_gx(xx1: np.ndarray, xx2: np.ndarray,
             yy1: np.ndarray, yy2: np.ndarray,
             r1: np.ndarray, r2: np.ndarray,
             lon: float, lat: float, r0: float,
             e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1st derivatives from an array of point masses (itype=1).

    Corresponds to TC_GUI.m L2283-L2314 (Point_gx).

    Note: uses global2local(point_lon,point_lat,point_r, lon,lat,r0)
    (prism -> station frame). gz = -res to compensate the flipped z-axis.
    """
    G = PhysicalConstants.G
    d2r = PhysicalConstants.d2r()

    point_r = (r1 + r2) / 2.0
    point_lon = (xx1 + xx2) / 2.0
    point_lat = (yy1 + yy2) / 2.0

    mass = _equivalent_mass(e, xx1, xx2, yy1, yy2, r1, r2)

    # MATLAB: [dx dy dz]=global2local(point_lon,point_lat,point_r,lon,lat,r0);
    dx, dy, dz = global2local(point_lon, point_lat, point_r,
                              np.array([lon]), np.array([lat]), np.array([r0]))
    l = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    l3 = l * l * l

    c = G * mass

    gx = c * (dx * l3)
    gy = c * (dy * l3)
    gz = -(c * (dz * l3))   # MATLAB: gz=-res;

    Vx = np.sum(gx)
    Vy = np.sum(gy)
    Vz = np.sum(gz)
    return Vx, Vy, Vz


def point_g_gx(xx1: np.ndarray, xx2: np.ndarray,
               yy1: np.ndarray, yy2: np.ndarray,
               r1: np.ndarray, r2: np.ndarray,
               lon: float, lat: float, r0: float,
               e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute potential + 1st derivatives from point masses (itype=4/103/104).

    Corresponds to TC_GUI.m L2316-L2353 (Point_g_gx).

    Note: uses global2local(point_lon,point_lat,point_r, lon,lat,r0).
    The intermediate gz=res (no sign flip), and Vz=-sum(gz) at the end.
    """
    G = PhysicalConstants.G
    d2r = PhysicalConstants.d2r()

    point_r = (r1 + r2) / 2.0
    point_lon = (xx1 + xx2) / 2.0
    point_lat = (yy1 + yy2) / 2.0

    mass = _equivalent_mass(e, xx1, xx2, yy1, yy2, r1, r2)

    dx, dy, dz = global2local(point_lon, point_lat, point_r,
                              np.array([lon]), np.array([lat]), np.array([r0]))
    l = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    l3 = l * l * l

    c = G * mass
    res = c * l
    g = -np.sum(res) * 1e-5

    gx = c * (dx * l3)
    gy = c * (dy * l3)
    gz = c * (dz * l3)        # MATLAB: gz=res; (no flip here)

    Vx = np.sum(gx)
    Vy = np.sum(gy)
    Vz = -np.sum(gz)          # MATLAB: Vz=-sum(gz);
    return g, Vx, Vy, Vz


def point_gxx(xx1: np.ndarray, xx2: np.ndarray,
              yy1: np.ndarray, yy2: np.ndarray,
              r1: np.ndarray, r2: np.ndarray,
              lon: float, lat: float, r0: float,
              e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2nd derivatives from an array of point masses (itype=2).

    Corresponds to TC_GUI.m L2355-L2397 (Point_gxx).

    Note: uses global2local(point_lon,point_lat,point_r, lon,lat,r0).
    Sign pattern: gxx=-res, gxy=-res, gxz=+res, gyy=-res, gyz=+res, gzz=-res.
    """
    G = PhysicalConstants.G
    d2r = PhysicalConstants.d2r()

    point_r = (r1 + r2) / 2.0
    point_lon = (xx1 + xx2) / 2.0
    point_lat = (yy1 + yy2) / 2.0

    mass = _equivalent_mass(e, xx1, xx2, yy1, yy2, r1, r2)

    dx, dy, dz = global2local(point_lon, point_lat, point_r,
                              np.array([lon]), np.array([lat]), np.array([r0]))
    l = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    l3 = l * l * l
    l5 = l3 * l * l

    c = G * mass

    gxx = -(c * (-l3 + 3.0 * dx * dx * l5))
    gxy = -(c * (3.0 * dx * dy * l5))
    gxz = +(c * (3.0 * dx * dz * l5))
    gyy = -(c * (-l3 + 3.0 * dy * dy * l5))
    gyz = +(c * (3.0 * dy * dz * l5))
    gzz = -(c * (-l3 + 3.0 * dz * dz * l5))

    Vxx = np.sum(gxx) * 1e4
    Vxy = np.sum(gxy) * 1e4
    Vxz = np.sum(gxz) * 1e4
    Vyy = np.sum(gyy) * 1e4
    Vyz = np.sum(gyz) * 1e4
    Vzz = np.sum(gzz) * 1e4
    return Vxx, Vxy, Vxz, Vyy, Vyz, Vzz


def point_mass(xx1: np.ndarray, xx2: np.ndarray,
               yy1: np.ndarray, yy2: np.ndarray,
               r1: np.ndarray, r2: np.ndarray,
               lon: float, lat: float, r0: float,
               e: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Compute all 10 gravity field components from point masses (itype=10).

    Corresponds to TC_GUI.m L2399-L2458 (Point_mass).

    Note: uses global2local(point_lon,point_lat,point_r, lon,lat,r0).
    Combines Point_g_gx (potential+1st) and Point_gxx (2nd) sign patterns.
    """
    G = PhysicalConstants.G
    d2r = PhysicalConstants.d2r()

    point_r = (r1 + r2) / 2.0
    point_lon = (xx1 + xx2) / 2.0
    point_lat = (yy1 + yy2) / 2.0

    mass = _equivalent_mass(e, xx1, xx2, yy1, yy2, r1, r2)

    dx, dy, dz = global2local(point_lon, point_lat, point_r,
                              np.array([lon]), np.array([lat]), np.array([r0]))
    l = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    l3 = l * l * l
    l5 = l3 * l * l

    c = G * mass

    # potential
    res = c * l
    g = -np.sum(res) * 1e-5

    # 1st derivatives (Point_g_gx style: gz=res, Vz=-sum(gz))
    gx = c * (dx * l3)
    gy = c * (dy * l3)
    gz = -(c * (dz * l3))     # MATLAB Point_mass: gz=-res;
    Vx = np.sum(gx)
    Vy = np.sum(gy)
    Vz = np.sum(gz)           # MATLAB Point_mass: Vz=sum(gz); (no extra minus)

    # 2nd derivatives (same sign pattern as Point_gxx)
    gxx = -(c * (-l3 + 3.0 * dx * dx * l5))
    gxy = -(c * (3.0 * dx * dy * l5))
    gxz = +(c * (3.0 * dx * dz * l5))
    gyy = -(c * (-l3 + 3.0 * dy * dy * l5))
    gyz = +(c * (3.0 * dy * dz * l5))
    gzz = -(c * (-l3 + 3.0 * dz * dz * l5))

    Vxx = np.sum(gxx) * 1e4
    Vxy = np.sum(gxy) * 1e4
    Vxz = np.sum(gxz) * 1e4
    Vyy = np.sum(gyy) * 1e4
    Vyz = np.sum(gyz) * 1e4
    Vzz = np.sum(gzz) * 1e4

    return g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz
