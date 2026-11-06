"""
Numba-accelerated Point Mass kernel (Zone 4/5).

Fuses _equivalent_mass + global2local + point mass formulas into a
single @njit loop.  Eliminates ~25 intermediate numpy arrays +
global2local's 12 trig array ops per call.

Mirrors TC_GUI.m L2399-L2458 (Point_mass).
"""

import numpy as np
from numba import njit
import math

_PI = 3.141592654
_D2R = _PI / 180.0
_G = 6.67384e-3


# ------------------------------------------------------------------
# itype=10: all 10 components
# ------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _pm_kernel_10(xx1, xx2, yy1, yy2, r1, r2,
                   lon, lat, r0, e):
    """
    Point mass kernel — all 10 components (itype=10).

    Inlines global2local as scalar ops. Accumulates into local variables.

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz)
    """
    N = xx1.shape[0]
    if N == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # station trig (computed once)
    lat_r = lat * _D2R
    lon_r = lon * _D2R
    coslat = math.cos(lat_r)
    sinlat = math.sin(lat_r)
    coslon = math.cos(lon_r)
    sinlon = math.sin(lon_r)

    # station Cartesian position (r0 is used directly, matching numpy's
    # global2local convention where r0=orthometric height, not geocentric)
    sx = r0 * coslat * coslon
    sy = r0 * coslat * sinlon
    sz = r0 * sinlat

    sum_V = 0.0
    sum_Vx = 0.0; sum_Vy = 0.0; sum_Vz = 0.0
    sum_Vxx = 0.0; sum_Vxy = 0.0; sum_Vxz = 0.0
    sum_Vyy = 0.0; sum_Vyz = 0.0; sum_Vzz = 0.0

    G = _G
    d2r = _D2R

    for i in range(N):
        # point mass center
        plon = (xx1[i] + xx2[i]) * 0.5
        plat = (yy1[i] + yy2[i]) * 0.5
        pr = (r1[i] + r2[i]) * 0.5

        # equivalent mass
        mass = (d2r * e[i] * (xx2[i] - xx1[i])
                * (r1[i] * r1[i] * r1[i] - r2[i] * r2[i] * r2[i])
                * (math.sin(d2r * yy2[i]) - math.sin(d2r * yy1[i])) / 3.0)

        # global2local: matches numpy convention
        # global2local(station_lon, station_lat, station_r,
        #              prism_lon, prism_lat, prism_r)
        # Local frame is at the STATION; prism is transformed into it.
        plon_r = plon * d2r
        plat_r = plat * d2r
        cplon = math.cos(plon_r)
        splon = math.sin(plon_r)
        cplat = math.cos(plat_r)
        splat = math.sin(plat_r)

        # Prism Cartesian position
        px_r = pr * cplat * cplon
        py_r = pr * cplat * splon
        pz_r = pr * splat

        gx = px_r - sx
        gy = py_r - sy
        gz = pz_r - sz

        # Rotate into local frame using STATION trig
        dx = -gx * sinlat * coslon - gy * sinlat * sinlon + gz * coslat
        dy = -gx * sinlon + gy * coslon
        dz = -(gx * coslat * coslon + gy * coslat * sinlon + gz * sinlat)

        # distance
        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 < 1e-300:
            continue
        l = 1.0 / math.sqrt(dist2)
        l3 = l * l * l
        l5 = l3 * l * l

        c = G * mass

        # potential
        sum_V += c * l

        # 1st derivatives (Point_mass convention: gz=-c*dz*l3)
        sum_Vx += c * dx * l3
        sum_Vy += c * dy * l3
        sum_Vz += -(c * dz * l3)   # gz = -res

        # 2nd derivatives
        sum_Vxx += -(c * (-l3 + 3.0 * dx * dx * l5))
        sum_Vxy += -(c * (3.0 * dx * dy * l5))
        sum_Vxz += (c * (3.0 * dx * dz * l5))
        sum_Vyy += -(c * (-l3 + 3.0 * dy * dy * l5))
        sum_Vyz += (c * (3.0 * dy * dz * l5))
        sum_Vzz += -(c * (-l3 + 3.0 * dz * dz * l5))

    return (-sum_V * 1e-5,
            sum_Vx, sum_Vy, sum_Vz,
            sum_Vxx * 1e4, sum_Vxy * 1e4, sum_Vxz * 1e4,
            sum_Vyy * 1e4, sum_Vyz * 1e4, sum_Vzz * 1e4)


# ------------------------------------------------------------------
# itype=0/1/4: lightweight variants
# ------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _pm_kernel_04(xx1, xx2, yy1, yy2, r1, r2,
                   lon, lat, r0, e):
    """Potential + 1st derivatives (itype=0/1/4)."""
    N = xx1.shape[0]
    if N == 0:
        return (0.0, 0.0, 0.0, 0.0)

    lat_r = lat * _D2R
    lon_r = lon * _D2R
    coslat = math.cos(lat_r)
    sinlat = math.sin(lat_r)
    coslon = math.cos(lon_r)
    sinlon = math.sin(lon_r)

    sx = r0 * coslat * coslon
    sy = r0 * coslat * sinlon
    sz = r0 * sinlat

    sum_V = 0.0
    sum_Vx = 0.0; sum_Vy = 0.0; sum_Vz = 0.0

    G = _G
    d2r = _D2R

    for i in range(N):
        plon = (xx1[i] + xx2[i]) * 0.5
        plat = (yy1[i] + yy2[i]) * 0.5
        pr = (r1[i] + r2[i]) * 0.5

        mass = (d2r * e[i] * (xx2[i] - xx1[i])
                * (r1[i] ** 3 - r2[i] ** 3)
                * (math.sin(d2r * yy2[i]) - math.sin(d2r * yy1[i])) / 3.0)

        plon_r = plon * d2r
        plat_r = plat * d2r
        cplon = math.cos(plon_r)
        splon = math.sin(plon_r)
        cplat = math.cos(plat_r)
        splat = math.sin(plat_r)

        px_r = pr * cplat * cplon
        py_r = pr * cplat * splon
        pz_r = pr * splat

        gx = px_r - sx
        gy = py_r - sy
        gz = pz_r - sz

        dx = -gx * sinlat * coslon - gy * sinlat * sinlon + gz * coslat
        dy = -gx * sinlon + gy * coslon
        dz = -(gx * coslat * coslon + gy * coslat * sinlon + gz * sinlat)

        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 < 1e-300:
            continue
        l = 1.0 / math.sqrt(dist2)
        l3 = l * l * l

        c = G * mass

        sum_V += c * l
        sum_Vx += c * dx * l3
        sum_Vy += c * dy * l3
        sum_Vz += -(c * dz * l3)

    return (-sum_V * 1e-5, sum_Vx, sum_Vy, sum_Vz)


# ------------------------------------------------------------------
# itype=2: 2nd derivatives only
# ------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _pm_kernel_2(xx1, xx2, yy1, yy2, r1, r2,
                  lon, lat, r0, e):
    """2nd derivatives only (itype=2)."""
    N = xx1.shape[0]
    if N == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    lat_r = lat * _D2R
    lon_r = lon * _D2R
    coslat = math.cos(lat_r)
    sinlat = math.sin(lat_r)
    coslon = math.cos(lon_r)
    sinlon = math.sin(lon_r)

    sx = r0 * coslat * coslon
    sy = r0 * coslat * sinlon
    sz = r0 * sinlat

    sum_Vxx = 0.0; sum_Vxy = 0.0; sum_Vxz = 0.0
    sum_Vyy = 0.0; sum_Vyz = 0.0; sum_Vzz = 0.0

    G = _G
    d2r = _D2R

    for i in range(N):
        plon = (xx1[i] + xx2[i]) * 0.5
        plat = (yy1[i] + yy2[i]) * 0.5
        pr = (r1[i] + r2[i]) * 0.5

        mass = (d2r * e[i] * (xx2[i] - xx1[i])
                * (r1[i] ** 3 - r2[i] ** 3)
                * (math.sin(d2r * yy2[i]) - math.sin(d2r * yy1[i])) / 3.0)

        plon_r = plon * d2r
        plat_r = plat * d2r
        cplon = math.cos(plon_r)
        splon = math.sin(plon_r)
        cplat = math.cos(plat_r)
        splat = math.sin(plat_r)

        px_r = pr * cplat * cplon
        py_r = pr * cplat * splon
        pz_r = pr * splat

        gx = px_r - sx
        gy = py_r - sy
        gz = pz_r - sz

        dx = -gx * sinlat * coslon - gy * sinlat * sinlon + gz * coslat
        dy = -gx * sinlon + gy * coslon
        dz = -(gx * coslat * coslon + gy * coslat * sinlon + gz * sinlat)

        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 < 1e-300:
            continue
        l = 1.0 / math.sqrt(dist2)
        l3 = l * l * l
        l5 = l3 * l * l

        c = G * mass

        sum_Vxx += -(c * (-l3 + 3.0 * dx * dx * l5))
        sum_Vxy += -(c * (3.0 * dx * dy * l5))
        sum_Vxz += (c * (3.0 * dx * dz * l5))
        sum_Vyy += -(c * (-l3 + 3.0 * dy * dy * l5))
        sum_Vyz += (c * (3.0 * dy * dz * l5))
        sum_Vzz += -(c * (-l3 + 3.0 * dz * dz * l5))

    return (sum_Vxx * 1e4, sum_Vxy * 1e4, sum_Vxz * 1e4,
            sum_Vyy * 1e4, sum_Vyz * 1e4, sum_Vzz * 1e4)


# ------------------------------------------------------------------
# Public wrappers matching point_mass.py signatures
# ------------------------------------------------------------------

def _prepare_pm(xx1, xx2, yy1, yy2, r1, r2, e):
    return (np.ascontiguousarray(xx1, dtype=np.float64),
            np.ascontiguousarray(xx2, dtype=np.float64),
            np.ascontiguousarray(yy1, dtype=np.float64),
            np.ascontiguousarray(yy2, dtype=np.float64),
            np.ascontiguousarray(r1, dtype=np.float64),
            np.ascontiguousarray(r2, dtype=np.float64),
            np.ascontiguousarray(e, dtype=np.float64))


def pm_kernel_g(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e):
    """Potential only (itype=0)."""
    xx1, xx2, yy1, yy2, r1, r2, e = _prepare_pm(xx1, xx2, yy1, yy2, r1, r2, e)
    V, _, _, _ = _pm_kernel_04(xx1, xx2, yy1, yy2, r1, r2,
                                float(lon), float(lat), float(r0), e)
    return (V,)


def pm_kernel_gx(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e):
    """1st derivatives only (itype=1)."""
    xx1, xx2, yy1, yy2, r1, r2, e = _prepare_pm(xx1, xx2, yy1, yy2, r1, r2, e)
    _, Vx, Vy, Vz = _pm_kernel_04(xx1, xx2, yy1, yy2, r1, r2,
                                   float(lon), float(lat), float(r0), e)
    return (Vx, Vy, Vz)


def pm_kernel_g_gx(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e):
    """Potential + 1st derivatives (itype=4/103/104)."""
    xx1, xx2, yy1, yy2, r1, r2, e = _prepare_pm(xx1, xx2, yy1, yy2, r1, r2, e)
    return _pm_kernel_04(xx1, xx2, yy1, yy2, r1, r2,
                          float(lon), float(lat), float(r0), e)


def pm_kernel_gxx(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e):
    """2nd derivatives only (itype=2)."""
    xx1, xx2, yy1, yy2, r1, r2, e = _prepare_pm(xx1, xx2, yy1, yy2, r1, r2, e)
    return _pm_kernel_2(xx1, xx2, yy1, yy2, r1, r2,
                         float(lon), float(lat), float(r0), e)


def pm_kernel_10(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e):
    """All 10 components (itype=10)."""
    xx1, xx2, yy1, yy2, r1, r2, e = _prepare_pm(xx1, xx2, yy1, yy2, r1, r2, e)
    return _pm_kernel_10(xx1, xx2, yy1, yy2, r1, r2,
                          float(lon), float(lat), float(r0), e)
