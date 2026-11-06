"""
Numba-accelerated Tesseroid kernel (Zone 3).

Fuses tess_geometry + Taylor-series formulas into a single @njit loop.
Eliminates ~150 intermediate numpy arrays + ~60 zero-filled dummies +
dict allocation overhead from the numpy version.

Reference: Grombein et al. (2013) J. Geodesy 87(7), 645-660.
"""

import numpy as np
from numba import njit, prange
import math

_PI = 3.141592654
_D2R = _PI / 180.0
_G = 6.67384e-3


# ------------------------------------------------------------------
# Scalar math helpers
# ------------------------------------------------------------------

@njit(cache=True)
def _safe_acos(x):
    """Clamp to [-1,1] then acos."""
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return math.acos(x)


# ------------------------------------------------------------------
# itype=10: all 10 components
# ------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _tess_kernel_10(lon1, lat1, h1, ref,
                     lon, lat, r0, rx, ry, e):
    """
    Tesseroid kernel — all 10 components (itype=10).

    Combines _tess_kernel_04 (V/Vx/Vy/Vz) and _tess_kernel_2 (Vxx..Vzz)
    to avoid code duplication and ensure consistency.

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz)
    """
    V, Vx, Vy, Vz = _tess_kernel_04(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = _tess_kernel_2(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    return (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz)


# OLD_TESS_KERNEL_10_MARKER

# ------------------------------------------------------------------
# itype=0/1/4: V, Vx, Vy, Vz (NO b_iJk or gamaijk needed)
# ------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _tess_kernel_04(lon1, lat1, h1, ref,
                     lon, lat, r0, rx, ry, e):
    """
    Tesseroid kernel — potential + 1st derivatives (itype=0/1/4).

    Computes V, Vx, Vy, Vz only. Skips expensive b_iJk and gamaijk
    (3rd/4th-order Taylor terms) that are only needed for 2nd derivatives.

    Returns:
        (V, Vx, Vy, Vz)
    """
    N = lon1.shape[0]
    if N == 0:
        return (0.0, 0.0, 0.0, 0.0)

    lat_r = lat * _D2R
    coslat = math.cos(lat_r)
    sinlat = math.sin(lat_r)

    sum_V = 0.0
    sum_Vx = 0.0; sum_Vy = 0.0; sum_Vz = 0.0

    half_rx = rx * 0.5
    half_ry = ry * 0.5
    d2r = _D2R
    G = _G

    for i in range(N):
        zz1 = lon1[i] - half_rx
        zz2 = lon1[i] + half_rx
        yy1 = lat1[i] - half_ry
        yy2 = lat1[i] + half_ry
        xx1 = h1[i]
        xx2 = ref[i]
        r1 = (xx1 + xx2) * 0.5

        lat1_r = lat1[i] * d2r
        coslat1 = math.cos(lat1_r)
        sinlat1 = math.sin(lat1_r)

        d_lon = (lon1[i] - lon) * d2r
        cosdlon = math.cos(d_lon)
        sindlon = math.sin(d_lon)

        cosphicosa = coslat1 * cosdlon
        coslatsinlat1 = coslat * sinlat1
        sinphicosa = coslatsinlat1 - sinlat * cosphicosa
        sinphisina = coslat1 * sindlon
        sinlatsinlat1 = sinlat * sinlat1
        cosphi = sinlatsinlat1 + coslat * cosphicosa

        dx = r1 * sinphicosa
        dy = r1 * sinphisina
        dz = r1 * cosphi - r0

        dxx = dx * dx; dxy = dx * dy; dxz = dx * dz
        dyy = dy * dy; dyz = dy * dz; dzz = dz * dz

        l = math.sqrt(dxx + dyy + dzz)
        if l < 1e-300:
            continue

        xx = xx2 - xx1
        yy = (yy2 - yy1) * d2r
        zz = (zz2 - zz1) * d2r
        c = G * e[i] * zz * xx * yy
        l2 = l * l; l3 = 1.0 / (l2 * l)

        r12 = r1 * r1
        toi = r12 * coslat1
        toi1 = 2.0 * r1 * coslat1
        toi2 = -r12 * sinlat1
        toi12 = 2.0 * coslat1
        toi22 = -toi

        # x_i_J (first-order displacement derivatives)
        x1_1 = sinphicosa; x2_1 = sinphisina; x3_1 = cosphi
        x1_2 = r1 * (coslat * coslat1 + sinlatsinlat1 * cosdlon)
        x2_2 = -r1 * sinlat1 * sindlon
        x3_2 = r1 * (sinlat * coslat1 - coslatsinlat1 * cosdlon)
        x1_22 = -dx; x2_22 = -dy; x3_22 = -r1 * cosphi
        x1_3 = dy * sinlat; x2_3 = r1 * cosphicosa; x3_3 = -dy * coslat
        x1_32 = x2_3 * sinlat; x2_32 = -dy; x3_32 = -x2_3 * coslat

        # ml_J (projections)
        ml1 = r0 * cosphi - r1; ml2 = r0 * x3_2; ml3 = r0 * x3_3
        ml22 = r0 * x3_22; ml32 = r0 * x3_32

        # a_iJ (first derivatives of 1/l)
        l23 = 3.0 / l2
        a1 = dx; a2 = dy; a3 = dz
        a11 = l23 * dxx - 1.0; a12 = l23 * dxy; a13 = l23 * dxz
        a22 = l23 * dyy - 1.0; a23 = l23 * dyz; a33 = l23 * dzz - 1.0

        # b_iJ (second derivatives of l^2/2)
        b1 = 2.0 * ml1; b2 = 2.0 * ml2; b3 = 2.0 * ml3
        b11 = 2.0 * (x1_1 + l23 * dx * ml1)
        b12 = 2.0 * (x1_2 + l23 * dx * ml2)
        b21 = 2.0 * (x2_1 + l23 * dy * ml1)
        b13 = 2.0 * (x1_3 + l23 * dx * ml3)
        b31 = 2.0 * (x3_1 + l23 * dz * ml1)
        b22 = 2.0 * (x2_2 + l23 * dy * ml2)
        b23 = 2.0 * (x2_3 + l23 * dy * ml3)
        b32 = 2.0 * (x3_2 + l23 * dz * ml2)
        b33 = 2.0 * (x3_3 + l23 * dz * ml3)

        # gama_iJ (correction coefficients)
        ml1_2 = ml1 * ml1; ml2_2 = ml2 * ml2; ml3_2 = ml3 * ml3
        l25 = 5.0 / l2

        gama1 = -1.0 + l23 * ml1_2
        gama2 = ml22 + l23 * ml2_2
        gama3 = ml32 + l23 * ml3_2

        gama11 = l23 * (2.0 * x1_1 * ml1 + dx * (-1.0 + l25 * ml1_2))
        gama12 = x1_22 + l23 * (2.0 * x1_2 * ml2 + dx * (ml22 + l25 * ml2_2))
        gama21 = l23 * (2.0 * x2_1 * ml1 + dy * (-1.0 + l25 * ml1_2))
        gama13 = x1_32 + l23 * (2.0 * x1_3 * ml3 + dx * (ml32 + l25 * ml3_2))
        gama31 = l23 * (2.0 * x3_1 * ml1 + dz * (-1.0 + l25 * ml1_2))
        gama22 = x2_22 + l23 * (2.0 * x2_2 * ml2 + dy * (ml22 + l25 * ml2_2))
        gama23 = x2_32 + l23 * (2.0 * x2_3 * ml3 + dy * (ml32 + l25 * ml3_2))
        gama32 = x3_22 + l23 * (2.0 * x3_2 * ml2 + dz * (ml22 + l25 * ml2_2))
        gama33 = x3_32 + l23 * (2.0 * x3_3 * ml3 + dz * (ml32 + l25 * ml3_2))

        xx_2 = xx * xx; yy_2 = yy * yy; zz_2 = zz * zz

        # ---- V: potential ----
        K000 = (1.0 / l) * toi
        corr = (xx_2 * (toi12 * l2 + toi1 * b1 + toi * gama1)
                 + yy_2 * (toi22 * l2 + toi2 * b2 + toi * gama2)
                 + zz_2 * (0.0 + 0.0 + toi * gama3))
        pot = c * (K000 + (1.0 / 24.0) * l3 * corr)
        sum_V += pot

        # ---- Vx, Vy, Vz: 1st derivatives ----
        Lx = l3 * dx * toi
        Ly = l3 * dy * toi
        Lz = l3 * dz * toi

        gx = c * (Lx + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a1 + toi1 * b11 + toi * gama11)
            + yy_2 * (toi22 * a1 + toi2 * b12 + toi * gama12)
            + zz_2 * (0.0 + 0.0 + toi * gama13)))
        gy = c * (Ly + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a2 + toi1 * b21 + toi * gama21)
            + yy_2 * (toi22 * a2 + toi2 * b22 + toi * gama22)
            + zz_2 * (0.0 + 0.0 + toi * gama23)))
        gz = c * (Lz + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a3 + toi1 * b31 + toi * gama31)
            + yy_2 * (toi22 * a3 + toi2 * b32 + toi * gama32)
            + zz_2 * (0.0 + 0.0 + toi * gama33)))
        sum_Vx -= gx
        sum_Vy -= gy
        sum_Vz -= gz

    return (sum_V * 1e-5, sum_Vx, sum_Vy, sum_Vz)


# ------------------------------------------------------------------
# itype=2: 2nd derivatives only (full b_iJk + gamaijk required)
# ------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _tess_kernel_2(lon1, lat1, h1, ref,
                    lon, lat, r0, rx, ry, e):
    """
    Tesseroid kernel — 2nd derivatives only (itype=2).

    Same computation as _tess_kernel_10 but returns only (Vxx,Vxy,Vxz,Vyy,Vyz,Vzz).

    Returns:
        (Vxx, Vxy, Vxz, Vyy, Vyz, Vzz)
    """
    N = lon1.shape[0]
    if N == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    lat_r = lat * _D2R
    coslat = math.cos(lat_r)
    sinlat = math.sin(lat_r)

    sum_Vxx = 0.0; sum_Vxy = 0.0; sum_Vxz = 0.0
    sum_Vyy = 0.0; sum_Vyz = 0.0; sum_Vzz = 0.0

    half_rx = rx * 0.5
    half_ry = ry * 0.5
    d2r = _D2R
    G = _G

    for i in range(N):
        zz1 = lon1[i] - half_rx
        zz2 = lon1[i] + half_rx
        yy1 = lat1[i] - half_ry
        yy2 = lat1[i] + half_ry
        xx1 = h1[i]
        xx2 = ref[i]
        r1 = (xx1 + xx2) * 0.5

        lat1_r = lat1[i] * d2r
        coslat1 = math.cos(lat1_r)
        sinlat1 = math.sin(lat1_r)

        d_lon = (lon1[i] - lon) * d2r
        cosdlon = math.cos(d_lon)
        sindlon = math.sin(d_lon)

        cosphicosa = coslat1 * cosdlon
        coslatsinlat1 = coslat * sinlat1
        sinphicosa = coslatsinlat1 - sinlat * cosphicosa
        sinphisina = coslat1 * sindlon
        sinlatsinlat1 = sinlat * sinlat1
        cosphi = sinlatsinlat1 + coslat * cosphicosa

        dx = r1 * sinphicosa
        dy = r1 * sinphisina
        dz = r1 * cosphi - r0

        dxx = dx * dx; dxy = dx * dy; dxz = dx * dz
        dyy = dy * dy; dyz = dy * dz; dzz = dz * dz

        l = math.sqrt(dxx + dyy + dzz)
        if l < 1e-300:
            continue

        xx = xx2 - xx1
        yy = (yy2 - yy1) * d2r
        zz = (zz2 - zz1) * d2r
        c = G * e[i] * zz * xx * yy
        l2 = l * l; l3 = 1.0 / (l2 * l)

        r12 = r1 * r1
        toi = r12 * coslat1
        toi1 = 2.0 * r1 * coslat1
        toi2 = -r12 * sinlat1
        toi12 = 2.0 * coslat1
        toi22 = -toi

        x1_1 = sinphicosa; x2_1 = sinphisina; x3_1 = cosphi
        x1_2 = r1 * (coslat * coslat1 + sinlatsinlat1 * cosdlon)
        x2_2 = -r1 * sinlat1 * sindlon
        x3_2 = r1 * (sinlat * coslat1 - coslatsinlat1 * cosdlon)
        x1_22 = -dx; x2_22 = -dy; x3_22 = -r1 * cosphi
        x1_3 = dy * sinlat; x2_3 = r1 * cosphicosa; x3_3 = -dy * coslat
        x1_32 = x2_3 * sinlat; x2_32 = -dy; x3_32 = -x2_3 * coslat

        ml1 = r0 * cosphi - r1; ml2 = r0 * x3_2; ml3 = r0 * x3_3
        ml22 = r0 * x3_22; ml32 = r0 * x3_32

        l23 = 3.0 / l2
        a1 = dx; a2 = dy; a3 = dz
        a11 = l23 * dxx - 1.0; a12 = l23 * dxy; a13 = l23 * dxz
        a22 = l23 * dyy - 1.0; a23 = l23 * dyz; a33 = l23 * dzz - 1.0

        b1 = 2.0 * ml1; b2 = 2.0 * ml2; b3 = 2.0 * ml3
        b11 = 2.0 * (x1_1 + l23 * dx * ml1)
        b12 = 2.0 * (x1_2 + l23 * dx * ml2)
        b21 = 2.0 * (x2_1 + l23 * dy * ml1)
        b13 = 2.0 * (x1_3 + l23 * dx * ml3)
        b31 = 2.0 * (x3_1 + l23 * dz * ml1)
        b22 = 2.0 * (x2_2 + l23 * dy * ml2)
        b23 = 2.0 * (x2_3 + l23 * dy * ml3)
        b32 = 2.0 * (x3_2 + l23 * dz * ml2)
        b33 = 2.0 * (x3_3 + l23 * dz * ml3)

        ml1_2 = ml1 * ml1; ml2_2 = ml2 * ml2; ml3_2 = ml3 * ml3
        l25 = 5.0 / l2

        gama1 = -1.0 + l23 * ml1_2
        gama2 = ml22 + l23 * ml2_2
        gama3 = ml32 + l23 * ml3_2

        gama11 = l23 * (2.0 * x1_1 * ml1 + dx * (-1.0 + l25 * ml1_2))
        gama12 = x1_22 + l23 * (2.0 * x1_2 * ml2 + dx * (ml22 + l25 * ml2_2))
        gama21 = l23 * (2.0 * x2_1 * ml1 + dy * (-1.0 + l25 * ml1_2))
        gama13 = x1_32 + l23 * (2.0 * x1_3 * ml3 + dx * (ml32 + l25 * ml3_2))
        gama31 = l23 * (2.0 * x3_1 * ml1 + dz * (-1.0 + l25 * ml1_2))
        gama22 = x2_22 + l23 * (2.0 * x2_2 * ml2 + dy * (ml22 + l25 * ml2_2))
        gama23 = x2_32 + l23 * (2.0 * x2_3 * ml3 + dy * (ml32 + l25 * ml3_2))
        gama32 = x3_22 + l23 * (2.0 * x3_2 * ml2 + dz * (ml22 + l25 * ml2_2))
        gama33 = x3_32 + l23 * (2.0 * x3_3 * ml3 + dz * (ml32 + l25 * ml3_2))

        # ---- b_iJk (3rd-order) ----
        const_1 = 6.0 / l2
        dxx5 = 5.0 * dxx; dxy5 = 5.0 * dxy; dxz5 = 5.0 * dxz
        dyy5 = 5.0 * dyy; dyz5 = 5.0 * dyz; dzz5 = 5.0 * dzz

        dx_x11 = dx * x1_1; dx_x12 = dx * x1_2; dx_x13 = dx * x1_3
        dx_x21 = dx * x2_1; dx_x22 = dx * x2_2; dx_x23 = dx * x2_3
        dy_x11 = dy * x1_1; dy_x12 = dy * x1_2; dy_x13 = dy * x1_3
        dz_x11 = dz * x1_1; dz_x12 = dz * x1_2; dz_x13 = dz * x1_3
        dy_x21 = dy * x2_1; dy_x22 = dy * x2_2; dy_x23 = dy * x2_3
        dz_x21 = dz * x2_1; dz_x22 = dz * x2_2; dz_x23 = dz * x2_3
        dx_x31 = dx * x3_1; dx_x32 = dx * x3_2; dx_x33 = dx * x3_3
        dy_x31 = dy * x3_1; dy_x32 = dy * x3_2; dy_x33 = dy * x3_3
        dz_x31 = dz * x3_1; dz_x32 = dz * x3_2; dz_x33 = dz * x3_3

        b111 = const_1 * (dx_x11 + dx_x11 + dxx5 * ml1 / l2 - ml1)
        b112 = const_1 * (dx_x12 + dx_x12 + dxx5 * ml2 / l2 - ml2)
        b113 = const_1 * (dx_x13 + dx_x13 + dxx5 * ml3 / l2 - ml3)
        b121 = const_1 * (dx_x21 + dy_x11 + dxy5 * ml1 / l2)
        b122 = const_1 * (dx_x22 + dy_x12 + dxy5 * ml2 / l2)
        b123 = const_1 * (dx_x23 + dy_x13 + dxy5 * ml3 / l2)
        b131 = const_1 * (dx_x31 + dz_x11 + dxz5 * ml1 / l2)
        b132 = const_1 * (dx_x32 + dz_x12 + dxz5 * ml2 / l2)
        b133 = const_1 * (dx_x33 + dz_x13 + dxz5 * ml3 / l2)
        b221 = const_1 * (dy_x21 + dy_x21 + dyy5 * ml1 / l2 - ml1)
        b222 = const_1 * (dy_x22 + dy_x22 + dyy5 * ml2 / l2 - ml2)
        b223 = const_1 * (dy_x23 + dy_x23 + dyy5 * ml3 / l2 - ml3)
        b231 = const_1 * (dy_x31 + dz_x21 + dyz5 * ml1 / l2)
        b232 = const_1 * (dy_x32 + dz_x22 + dyz5 * ml2 / l2)
        b233 = const_1 * (dy_x33 + dz_x23 + dyz5 * ml3 / l2)
        b331 = const_1 * (dz_x31 + dz_x31 + dzz5 * ml1 / l2 - ml1)
        b332 = const_1 * (dz_x32 + dz_x32 + dzz5 * ml2 / l2 - ml2)
        b333 = const_1 * (dz_x33 + dz_x33 + dzz5 * ml3 / l2 - ml3)

        # ---- gamaijk (4th-order) ----
        dx_x122 = dx * x1_22; dx_x132 = dx * x1_32
        dx_x222 = dx * x2_22; dx_x232 = dx * x2_32
        dx_x322 = dx * x3_22; dx_x332 = dx * x3_32
        dy_x122 = dy * x1_22; dy_x132 = dy * x1_32
        dy_x222 = dy * x2_22; dy_x232 = dy * x2_32
        dy_x322 = dy * x3_22; dy_x332 = dy * x3_32
        dz_x122 = dz * x1_22; dz_x132 = dz * x1_32
        dz_x222 = dz * x2_22; dz_x232 = dz * x2_32
        dz_x322 = dz * x3_22; dz_x332 = dz * x3_32

        l25 = 5.0 / l2; l27 = 7.0 / l2

        gama111 = l23 * (2.0 * x1_1 * x1_1
                         + l25 * (b1 * (2.0 * dx_x11) + dxx * (-1.0 + l27 * ml1_2))
                         - (-1.0 + l25 * ml1_2))
        gama112 = l23 * (2.0 * dx_x122 + 2.0 * x1_2 * x1_2
                         + l25 * (b2 * (2.0 * dx_x12) + dxx * (ml22 + l27 * ml2_2))
                         - (ml22 + l25 * ml2_2))
        gama113 = l23 * (2.0 * dx_x132 + 2.0 * x1_3 * x1_3
                         + l25 * (b3 * (2.0 * dx_x13) + dxx * (ml32 + l27 * ml3_2))
                         - (ml32 + l25 * ml3_2))
        gama121 = l23 * (2.0 * x1_1 * x2_1
                         + l25 * (b1 * (dx_x21 + dy_x11) + dxy * (-1.0 + l27 * ml1_2)))
        gama122 = l23 * (dx_x222 + dy_x122 + 2.0 * x1_2 * x2_2
                         + l25 * (b2 * (dx_x22 + dy_x12) + dxy * (ml22 + l27 * ml2_2)))
        gama123 = l23 * (dx_x232 + dy_x132 + 2.0 * x1_3 * x2_3
                         + l25 * (b3 * (dx_x23 + dy_x13) + dxy * (ml32 + l27 * ml3_2)))
        gama131 = l23 * (2.0 * x1_1 * x3_1
                         + l25 * (b1 * (dx_x31 + dz_x11) + dxz * (-1.0 + l27 * ml1_2)))
        gama132 = l23 * (dx_x322 + dz_x122 + 2.0 * x1_2 * x3_2
                         + l25 * (b2 * (dx_x32 + dz_x12) + dxz * (ml22 + l27 * ml2_2)))
        gama133 = l23 * (dx_x332 + dz_x132 + 2.0 * x1_3 * x3_3
                         + l25 * (b3 * (dx_x33 + dz_x13) + dxz * (ml32 + l27 * ml3_2)))
        gama221 = l23 * (2.0 * x2_1 * x2_1
                         + l25 * (b1 * (2.0 * dy_x21) + dyy * (-1.0 + l27 * ml1_2))
                         - (-1.0 + l25 * ml1_2))
        gama222 = l23 * (2.0 * dy_x222 + 2.0 * x2_2 * x2_2
                         + l25 * (b2 * (2.0 * dy_x22) + dyy * (ml22 + l27 * ml2_2))
                         - (ml22 + l25 * ml2_2))
        gama223 = l23 * (2.0 * dy_x232 + 2.0 * x2_3 * x2_3
                         + l25 * (b3 * (2.0 * dy_x23) + dyy * (ml32 + l27 * ml3_2))
                         - (ml32 + l25 * ml3_2))
        gama231 = l23 * (2.0 * x2_1 * x3_1
                         + l25 * (b1 * (dy_x31 + dz_x21) + dyz * (-1.0 + l27 * ml1_2)))
        gama232 = l23 * (dy_x322 + dz_x222 + 2.0 * x2_2 * x3_2
                         + l25 * (b2 * (dy_x32 + dz_x22) + dyz * (ml22 + l27 * ml2_2)))
        gama233 = l23 * (dy_x332 + dz_x232 + 2.0 * x2_3 * x3_3
                         + l25 * (b3 * (dy_x33 + dz_x23) + dyz * (ml32 + l27 * ml3_2)))
        gama331 = l23 * (2.0 * x3_1 * x3_1
                         + l25 * (b1 * (2.0 * dz_x31) + dzz * (-1.0 + l27 * ml1_2))
                         - (-1.0 + l25 * ml1_2))
        gama332 = l23 * (2.0 * dz_x322 + 2.0 * x3_2 * x3_2
                         + l25 * (b2 * (2.0 * dz_x32) + dzz * (ml22 + l27 * ml2_2))
                         - (ml22 + l25 * ml2_2))
        gama333 = l23 * (2.0 * dz_x332 + 2.0 * x3_3 * x3_3
                         + l25 * (b3 * (2.0 * dz_x33) + dzz * (ml32 + l27 * ml3_2))
                         - (ml32 + l25 * ml3_2))

        xx_2 = xx * xx; yy_2 = yy * yy; zz_2 = zz * zz

        # ---- 2nd derivatives ----
        Kxx = l3 * (l23 * dxx - 1.0) * toi
        Kxy = l3 * l23 * dxy * toi
        Kxz = l3 * l23 * dxz * toi
        Kyy = l3 * (l23 * dyy - 1.0) * toi
        Kyz = l3 * l23 * dyz * toi
        Kzz = l3 * (l23 * dzz - 1.0) * toi

        Lxx = c * (Kxx + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a11 + toi1 * b111 + toi * gama111)
            + yy_2 * (toi22 * a11 + toi2 * b112 + toi * gama112)
            + zz_2 * (0.0 + 0.0 + toi * gama113)))
        Lxy = c * (Kxy + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a12 + toi1 * b121 + toi * gama121)
            + yy_2 * (toi22 * a12 + toi2 * b122 + toi * gama122)
            + zz_2 * (0.0 + 0.0 + toi * gama123)))
        Lxz = c * (Kxz + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a13 + toi1 * b131 + toi * gama131)
            + yy_2 * (toi22 * a13 + toi2 * b132 + toi * gama132)
            + zz_2 * (0.0 + 0.0 + toi * gama133)))
        Lyy = c * (Kyy + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a22 + toi1 * b221 + toi * gama221)
            + yy_2 * (toi22 * a22 + toi2 * b222 + toi * gama222)
            + zz_2 * (0.0 + 0.0 + toi * gama223)))
        Lyz = c * (Kyz + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a23 + toi1 * b231 + toi * gama231)
            + yy_2 * (toi22 * a23 + toi2 * b232 + toi * gama232)
            + zz_2 * (0.0 + 0.0 + toi * gama233)))
        Lzz = c * (Kzz + (1.0 / 24.0) * l3 * (
            xx_2 * (toi12 * a33 + toi1 * b331 + toi * gama331)
            + yy_2 * (toi22 * a33 + toi2 * b332 + toi * gama332)
            + zz_2 * (0.0 + 0.0 + toi * gama333)))

        sum_Vxx += Lxx; sum_Vxy += Lxy; sum_Vxz += Lxz
        sum_Vyy += Lyy; sum_Vyz += Lyz; sum_Vzz += Lzz

    return (sum_Vxx * 1e4, sum_Vxy * 1e4, sum_Vxz * 1e4,
            sum_Vyy * 1e4, sum_Vyz * 1e4, sum_Vzz * 1e4)


# ------------------------------------------------------------------
# Public wrappers matching tesseroid.py signatures
# ------------------------------------------------------------------

def _prepare_arrays(lon1, lat1, h1, ref, e):
    """Ensure contiguous float64 arrays."""
    return (np.ascontiguousarray(lon1, dtype=np.float64),
            np.ascontiguousarray(lat1, dtype=np.float64),
            np.ascontiguousarray(h1, dtype=np.float64),
            np.ascontiguousarray(ref, dtype=np.float64),
            np.ascontiguousarray(e, dtype=np.float64))


def tess_kernel_g(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e):
    """Potential only (itype=0)."""
    lon1, lat1, h1, ref, e = _prepare_arrays(lon1, lat1, h1, ref, e)
    V, _, _, _ = _tess_kernel_04(lon1, lat1, h1, ref,
                                  float(lon), float(lat), float(r0),
                                  float(rx), float(ry), e)
    return (V,)


def tess_kernel_gx(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e):
    """1st derivatives only (itype=1)."""
    lon1, lat1, h1, ref, e = _prepare_arrays(lon1, lat1, h1, ref, e)
    _, Vx, Vy, Vz = _tess_kernel_04(lon1, lat1, h1, ref,
                                     float(lon), float(lat), float(r0),
                                     float(rx), float(ry), e)
    return (Vx, Vy, Vz)


def tess_kernel_g_gx(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e):
    """Potential + 1st derivatives (itype=4/103/104)."""
    lon1, lat1, h1, ref, e = _prepare_arrays(lon1, lat1, h1, ref, e)
    return _tess_kernel_04(lon1, lat1, h1, ref,
                            float(lon), float(lat), float(r0),
                            float(rx), float(ry), e)


def tess_kernel_gxx(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e):
    """2nd derivatives only (itype=2)."""
    lon1, lat1, h1, ref, e = _prepare_arrays(lon1, lat1, h1, ref, e)
    return _tess_kernel_2(lon1, lat1, h1, ref,
                           float(lon), float(lat), float(r0),
                           float(rx), float(ry), e)


def tess_kernel_10(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e):
    """All 10 components (itype=10)."""
    lon1, lat1, h1, ref, e = _prepare_arrays(lon1, lat1, h1, ref, e)
    return _tess_kernel_10(lon1, lat1, h1, ref,
                            float(lon), float(lat), float(r0),
                            float(rx), float(ry), e)
