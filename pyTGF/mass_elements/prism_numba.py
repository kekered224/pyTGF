"""
Numba-accelerated prism kernel (Zone 2, itype=10 — all 10 components).

Fuses tess2prism + global2local + prism geometry + g/gx/gy/gz/gxx/gxy/gxz/gyy/gyz/gzz
+ rotation (g_prism2point + ggt_prism2point) into a single @njit function.

Eliminates ~250 temporary numpy array allocations per call by operating on scalars
in a per-prism loop compiled by Numba.

Reference: Nagy, Papp, Benedek (2000/2002) J. Geodesy 74/76.
"""

import numpy as np
from numba import njit, prange
import math

# Physical constants matching PhysicalConstants
_PI = 3.141592654  # MATLAB uses this truncated value
_D2R = _PI / 180.0
_G = 6.67384e-3
_SING_THRESH = 0.05


@njit(cache=True, fastmath=False)
def _safe_atan2(y, x):
    """atan(y/x) with singular-point protection."""
    if abs(x) < 1e-300:
        return 0.0
    v = y / x
    if math.isfinite(v):
        return math.atan(v)
    return 0.0


@njit(cache=True, fastmath=False)
def _safe_log(v):
    """log(v) with NaN/Inf protection."""
    if v > 0 and math.isfinite(v):
        return math.log(v)
    return 0.0


@njit(cache=True, fastmath=False)
def _g6term_sum(r, x, y, z, nx_sing, ny_sing, nz_sing):
    """
    Compute the sum of 6 potential terms for one prism vertex.

    t1 = x*y*ln(z+r)
    t2 = y*z*ln(x+r)
    t3 = x*z*ln(y+r)
    t4 = -x^2/2 * atan(y*z/(x*r))
    t5 = -y^2/2 * atan(x*z/(y*r))
    t6 = -z^2/2 * atan(y*x/(z*r))

    Singular points zeroed per MATLAB convention.
    """
    t1 = x * y * _safe_log(z + r)
    t2 = y * z * _safe_log(x + r)
    t3 = x * z * _safe_log(y + r)
    t4 = -0.5 * x * x * _safe_atan2(y * z, x * r)
    t5 = -0.5 * y * y * _safe_atan2(x * z, y * r)
    t6 = -0.5 * z * z * _safe_atan2(y * x, z * r)

    # Zero singular contributions
    if nx_sing:
        t1 = 0.0; t3 = 0.0; t4 = 0.0
    if ny_sing:
        t1 = 0.0; t2 = 0.0; t5 = 0.0
    if nz_sing:
        t2 = 0.0; t3 = 0.0; t6 = 0.0

    return t1 + t2 + t3 + t4 + t5 + t6


@njit(cache=True, fastmath=False)
def _gz3term(r, x, y, z, nz_sing):
    """
    Compute -z * atan(y*x/(z*r)) for one vertex, zeroed at singular points.
    Used in gz/gy/gx calculations.
    """
    if nz_sing:
        return 0.0
    v = _safe_atan2(y * x, z * r)
    return -z * v


@njit(cache=True, fastmath=False)
def _prism_numba_kernel(x1, x2, y1, y2, d1, d2, e,
                        sta_lon, sta_lat, sta_r):
    """
    Full prism kernel: compute all 10 gravity components for N prisms.

    Args:
        x1, x2, y1, y2: Prism lon/lat bounds [deg], shape (N,)
        d1, d2: Upper/lower radial bounds [m], shape (N,)
        e: Density [g/cm^3], shape (N,)
        sta_lon, sta_lat, sta_r: Station coordinates (scalars)

    Returns:
        result[0..9] = [V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz]
    """
    N = x1.shape[0]

    # Station coords in radians (constant for all prisms)
    slon_rad = _D2R * sta_lon
    slat_rad = _D2R * sta_lat
    cos_slat = math.cos(slat_rad)
    sin_slat = math.sin(slat_rad)
    cos_slon = math.cos(slon_rad)
    sin_slon = math.sin(slon_rad)

    # Accumulators
    sum_g = 0.0
    sum_Vx = 0.0; sum_Vy = 0.0; sum_Vz = 0.0
    sum_Vxx = 0.0; sum_Vxy = 0.0; sum_Vxz = 0.0
    sum_Vyy = 0.0; sum_Vyz = 0.0; sum_Vzz = 0.0

    for i in range(N):
        # ---- tess2prism ----
        r1_val = d1[i]
        r2_val = d2[i]
        r0 = 0.5 * (r1_val + r2_val)
        dx = r0 * _D2R * (y2[i] - y1[i])
        half_lat_sum = _D2R * 0.5 * (y1[i] + y2[i])
        dy = r0 * math.cos(half_lat_sum) * _D2R * (x2[i] - x1[i])

        px1 = -0.5 * dx
        px2 = -px1
        py1 = -0.5 * dy
        py2 = -py1
        pz1 = (r2_val - r1_val) / 2.0
        pz2 = (r1_val - r2_val) / 2.0

        # Volume-equivalent density
        tess_vol = (_D2R * (x2[i] - x1[i]) *
                    (r1_val * r1_val * r1_val - r2_val * r2_val * r2_val) *
                    (math.sin(_D2R * y2[i]) - math.sin(_D2R * y1[i])) / 3.0)
        prism_vol = (px2 - px1) * (py2 - py1) * (pz2 - pz1)
        if abs(prism_vol) > 1e-300:
            prism_e = e[i] * tess_vol / prism_vol
            if not math.isfinite(prism_e):
                prism_e = e[i]
        else:
            prism_e = e[i]

        c = _G * prism_e

        prism_lon = 0.5 * (x1[i] + x2[i])
        prism_lat = 0.5 * (y1[i] + y2[i])

        # ---- global2local: station -> prism frame ----
        plon_rad = _D2R * prism_lon
        plat_rad = _D2R * prism_lat
        cos_plat = math.cos(plat_rad)
        sin_plat = math.sin(plat_rad)
        cos_plon = math.cos(plon_rad)
        sin_plon = math.sin(plon_rad)

        gx = sta_r * cos_slat * cos_slon - r0 * cos_plat * cos_plon
        gy = sta_r * cos_slat * sin_slon - r0 * cos_plat * sin_plon
        gz = sta_r * sin_slat - r0 * sin_plat

        sx = -gx * sin_plat * cos_plon - gy * sin_plat * sin_plon + gz * cos_plat
        sy = -gx * sin_plon + gy * cos_plon
        sz = -(gx * cos_plat * cos_plon + gy * cos_plat * sin_plon + gz * sin_plat)

        # ---- Vertex coordinates relative to station ----
        # MATLAB convention: zpos2 = prism_z1 - z; zpos1 = prism_z2 - z
        # (note the swap! prism_z1=(d2-d1)/2 is lower face, prism_z2=(d1-d2)/2 is upper face)
        xp1 = px1 - sx
        xp2 = px2 - sx
        yp1 = py1 - sy
        yp2 = py2 - sy
        zp1 = pz2 - sz  # upper face relative to station
        zp2 = pz1 - sz  # lower face relative to station

        # Singular point detection
        nx1_s = abs(xp1) <= _SING_THRESH
        nx2_s = abs(xp2) <= _SING_THRESH
        ny1_s = abs(yp1) <= _SING_THRESH
        ny2_s = abs(yp2) <= _SING_THRESH
        nz1_s = abs(zp1) <= _SING_THRESH
        nz2_s = abs(zp2) <= _SING_THRESH

        # ---- 8 vertex distances ----
        r1 = math.sqrt(xp1*xp1 + yp1*yp1 + zp1*zp1)
        r2 = math.sqrt(xp2*xp2 + yp1*yp1 + zp1*zp1)
        r3 = math.sqrt(xp1*xp1 + yp2*yp2 + zp1*zp1)
        r4 = math.sqrt(xp2*xp2 + yp2*yp2 + zp1*zp1)
        r5 = math.sqrt(xp1*xp1 + yp1*yp1 + zp2*zp2)
        r6 = math.sqrt(xp2*xp2 + yp1*yp1 + zp2*zp2)
        r7 = math.sqrt(xp1*xp1 + yp2*yp2 + zp2*zp2)
        r8 = math.sqrt(xp2*xp2 + yp2*yp2 + zp2*zp2)

        # ============================================================
        # POTENTIAL (g) — 8 calls to g6term, signs: + - - + - + + -
        # ============================================================
        res_g = 0.0
        res_g += _g6term_sum(r1, xp1, yp1, zp1, nx1_s, ny1_s, nz1_s)
        res_g -= _g6term_sum(r2, xp2, yp1, zp1, nx2_s, ny1_s, nz1_s)
        res_g -= _g6term_sum(r3, xp1, yp2, zp1, nx1_s, ny2_s, nz1_s)
        res_g += _g6term_sum(r4, xp2, yp2, zp1, nx2_s, ny2_s, nz1_s)
        res_g -= _g6term_sum(r5, xp1, yp1, zp2, nx1_s, ny1_s, nz2_s)
        res_g += _g6term_sum(r6, xp2, yp1, zp2, nx2_s, ny1_s, nz2_s)
        res_g += _g6term_sum(r7, xp1, yp2, zp2, nx1_s, ny2_s, nz2_s)
        res_g -= _g6term_sum(r8, xp2, yp2, zp2, nx2_s, ny2_s, nz2_s)
        sum_g += c * res_g

        # ============================================================
        # gz — INLINED log terms + gz3term (matches numpy _compute_gz)
        # ============================================================
        # s1 = xpos1 * log((ypos1+r1)*(ypos2+r7) / ((ypos2+r3)*(ypos1+r5)))
        _den = (yp2 + r3) * (yp1 + r5)
        if abs(_den) > 1e-300 and not nx1_s:
            _v = _safe_log((yp1 + r1) * (yp2 + r7) / _den)
            gz_s1 = xp1 * _v if math.isfinite(_v) else 0.0
        else:
            gz_s1 = 0.0
        # s2 = xpos2 * log((ypos1+r2)*(ypos2+r8) / ((ypos2+r4)*(ypos1+r6)))
        _den = (yp2 + r4) * (yp1 + r6)
        if abs(_den) > 1e-300 and not nx2_s:
            _v = _safe_log((yp1 + r2) * (yp2 + r8) / _den)
            gz_s2 = xp2 * _v if math.isfinite(_v) else 0.0
        else:
            gz_s2 = 0.0
        gz_s = gz_s1 - gz_s2
        # s1b = ypos1 * log((xpos1+r1)*(xpos2+r6) / ((xpos2+r2)*(xpos1+r5)))
        _den = (xp2 + r2) * (xp1 + r5)
        if abs(_den) > 1e-300 and not ny1_s:
            _v = _safe_log((xp1 + r1) * (xp2 + r6) / _den)
            gz_s1b = yp1 * _v if math.isfinite(_v) else 0.0
        else:
            gz_s1b = 0.0
        # s2b = ypos2 * log((xpos1+r3)*(xpos2+r8) / ((xpos2+r4)*(xpos1+r7)))
        _den = (xp2 + r4) * (xp1 + r7)
        if abs(_den) > 1e-300 and not ny2_s:
            _v = _safe_log((xp1 + r3) * (xp2 + r8) / _den)
            gz_s2b = yp2 * _v if math.isfinite(_v) else 0.0
        else:
            gz_s2b = 0.0
        gz_s1b_val = gz_s1b - gz_s2b
        gz_val = gz_s + gz_s1b_val

        # gz3term: + - - + - + + -
        gz_val += _gz3term(r1, xp1, yp1, zp1, nz1_s)
        gz_val -= _gz3term(r2, xp2, yp1, zp1, nz1_s)
        gz_val -= _gz3term(r3, xp1, yp2, zp1, nz1_s)
        gz_val += _gz3term(r4, xp2, yp2, zp1, nz1_s)
        gz_val -= _gz3term(r5, xp1, yp1, zp2, nz2_s)
        gz_val += _gz3term(r6, xp2, yp1, zp2, nz2_s)
        gz_val += _gz3term(r7, xp1, yp2, zp2, nz2_s)
        gz_val -= _gz3term(r8, xp2, yp2, zp2, nz2_s)
        gz_total = c * gz_val

        # ============================================================
        # gy — INLINED log terms + gz3term (matches numpy _compute_gy)
        # NOTE: s1b uses r4/r3 — UNIQUE pattern, not r7/r3 or r6/r2!
        # ============================================================
        # s1 = xpos1 * log((zpos1+r1)*(zpos2+r7) / ((zpos1+r3)*(zpos2+r5)))
        _den = (zp1 + r3) * (zp2 + r5)
        if abs(_den) > 1e-300 and not nx1_s:
            _v = _safe_log((zp1 + r1) * (zp2 + r7) / _den)
            gy_s1 = xp1 * _v if math.isfinite(_v) else 0.0
        else:
            gy_s1 = 0.0
        # s2 = xpos2 * log((zpos1+r2)*(zpos2+r8) / ((zpos1+r4)*(zpos2+r6)))
        _den = (zp1 + r4) * (zp2 + r6)
        if abs(_den) > 1e-300 and not nx2_s:
            _v = _safe_log((zp1 + r2) * (zp2 + r8) / _den)
            gy_s2 = xp2 * _v if math.isfinite(_v) else 0.0
        else:
            gy_s2 = 0.0
        gy_s = gy_s1 - gy_s2
        # s1b = zpos1 * log((xpos1+r1)*(xpos2+r4) / ((xpos2+r2)*(xpos1+r3)))  *** r4/r3 ***
        _den = (xp2 + r2) * (xp1 + r3)
        if abs(_den) > 1e-300 and not nz1_s:
            _v = _safe_log((xp1 + r1) * (xp2 + r4) / _den)
            gy_s1b = zp1 * _v if math.isfinite(_v) else 0.0
        else:
            gy_s1b = 0.0
        # s2b = zpos2 * log((xpos1+r5)*(xpos2+r8) / ((xpos2+r6)*(xpos1+r7)))
        _den = (xp2 + r6) * (xp1 + r7)
        if abs(_den) > 1e-300 and not nz2_s:
            _v = _safe_log((xp1 + r5) * (xp2 + r8) / _den)
            gy_s2b = zp2 * _v if math.isfinite(_v) else 0.0
        else:
            gy_s2b = 0.0
        gy_s1b_val = gy_s1b - gy_s2b
        gy_val = gy_s + gy_s1b_val

        # gz3term with swapped args: (r, xp, zp, yp, ny)
        gy_val += _gz3term(r1, xp1, zp1, yp1, ny1_s)
        gy_val -= _gz3term(r2, xp2, zp1, yp1, ny1_s)
        gy_val -= _gz3term(r3, xp1, zp1, yp2, ny2_s)
        gy_val += _gz3term(r4, xp2, zp1, yp2, ny2_s)
        gy_val -= _gz3term(r5, xp1, zp2, yp1, ny1_s)
        gy_val += _gz3term(r6, xp2, zp2, yp1, ny1_s)
        gy_val += _gz3term(r7, xp1, zp2, yp2, ny2_s)
        gy_val -= _gz3term(r8, xp2, zp2, yp2, ny2_s)
        gy_total = -c * gy_val  # Note: gy has leading minus

        # ============================================================
        # gx — INLINED log terms + gz3term (matches numpy _compute_gx)
        # NOTE: s-part uses r6/r2 — UNIQUE pattern!
        # ============================================================
        # s1 = ypos1 * log((zpos1+r1)*(zpos2+r6) / ((zpos1+r2)*(zpos2+r5)))
        _den = (zp1 + r2) * (zp2 + r5)
        if abs(_den) > 1e-300 and not ny1_s:
            _v = _safe_log((zp1 + r1) * (zp2 + r6) / _den)
            gx_s1 = yp1 * _v if math.isfinite(_v) else 0.0
        else:
            gx_s1 = 0.0
        # s2 = ypos2 * log((zpos1+r3)*(zpos2+r8) / ((zpos1+r4)*(zpos2+r7)))
        _den = (zp1 + r4) * (zp2 + r7)
        if abs(_den) > 1e-300 and not ny2_s:
            _v = _safe_log((zp1 + r3) * (zp2 + r8) / _den)
            gx_s2 = yp2 * _v if math.isfinite(_v) else 0.0
        else:
            gx_s2 = 0.0
        gx_s = gx_s1 - gx_s2
        # s1b = zpos1 * log((ypos1+r1)*(ypos2+r4) / ((ypos1+r2)*(ypos2+r3)))
        _den = (yp1 + r2) * (yp2 + r3)
        if abs(_den) > 1e-300 and not nz1_s:
            _v = _safe_log((yp1 + r1) * (yp2 + r4) / _den)
            gx_s1b = zp1 * _v if math.isfinite(_v) else 0.0
        else:
            gx_s1b = 0.0
        # s2b = zpos2 * log((ypos1+r5)*(ypos2+r8) / ((ypos2+r7)*(ypos1+r6)))
        _den = (yp2 + r7) * (yp1 + r6)
        if abs(_den) > 1e-300 and not nz2_s:
            _v = _safe_log((yp1 + r5) * (yp2 + r8) / _den)
            gx_s2b = zp2 * _v if math.isfinite(_v) else 0.0
        else:
            gx_s2b = 0.0
        gx_s1b_val = gx_s1b - gx_s2b
        gx_val = gx_s + gx_s1b_val

        # gz3term with swapped args: (r, yp, zp, xp, nx)
        gx_val += _gz3term(r1, yp1, zp1, xp1, nx1_s)
        gx_val -= _gz3term(r2, yp1, zp1, xp2, nx2_s)
        gx_val -= _gz3term(r3, yp2, zp1, xp1, nx1_s)
        gx_val += _gz3term(r4, yp2, zp1, xp2, nx2_s)
        gx_val -= _gz3term(r5, yp1, zp2, xp1, nx1_s)
        gx_val += _gz3term(r6, yp1, zp2, xp2, nx2_s)
        gx_val += _gz3term(r7, yp2, zp2, xp1, nx1_s)
        gx_val -= _gz3term(r8, yp2, zp2, xp2, nx2_s)
        gx_total = -c * gx_val  # Note: gx has leading minus

        # ============================================================
        # Vxx — 8 arctan terms: atan(z*y/(x*r)), signs + - - + - + + -
        # Singular index: nx (x-direction) — per-vertex zeroing
        # ============================================================
        vxx = 0.0
        v = _safe_atan2(zp1 * yp1, xp1 * r1)
        vxx += v if not nx1_s else 0.0
        v = _safe_atan2(zp1 * yp1, xp2 * r2)
        vxx -= v if not nx2_s else 0.0
        v = _safe_atan2(zp1 * yp2, xp1 * r3)
        vxx -= v if not nx1_s else 0.0
        v = _safe_atan2(zp1 * yp2, xp2 * r4)
        vxx += v if not nx2_s else 0.0
        v = _safe_atan2(zp2 * yp1, xp1 * r5)
        vxx -= v if not nx1_s else 0.0
        v = _safe_atan2(zp2 * yp1, xp2 * r6)
        vxx += v if not nx2_s else 0.0
        v = _safe_atan2(zp2 * yp2, xp1 * r7)
        vxx += v if not nx1_s else 0.0
        v = _safe_atan2(zp2 * yp2, xp2 * r8)
        vxx -= v if not nx2_s else 0.0
        gxx = (c * vxx) * 1e4

        # ============================================================
        # Vyy — 8 arctan terms: atan(z*x/(y*r)), signs + - - + - + + -
        # Singular index: ny — per-vertex zeroing
        # ============================================================
        vyy = 0.0
        v = _safe_atan2(zp1 * xp1, yp1 * r1)
        vyy += v if not ny1_s else 0.0
        v = _safe_atan2(zp1 * xp2, yp1 * r2)
        vyy -= v if not ny1_s else 0.0
        v = _safe_atan2(zp1 * xp1, yp2 * r3)
        vyy -= v if not ny2_s else 0.0
        v = _safe_atan2(zp1 * xp2, yp2 * r4)
        vyy += v if not ny2_s else 0.0
        v = _safe_atan2(zp2 * xp1, yp1 * r5)
        vyy -= v if not ny1_s else 0.0
        v = _safe_atan2(zp2 * xp2, yp1 * r6)
        vyy += v if not ny1_s else 0.0
        v = _safe_atan2(zp2 * xp1, yp2 * r7)
        vyy += v if not ny2_s else 0.0
        v = _safe_atan2(zp2 * xp2, yp2 * r8)
        vyy -= v if not ny2_s else 0.0
        gyy = (c * vyy) * 1e4

        # ============================================================
        # Vzz — 8 arctan terms: atan(x*y/(z*r)), signs + - - + - + + -
        # Singular index: nz — per-vertex zeroing
        # ============================================================
        vzz = 0.0
        v = _safe_atan2(xp1 * yp1, zp1 * r1)
        vzz += v if not nz1_s else 0.0
        v = _safe_atan2(xp2 * yp1, zp1 * r2)
        vzz -= v if not nz1_s else 0.0
        v = _safe_atan2(xp1 * yp2, zp1 * r3)
        vzz -= v if not nz1_s else 0.0
        v = _safe_atan2(xp2 * yp2, zp1 * r4)
        vzz += v if not nz1_s else 0.0
        v = _safe_atan2(xp1 * yp1, zp2 * r5)
        vzz -= v if not nz2_s else 0.0
        v = _safe_atan2(xp2 * yp1, zp2 * r6)
        vzz += v if not nz2_s else 0.0
        v = _safe_atan2(xp1 * yp2, zp2 * r7)
        vzz += v if not nz2_s else 0.0
        v = _safe_atan2(xp2 * yp2, zp2 * r8)
        vzz -= v if not nz2_s else 0.0
        gzz = (c * vzz) * 1e4

        # ============================================================
        # Vxy — log of ratio products
        # ============================================================
        s1_xy = (r1 + zp1) * (r4 + zp1) * (r6 + zp2) * (r7 + zp2)
        s2_xy = (r2 + zp1) * (r3 + zp1) * (r5 + zp2) * (r8 + zp2)
        if abs(s2_xy) > 0.001 and abs(s1_xy) > 0.001:
            v = _safe_log(s1_xy / s2_xy)
            if math.isfinite(v):
                gxy = -c * v * 1e4
            else:
                gxy = 0.0
        else:
            gxy = 0.0

        # ============================================================
        # Vxz — log of ratio products
        # ============================================================
        s1_xz = (r1 + yp1) * (r4 + yp2) * (r6 + yp1) * (r7 + yp2)
        s2_xz = (r2 + yp1) * (r3 + yp2) * (r5 + yp1) * (r8 + yp2)
        if abs(s2_xz) > 0.001 and abs(s1_xz) > 0.001:
            v = _safe_log(s1_xz / s2_xz)
            if math.isfinite(v):
                gxz = c * v * 1e4
            else:
                gxz = 0.0
        else:
            gxz = 0.0

        # ============================================================
        # Vyz — log of ratio products
        # ============================================================
        s1_yz = (r1 + xp1) * (r4 + xp2) * (r6 + xp2) * (r7 + xp1)
        s2_yz = (r2 + xp2) * (r3 + xp1) * (r5 + xp1) * (r8 + xp2)
        if abs(s2_yz) > 0.001 and abs(s1_yz) > 0.001:
            v = _safe_log(s1_yz / s2_yz)
            if math.isfinite(v):
                gyz = c * v * 1e4
            else:
                gyz = 0.0
        else:
            gyz = 0.0

        # ============================================================
        # ROTATION: prism frame -> station frame
        # ============================================================
        # --- g_prism2point: rotate (gx, gy, gz) ---
        delta_lon = plon_rad - slon_rad
        cosbeta = math.cos(delta_lon)
        sinbeta = math.sin(delta_lon)

        cosbetasinphi = cosbeta * sin_slat
        cosbetacosphi = cosbeta * cos_slat

        R0 = cosbetasinphi * sin_plat + cos_slat * cos_plat
        R1 = sinbeta * sin_slat
        R2 = -cosbetasinphi * cos_plat + cos_slat * sin_plat
        R3 = -sinbeta * sin_plat
        R4 = cosbeta
        R5 = sinbeta * cos_plat
        R6 = -cosbetacosphi * sin_plat + sin_slat * cos_plat
        R7 = -sinbeta * cos_slat
        R8 = cosbetacosphi * cos_plat + sin_slat * sin_plat

        sum_Vx += R0 * gx_total + R1 * gy_total + R2 * gz_total
        sum_Vy += R3 * gx_total + R4 * gy_total + R5 * gz_total
        sum_Vz += R6 * gx_total + R7 * gy_total + R8 * gz_total

        # --- ggt_prism2point: rotate (gxx, gxy, gxz, gyy, gyz, gzz) ---
        # Reconstruct full 3x3 tensor (symmetric)
        a0 = gxx; a1 = gxy; a2 = gxz
        a3 = gxy; a4 = gyy; a5 = gyz
        a6 = gxz; a7 = gyz; a8 = gzz

        # T = R * A
        t0 = R0 * a0 + R1 * a3 + R2 * a6
        t1 = R0 * a1 + R1 * a4 + R2 * a7
        t2 = R0 * a2 + R1 * a5 + R2 * a8
        t3 = R3 * a0 + R4 * a3 + R5 * a6
        t4 = R3 * a1 + R4 * a4 + R5 * a7
        t5 = R3 * a2 + R4 * a5 + R5 * a8
        t6 = R6 * a0 + R7 * a3 + R8 * a6
        t7 = R6 * a1 + R7 * a4 + R8 * a7
        t8 = R6 * a2 + R7 * a5 + R8 * a8

        # V = T * R^T (extract 6 unique components)
        sum_Vxx += t0 * R0 + t1 * R1 + t2 * R2
        sum_Vxy += t0 * R3 + t1 * R4 + t2 * R5
        sum_Vxz += t0 * R6 + t1 * R7 + t2 * R8
        sum_Vyy += t3 * R3 + t4 * R4 + t5 * R5
        sum_Vyz += t6 * R3 + t7 * R4 + t8 * R5
        sum_Vzz += t6 * R6 + t7 * R7 + t8 * R8

    # Apply scaling factors
    V = -sum_g * 1e-5

    result = np.empty(10, dtype=np.float64)
    result[0] = V
    result[1] = sum_Vx
    result[2] = sum_Vy
    result[3] = sum_Vz
    result[4] = sum_Vxx
    result[5] = sum_Vxy
    result[6] = sum_Vxz
    result[7] = sum_Vyy
    result[8] = sum_Vyz
    result[9] = sum_Vzz
    return result


def prism_numba(x1, x2, y1, y2, d1, d2,
                lon, lat, r0, e,
                resolution_x, resolution_y):
    """
    Numba-accelerated prism function for itype=10.

    Same interface as prism() — returns tuple of 10 floats.
    """
    # Ensure contiguous float64 arrays
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    y1 = np.ascontiguousarray(y1, dtype=np.float64)
    y2 = np.ascontiguousarray(y2, dtype=np.float64)
    d1 = np.ascontiguousarray(d1, dtype=np.float64)
    d2 = np.ascontiguousarray(d2, dtype=np.float64)
    e = np.ascontiguousarray(e, dtype=np.float64)

    result = _prism_numba_kernel(x1, x2, y1, y2, d1, d2, e,
                                 float(lon), float(lat), float(r0))
    return (float(result[0]), float(result[1]), float(result[2]),
            float(result[3]), float(result[4]), float(result[5]),
            float(result[6]), float(result[7]), float(result[8]),
            float(result[9]))
