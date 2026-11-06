"""
Polyhedron mass element (Zone 1, nearest, most precise)

Mirrors TC_GUI.m:
    Polyhedron_full (L2460-L2628)  all 10 components, itype=10
    Polyhedron_g    (L2630-L2783)  potential only, itype=0
    Polyhedron_g_gx (L2785-L2934)  potential + 1st derivs, itype=4/103/104
    Polyhedron_gx   (L2948-L3095)  1st derivatives, itype=1
    Polyhedron_gxx  (L3110-L3260)  2nd derivatives, itype=2

Reference: D. Tsoulis (2012) Geophysics 77(2), F1-F11.
           "Analytical computation of the full gravity tensor of a
            uniformly regularized polyhedral source"

The polyhedron is an 8-vertex "brick" (top face 1-2-3-4, bottom face 5-6-7-8)
constructed from a DEM cell. The computation point P is translated to the
origin of a local Cartesian frame. The algorithm decomposes the polyhedron
into 7 planar faces (2 top triangles, 1 bottom quad, 4 side quads) and
evaluates the line-integral formulas (Eqs. 10-12 of Tsoulis 2012):

    V   = sum_p  s_p * h_p * ( sum_q h_pq*c1 + h_p*sum_q c2 - corr )
    Vi  = sum_p  cs_p,i * ( sum_q h_pq*c1 + h_p*sum_q c2 - corr )
    Vij = sum_p  cs_p,i * ( sum_q cspq_pq,j*c1 + s_p*cs_p,j*(sum_q c2 - corr') )

where c1 = ln((s2+r2)/(s1+r1)), c2 = atan(h*t/(hp*r2))-atan(h*s/(hp*r1)),
and corr accounts for singular configurations (P' inside face / on edge / vertex).

MATLAB calls Fortran mex routines (polyhedron_full.F etc.); this is a native
Python reimplementation of the same algorithm.
"""

from typing import Tuple
import numpy as np
from numba import njit

from ..config import PhysicalConstants
from ..utils.coordinate import global2local, g_prism2point, ggt_prism2point


# ---------------------------------------------------------------------------
# Topology of the 8-vertex polyhedron: 7 faces, vertex indices per face.
# Matches Fortran polyhedron_full.F lines 104-110.
#   Face 1: top triangle  (1,2,3)
#   Face 2: top triangle  (3,4,1)
#   Face 3: bottom quad   (6,5,8,7)
#   Face 4: side quad     (8,5,1,4)
#   Face 5: side quad     (6,7,3,2)
#   Face 6: side quad     (5,6,2,1)
#   Face 7: side quad     (7,8,4,3)
# (indices are 1-based in Fortran; we keep 1-based here and subtract when
#  indexing into Python arrays.)
# ---------------------------------------------------------------------------
_NOE = np.array([3, 3, 4, 4, 4, 4, 4], dtype=np.int64)  # edges per face
_K = np.array([
    [1, 2, 3, 0],   # face 1
    [3, 4, 1, 0],   # face 2
    [6, 5, 8, 7],   # face 3
    [8, 5, 1, 4],   # face 4
    [6, 7, 3, 2],   # face 5
    [5, 6, 2, 1],   # face 6
    [7, 8, 4, 3],   # face 7
], dtype=np.int64)  # 1-based vertex indices; unused entries are 0
_NOP = 7
_NOV = 8


# ---------------------------------------------------------------------------
# Small vector helpers (mirror Fortran statement functions snorm/fln/fan)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _snorm(a, b, c):
    return np.sqrt(a * a + b * b + c * c)


@njit(cache=True)
def _fln(a, b, c, d):
    """ln((a+b)/(c+d)). Returns NaN for non-positive arguments (matching
    Fortran dlog behavior). MATLAB nansum then skips these contributions."""
    num = a + b
    den = c + d
    if num <= 0.0 or den <= 0.0:
        return np.nan
    return np.log(num / den)


@njit(cache=True)
def _fan(e, f, g, t, yf, w):
    """atan((e*t)/(f*w)) - atan((e*g)/(f*yf)). Guards zero denominator."""
    if f == 0.0 or w == 0.0 or yf == 0.0:
        return 0.0
    return np.arctan((e * t) / (f * w)) - np.arctan((e * g) / (f * yf))


@njit(cache=True)
def _cross(a1, a2, a3, b1, b2, b3):
    return (a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1)


@njit(cache=True)
def _dot(a1, a2, a3, b1, b2, b3):
    return a1 * b1 + a2 * b2 + a3 * b3


@njit(cache=True)
def _planedist(x, y, z, kp1, kp2, kp3):
    """
    Distance from origin P to the plane through vertices kp1,kp2,kp3.
    Returns (h, xo, yo, zo) where xo/yo/zo are axis intersections.
    Mirrors Fortran subroutine `planedist`.
    """
    a = (y[kp2 - 1] - y[kp1 - 1]) * (z[kp3 - 1] - z[kp1 - 1]) - \
        (y[kp3 - 1] - y[kp1 - 1]) * (z[kp2 - 1] - z[kp1 - 1])
    b = (z[kp2 - 1] - z[kp1 - 1]) * (x[kp3 - 1] - x[kp1 - 1]) - \
        (z[kp3 - 1] - z[kp1 - 1]) * (x[kp2 - 1] - x[kp1 - 1])
    c = (x[kp2 - 1] - x[kp1 - 1]) * (y[kp3 - 1] - y[kp1 - 1]) - \
        (x[kp3 - 1] - x[kp1 - 1]) * (y[kp2 - 1] - y[kp1 - 1])
    d = -a * x[kp1 - 1] - b * y[kp1 - 1] - c * z[kp1 - 1]
    norm = np.sqrt(a * a + b * b + c * c)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    hp = abs(d / norm)
    xo = (-d / a) if a != 0.0 else 0.0
    yo = (-d / b) if b != 0.0 else 0.0
    zo = (-d / c) if c != 0.0 else 0.0
    return hp, xo, yo, zo


@njit(cache=True)
def _segmproj(x1, y1, z1, x2, y2, z2, xo, yo, zo):
    """
    Projection of P'=(xo,yo,zo) onto segment (x1,y1,z1)-(x2,y2,z2).
    Returns (xs, ys, zs). Mirrors Fortran subroutine `segmproj`.
    """
    r1 = x2 - x1
    r2 = y2 - y1
    r3 = z2 - z1
    e1 = x1 - xo
    e2 = y1 - yo
    e3 = z1 - zo
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    d1 = a1 * xo + b1 * yo + c1 * zo
    a2, b2, c2 = _cross(e1, e2, e3, r1, r2, r3)
    a3, b3, c3 = _cross(a2, b2, c2, r1, r2, r3)
    d2 = _dot(a2, b2, c2, xo, yo, zo)
    d3 = _dot(a3, b3, c3, x1, y1, z1)
    det = a1 * b2 * c3 + b1 * c2 * a3 + c1 * a2 * b3 - \
          a3 * b2 * c1 - b3 * c2 * a1 - c3 * a2 * b1
    if det == 0.0:
        return xo, yo, zo
    dx = d1 * b2 * c3 + b1 * c2 * d3 + c1 * d2 * b3 - \
         d3 * b2 * c1 - b3 * c2 * d1 - c3 * d2 * b1
    dy = a1 * d2 * c3 + d1 * c2 * a3 + c1 * a2 * d3 - \
         a3 * d2 * c1 - d3 * c2 * a1 - c3 * a2 * d1
    dz = a1 * b2 * d3 + b1 * d2 * a3 + d1 * a2 * b3 - \
         a3 * b2 * d1 - b3 * d2 * a1 - d3 * a2 * b1
    return dx / det, dy / det, dz / det


# ---------------------------------------------------------------------------
# Core: compute V / Vi / Vij for ONE polyhedron given 8 vertices.
# P (computation point) is at the origin.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _one_polyhedron(x, y, z, compute_v, compute_vi, compute_vij):
    """
    Compute gravity field of one uniformly regularized polyhedron at origin.

    Args:
        x, y, z: 1D arrays of length 8 (vertex coordinates, P at origin).
        compute_v:   if True, compute potential V.
        compute_vi:  if True, compute 1st derivatives (Vx,Vy,Vz).
        compute_vij: if True, compute 2nd derivatives (Vxx..Vzz).

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz) — uncomputed items are 0.
        Units are SI (V in m^2/s^2 per (kg/m^3), derivatives per (kg/m^3));
        density and unit scaling are applied by the caller.
    """
    pi = np.pi

    # --- Build edge vectors Gij (Eq 17) -------------------------------
    g = np.zeros((_NOP, 4, 3))
    for i in range(_NOP):
        for j in range(_NOE[i]):
            nxt = _K[i, 0] if j + 1 == _NOE[i] else _K[i, j + 1]
            cur = _K[i, j]
            g[i, j, 0] = x[nxt - 1] - x[cur - 1]
            g[i, j, 1] = y[nxt - 1] - y[cur - 1]
            g[i, j, 2] = z[nxt - 1] - z[cur - 1]

    # --- Plane normals Np (Eq 18) and direction cosines ---------------
    plnorm = np.zeros((_NOP, 3))
    for i in range(_NOP):
        cr0, cr1, cr2 = _cross(g[i, 0, 0], g[i, 0, 1], g[i, 0, 2],
                               g[i, 1, 0], g[i, 1, 1], g[i, 1, 2])
        qn = _snorm(cr0, cr1, cr2)
        if qn != 0.0:
            plnorm[i, 0] = cr0 / qn
            plnorm[i, 1] = cr1 / qn
            plnorm[i, 2] = cr2 / qn
    cs = plnorm.copy()  # direction cosines cos(Np, ei)

    # --- Line segment normals npq (Eq 19) and direction cosines -------
    segnorm = np.zeros((_NOP, 4, 3))
    for i in range(_NOP):
        for j in range(_NOE[i]):
            cr0, cr1, cr2 = _cross(g[i, j, 0], g[i, j, 1], g[i, j, 2],
                                   plnorm[i, 0], plnorm[i, 1], plnorm[i, 2])
            qn = _snorm(cr0, cr1, cr2)
            if qn != 0.0:
                segnorm[i, j, 0] = cr0 / qn
                segnorm[i, j, 1] = cr1 / qn
                segnorm[i, j, 2] = cr2 / qn
    cspq = segnorm  # direction cosines cos(npq, em)

    # --- Plane distances h(i) and axis intersections ------------------
    h = np.zeros(_NOP)
    xo = np.zeros(_NOP)
    yo = np.zeros(_NOP)
    zo = np.zeros(_NOP)
    for i in range(_NOP):
        h[i], xo[i], yo[i], zo[i] = _planedist(
            x, y, z, _K[i, 0], _K[i, 1], _K[i, 2])

    # --- Plane normal orientation s(i) (Eq 20) -----------------------
    s = np.zeros(_NOP)
    for i in range(_NOP):
        plnrmor = _dot(plnorm[i, 0], plnorm[i, 1], plnorm[i, 2],
                        -x[_K[i, 0] - 1], -y[_K[i, 0] - 1], -z[_K[i, 0] - 1])
        if plnrmor > 0:
            s[i] = -1.0
        elif plnrmor < 0:
            s[i] = 1.0

    # --- Projections P' on planes (Eq 21) ----------------------------
    xx = np.zeros(_NOP)
    yy = np.zeros(_NOP)
    zz = np.zeros(_NOP)
    for i in range(_NOP):
        dc1, dc2, dc3 = plnorm[i, 0], plnorm[i, 1], plnorm[i, 2]
        hi = h[i]
        if xo[i] >= 0 and yo[i] >= 0 and zo[i] >= 0:
            xx[i] = abs(dc1 * hi)
            yy[i] = abs(dc2 * hi)
            zz[i] = abs(dc3 * hi)
        elif xo[i] < 0 and yo[i] < 0 and zo[i] < 0:
            xx[i] = -dc1 * hi if dc1 > 0 else dc1 * hi
            yy[i] = -dc2 * hi if dc2 > 0 else dc2 * hi
            zz[i] = -dc3 * hi if dc3 > 0 else dc3 * hi
        elif xo[i] < 0 and yo[i] < 0:
            xx[i] = -dc1 * hi if dc1 > 0 else dc1 * hi
            yy[i] = -dc2 * hi if dc2 > 0 else dc2 * hi
            zz[i] = abs(dc3 * hi)
        elif yo[i] < 0 and zo[i] < 0:
            xx[i] = abs(dc1 * hi)
            yy[i] = -dc2 * hi if dc2 > 0 else dc2 * hi
            zz[i] = -dc3 * hi if dc3 > 0 else dc3 * hi
        elif xo[i] < 0 and zo[i] < 0:
            xx[i] = -dc1 * hi if dc1 > 0 else dc1 * hi
            yy[i] = abs(dc2 * hi)
            zz[i] = -dc3 * hi if dc3 > 0 else dc3 * hi
        elif xo[i] < 0:
            xx[i] = -dc1 * hi if dc1 > 0 else dc1 * hi
            yy[i] = abs(dc2 * hi)
            zz[i] = abs(dc3 * hi)
        elif yo[i] < 0:
            xx[i] = abs(dc1 * hi)
            yy[i] = -dc2 * hi if dc2 > 0 else dc2 * hi
            zz[i] = abs(dc3 * hi)
        elif zo[i] < 0:
            xx[i] = abs(dc1 * hi)
            yy[i] = abs(dc2 * hi)
            zz[i] = -dc3 * hi if dc3 > 0 else dc3 * hi

    # --- Segment normal orientation ss(i,j) (Eq 22) ------------------
    ss = np.zeros((_NOP, 4))
    for i in range(_NOP):
        for j in range(_NOE[i]):
            senrmor = _dot(segnorm[i, j, 0], segnorm[i, j, 1], segnorm[i, j, 2],
                            xx[i] - x[_K[i, j] - 1],
                            yy[i] - y[_K[i, j] - 1],
                            zz[i] - z[_K[i, j] - 1])
            if abs(senrmor) < 1e-10:
                ss[i, j] = 0.0
            elif senrmor > 0:
                ss[i, j] = -1.0
            else:
                ss[i, j] = 1.0

    # --- Segment distances hh(i,j) and projections P'' (Eq 23-25) ----
    hh = np.zeros((_NOP, 4))
    xxx = np.zeros((_NOP, 4))
    yyy = np.zeros((_NOP, 4))
    zzz = np.zeros((_NOP, 4))
    for i in range(_NOP):
        for j in range(_NOE[i]):
            if ss[i, j] == 0.0:
                hh[i, j] = 0.0
                xxx[i, j] = xx[i]
                yyy[i, j] = yy[i]
                zzz[i, j] = zz[i]
                continue
            if j + 1 == _NOE[i]:
                xs, ys, zs = _segmproj(
                    x[_K[i, j] - 1], y[_K[i, j] - 1], z[_K[i, j] - 1],
                    x[_K[i, 0] - 1], y[_K[i, 0] - 1], z[_K[i, 0] - 1],
                    xx[i], yy[i], zz[i])
            else:
                xs, ys, zs = _segmproj(
                    x[_K[i, j] - 1], y[_K[i, j] - 1], z[_K[i, j] - 1],
                    x[_K[i, j + 1] - 1], y[_K[i, j + 1] - 1], z[_K[i, j + 1] - 1],
                    xx[i], yy[i], zz[i])
            hh[i, j] = _snorm(xs - xx[i], ys - yy[i], zs - zz[i])
            xxx[i, j] = xs
            yyy[i, j] = ys
            zzz[i, j] = zs

    # --- Compute V and Vi (Eq 10-11) ---------------------------------
    V = Vx = Vy = Vz = 0.0
    if compute_v or compute_vi:
        for i in range(_NOP):
            sum1 = 0.0
            sum2 = 0.0
            singarea = 0
            singsegm = 0
            singvert = 0
            theta = 0.0
            for j in range(_NOE[i]):
                thisvert = 0
                gpqnorm = _snorm(g[i, j, 0], g[i, j, 1], g[i, j, 2])

                if ss[i, j] == 1.0:
                    singarea += 1

                if ss[i, j] == 0.0:
                    # Check if P' is on vertex or inside segment
                    if j + 1 == _NOE[i]:
                        gx1 = xx[i] - x[_K[i, 0] - 1]
                        gx2 = xx[i] - x[_K[i, j] - 1]
                        gy1 = yy[i] - y[_K[i, 0] - 1]
                        gy2 = yy[i] - y[_K[i, j] - 1]
                        gz1 = zz[i] - z[_K[i, 0] - 1]
                        gz2 = zz[i] - z[_K[i, j] - 1]
                    else:
                        gx2 = xx[i] - x[_K[i, j] - 1]
                        gx1 = xx[i] - x[_K[i, j + 1] - 1]
                        gy2 = yy[i] - y[_K[i, j] - 1]
                        gy1 = yy[i] - y[_K[i, j + 1] - 1]
                        gz2 = zz[i] - z[_K[i, j] - 1]
                        gz1 = zz[i] - z[_K[i, j + 1] - 1]
                    enorm1 = _snorm(gx1, gy1, gz1)
                    enorm2 = _snorm(gx2, gy2, gz2)

                    # Vertex check (i,j)
                    if enorm1 == 0.0:
                        singvert += 1
                        thisvert += 1
                        if j + 1 == _NOE[i]:
                            g1n = _snorm(g[i, j, 0], g[i, j, 1], g[i, j, 2])
                            g2n = _snorm(g[i, 0, 0], g[i, 0, 1], g[i, 0, 2])
                            gd = _dot(-g[i, j, 0], -g[i, j, 1], -g[i, j, 2],
                                      g[i, 0, 0], g[i, 0, 1], g[i, 0, 2])
                        else:
                            g1n = _snorm(g[i, j, 0], g[i, j, 1], g[i, j, 2])
                            g2n = _snorm(g[i, j + 1, 0], g[i, j + 1, 1], g[i, j + 1, 2])
                            gd = _dot(-g[i, j, 0], -g[i, j, 1], -g[i, j, 2],
                                      g[i, j + 1, 0], g[i, j + 1, 1], g[i, j + 1, 2])
                        if gd == 0.0 or g1n * g2n == 0.0:
                            theta = pi / 2
                        else:
                            theta = np.arccos(min(1.0, max(-1.0, gd / (g1n * g2n))))

                    # Vertex check (i,j+1 or i,1)
                    if enorm2 == 0.0:
                        singvert += 1
                        thisvert += 1
                        if j == 0:
                            g1n = _snorm(g[i, _NOE[i] - 1, 0], g[i, _NOE[i] - 1, 1], g[i, _NOE[i] - 1, 2])
                            g2n = _snorm(g[i, 0, 0], g[i, 0, 1], g[i, 0, 2])
                            gd = _dot(-g[i, _NOE[i] - 1, 0], -g[i, _NOE[i] - 1, 1], -g[i, _NOE[i] - 1, 2],
                                      g[i, 0, 0], g[i, 0, 1], g[i, 0, 2])
                        else:
                            g1n = _snorm(g[i, j - 1, 0], g[i, j - 1, 1], g[i, j - 1, 2])
                            g2n = _snorm(g[i, j, 0], g[i, j, 1], g[i, j, 2])
                            gd = _dot(-g[i, j - 1, 0], -g[i, j - 1, 1], -g[i, j - 1, 2],
                                      g[i, j, 0], g[i, j, 1], g[i, j, 2])
                        if gd == 0.0 or g1n * g2n == 0.0:
                            theta = pi / 2
                        else:
                            theta = np.arccos(min(1.0, max(-1.0, gd / (g1n * g2n))))

                    if enorm1 < gpqnorm and enorm2 < gpqnorm:
                        singsegm += 1

                # Compute s1,s2,r1,r2 (1D coordinates of segment endpoints
                # relative to P'' projection on the segment line)
                if j + 1 == _NOE[i]:
                    d1 = _snorm(xxx[i, j] - x[_K[i, j] - 1],
                                yyy[i, j] - y[_K[i, j] - 1],
                                zzz[i, j] - z[_K[i, j] - 1])
                    d2 = _snorm(xxx[i, j] - x[_K[i, 0] - 1],
                                yyy[i, j] - y[_K[i, 0] - 1],
                                zzz[i, j] - z[_K[i, 0] - 1])
                    f1 = _snorm(x[_K[i, j] - 1], y[_K[i, j] - 1], z[_K[i, j] - 1])
                    f2 = _snorm(x[_K[i, 0] - 1], y[_K[i, 0] - 1], z[_K[i, 0] - 1])
                else:
                    d1 = _snorm(xxx[i, j] - x[_K[i, j] - 1],
                                yyy[i, j] - y[_K[i, j] - 1],
                                zzz[i, j] - z[_K[i, j] - 1])
                    d2 = _snorm(xxx[i, j] - x[_K[i, j + 1] - 1],
                                yyy[i, j] - y[_K[i, j + 1] - 1],
                                zzz[i, j] - z[_K[i, j + 1] - 1])
                    f1 = _snorm(x[_K[i, j] - 1], y[_K[i, j] - 1], z[_K[i, j] - 1])
                    f2 = _snorm(x[_K[i, j + 1] - 1], y[_K[i, j + 1] - 1], z[_K[i, j + 1] - 1])

                # Determine s1,s2,r1,r2 based on P'' position
                if abs(d1 - f1) < 1e-10 and abs(d2 - f2) < 1e-10:
                    # P'' coincides with a vertex (degenerate)
                    if d1 < d2:
                        s1, s2, r1, r2 = d1, d2, f1, f2
                    elif d2 < d1:
                        s1, s2, r1, r2 = -d1, -d2, -f1, -f2
                    else:  # d1 == d2
                        s1, s2, r1, r2 = -d1, d2, -f1, f2
                else:
                    if d1 < gpqnorm and d2 < gpqnorm:
                        # P'' inside segment
                        s1, s2, r1, r2 = -d1, d2, f1, f2
                    elif d1 < d2:
                        s1, s2, r1, r2 = d1, d2, f1, f2
                    elif d2 < d1:
                        s1, s2, r1, r2 = -d1, -d2, f1, f2
                    else:
                        s1, s2, r1, r2 = 0.0, 0.0, f1, f2

                # Compute c1 (log term) and c2 (atan term)
                if h[i] == 0.0:
                    if thisvert > 0:
                        c1 = 0.0
                        c2 = 0.0
                    elif (s1 + s2) < 1e-10 and (r1 + r2) < 1e-10:
                        c1 = 0.0
                        c2 = 0.0
                    else:
                        c1 = _fln(s2, r2, s1, r1)
                        c2 = 0.0
                else:
                    if thisvert > 0:
                        c1 = 0.0
                    else:
                        c1 = _fln(s2, r2, s1, r1)
                    if hh[i, j] == 0.0:
                        c2 = 0.0
                    else:
                        c2 = _fan(h[i], hh[i, j], s1, s2, r1, r2)

                sum1 += ss[i, j] * hh[i, j] * c1
                sum2 += ss[i, j] * c2

            # Singular corrections
            area = 2 * pi * h[i] if singarea == _NOE[i] else 0.0
            edge = pi * h[i] if singsegm > 0 else 0.0
            vertex = theta * h[i] if singvert > 0 else 0.0

            bracket = sum1 + h[i] * sum2 - area - edge - vertex
            if compute_v:
                V += s[i] * h[i] * bracket
            if compute_vi:
                Vx += cs[i, 0] * bracket
                Vy += cs[i, 1] * bracket
                Vz += cs[i, 2] * bracket

    # --- Compute Vij (Eq 12) -----------------------------------------
    vij = np.zeros((3, 3))
    if compute_vij:
        for i in range(3):
            for j in range(3):
                total = 0.0
                for m in range(_NOP):
                    sum1 = 0.0
                    sum2 = 0.0
                    singarea = 0
                    singsegm = 0
                    singvert = 0
                    theta = 0.0
                    for n in range(_NOE[m]):
                        thisvert = 0
                        gpqnorm = _snorm(g[m, n, 0], g[m, n, 1], g[m, n, 2])

                        if ss[m, n] == 1.0:
                            singarea += 1

                        if ss[m, n] == 0.0:
                            if n + 1 == _NOE[m]:
                                gx1 = xx[m] - x[_K[m, 0] - 1]
                                gx2 = xx[m] - x[_K[m, n] - 1]
                                gy1 = yy[m] - y[_K[m, 0] - 1]
                                gy2 = yy[m] - y[_K[m, n] - 1]
                                gz1 = zz[m] - z[_K[m, 0] - 1]
                                gz2 = zz[m] - z[_K[m, n] - 1]
                            else:
                                gx2 = xx[m] - x[_K[m, n] - 1]
                                gx1 = xx[m] - x[_K[m, n + 1] - 1]
                                gy2 = yy[m] - y[_K[m, n] - 1]
                                gy1 = yy[m] - y[_K[m, n + 1] - 1]
                                gz2 = zz[m] - z[_K[m, n] - 1]
                                gz1 = zz[m] - z[_K[m, n + 1] - 1]
                            enorm1 = _snorm(gx1, gy1, gz1)
                            enorm2 = _snorm(gx2, gy2, gz2)

                            if enorm1 == 0.0:
                                singvert += 1
                                thisvert += 1
                                if n + 1 == _NOE[m]:
                                    g1n = _snorm(g[m, n, 0], g[m, n, 1], g[m, n, 2])
                                    g2n = _snorm(g[m, 0, 0], g[m, 0, 1], g[m, 0, 2])
                                    gd = _dot(-g[m, n, 0], -g[m, n, 1], -g[m, n, 2],
                                              g[m, 0, 0], g[m, 0, 1], g[m, 0, 2])
                                else:
                                    g1n = _snorm(g[m, n, 0], g[m, n, 1], g[m, n, 2])
                                    g2n = _snorm(g[m, n + 1, 0], g[m, n + 1, 1], g[m, n + 1, 2])
                                    gd = _dot(-g[m, n, 0], -g[m, n, 1], -g[m, n, 2],
                                              g[m, n + 1, 0], g[m, n + 1, 1], g[m, n + 1, 2])
                                if gd == 0.0 or g1n * g2n == 0.0:
                                    theta = pi / 2
                                else:
                                    theta = np.arccos(min(1.0, max(-1.0, gd / (g1n * g2n))))

                            if enorm2 == 0.0:
                                singvert += 1
                                thisvert += 1
                                if n == 0:
                                    g1n = _snorm(g[m, _NOE[m] - 1, 0], g[m, _NOE[m] - 1, 1], g[m, _NOE[m] - 1, 2])
                                    g2n = _snorm(g[m, 0, 0], g[m, 0, 1], g[m, 0, 2])
                                    gd = _dot(-g[m, _NOE[m] - 1, 0], -g[m, _NOE[m] - 1, 1], -g[m, _NOE[m] - 1, 2],
                                              g[m, 0, 0], g[m, 0, 1], g[m, 0, 2])
                                else:
                                    g1n = _snorm(g[m, n - 1, 0], g[m, n - 1, 1], g[m, n - 1, 2])
                                    g2n = _snorm(g[m, n, 0], g[m, n, 1], g[m, n, 2])
                                    gd = _dot(-g[m, n - 1, 0], -g[m, n - 1, 1], -g[m, n - 1, 2],
                                              g[m, n, 0], g[m, n, 1], g[m, n, 2])
                                if gd == 0.0 or g1n * g2n == 0.0:
                                    theta = pi / 2
                                else:
                                    theta = np.arccos(min(1.0, max(-1.0, gd / (g1n * g2n))))

                            if enorm1 < gpqnorm and enorm2 < gpqnorm:
                                singsegm += 1

                        # s1,s2,r1,r2 for Vij (same logic as V/Vi)
                        if n + 1 == _NOE[m]:
                            d1 = _snorm(xxx[m, n] - x[_K[m, n] - 1],
                                        yyy[m, n] - y[_K[m, n] - 1],
                                        zzz[m, n] - z[_K[m, n] - 1])
                            d2 = _snorm(xxx[m, n] - x[_K[m, 0] - 1],
                                        yyy[m, n] - y[_K[m, 0] - 1],
                                        zzz[m, n] - z[_K[m, 0] - 1])
                            f1 = _snorm(x[_K[m, n] - 1], y[_K[m, n] - 1], z[_K[m, n] - 1])
                            f2 = _snorm(x[_K[m, 0] - 1], y[_K[m, 0] - 1], z[_K[m, 0] - 1])
                        else:
                            d1 = _snorm(xxx[m, n] - x[_K[m, n] - 1],
                                        yyy[m, n] - y[_K[m, n] - 1],
                                        zzz[m, n] - z[_K[m, n] - 1])
                            d2 = _snorm(xxx[m, n] - x[_K[m, n + 1] - 1],
                                        yyy[m, n] - y[_K[m, n + 1] - 1],
                                        zzz[m, n] - z[_K[m, n + 1] - 1])
                            f1 = _snorm(x[_K[m, n] - 1], y[_K[m, n] - 1], z[_K[m, n] - 1])
                            f2 = _snorm(x[_K[m, n + 1] - 1], y[_K[m, n + 1] - 1], z[_K[m, n + 1] - 1])

                        if abs(d1 - f1) < 1e-10 and abs(d2 - f2) < 1e-10:
                            if d1 < d2:
                                s1, s2, r1, r2 = d1, d2, f1, f2
                            elif d2 < d1:
                                s1, s2, r1, r2 = -d1, -d2, -f1, -f2
                            else:
                                s1, s2, r1, r2 = -d1, d2, -f1, f2
                        else:
                            if d1 < gpqnorm and d2 < gpqnorm:
                                s1, s2, r1, r2 = -d1, d2, f1, f2
                            elif d1 < d2:
                                s1, s2, r1, r2 = d1, d2, f1, f2
                            elif d2 < d1:
                                s1, s2, r1, r2 = -d1, -d2, f1, f2
                            else:
                                s1, s2, r1, r2 = 0.0, 0.0, f1, f2

                        if h[m] == 0.0:
                            # Fortran Vij: no (s1+s2)<1e-10 check (unlike V/Vi)
                            if thisvert > 0:
                                c1 = 0.0
                                c2 = 0.0
                            else:
                                c1 = _fln(s2, r2, s1, r1)
                                c2 = 0.0
                        else:
                            if thisvert > 0:
                                c1 = 0.0
                            else:
                                c1 = _fln(s2, r2, s1, r1)
                            if hh[m, n] == 0.0:
                                c2 = 0.0
                            else:
                                c2 = _fan(h[m], hh[m, n], s1, s2, r1, r2)

                        # NOTE: Vij uses cspq for sum1 (not ss*hh*c1)
                        sum1 += cspq[m, n, j] * c1
                        sum2 += ss[m, n] * c2

                    # Corrections for Vij: NO h(m) factor
                    area = 2 * pi if singarea == _NOE[m] else 0.0
                    edge = pi if singsegm > 0 else 0.0
                    vertex = theta if singvert > 0 else 0.0

                    total += cs[m, i] * (sum1 + s[m] * cs[m, j] * (sum2 - area - edge - vertex))
                vij[i, j] = total

    Vxx = vij[0, 0]
    Vxy = vij[0, 1]
    Vxz = vij[0, 2]
    Vyy = vij[1, 1]
    Vyz = vij[1, 2]
    Vzz = vij[2, 2]

    return V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz


@njit(cache=True)
def _polyhedron_batch(x, y, z, compute_v, compute_vi, compute_vij):
    """
    Loop over `num` polyhedra. x,y,z have shape (8, num) (each column is one
    polyhedron's 8 vertices). Returns arrays of shape (num, 10).
    """
    num = x.shape[1]
    out = np.zeros((num, 10))
    for p in range(num):
        V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = _one_polyhedron(
            x[:, p], y[:, p], z[:, p],
            compute_v, compute_vi, compute_vij)
        out[p, 0] = V
        out[p, 1] = Vx
        out[p, 2] = Vy
        out[p, 3] = Vz
        out[p, 4] = Vxx
        out[p, 5] = Vxy
        out[p, 6] = Vxz
        out[p, 7] = Vyy
        out[p, 8] = Vyz
        out[p, 9] = Vzz
    return out


# ---------------------------------------------------------------------------
# DEM-grid processing: build 8-vertex polyhedra from a DEM sub-grid.
# Mirrors the common preamble of Polyhedron_full/g/gx/g_gx/gxx in TC_GUI.m.
# ---------------------------------------------------------------------------
def _build_polyhedra(x1_grid, y1_grid, z1_grid, z1_ref,
                     dov_lat, dov_lon, dov_h,
                     extend_x1, extend_y1, e1,
                     resolution_x, resolution_y):
    """
    For each computation point, extract the DEM sub-grid, build polyhedron
    vertices, detect & split irregular polyhedra, and return per-point
    vertex arrays plus density and centroid lon/lat.

    Returns a list of dicts (one per computation point) with keys:
        x, y, z   : (num_poly, 8) arrays of vertex coordinates (P at origin)
        e_d       : (num_poly,) density values
        lonp, latp: (num_poly,) centroid lon/lat [deg]
        lon, lat  : scalar computation point lon/lat [deg]
    """
    d2r = np.pi / 180.0

    lat_p = y1_grid[:, 0]
    lon_p = x1_grid[0, :]

    points = []
    for i in range(len(dov_lon)):
        lon = dov_lon[i]
        lat = dov_lat[i]
        r0 = dov_h[i]

        # Extract sub-grid
        n_idx = np.where((lon_p <= lon + extend_x1) & (lon_p >= lon - extend_x1))[0]
        m_idx = np.where((lat_p <= lat + extend_y1) & (lat_p >= lat - extend_y1))[0]
        lon1 = lon_p[n_idx]
        lat1 = lat_p[m_idx]
        zz_bottom = z1_ref[np.ix_(m_idx, n_idx)]
        zz = z1_grid[np.ix_(m_idx, n_idx)]
        e = e1[np.ix_(m_idx, n_idx)]

        # 2D meshgrid then cell centroids
        lon1g, lat1g = np.meshgrid(lon1, lat1)
        lonp = 0.5 * (lon1g[:-1, :-1] + lon1g[1:, 1:])
        latp = 0.5 * (lat1g[:-1, :-1] + lat1g[1:, 1:])
        zzp = 0.25 * (zz[:-1, :-1] + zz[1:, 1:] +
                      zz[:-1, 1:] + zz[1:, :-1])

        # Local Cartesian coordinates of cell centroids (P at origin)
        x, y, z = global2local(lon, lat, r0, lonp, latp, zzp)

        # Cell half-widths in local meters
        Dx = zzp * d2r * resolution_y
        Dy = zzp * np.cos(d2r * latp) * d2r * resolution_x
        dx = 0.5 * Dx
        dy = 0.5 * Dy

        # 8 vertices of each polyhedron (top 1-4, bottom 5-8)
        xx1 = -dx - x;  yy1 = -dy - y;  zz1 = zz[:-1, :-1] - zzp + z
        xx2 = xx1;       yy2 = dy - y;   zz2 = zz[:-1, 1:] - zzp + z
        xx3 = dx - x;    yy3 = yy2;      zz3 = zz[1:, 1:] - zzp + z
        xx4 = xx3;       yy4 = yy1;      zz4 = zz[1:, :-1] - zzp + z
        xx5 = xx1;       yy5 = yy1;      zz5 = zz_bottom[:-1, :-1] - zzp + z
        xx6 = xx2;       yy6 = yy2;      zz6 = zz_bottom[:-1, 1:] - zzp + z
        xx7 = xx3;       yy7 = yy3;      zz7 = zz_bottom[1:, 1:] - zzp + z
        xx8 = xx4;       yy8 = yy4;      zz8 = zz_bottom[1:, :-1] - zzp + z

        e_sub = e[:-1, :-1]
        e_d = e_sub.flatten()

        # --- Irregular polyhedron detection & splitting ---
        f1 = (zz1 - zz5) * (zz2 - zz6) < 0
        f2 = (zz2 - zz6) * (zz3 - zz7) < 0
        f3 = (zz3 - zz7) * (zz4 - zz8) < 0
        f4 = (zz1 - zz5) * (zz4 - zz8) < 0
        n1 = np.flatnonzero(f1 & f2)
        n2 = np.flatnonzero(f3 & f2)
        n3 = np.flatnonzero(f3 & f4)
        n4 = np.flatnonzero(f1 & f4)
        n5 = np.flatnonzero(f4 & f2)
        n6 = np.flatnonzero(f1 & f3)
        n_all = np.unique(np.concatenate([n1, n2, n3, n4, n5, n6]))
        p = len(n_all)

        # Flatten main grids for stacking
        xx1f, yy1f, zz1f = xx1.flatten(), yy1.flatten(), zz1.flatten()
        xx2f, yy2f, zz2f = xx2.flatten(), yy2.flatten(), zz2.flatten()
        xx3f, yy3f, zz3f = xx3.flatten(), yy3.flatten(), zz3.flatten()
        xx4f, yy4f, zz4f = xx4.flatten(), yy4.flatten(), zz4.flatten()
        xx5f, yy5f, zz5f = xx5.flatten(), yy5.flatten(), zz5.flatten()
        xx6f, yy6f, zz6f = xx6.flatten(), yy6.flatten(), zz6.flatten()
        xx7f, yy7f, zz7f = xx7.flatten(), yy7.flatten(), zz7.flatten()
        xx8f, yy8f, zz8f = xx8.flatten(), yy8.flatten(), zz8.flatten()

        # Assemble main polyhedra vertex arrays (num_main, 8)
        x_main = np.column_stack([xx1f, xx2f, xx3f, xx4f, xx5f, xx6f, xx7f, xx8f])
        y_main = np.column_stack([yy1f, yy2f, yy3f, yy4f, yy5f, yy6f, yy7f, yy8f])
        z_main = np.column_stack([zz1f, zz2f, zz3f, zz4f, zz5f, zz6f, zz7f, zz8f])
        lonp_f = lonp.flatten()
        latp_f = latp.flatten()

        if p > 0:
            # Build split polyhedra for irregular cells (top set to -10000)
            temp = np.full(p, -10000.0)
            x_s = np.column_stack([xx1f[n_all], xx2f[n_all], xx3f[n_all], xx4f[n_all],
                                   xx5f[n_all], xx6f[n_all], xx7f[n_all], xx8f[n_all]])
            y_s = np.column_stack([yy1f[n_all], yy2f[n_all], yy3f[n_all], yy4f[n_all],
                                   yy5f[n_all], yy6f[n_all], yy7f[n_all], yy8f[n_all]])
            z_s = np.column_stack([temp, temp, temp, temp,
                                   zz5f[n_all], zz6f[n_all], zz7f[n_all], zz8f[n_all]])
            lon_s = lonp_f[n_all]
            lat_s = latp_f[n_all]

            # Zero out bottom of irregular cells in the main set
            zz5f[n_all] = -10000.0
            zz6f[n_all] = -10000.0
            zz7f[n_all] = -10000.0
            zz8f[n_all] = -10000.0
            z_main = np.column_stack([zz1f, zz2f, zz3f, zz4f,
                                      zz5f, zz6f, zz7f, zz8f])

            x_all = np.vstack([x_main, x_s])
            y_all = np.vstack([y_main, y_s])
            z_all = np.vstack([z_main, z_s])
            lonp_all = np.concatenate([lonp_f, lon_s])
            latp_all = np.concatenate([latp_f, lat_s])
            e_d_all = np.concatenate([e_d, e_d[n_all]])
        else:
            x_all = x_main
            y_all = y_main
            z_all = z_main
            lonp_all = lonp_f
            latp_all = latp_f
            e_d_all = e_d

        # Avoid zero x/y (singular at computation point) — perturb slightly
        x_all[np.abs(x_all) < 0.001] += 0.1
        y_all[np.abs(y_all) < 0.001] += 0.1

        points.append({
            'x': x_all,  # (num_poly, 8)
            'y': y_all,
            'z': z_all,
            'e_d': e_d_all,
            'lonp': lonp_all,
            'latp': latp_all,
            'lon': lon,
            'lat': lat,
        })
    return points


# ---------------------------------------------------------------------------
# Public API: 5 variants matching TC_GUI.m
# ---------------------------------------------------------------------------
# Physical constants (matching TC_GUI.m)
_G = 6.67384e-8   # gravitational constant [cm^3 g^-1 s^-2] -> gives m^2/s^2 per g/cm^3 * m


def _run_variant(x1_grid, y1_grid, z1_grid, z1_ref,
                 dov_lat, dov_lon, dov_h,
                 extend_x1, extend_y1, e1,
                 resolution_x, resolution_y,
                 compute_v, compute_vi, compute_vij,
                 need_rotation):
    """
    Common driver for all 5 polyhedron variants.

    Args:
        compute_v / compute_vi / compute_vij: which field components to compute.
        need_rotation: if True, rotate Vi/Vij from polyhedron frame to station
            frame via g_prism2point / ggt_prism2point (used by full, g_gx, gxx).
            Variants g and gx apply their own sign conventions without rotation.

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz) arrays of length n_points.
        Final unit scaling matches TC_GUI.m:
            V  -> m^2/s^2 (after /2)
            Vx,Vy,Vz -> mGal (after *1e5)
            Vxx..Vzz -> E   (after *1e9)
    """
    size_com = len(dov_lon)
    V = np.zeros(size_com)
    Vx = np.zeros(size_com)
    Vy = np.zeros(size_com)
    Vz = np.zeros(size_com)
    Vxx = np.zeros(size_com)
    Vxy = np.zeros(size_com)
    Vxz = np.zeros(size_com)
    Vyy = np.zeros(size_com)
    Vyz = np.zeros(size_com)
    Vzz = np.zeros(size_com)

    points = _build_polyhedra(x1_grid, y1_grid, z1_grid, z1_ref,
                              dov_lat, dov_lon, dov_h,
                              extend_x1, extend_y1, e1,
                              resolution_x, resolution_y)

    for i, pt in enumerate(points):
        # x,y,z have shape (num_poly, 8); transpose to (8, num_poly) for batch
        xb = pt['x'].T
        yb = pt['y'].T
        zb = pt['z'].T
        g = _polyhedron_batch(xb, yb, zb, compute_v, compute_vi, compute_vij)
        # g: (num_poly, 10) -> [V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz]
        num = g.shape[0]
        c = _G * pt['e_d']

        lon = pt['lon']
        lat = pt['lat']
        lonp = pt['lonp']
        latp = pt['latp']

        if compute_v:
            V[i] = np.nansum(c * g[:, 0])

        if compute_vi:
            # Fortran returns Vz with opposite sign; MATLAB does g(:,4) = -g(:,4)
            gx_raw = g[:, 1]
            gy_raw = g[:, 2]
            gz_raw = -g[:, 3]
            if need_rotation:
                tvx, tvy, tvz = g_prism2point(gx_raw, gy_raw, gz_raw,
                                              lon, lat, lonp, latp)
                Vx[i] = np.nansum(c * tvx)
                Vy[i] = np.nansum(c * tvy)
                # MATLAB: Vz(i)=-nansum(c.*temp_Vz) for g_gx/gx;
                #         Vz=-Vz*1e5 for full. Net effect: negate.
                Vz[i] = -np.nansum(c * tvz)
            else:
                # Variant g: no rotation, no Vz computed (compute_vi=False)
                Vx[i] = np.nansum(c * gx_raw)
                Vy[i] = np.nansum(c * gy_raw)
                Vz[i] = -np.nansum(c * gz_raw)

        if compute_vij:
            if need_rotation:
                tvxx, tvxy, tvxz, tvyy, tvyz, tvzz = ggt_prism2point(
                    g[:, 4], g[:, 5], g[:, 6], g[:, 7], g[:, 8], g[:, 9],
                    lon, lat, lonp, latp)
                Vxx[i] = np.nansum(c * tvxx)
                Vxy[i] = np.nansum(c * tvxy)
                Vxz[i] = np.nansum(c * tvxz)
                Vyy[i] = np.nansum(c * tvyy)
                Vyz[i] = np.nansum(c * tvyz)
                Vzz[i] = np.nansum(c * tvzz)
            else:
                Vxx[i] = np.nansum(c * g[:, 4])
                Vxy[i] = np.nansum(c * g[:, 5])
                Vxz[i] = np.nansum(c * g[:, 6])
                Vyy[i] = np.nansum(c * g[:, 7])
                Vyz[i] = np.nansum(c * g[:, 8])
                Vzz[i] = np.nansum(c * g[:, 9])

    # Final unit scaling (matching TC_GUI.m)
    if compute_v:
        V = V / 2.0
    if compute_vi:
        Vx = Vx * 1e5
        Vy = Vy * 1e5
        Vz = Vz * 1e5
    if compute_vij:
        Vxx = Vxx * 1e9
        Vxy = Vxy * 1e9
        Vxz = Vxz * 1e9
        Vyy = Vyy * 1e9
        Vyz = Vyz * 1e9
        Vzz = Vzz * 1e9

    return V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz


def polyhedron_full(x1_grid: np.ndarray, y1_grid: np.ndarray,
                    z1_grid: np.ndarray, z1_ref: np.ndarray,
                    dov_lat: np.ndarray, dov_lon: np.ndarray, dov_h: np.ndarray,
                    extend_x1: float, extend_y1: float,
                    e1: np.ndarray,
                    resolution_x: float, resolution_y: float
                    ) -> Tuple[np.ndarray, ...]:
    """
    Compute all 10 gravity field components via polyhedron method (itype=10).

    Corresponds to TC_GUI.m L2460-L2628 (Polyhedron_full).

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz): 10 arrays of length n_points.
        Units: V in m^2/s^2, Vx/Vy/Vz in mGal, Vxx..Vzz in E.
    """
    return _run_variant(x1_grid, y1_grid, z1_grid, z1_ref,
                        dov_lat, dov_lon, dov_h,
                        extend_x1, extend_y1, e1,
                        resolution_x, resolution_y,
                        compute_v=True, compute_vi=True, compute_vij=True,
                        need_rotation=True)


def polyhedron_g(x1_grid: np.ndarray, y1_grid: np.ndarray,
                 z1_grid: np.ndarray, z1_ref: np.ndarray,
                 dov_lat: np.ndarray, dov_lon: np.ndarray, dov_h: np.ndarray,
                 extend_x1: float, extend_y1: float,
                 e1: np.ndarray,
                 resolution_x: float, resolution_y: float
                 ) -> Tuple[np.ndarray, ...]:
    """
    Compute potential only via polyhedron method (itype=0).

    Corresponds to TC_GUI.m L2630-L2783 (Polyhedron_g).

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz) with only V computed;
        other arrays are zeros (kept for unified interface).
    """
    return _run_variant(x1_grid, y1_grid, z1_grid, z1_ref,
                        dov_lat, dov_lon, dov_h,
                        extend_x1, extend_y1, e1,
                        resolution_x, resolution_y,
                        compute_v=True, compute_vi=False, compute_vij=False,
                        need_rotation=False)


def polyhedron_g_gx(x1_grid: np.ndarray, y1_grid: np.ndarray,
                    z1_grid: np.ndarray, z1_ref: np.ndarray,
                    dov_lat: np.ndarray, dov_lon: np.ndarray, dov_h: np.ndarray,
                    extend_x1: float, extend_y1: float,
                    e1: np.ndarray,
                    resolution_x: float, resolution_y: float
                    ) -> Tuple[np.ndarray, ...]:
    """
    Compute potential + 1st derivatives via polyhedron (itype=4/103/104).

    Corresponds to TC_GUI.m L2785-L2934 (Polyhedron_g_gx).
    """
    return _run_variant(x1_grid, y1_grid, z1_grid, z1_ref,
                        dov_lat, dov_lon, dov_h,
                        extend_x1, extend_y1, e1,
                        resolution_x, resolution_y,
                        compute_v=True, compute_vi=True, compute_vij=False,
                        need_rotation=True)


def polyhedron_gx(x1_grid: np.ndarray, y1_grid: np.ndarray,
                  z1_grid: np.ndarray, z1_ref: np.ndarray,
                  dov_lat: np.ndarray, dov_lon: np.ndarray, dov_h: np.ndarray,
                  extend_x1: float, extend_y1: float,
                  e1: np.ndarray,
                  resolution_x: float, resolution_y: float
                  ) -> Tuple[np.ndarray, ...]:
    """
    Compute 1st derivatives only via polyhedron (itype=1).

    Corresponds to TC_GUI.m L2948-L3095 (Polyhedron_gx).
    """
    return _run_variant(x1_grid, y1_grid, z1_grid, z1_ref,
                        dov_lat, dov_lon, dov_h,
                        extend_x1, extend_y1, e1,
                        resolution_x, resolution_y,
                        compute_v=False, compute_vi=True, compute_vij=False,
                        need_rotation=True)


def polyhedron_gxx(x1_grid: np.ndarray, y1_grid: np.ndarray,
                   z1_grid: np.ndarray, z1_ref: np.ndarray,
                   dov_lat: np.ndarray, dov_lon: np.ndarray, dov_h: np.ndarray,
                   extend_x1: float, extend_y1: float,
                   e1: np.ndarray,
                   resolution_x: float, resolution_y: float
                   ) -> Tuple[np.ndarray, ...]:
    """
    Compute 2nd derivatives only via polyhedron (itype=2).

    Corresponds to TC_GUI.m L3110-L3260 (Polyhedron_gxx).
    """
    return _run_variant(x1_grid, y1_grid, z1_grid, z1_ref,
                        dov_lat, dov_lon, dov_h,
                        extend_x1, extend_y1, e1,
                        resolution_x, resolution_y,
                        compute_v=False, compute_vi=False, compute_vij=True,
                        need_rotation=True)
