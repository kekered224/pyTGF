"""
Tesseroid mass element (Zone 3, medium, efficient)

Mirrors TC_GUI.m:
    Tesseroid       (L4183-L4495)  all 10 components, itype=10
    Tesseroid_g     (L4497-L4592)  potential only, itype=0
    Tesseroid_g_gx  (L4594-L4749)  potential + 1st derivs, itype=4/103/104
    Tesseroid_gx    (L4751-L4889)  1st derivatives, itype=1
    Tesseroid_gxx   (L4891-L5139)  2nd derivatives, itype=2

Reference: T. Grombein, K. Seitz, B. Heck (2013) J. Geodesy 87(7), 645-660.
           X.L. Deng et al. (2016) corrections.

The tesseroid is centered at (lon1, lat1) with angular size
(resolution_x1, resolution_y1) and radial extent [h1, ref] (h1 = upper
radius, ref = lower radius). The computation point is (lon, lat, r0).

All formulas use the Taylor-expanded kernel up to 2nd order in the
tesseroid dimensions (xx, yy, zz), with coefficients toi, a, b, gama
evaluated at the tesseroid center. See Grombein et al. (2013) for the
derivation.

Output unit conventions (matching MATLAB):
    V   : m²/s² × 1e-5  (i.e., 10 m²/s² units, potential)
    Vx..: m/s²          (1st derivatives, no scaling)
    Vxx.: Eötvös        (2nd derivatives, ×1e4)
"""

from typing import Tuple, Dict
import numpy as np

from ..config import PhysicalConstants


# =============================================================================
#  Shared geometry kernel
# =============================================================================

def _tess_geometry(lon1: np.ndarray, lat1: np.ndarray,
                   h1: np.ndarray, ref: np.ndarray,
                   lon: float, lat: float, r0: float,
                   resolution_x1: float, resolution_y1: float,
                   e: np.ndarray,
                   need_gxx: bool = True) -> Dict[str, np.ndarray]:
    """
    Build all geometric quantities shared by the 5 tesseroid variants.

    Returns a dict with keys matching MATLAB variable names so the
    per-itype functions can be written as near-verbatim translations.

    When need_gxx=False, skips the expensive 3rd/4th-order derivatives
    (bijk, gamaijk) that are only needed for 2nd-order gravity gradients
    (itype=2/3/5/6). This roughly halves the compute for itype=0/1/4/103/104.
    """
    Pi = PhysicalConstants.PI
    d2r = Pi / 180.0
    G = PhysicalConstants.G

    # build tesseroid bounds
    zz1 = lon1 - resolution_x1 / 2.0
    zz2 = lon1 + resolution_x1 / 2.0
    yy1 = lat1 - resolution_y1 / 2.0
    yy2 = lat1 + resolution_y1 / 2.0
    xx1 = h1
    xx2 = ref

    r1 = (xx1 + xx2) / 2.0

    # variable definition
    lat_r = lat * d2r
    lat1_r = lat1 * d2r
    coslat = np.cos(lat_r)
    coslat1 = np.cos(lat1_r)
    sinlat = np.sin(lat_r)
    sinlat1 = np.sin(lat1_r)

    d_lon = (lon1 - lon) * d2r
    cosdlon = np.cos(d_lon)
    sindlon = np.sin(d_lon)

    cosphicosa = coslat1 * cosdlon
    coslatsinlat1 = coslat * sinlat1
    sinphicosa = coslatsinlat1 - sinlat * cosphicosa
    sinphisina = coslat1 * sindlon
    sinlatsinlat1 = sinlat * sinlat1
    cosphi = sinlatsinlat1 + coslat * cosphicosa

    dx = r1 * sinphicosa
    dy = r1 * sinphisina
    dz = r1 * cosphi - r0

    dxx = dx * dx
    dxy = dx * dy
    dxz = dx * dz
    dyy = dy * dy
    dyz = dy * dz
    dzz = dz * dz

    l = np.sqrt(dxx + dyy + dzz)

    xx = xx2 - xx1
    yy = (yy2 - yy1) * d2r
    zz = (zz2 - zz1) * d2r

    c = G * e * zz * xx * yy
    l2 = l * l

    # Toi (kernel coefficient at center)
    r12 = r1 * r1
    toi = r12 * coslat1
    toi1 = 2.0 * r1 * coslat1
    toi2 = -r12 * sinlat1
    toi3 = 0.0
    toi12 = 2.0 * coslat1
    toi22 = -toi
    toi32 = 0.0

    # dxi (1st derivatives of position vector x_i wrt tesseroid coords)
    x1_1 = sinphicosa
    x2_1 = sinphisina
    x3_1 = cosphi

    x1_12 = 0.0
    x2_12 = 0.0
    x3_12 = 0.0

    x1_2 = r1 * (coslat * coslat1 + sinlatsinlat1 * cosdlon)
    x2_2 = -r1 * sinlat1 * sindlon
    x3_2 = r1 * (sinlat * coslat1 - coslatsinlat1 * cosdlon)

    x1_22 = -dx
    x2_22 = -dy
    x3_22 = -r1 * cosphi

    x1_3 = (dy) * sinlat
    x2_3 = r1 * cosphicosa
    x3_3 = -(dy) * coslat

    x1_32 = x2_3 * sinlat
    x2_32 = -(dy)
    x3_32 = -x2_3 * coslat

    # ml (1st derivatives of l)
    ml1 = r0 * cosphi - r1
    ml2 = r0 * x3_2
    ml3 = r0 * x3_3

    ml12 = -1.0
    ml22 = r0 * x3_22
    ml32 = r0 * x3_32

    # a (2nd derivatives of l)
    a = l2
    l23 = 3.0 / l2
    a1 = dx
    a2 = dy
    a3 = dz
    a11 = l23 * dxx - 1.0
    a12 = l23 * dxy
    a13 = l23 * dxz
    a22 = l23 * dyy - 1.0
    a23 = l23 * dyz
    a33 = l23 * dzz - 1.0

    # b (3rd derivatives of l, first index pair)
    b1 = 2.0 * ml1
    b2 = 2.0 * ml2
    b3 = 2.0 * ml3

    b11 = 2.0 * (x1_1 + l23 * dx * ml1)
    b12 = 2.0 * (x1_2 + l23 * dx * ml2)
    b21 = 2.0 * (x2_1 + l23 * dy * ml1)
    b13 = 2.0 * (x1_3 + l23 * dx * ml3)
    b31 = 2.0 * (x3_1 + l23 * dz * ml1)
    b22 = 2.0 * (x2_2 + l23 * dy * ml2)
    b23 = 2.0 * (x2_3 + l23 * dy * ml3)
    b32 = 2.0 * (x3_2 + l23 * dz * ml2)
    b33 = 2.0 * (x3_3 + l23 * dz * ml3)

    # gama (4th derivatives of l, first index triple)
    ml1_2 = ml1 * ml1
    ml2_2 = ml2 * ml2
    ml3_2 = ml3 * ml3

    l25 = 5.0 / l2
    l27 = 7.0 / l2

    gama1 = ml12 + l23 * ml1_2
    gama2 = ml22 + l23 * ml2_2
    gama3 = ml32 + l23 * ml3_2

    gama11 = x1_12 + l23 * (2.0 * x1_1 * ml1 + dx * (ml12 + l25 * ml1_2))
    gama12 = x1_22 + l23 * (2.0 * x1_2 * ml2 + dx * (ml22 + l25 * ml2_2))
    gama21 = x2_12 + l23 * (2.0 * x2_1 * ml1 + dy * (ml12 + l25 * ml1_2))
    gama13 = x1_32 + l23 * (2.0 * x1_3 * ml3 + dx * (ml32 + l25 * ml3_2))
    gama31 = x3_12 + l23 * (2.0 * x3_1 * ml1 + dz * (ml12 + l25 * ml1_2))
    gama22 = x2_22 + l23 * (2.0 * x2_2 * ml2 + dy * (ml22 + l25 * ml2_2))
    gama23 = x2_32 + l23 * (2.0 * x2_3 * ml3 + dy * (ml32 + l25 * ml3_2))
    gama32 = x3_22 + l23 * (2.0 * x3_2 * ml2 + dz * (ml22 + l25 * ml2_2))
    gama33 = x3_32 + l23 * (2.0 * x3_3 * ml3 + dz * (ml32 + l25 * ml3_2))

    # 3rd-order b derivatives (bijk) needed only by gxx/full
    if need_gxx:
        const_1 = 6.0 / l2
        dxx5 = 5.0 * dxx
        dxy5 = 5.0 * dxy
        dxz5 = 5.0 * dxz
        dyy5 = 5.0 * dyy
        dyz5 = 5.0 * dyz
        dzz5 = 5.0 * dzz

        dx_x11 = dx * x1_1
        dx_x12 = dx * x1_2
        dx_x13 = dx * x1_3
        dx_x21 = dx * x2_1
        dx_x22 = dx * x2_2
        dx_x23 = dx * x2_3
        dy_x11 = dy * x1_1
        dy_x12 = dy * x1_2
        dy_x13 = dy * x1_3
        dz_x11 = dz * x1_1
        dz_x12 = dz * x1_2
        dz_x13 = dz * x1_3
        dy_x21 = dy * x2_1
        dy_x22 = dy * x2_2
        dy_x23 = dy * x2_3
        dz_x21 = dz * x2_1
        dz_x22 = dz * x2_2
        dz_x23 = dz * x2_3
        dx_x31 = dx * x3_1
        dx_x32 = dx * x3_2
        dx_x33 = dx * x3_3
        dy_x31 = dy * x3_1
        dy_x32 = dy * x3_2
        dy_x33 = dy * x3_3
        dz_x31 = dz * x3_1
        dz_x32 = dz * x3_2
        dz_x33 = dz * x3_3

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

        # 4th-order gama derivatives (gamaijk) needed only by gxx/full
        dx_x112 = dx * x1_12
        dx_x122 = dx * x1_22
        dx_x132 = dx * x1_32
        dx_x212 = dx * x2_12
        dx_x222 = dx * x2_22
        dx_x232 = dx * x2_32
        dx_x312 = dx * x3_12
        dx_x322 = dx * x3_22
        dx_x332 = dx * x3_32

        dy_x112 = dy * x1_12
        dy_x122 = dy * x1_22
        dy_x132 = dy * x1_32
        dy_x212 = dy * x2_12
        dy_x222 = dy * x2_22
        dy_x232 = dy * x2_32
        dy_x312 = dy * x3_12
        dy_x322 = dy * x3_22
        dy_x332 = dy * x3_32

        dz_x112 = dz * x1_12
        dz_x122 = dz * x1_22
        dz_x132 = dz * x1_32
        dz_x212 = dz * x2_12
        dz_x222 = dz * x2_22
        dz_x232 = dz * x2_32
        dz_x312 = dz * x3_12
        dz_x322 = dz * x3_22
        dz_x332 = dz * x3_32

        gama111 = l23 * (dx_x112 + dx_x112 + 2.0 * x1_1 * x1_1
                         + l25 * (b1 * (dx_x11 + dx_x11) + dxx * (ml12 + l27 * ml1_2))
                         - (ml12 + l25 * ml1_2))
        gama112 = l23 * (dx_x122 + dx_x122 + 2.0 * x1_2 * x1_2
                         + l25 * (b2 * (dx_x12 + dx_x12) + dxx * (ml22 + l27 * ml2_2))
                         - (ml22 + l25 * ml2_2))
        gama113 = l23 * (dx_x132 + dx_x132 + 2.0 * x1_3 * x1_3
                         + l25 * (b3 * (dx_x13 + dx_x13) + dxx * (ml32 + l27 * ml3_2))
                         - (ml32 + l25 * ml3_2))
        gama121 = l23 * (dx_x212 + dy_x112 + 2.0 * x1_1 * x2_1
                         + l25 * (b1 * (dx_x21 + dy_x11) + dxy * (ml12 + l27 * ml1_2)))
        gama122 = l23 * (dx_x222 + dy_x122 + 2.0 * x1_2 * x2_2
                         + l25 * (b2 * (dx_x22 + dy_x12) + dxy * (ml22 + l27 * ml2_2)))
        gama123 = l23 * (dx_x232 + dy_x132 + 2.0 * x1_3 * x2_3
                         + l25 * (b3 * (dx_x23 + dy_x13) + dxy * (ml32 + l27 * ml3_2)))
        gama131 = l23 * (dx_x312 + dz_x112 + 2.0 * x1_1 * x3_1
                         + l25 * (b1 * (dx_x31 + dz_x11) + dxz * (ml12 + l27 * ml1_2)))
        gama132 = l23 * (dx_x322 + dz_x122 + 2.0 * x1_2 * x3_2
                         + l25 * (b2 * (dx_x32 + dz_x12) + dxz * (ml22 + l27 * ml2_2)))
        gama133 = l23 * (dx_x332 + dz_x132 + 2.0 * x1_3 * x3_3
                         + l25 * (b3 * (dx_x33 + dz_x13) + dxz * (ml32 + l27 * ml3_2)))
        gama221 = l23 * (dy_x212 + dy_x212 + 2.0 * x2_1 * x2_1
                         + l25 * (b1 * (dy_x21 + dy_x21) + dyy * (ml12 + l27 * ml1_2))
                         - (ml12 + l25 * ml1_2))
        gama222 = l23 * (dy_x222 + dy_x222 + 2.0 * x2_2 * x2_2
                         + l25 * (b2 * (dy_x22 + dy_x22) + dyy * (ml22 + l27 * ml2_2))
                         - (ml22 + l25 * ml2_2))
        gama223 = l23 * (dy_x232 + dy_x232 + 2.0 * x2_3 * x2_3
                         + l25 * (b3 * (dy_x23 + dy_x23) + dyy * (ml32 + l27 * ml3_2))
                         - (ml32 + l25 * ml3_2))
        gama231 = l23 * (dy_x312 + dz_x212 + 2.0 * x2_1 * x3_1
                         + l25 * (b1 * (dy_x31 + dz_x21) + dyz * (ml12 + l27 * ml1_2)))
        gama232 = l23 * (dy_x322 + dz_x222 + 2.0 * x2_2 * x3_2
                         + l25 * (b2 * (dy_x32 + dz_x22) + dyz * (ml22 + l27 * ml2_2)))
        gama233 = l23 * (dy_x332 + dz_x232 + 2.0 * x2_3 * x3_3
                         + l25 * (b3 * (dy_x33 + dz_x23) + dyz * (ml32 + l27 * ml3_2)))
        gama331 = l23 * (dz_x312 + dz_x312 + 2.0 * x3_1 * x3_1
                         + l25 * (b1 * (dz_x31 + dz_x31) + dzz * (ml12 + l27 * ml1_2))
                         - (ml12 + l25 * ml1_2))
        gama332 = l23 * (dz_x322 + dz_x322 + 2.0 * x3_2 * x3_2
                         + l25 * (b2 * (dz_x32 + dz_x32) + dzz * (ml22 + l27 * ml2_2))
                         - (ml22 + l25 * ml2_2))
        gama333 = l23 * (dz_x332 + dz_x332 + 2.0 * x3_3 * x3_3
                         + l25 * (b3 * (dz_x33 + dz_x33) + dzz * (ml32 + l27 * ml3_2))
                         - (ml32 + l25 * ml3_2))

    else:
        const_1 = np.zeros_like(l)
        dxx5 = np.zeros_like(l)
        dxy5 = np.zeros_like(l)
        dxz5 = np.zeros_like(l)
        dyy5 = np.zeros_like(l)
        dyz5 = np.zeros_like(l)
        dzz5 = np.zeros_like(l)
        dx_x11 = np.zeros_like(l)
        dx_x12 = np.zeros_like(l)
        dx_x13 = np.zeros_like(l)
        dx_x21 = np.zeros_like(l)
        dx_x22 = np.zeros_like(l)
        dx_x23 = np.zeros_like(l)
        dy_x11 = np.zeros_like(l)
        dy_x12 = np.zeros_like(l)
        dy_x13 = np.zeros_like(l)
        dz_x11 = np.zeros_like(l)
        dz_x12 = np.zeros_like(l)
        dz_x13 = np.zeros_like(l)
        dy_x21 = np.zeros_like(l)
        dy_x22 = np.zeros_like(l)
        dy_x23 = np.zeros_like(l)
        dz_x21 = np.zeros_like(l)
        dz_x22 = np.zeros_like(l)
        dz_x23 = np.zeros_like(l)
        dx_x31 = np.zeros_like(l)
        dx_x32 = np.zeros_like(l)
        dx_x33 = np.zeros_like(l)
        dy_x31 = np.zeros_like(l)
        dy_x32 = np.zeros_like(l)
        dy_x33 = np.zeros_like(l)
        dz_x31 = np.zeros_like(l)
        dz_x32 = np.zeros_like(l)
        dz_x33 = np.zeros_like(l)
        b111 = np.zeros_like(l)
        b112 = np.zeros_like(l)
        b113 = np.zeros_like(l)
        b121 = np.zeros_like(l)
        b122 = np.zeros_like(l)
        b123 = np.zeros_like(l)
        b131 = np.zeros_like(l)
        b132 = np.zeros_like(l)
        b133 = np.zeros_like(l)
        b221 = np.zeros_like(l)
        b222 = np.zeros_like(l)
        b223 = np.zeros_like(l)
        b231 = np.zeros_like(l)
        b232 = np.zeros_like(l)
        b233 = np.zeros_like(l)
        b331 = np.zeros_like(l)
        b332 = np.zeros_like(l)
        b333 = np.zeros_like(l)
        dx_x112 = np.zeros_like(l)
        dx_x122 = np.zeros_like(l)
        dx_x132 = np.zeros_like(l)
        dx_x212 = np.zeros_like(l)
        dx_x222 = np.zeros_like(l)
        dx_x232 = np.zeros_like(l)
        dx_x312 = np.zeros_like(l)
        dx_x322 = np.zeros_like(l)
        dx_x332 = np.zeros_like(l)
        dy_x112 = np.zeros_like(l)
        dy_x122 = np.zeros_like(l)
        dy_x132 = np.zeros_like(l)
        dy_x212 = np.zeros_like(l)
        dy_x222 = np.zeros_like(l)
        dy_x232 = np.zeros_like(l)
        dy_x312 = np.zeros_like(l)
        dy_x322 = np.zeros_like(l)
        dy_x332 = np.zeros_like(l)
        dz_x112 = np.zeros_like(l)
        dz_x122 = np.zeros_like(l)
        dz_x132 = np.zeros_like(l)
        dz_x212 = np.zeros_like(l)
        dz_x222 = np.zeros_like(l)
        dz_x232 = np.zeros_like(l)
        dz_x312 = np.zeros_like(l)
        dz_x322 = np.zeros_like(l)
        dz_x332 = np.zeros_like(l)
        gama111 = np.zeros_like(l)
        gama112 = np.zeros_like(l)
        gama113 = np.zeros_like(l)
        gama121 = np.zeros_like(l)
        gama122 = np.zeros_like(l)
        gama123 = np.zeros_like(l)
        gama131 = np.zeros_like(l)
        gama132 = np.zeros_like(l)
        gama133 = np.zeros_like(l)
        gama221 = np.zeros_like(l)
        gama222 = np.zeros_like(l)
        gama223 = np.zeros_like(l)
        gama231 = np.zeros_like(l)
        gama232 = np.zeros_like(l)
        gama233 = np.zeros_like(l)
        gama331 = np.zeros_like(l)
        gama332 = np.zeros_like(l)
        gama333 = np.zeros_like(l)
    # common powers
    xx_2 = xx * xx
    yy_2 = yy * yy
    zz_2 = zz * zz
    l3 = 1.0 / (l2 * l)   # = 1/l^3  (MATLAB redefines l3 = 1./l.^3 later)

    return dict(
        # scalar inputs / derived
        d2r=d2r, G=G, c=c, l=l, l2=l2, l3=l3, l23=l23,
        dx=dx, dy=dy, dz=dz, dxx=dxx, dxy=dxy, dxz=dxz,
        dyy=dyy, dyz=dyz, dzz=dzz,
        xx_2=xx_2, yy_2=yy_2, zz_2=zz_2,
        # toi
        toi=toi, toi1=toi1, toi2=toi2, toi3=toi3,
        toi12=toi12, toi22=toi22, toi32=toi32,
        # a
        a=a, a1=a1, a2=a2, a3=a3,
        a11=a11, a12=a12, a13=a13, a22=a22, a23=a23, a33=a33,
        # b
        b1=b1, b2=b2, b3=b3,
        b11=b11, b12=b12, b13=b13,
        b21=b21, b22=b22, b23=b23,
        b31=b31, b32=b32, b33=b33,
        b111=b111, b112=b112, b113=b113,
        b121=b121, b122=b122, b123=b123,
        b131=b131, b132=b132, b133=b133,
        b221=b221, b222=b222, b223=b223,
        b231=b231, b232=b232, b233=b233,
        b331=b331, b332=b332, b333=b333,
        # gama
        gama1=gama1, gama2=gama2, gama3=gama3,
        gama11=gama11, gama12=gama12, gama13=gama13,
        gama21=gama21, gama22=gama22, gama23=gama23,
        gama31=gama31, gama32=gama32, gama33=gama33,
        gama111=gama111, gama112=gama112, gama113=gama113,
        gama121=gama121, gama122=gama122, gama123=gama123,
        gama131=gama131, gama132=gama132, gama133=gama133,
        gama221=gama221, gama222=gama222, gama223=gama223,
        gama231=gama231, gama232=gama232, gama233=gama233,
        gama331=gama331, gama332=gama332, gama333=gama333,
    )


# =============================================================================
#  itype = 0  (potential only)
# =============================================================================

def tesseroid_g(lon1: np.ndarray, lat1: np.ndarray,
                h1: np.ndarray, ref: np.ndarray,
                lon: float, lat: float, r0: float,
                resolution_x1: float, resolution_y1: float,
                e: np.ndarray) -> np.ndarray:
    """
    Compute potential only for an array of tesseroids (itype=0).

    Corresponds to TC_GUI.m L4497-L4592 (Tesseroid_g).
    """
    g = _tess_geometry(lon1, lat1, h1, ref, lon, lat, r0,
                       resolution_x1, resolution_y1, e, need_gxx=False)

    K000 = (1.0 / g['l']) * g['toi']
    # MATLAB uses l2 (not a) in the correction term for Tesseroid_g
    corr = (g['xx_2'] * (g['toi12'] * g['l2'] + g['toi1'] * g['b1'] + g['toi'] * g['gama1'])
            + g['yy_2'] * (g['toi22'] * g['l2'] + g['toi2'] * g['b2'] + g['toi'] * g['gama2'])
            + g['zz_2'] * (g['toi32'] * g['l2'] + g['toi3'] * g['b3'] + g['toi'] * g['gama3']))
    pot = g['c'] * (K000 + (1.0 / 24.0) * g['l3'] * corr)
    V = np.sum(pot) * 1e-5
    return (V,)


# =============================================================================
#  itype = 1  (1st derivatives only)
# =============================================================================

def tesseroid_gx(lon1: np.ndarray, lat1: np.ndarray,
                 h1: np.ndarray, ref: np.ndarray,
                 lon: float, lat: float, r0: float,
                 resolution_x1: float, resolution_y1: float,
                 e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1st derivatives only for an array of tesseroids (itype=1).

    Corresponds to TC_GUI.m L4751-L4889 (Tesseroid_gx).
    """
    g = _tess_geometry(lon1, lat1, h1, ref, lon, lat, r0,
                       resolution_x1, resolution_y1, e, need_gxx=False)

    Lx = g['l3'] * g['dx'] * g['toi']
    Ly = g['l3'] * g['dy'] * g['toi']
    Lz = g['l3'] * g['dz'] * g['toi']

    gx = g['c'] * (Lx + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a1'] + g['toi1'] * g['b11'] + g['toi'] * g['gama11'])
        + g['yy_2'] * (g['toi22'] * g['a1'] + g['toi2'] * g['b12'] + g['toi'] * g['gama12'])
        + g['zz_2'] * (g['toi32'] * g['a1'] + g['toi3'] * g['b13'] + g['toi'] * g['gama13'])))
    gy = g['c'] * (Ly + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a2'] + g['toi1'] * g['b21'] + g['toi'] * g['gama21'])
        + g['yy_2'] * (g['toi22'] * g['a2'] + g['toi2'] * g['b22'] + g['toi'] * g['gama22'])
        + g['zz_2'] * (g['toi32'] * g['a2'] + g['toi3'] * g['b23'] + g['toi'] * g['gama23'])))
    gz = g['c'] * (Lz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a3'] + g['toi1'] * g['b31'] + g['toi'] * g['gama31'])
        + g['yy_2'] * (g['toi22'] * g['a3'] + g['toi2'] * g['b32'] + g['toi'] * g['gama32'])
        + g['zz_2'] * (g['toi32'] * g['a3'] + g['toi3'] * g['b33'] + g['toi'] * g['gama33'])))

    Vx = -np.sum(gx)
    Vy = -np.sum(gy)
    Vz = -np.sum(gz)
    return Vx, Vy, Vz


# =============================================================================
#  itype = 4/103/104  (potential + 1st derivatives)
# =============================================================================

def tesseroid_g_gx(lon1: np.ndarray, lat1: np.ndarray,
                   h1: np.ndarray, ref: np.ndarray,
                   lon: float, lat: float, r0: float,
                   resolution_x1: float, resolution_y1: float,
                   e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute potential + 1st derivatives for an array of tesseroids.

    Corresponds to TC_GUI.m L4594-L4749 (Tesseroid_g_gx).
    """
    g = _tess_geometry(lon1, lat1, h1, ref, lon, lat, r0,
                       resolution_x1, resolution_y1, e, need_gxx=False)

    K000 = (1.0 / g['l']) * g['toi']
    Lx = g['l3'] * g['dx'] * g['toi']
    Ly = g['l3'] * g['dy'] * g['toi']
    Lz = g['l3'] * g['dz'] * g['toi']

    # potential (uses a = l2 here, like Tesseroid_g_gx in MATLAB)
    pot = g['c'] * (K000 + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a'] + g['toi1'] * g['b1'] + g['toi'] * g['gama1'])
        + g['yy_2'] * (g['toi22'] * g['a'] + g['toi2'] * g['b2'] + g['toi'] * g['gama2'])
        + g['zz_2'] * (g['toi32'] * g['a'] + g['toi3'] * g['b3'] + g['toi'] * g['gama3'])))
    V = np.sum(pot) * 1e-5

    gx = g['c'] * (Lx + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a1'] + g['toi1'] * g['b11'] + g['toi'] * g['gama11'])
        + g['yy_2'] * (g['toi22'] * g['a1'] + g['toi2'] * g['b12'] + g['toi'] * g['gama12'])
        + g['zz_2'] * (g['toi32'] * g['a1'] + g['toi3'] * g['b13'] + g['toi'] * g['gama13'])))
    gy = g['c'] * (Ly + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a2'] + g['toi1'] * g['b21'] + g['toi'] * g['gama21'])
        + g['yy_2'] * (g['toi22'] * g['a2'] + g['toi2'] * g['b22'] + g['toi'] * g['gama22'])
        + g['zz_2'] * (g['toi32'] * g['a2'] + g['toi3'] * g['b23'] + g['toi'] * g['gama23'])))
    gz = g['c'] * (Lz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a3'] + g['toi1'] * g['b31'] + g['toi'] * g['gama31'])
        + g['yy_2'] * (g['toi22'] * g['a3'] + g['toi2'] * g['b32'] + g['toi'] * g['gama32'])
        + g['zz_2'] * (g['toi32'] * g['a3'] + g['toi3'] * g['b33'] + g['toi'] * g['gama33'])))

    Vx = -np.sum(gx)
    Vy = -np.sum(gy)
    Vz = -np.sum(gz)
    return V, Vx, Vy, Vz


# =============================================================================
#  itype = 2  (2nd derivatives only)
# =============================================================================

def tesseroid_gxx(lon1: np.ndarray, lat1: np.ndarray,
                  h1: np.ndarray, ref: np.ndarray,
                  lon: float, lat: float, r0: float,
                  resolution_x1: float, resolution_y1: float,
                  e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2nd derivatives only for an array of tesseroids (itype=2).

    Corresponds to TC_GUI.m L4891-L5139 (Tesseroid_gxx).
    """
    g = _tess_geometry(lon1, lat1, h1, ref, lon, lat, r0,
                       resolution_x1, resolution_y1, e)

    Kxx = g['l3'] * (g['l23'] * g['dxx'] - 1.0) * g['toi']
    Kxy = g['l3'] * (g['l23'] * g['dxy']) * g['toi']
    Kxz = g['l3'] * (g['l23'] * g['dxz']) * g['toi']
    Kyy = g['l3'] * (g['l23'] * g['dyy'] - 1.0) * g['toi']
    Kyz = g['l3'] * (g['l23'] * g['dyz']) * g['toi']
    Kzz = g['l3'] * (g['l23'] * g['dzz'] - 1.0) * g['toi']

    Lxx = g['c'] * (Kxx + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a11'] + g['toi1'] * g['b111'] + g['toi'] * g['gama111'])
        + g['yy_2'] * (g['toi22'] * g['a11'] + g['toi2'] * g['b112'] + g['toi'] * g['gama112'])
        + g['zz_2'] * (g['toi32'] * g['a11'] + g['toi3'] * g['b113'] + g['toi'] * g['gama113'])))
    Lxy = g['c'] * (Kxy + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a12'] + g['toi1'] * g['b121'] + g['toi'] * g['gama121'])
        + g['yy_2'] * (g['toi22'] * g['a12'] + g['toi2'] * g['b122'] + g['toi'] * g['gama122'])
        + g['zz_2'] * (g['toi32'] * g['a12'] + g['toi3'] * g['b123'] + g['toi'] * g['gama123'])))
    Lxz = g['c'] * (Kxz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a13'] + g['toi1'] * g['b131'] + g['toi'] * g['gama131'])
        + g['yy_2'] * (g['toi22'] * g['a13'] + g['toi2'] * g['b132'] + g['toi'] * g['gama132'])
        + g['zz_2'] * (g['toi32'] * g['a13'] + g['toi3'] * g['b133'] + g['toi'] * g['gama133'])))
    Lyy = g['c'] * (Kyy + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a22'] + g['toi1'] * g['b221'] + g['toi'] * g['gama221'])
        + g['yy_2'] * (g['toi22'] * g['a22'] + g['toi2'] * g['b222'] + g['toi'] * g['gama222'])
        + g['zz_2'] * (g['toi32'] * g['a22'] + g['toi3'] * g['b223'] + g['toi'] * g['gama223'])))
    Lyz = g['c'] * (Kyz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a23'] + g['toi1'] * g['b231'] + g['toi'] * g['gama231'])
        + g['yy_2'] * (g['toi22'] * g['a23'] + g['toi2'] * g['b232'] + g['toi'] * g['gama232'])
        + g['zz_2'] * (g['toi32'] * g['a23'] + g['toi3'] * g['b233'] + g['toi'] * g['gama233'])))
    Lzz = g['c'] * (Kzz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a33'] + g['toi1'] * g['b331'] + g['toi'] * g['gama331'])
        + g['yy_2'] * (g['toi22'] * g['a33'] + g['toi2'] * g['b332'] + g['toi'] * g['gama332'])
        + g['zz_2'] * (g['toi32'] * g['a33'] + g['toi3'] * g['b333'] + g['toi'] * g['gama333'])))

    Vxx = np.sum(Lxx) * 1e4
    Vxy = np.sum(Lxy) * 1e4
    Vxz = np.sum(Lxz) * 1e4
    Vyy = np.sum(Lyy) * 1e4
    Vyz = np.sum(Lyz) * 1e4
    Vzz = np.sum(Lzz) * 1e4
    return Vxx, Vxy, Vxz, Vyy, Vyz, Vzz


# =============================================================================
#  itype = 10  (all 10 components)
# =============================================================================

def tesseroid(lon1: np.ndarray, lat1: np.ndarray,
              h1: np.ndarray, ref: np.ndarray,
              lon: float, lat: float, r0: float,
              resolution_x1: float, resolution_y1: float,
              e: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Compute all 10 gravity field components for an array of tesseroids.

    Corresponds to TC_GUI.m L4183-L4495 (Tesseroid).

    Returns:
        (V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz): 10 scalars.
    """
    g = _tess_geometry(lon1, lat1, h1, ref, lon, lat, r0,
                       resolution_x1, resolution_y1, e)

    K000 = (1.0 / g['l']) * g['toi']
    Lx = g['l3'] * g['dx'] * g['toi']
    Ly = g['l3'] * g['dy'] * g['toi']
    Lz = g['l3'] * g['dz'] * g['toi']

    # potential (uses a = l2)
    pot = g['c'] * (K000 + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a'] + g['toi1'] * g['b1'] + g['toi'] * g['gama1'])
        + g['yy_2'] * (g['toi22'] * g['a'] + g['toi2'] * g['b2'] + g['toi'] * g['gama2'])
        + g['zz_2'] * (g['toi32'] * g['a'] + g['toi3'] * g['b3'] + g['toi'] * g['gama3'])))
    V = np.sum(pot) * 1e-5

    # 1st derivatives
    gx = g['c'] * (Lx + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a1'] + g['toi1'] * g['b11'] + g['toi'] * g['gama11'])
        + g['yy_2'] * (g['toi22'] * g['a1'] + g['toi2'] * g['b12'] + g['toi'] * g['gama12'])
        + g['zz_2'] * (g['toi32'] * g['a1'] + g['toi3'] * g['b13'] + g['toi'] * g['gama13'])))
    gy = g['c'] * (Ly + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a2'] + g['toi1'] * g['b21'] + g['toi'] * g['gama21'])
        + g['yy_2'] * (g['toi22'] * g['a2'] + g['toi2'] * g['b22'] + g['toi'] * g['gama22'])
        + g['zz_2'] * (g['toi32'] * g['a2'] + g['toi3'] * g['b23'] + g['toi'] * g['gama23'])))
    gz = g['c'] * (Lz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a3'] + g['toi1'] * g['b31'] + g['toi'] * g['gama31'])
        + g['yy_2'] * (g['toi22'] * g['a3'] + g['toi2'] * g['b32'] + g['toi'] * g['gama32'])
        + g['zz_2'] * (g['toi32'] * g['a3'] + g['toi3'] * g['b33'] + g['toi'] * g['gama33'])))
    Vx = -np.sum(gx)
    Vy = -np.sum(gy)
    Vz = -np.sum(gz)

    # 2nd derivatives
    Kxx = g['l3'] * (g['l23'] * g['dxx'] - 1.0) * g['toi']
    Kxy = g['l3'] * (g['l23'] * g['dxy']) * g['toi']
    Kxz = g['l3'] * (g['l23'] * g['dxz']) * g['toi']
    Kyy = g['l3'] * (g['l23'] * g['dyy'] - 1.0) * g['toi']
    Kyz = g['l3'] * (g['l23'] * g['dyz']) * g['toi']
    Kzz = g['l3'] * (g['l23'] * g['dzz'] - 1.0) * g['toi']

    Lxx = g['c'] * (Kxx + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a11'] + g['toi1'] * g['b111'] + g['toi'] * g['gama111'])
        + g['yy_2'] * (g['toi22'] * g['a11'] + g['toi2'] * g['b112'] + g['toi'] * g['gama112'])
        + g['zz_2'] * (g['toi32'] * g['a11'] + g['toi3'] * g['b113'] + g['toi'] * g['gama113'])))
    Lxy = g['c'] * (Kxy + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a12'] + g['toi1'] * g['b121'] + g['toi'] * g['gama121'])
        + g['yy_2'] * (g['toi22'] * g['a12'] + g['toi2'] * g['b122'] + g['toi'] * g['gama122'])
        + g['zz_2'] * (g['toi32'] * g['a12'] + g['toi3'] * g['b123'] + g['toi'] * g['gama123'])))
    Lxz = g['c'] * (Kxz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a13'] + g['toi1'] * g['b131'] + g['toi'] * g['gama131'])
        + g['yy_2'] * (g['toi22'] * g['a13'] + g['toi2'] * g['b132'] + g['toi'] * g['gama132'])
        + g['zz_2'] * (g['toi32'] * g['a13'] + g['toi3'] * g['b133'] + g['toi'] * g['gama133'])))
    Lyy = g['c'] * (Kyy + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a22'] + g['toi1'] * g['b221'] + g['toi'] * g['gama221'])
        + g['yy_2'] * (g['toi22'] * g['a22'] + g['toi2'] * g['b222'] + g['toi'] * g['gama222'])
        + g['zz_2'] * (g['toi32'] * g['a22'] + g['toi3'] * g['b223'] + g['toi'] * g['gama223'])))
    Lyz = g['c'] * (Kyz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a23'] + g['toi1'] * g['b231'] + g['toi'] * g['gama231'])
        + g['yy_2'] * (g['toi22'] * g['a23'] + g['toi2'] * g['b232'] + g['toi'] * g['gama232'])
        + g['zz_2'] * (g['toi32'] * g['a23'] + g['toi3'] * g['b233'] + g['toi'] * g['gama233'])))
    Lzz = g['c'] * (Kzz + (1.0 / 24.0) * g['l3'] * (
        g['xx_2'] * (g['toi12'] * g['a33'] + g['toi1'] * g['b331'] + g['toi'] * g['gama331'])
        + g['yy_2'] * (g['toi22'] * g['a33'] + g['toi2'] * g['b332'] + g['toi'] * g['gama332'])
        + g['zz_2'] * (g['toi32'] * g['a33'] + g['toi3'] * g['b333'] + g['toi'] * g['gama333'])))

    Vxx = np.sum(Lxx) * 1e4
    Vxy = np.sum(Lxy) * 1e4
    Vxz = np.sum(Lxz) * 1e4
    Vyy = np.sum(Lyy) * 1e4
    Vyz = np.sum(Lyz) * 1e4
    Vzz = np.sum(Lzz) * 1e4

    return V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz
