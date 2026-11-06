"""
Coordinate transformations

Mirrors TC_GUI.m:
    global2local    (L2228-L2260)  spherical -> prism-based local Cartesian
    g_prism2point   (L2122-L2155)  rotate 1st-derivative vector prism->station
    ggt_prism2point (L2157-L2226)  rotate 2nd-derivative tensor prism->station

Note: MATLAB uses Pi = 3.141592654 (truncated) throughout. We use the same
constant via PhysicalConstants.PI to guarantee bit-for-bit parity.
"""

from typing import Tuple
import numpy as np

from ..config import PhysicalConstants


def global2local(lon: np.ndarray, lat: np.ndarray, r: np.ndarray,
                 prism_lon: np.ndarray, prism_lat: np.ndarray,
                 prism_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform the spherical coordinates of a station to prism-based local
    Cartesian coordinates.

    Corresponds to TC_GUI.m L2228-L2260 (global2local).

    The local frame is centered at (prism_lon, prism_lat, prism_r) with axes:
        x -> north, y -> east, z -> down (radial inward, note the leading '-').

    MATLAB:
        d2r = Pi/180;
        prism_lon=d2r*prism_lon; prism_lat=d2r*prism_lat;
        lon=d2r*lon; lat=d2r*lat;
        g_x = r.*coslat.*coslon - prism_r.*cosprism_lat.*cosprism_lon;
        g_y = r.*coslat.*sinlon - prism_r.*cosprism_lat.*sinprism_lon;
        g_z = r.*sinlat - prism_r.*sinprism_lat;
        x = -g_x.*sinprism_lat.*cosprism_lon - g_y.*sinprism_lat.*sinprism_lon + g_z.*cosprism_lat;
        y = -g_x.*sinprism_lon + g_y.*cosprism_lon;
        z = -(g_x.*cosprism_lat.*cosprism_lon + g_y.*cosprism_lat.*sinprism_lon + g_z.*sinprism_lat);

    Args:
        lon, lat, r: Station spherical coordinates (degrees, degrees, meters).
        prism_lon, prism_lat, prism_r: Prism center coordinates.

    Returns:
        (x, y, z): Local Cartesian coordinates of the station.
    """
    d2r = PhysicalConstants.d2r()

    prism_lon = d2r * prism_lon
    prism_lat = d2r * prism_lat
    lon = d2r * lon
    lat = d2r * lat

    coslat = np.cos(lat)
    coslon = np.cos(lon)
    sinlat = np.sin(lat)
    sinlon = np.sin(lon)

    cosprism_lon = np.cos(prism_lon)
    cosprism_lat = np.cos(prism_lat)
    sinprism_lon = np.sin(prism_lon)
    sinprism_lat = np.sin(prism_lat)

    g_x = r * coslat * coslon - prism_r * cosprism_lat * cosprism_lon
    g_y = r * coslat * sinlon - prism_r * cosprism_lat * sinprism_lon
    g_z = r * sinlat - prism_r * sinprism_lat

    x = (-g_x * sinprism_lat * cosprism_lon
         - g_y * sinprism_lat * sinprism_lon
         + g_z * cosprism_lat)
    y = -g_x * sinprism_lon + g_y * cosprism_lon
    z = -(g_x * cosprism_lat * cosprism_lon
          + g_y * cosprism_lat * sinprism_lon
          + g_z * sinprism_lat)

    return x, y, z


def g_prism2point(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
                  lon: float, lat: float,
                  prism_lon: np.ndarray, prism_lat: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate gravity vector from prism-based frame to station-based frame.

    Corresponds to TC_GUI.m L2122-L2155 (g_prism2point).

    Used to overcome Earth's curvature effect when summing prism
    contributions from different prism centers around a single station.

    MATLAB:
        cosbeta = cos(prism_lon - lon);
        sinbeta = sin(prism_lon - lon);
        cosphi = cos(lat); sinphi = sin(lat);
        cosphil = cos(prism_lat); sinphil = sin(prism_lat);
        R0 = cosbeta*sinphi*sinphil + cosphi*cosphil;
        R1 = sinbeta*sinphi;
        R2 = -cosbeta*sinphi*cosphil + cosphi*sinphil;
        R3 = -sinbeta*sinphil;
        R4 = cosbeta;
        R5 = sinbeta*cosphil;
        R6 = -cosbeta*cosphi*sinphil + sinphi*cosphil;
        R7 = -sinbeta*cosphi;
        R8 = cosbeta*cosphi*cosphil + sinphi*sinphil;
        Vx = R0*gx + R1*gy + R2*gz;
        Vy = R3*gx + R4*gy + R5*gz;
        Vz = R6*gx + R7*gy + R8*gz;

    Args:
        gx, gy, gz: Gravity components in prism-based frame (arrays).
        lon, lat: Station coordinates [deg] (scalars).
        prism_lon, prism_lat: Prism center coordinates [deg] (arrays).

    Returns:
        (Vx, Vy, Vz): Gravity components in station-based frame.
    """
    d2r = PhysicalConstants.d2r()

    prism_lon = d2r * prism_lon
    prism_lat = d2r * prism_lat
    lon = d2r * lon
    lat = d2r * lat

    cosbeta = np.cos(prism_lon - lon)
    sinbeta = np.sin(prism_lon - lon)
    cosphi = np.cos(lat)
    sinphi = np.sin(lat)
    cosphil = np.cos(prism_lat)
    sinphil = np.sin(prism_lat)

    cosbetasinphi = cosbeta * sinphi
    cosbetacosphi = cosbeta * cosphi

    R0 = cosbetasinphi * sinphil + cosphi * cosphil
    R1 = sinbeta * sinphi
    R2 = -cosbetasinphi * cosphil + cosphi * sinphil
    R3 = -sinbeta * sinphil
    R4 = cosbeta
    R5 = sinbeta * cosphil
    R6 = -cosbetacosphi * sinphil + sinphi * cosphil
    R7 = -sinbeta * cosphi
    R8 = cosbetacosphi * cosphil + sinphi * sinphil

    Vx = R0 * gx + R1 * gy + R2 * gz
    Vy = R3 * gx + R4 * gy + R5 * gz
    Vz = R6 * gx + R7 * gy + R8 * gz

    return Vx, Vy, Vz


def ggt_prism2point(gxx: np.ndarray, gxy: np.ndarray, gxz: np.ndarray,
                    gyy: np.ndarray, gyz: np.ndarray, gzz: np.ndarray,
                    lon: float, lat: float,
                    prism_lon: np.ndarray, prism_lat: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate gravity gradient tensor from prism-based to station-based frame.

    Corresponds to TC_GUI.m L2157-L2226 (ggt_prism2point).

    The symmetric tensor is stored as 6 unique components
    (gxx, gxy, gxz, gyy, gyz, gzz). Internally the full 3x3 matrix is
    reconstructed (atprism0..8), rotated by R, then the unique components
    of the result are returned.

    MATLAB:
        atprism0=gxx; atprism1=gxy; atprism2=gxz;
        atprism3=gxy; atprism4=gyy; atprism5=gyz;
        atprism6=gxz; atprism7=gyz; atprism8=gzz;
        (compute R0..R8 same as g_prism2point)
        t0=R0*atprism0+R1*atprism3+R2*atprism6;
        t1=R0*atprism1+R1*atprism4+R2*atprism7;
        t2=R0*atprism2+R1*atprism5+R2*atprism8;
        t3=R3*atprism0+R4*atprism3+R5*atprism6;
        t4=R3*atprism1+R4*atprism4+R5*atprism7;
        t5=R3*atprism2+R4*atprism5+R5*atprism8;
        t6=R6*atprism0+R7*atprism3+R8*atprism6;
        t7=R6*atprism1+R7*atprism4+R8*atprism7;
        t8=R6*atprism2+R7*atprism5+R8*atprism8;
        V0=t0*R0+t1*R1+t2*R2;   % Vxx
        V1=t0*R3+t1*R4+t2*R5;   % Vxy
        V2=t0*R6+t1*R7+t2*R8;   % Vxz
        V4=t3*R3+t4*R4+t5*R5;   % Vyy
        V7=t6*R3+t7*R4+t8*R5;   % Vyz
        V8=t6*R6+t7*R7+t8*R8;   % Vzz

    Args:
        gxx, gxy, gxz, gyy, gyz, gzz: Gradient components in prism frame.
        lon, lat: Station coordinates [deg] (scalars).
        prism_lon, prism_lat: Prism center coordinates [deg] (arrays).

    Returns:
        (Vxx, Vxy, Vxz, Vyy, Vyz, Vzz): Gradient tensor in station frame.
    """
    d2r = PhysicalConstants.d2r()

    atprism0 = gxx
    atprism1 = gxy
    atprism2 = gxz
    atprism3 = gxy
    atprism4 = gyy
    atprism5 = gyz
    atprism6 = gxz
    atprism7 = gyz
    atprism8 = gzz

    prism_lon = d2r * prism_lon
    prism_lat = d2r * prism_lat
    lon = d2r * lon
    lat = d2r * lat

    cosbeta = np.cos(prism_lon - lon)
    sinbeta = np.sin(prism_lon - lon)
    cosphi = np.cos(lat)
    sinphi = np.sin(lat)
    cosphil = np.cos(prism_lat)
    sinphil = np.sin(prism_lat)

    cosbetasinphi = cosbeta * sinphi
    cosbetacosphi = cosbeta * cosphi

    R0 = cosbetasinphi * sinphil + cosphi * cosphil
    R1 = sinbeta * sinphi
    R2 = -cosbetasinphi * cosphil + cosphi * sinphil
    R3 = -sinbeta * sinphil
    R4 = cosbeta
    R5 = sinbeta * cosphil
    R6 = -cosbetacosphi * sinphil + sinphi * cosphil
    R7 = -sinbeta * cosphi
    R8 = cosbetacosphi * cosphil + sinphi * sinphil

    t0 = R0 * atprism0 + R1 * atprism3 + R2 * atprism6
    t1 = R0 * atprism1 + R1 * atprism4 + R2 * atprism7
    t2 = R0 * atprism2 + R1 * atprism5 + R2 * atprism8
    t3 = R3 * atprism0 + R4 * atprism3 + R5 * atprism6
    t4 = R3 * atprism1 + R4 * atprism4 + R5 * atprism7
    t5 = R3 * atprism2 + R4 * atprism5 + R5 * atprism8
    t6 = R6 * atprism0 + R7 * atprism3 + R8 * atprism6
    t7 = R6 * atprism1 + R7 * atprism4 + R8 * atprism7
    t8 = R6 * atprism2 + R7 * atprism5 + R8 * atprism8

    Vxx = t0 * R0 + t1 * R1 + t2 * R2
    Vxy = t0 * R3 + t1 * R4 + t2 * R5
    Vxz = t0 * R6 + t1 * R7 + t2 * R8
    Vyy = t3 * R3 + t4 * R4 + t5 * R5
    Vyz = t6 * R3 + t7 * R4 + t8 * R5
    Vzz = t6 * R6 + t7 * R7 + t8 * R8

    return Vxx, Vxy, Vxz, Vyy, Vyz, Vzz
