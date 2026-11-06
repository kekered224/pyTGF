"""
Normal gravity computation

Mirrors TC_GUI.m L2041-L2070 (normal_gravity_generic).

Uses Somigliana formula (Torge 2002, Eq. 4.41b) for gamma_0 and the
vertical gradient (Torge 2002, Eq. 4.63) up to 2nd order in h.

Note: MATLAB uses Pi = 3.141592654 (truncated) and the GRS80 parameters
hard-coded inside the function. We replicate the exact same constants
to guarantee bit-for-bit parity.
"""

import numpy as np

from ..config import PhysicalConstants


def normal_gravity_generic(phi: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Calculate normal gravity at station height.

    Corresponds to TC_GUI.m L2041-L2070 (normal_gravity_generic).

    MATLAB:
        a = 6378137;
        b = 6356752.3141;
        gamma_a = 9.7803267715;
        gamma_b = 9.8321863685;
        m = 0.00344978600308;
        pi = 3.141592654;
        phi = phi*pi/180;
        k   = b*gamma_b/(a*gamma_a)-1;
        f   = (a-b)/a;
        a2  = a*a;
        e2  = (a2-b*b)/(a2);
        sinphi = sin(phi);
        gamma_0 = gamma_a * (1+k*sinphi.*sinphi)./sqrt(1-e2.*sinphi.*sinphi);
        delta_gamma_h = gamma_0 .* ( -2/a * (1+f+m-2*f*sinphi.*sinphi).*h ...
                                     + 3/(a2)*h.*h );
        gamma_h = gamma_0 + delta_gamma_h;

    Args:
        phi: Latitude in degrees.
        h:   Geodetic height in meters.

    Returns:
        gamma_h: Normal gravity at height h [m/s^2].
    """
    # GRS80 ellipsoid parameters (hard-coded to match MATLAB exactly)
    a = PhysicalConstants.A          # 6378137.0
    b = PhysicalConstants.B          # 6356752.3141
    gamma_a = PhysicalConstants.GAMMA_A  # 9.7803267715
    gamma_b = PhysicalConstants.GAMMA_B  # 9.8321863685
    m = PhysicalConstants.M          # 0.00344978600308
    pi = PhysicalConstants.PI        # 3.141592654

    # conversion angle from deg to rad
    phi = phi * pi / 180.0

    # auxiliary quantities k, f, e2
    k = b * gamma_b / (a * gamma_a) - 1
    f = (a - b) / a
    a2 = a * a
    e2 = (a2 - b * b) / a2

    # Somigliana formula (Torge 2002, Eq. 4.41b)
    sinphi = np.sin(phi)
    sinphi2 = sinphi * sinphi
    gamma_0 = gamma_a * (1 + k * sinphi2) / np.sqrt(1 - e2 * sinphi2)

    # Vertical gravity gradient, 2nd-order series (Torge 2002, Eq. 4.63)
    delta_gamma_h = gamma_0 * (
        -2.0 / a * (1 + f + m - 2 * f * sinphi2) * h
        + 3.0 / a2 * h * h
    )

    gamma_h = gamma_0 + delta_gamma_h
    return gamma_h
