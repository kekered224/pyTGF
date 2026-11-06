"""
Helper terms for prism gravity formulas (Nagy et al. 2000/2002)

Mirrors TC_GUI.m:
    gz3term (L3569-L3572)  3-term sub-expression for gz
    g6term  (L3574-L3595)  6-term sub-expression for potential g

These sub-expressions are the building blocks of the rectangular prism
gravity formula. The singular-point index arrays (nx1, ny1, nz1) mark
positions where a coordinate is zero, which would produce NaNs in the
log/atan terms; those entries are set to zero (the correct limit).
"""

from typing import Tuple
import numpy as np


def gz3term(r: np.ndarray, xpos1: np.ndarray, ypos1: np.ndarray,
            zpos1: np.ndarray, nz1: np.ndarray) -> np.ndarray:
    """
    Compute the 3-term sub-expression used in prism gz calculation.

    Corresponds to TC_GUI.m L3569-L3572 (gz3term).

    MATLAB:
        t3 = -(zpos1).*atan(ypos1.*xpos1./((zpos1).*r));
        t3(nz1) = 0;

    Here nz1 holds the linear indices where zpos1 == 0 (singular points);
    the atan term tends to 0 there, so the result is explicitly zeroed to
    avoid 0/0 NaNs.

    Args:
        r: Distance from station to prism vertex.
        xpos1, ypos1, zpos1: Vertex coordinates relative to station.
        nz1: Indices (into the flattened arrays) where zpos1 ~ 0.

    Returns:
        t3: Sub-expression value (same shape as inputs).
    """
    # Compute with divide-by-zero suppressed; singular points are zeroed below
    # (mirrors MATLAB behavior where Inf/NaN at singular indices is overwritten)
    with np.errstate(divide='ignore', invalid='ignore'):
        t3 = -(zpos1) * np.arctan(ypos1 * xpos1 / ((zpos1) * r))
    # Zero out singular points (MATLAB: t3(nz1) = 0)
    t3 = np.asarray(t3, dtype=np.float64).copy()
    # Replace any NaN/Inf produced at singular points before explicit zeroing
    t3 = np.where(np.isfinite(t3), t3, 0.0)
    if nz1 is not None and np.size(nz1) > 0:
        t3.flat[nz1] = 0.0
    return t3


def g6term(r: np.ndarray, xpos1: np.ndarray, ypos1: np.ndarray,
           zpos1: np.ndarray,
           nx1: np.ndarray, ny1: np.ndarray, nz1: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 6-term sub-expression used in prism potential calculation.

    Corresponds to TC_GUI.m L3574-L3595 (g6term).

    MATLAB:
        t1 = xpos1.*ypos1.*log(zpos1+r);      t1(nx1)=0; t1(ny1)=0;
        t2 = ypos1.*(zpos1).*log(xpos1+r);    t2(ny1)=0; t2(nz1)=0;
        t3 = xpos1.*(zpos1).*log(ypos1+r);    t3(nx1)=0; t3(nz1)=0;
        t4 = -xpos1.^2/2.*atan(ypos1.*(zpos1)./(xpos1.*r));   t4(nx1)=0;
        t5 = -ypos1.^2/2.*atan(xpos1.*(zpos1)./(ypos1.*r));   t5(ny1)=0;
        t6 = -(zpos1).^2/2.*atan(ypos1.*xpos1./((zpos1).*r)); t6(nz1)=0;

    The 6 terms are:
        t1 = x*y*ln(z+r)
        t2 = y*z*ln(x+r)
        t3 = x*z*ln(y+r)
        t4 = -x^2/2 * atan(y*z/(x*r))
        t5 = -y^2/2 * atan(x*z/(y*r))
        t6 = -z^2/2 * atan(y*x/(z*r))

    Singular points (where the relevant coordinate is zero) are zeroed
    via the index arrays nx1, ny1, nz1 to avoid 0/0 NaNs.

    Args:
        r: Distance from station to prism vertex.
        xpos1, ypos1, zpos1: Vertex coordinates relative to station.
        nx1, ny1, nz1: Indices (into flattened arrays) where xpos1/ypos1/zpos1 ~ 0.

    Returns:
        (t1, t2, t3, t4, t5, t6): Six sub-expression arrays.
    """
    # Compute with divide-by-zero suppressed; singular points are zeroed below
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = xpos1 * ypos1 * np.log(zpos1 + r)
        t2 = ypos1 * (zpos1) * np.log(xpos1 + r)
        t3 = xpos1 * (zpos1) * np.log(ypos1 + r)
        t4 = -xpos1 ** 2 / 2.0 * np.arctan(ypos1 * (zpos1) / (xpos1 * r))
        t5 = -ypos1 ** 2 / 2.0 * np.arctan(xpos1 * (zpos1) / (ypos1 * r))
        t6 = -(zpos1) ** 2 / 2.0 * np.arctan(ypos1 * xpos1 / ((zpos1) * r))

    # Mutate in-place to zero singular/NaN/Inf points
    for arr in (t1, t2, t3, t4, t5, t6):
        arr[~np.isfinite(arr)] = 0.0

    # Zero out singular points exactly as MATLAB does
    if nx1 is not None and np.size(nx1) > 0:
        t1.flat[nx1] = 0.0
        t3.flat[nx1] = 0.0
        t4.flat[nx1] = 0.0
    if ny1 is not None and np.size(ny1) > 0:
        t1.flat[ny1] = 0.0
        t2.flat[ny1] = 0.0
        t5.flat[ny1] = 0.0
    if nz1 is not None and np.size(nz1) > 0:
        t2.flat[nz1] = 0.0
        t3.flat[nz1] = 0.0
        t6.flat[nz1] = 0.0

    return t1, t2, t3, t4, t5, t6
