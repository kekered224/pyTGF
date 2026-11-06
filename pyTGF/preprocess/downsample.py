"""
Block-mean down-sampling

Python port of MATLAB boxmean.m (CH 2012-06-01):
    Y = boxmean(X, red_fac)

Computes mean values of input matrix X over blocks of size
red_fac x red_fac and returns the down-sampled output matrix Y.

Example:
    X        = 4000 x 6000 input matrix
    red_fac  = 5
    -> Y     = 800 x 1200 output matrix (5x5 block means)

Implemented with pure numpy (reshape + mean) so it works across all
scipy versions — scipy.ndimage.block_reduce was removed in newer
releases.
"""

import numpy as np

from ..io.dem_io import DEMData


def boxmean(x: np.ndarray, red_fac: int) -> np.ndarray:
    """
    Block-mean down-sample a 2D array by an integer factor.

    Args:
        x: 2D input array.
        red_fac: integer block size (rows and columns reduced by this factor).

    Returns:
        Down-sampled array of shape (n_rows//red_fac, n_cols//red_fac).

    Raises:
        ValueError: if red_fac < 1 or array dimensions are not divisible.

    Notes:
        Mirrors MATLAB boxmean.m behavior exactly: requires the input
        dimensions to be divisible by red_fac (no padding / trimming).
    """
    if red_fac < 1:
        raise ValueError(f"red_fac must be >= 1, got {red_fac}")
    if red_fac == 1:
        return x.copy()

    n_rows, n_cols = x.shape
    if n_rows % red_fac != 0 or n_cols % red_fac != 0:
        raise ValueError(
            f"Input shape {(n_rows, n_cols)} not divisible by red_fac={red_fac}. "
            f"MATLAB boxmean requires exact divisibility. "
            f"Trim or pad the array first."
        )

    # Reshape into (n_rows//rf, rf, n_cols//rf, rf) and mean over the
    # block axes — numerically identical to MATLAB's mat2cell->cellfun(@mean).
    new_rows = n_rows // red_fac
    new_cols = n_cols // red_fac
    return x.reshape(new_rows, red_fac, new_cols, red_fac).mean(axis=(1, 3))


def boxmean_dem(dem: DEMData, red_fac: int) -> DEMData:
    """
    Down-sample a DEMData grid by an integer factor, updating resolution.

    Returns a new DEMData with:
        z       = boxmean(dem.z, red_fac)
        reslat  = dem.reslat * red_fac
        reslon  = dem.reslon * red_fac
        bounds  = unchanged (the grid still covers the same region)
    """
    z_new = boxmean(dem.z, red_fac)
    return DEMData(
        minlat=dem.minlat,
        maxlat=dem.maxlat,
        minlon=dem.minlon,
        maxlon=dem.maxlon,
        reslat=dem.reslat * red_fac,
        reslon=dem.reslon * red_fac,
        z=z_new,
    )
