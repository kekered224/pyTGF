"""
End-to-end pyTGF pipeline — Python port of xxh_tgf2.m

This script reproduces the workflow of the MATLAB xxh_tgf2.m driver:
    1. Read computation points from an ASCII file.
    2. Read MERIT DEM (3") and REF (30") from the SIS tile database
       (port of dem2017YM_MERIT_R1.m).
    3. Resample REF onto the DEM grid (cubic), zero REF over ocean.
    4. Build the three DEM layers (Detailed, Tesseroid, Coarse) using the
       same indexing logic as xxh_tgf2.m.
    5. Write all layers to the TGF binary format.
    6. Call TC_GUI to perform the gravity-field calculation.

Usage:
    python examples/run_colorado.py

Edit the CONFIGURATION section below to match your data layout.
"""

import os
import sys
import time
import numpy as np

# Resolve paths relative to this script (works from anywhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyTGF import TGFConfig, TC_GUI
from pyTGF.io import write_com_points, write_dem
from pyTGF.io.dem_io import DEMData
from pyTGF.preprocess import (
    read_merit_dem, boxmean, read_compoints_ascii,
)


# ============================================================================
# CONFIGURATION  (edit these to match your data layout)
# ============================================================================

# --- Input data paths ---
INPUT_POINTS_FILE = r"D:\research2026\Colorado\TGF_anomly\compute_points\1.dat"
PATH_SIS_DATA     = r"D:\research2026\Colorado\DEM\SIS\DEMs_Earth"  # SIS tile database
DEM_NAME          = "MERIT2017_R1"   # 3" high-resolution DEM
REF_NAME          = "MERIT_2160"     # 30" reference surface

# --- Output base directory (all products go under here) ---
BASE_OUT_PATH     = r"D:\research2026\Colorado\TGF_anomly"

# --- Gravity-field calculation parameters (see TGF User Manual) ---
IKIND      = 2                        # 1=topographic, 2=RTM
ITYPE      = 104                      # 0/1/2/4/10/103/104
IDENSITY   = 0                        # 0=constant, 1=density map
FLAG_EARTH = 0                        # 0=sphere, 1=ellipsoid
RZONES     = [0.03, 0.5, 0.7, 2.0]  # [r1, r2, r3, r4] in degrees (match MATLAB xxh_tgf2.m)
E_DENSITY  = 2.67                     # g/cm^3

# --- DEM grid parameters ---
R1 = 3                                # Base DEM resolution [arc-seconds]
SF = 20                               # Smoothing factor for Coarse layer
FAC_X = R1 / 3.0                      # Interpolation factor X (match MATLAB)
FAC_Y = R1 / 3.0                      # Interpolation factor Y
METHOD    = 'linear'                  # Interpolation method
SEA       = 0.0                       # Ocean fill value

# --- Test control ---
# Set TEST_NUM = None to process all points, or an int to limit the run
TEST_NUM = 10


# ============================================================================
# Helpers
# ============================================================================

def get_filename(lat: float, lon: float) -> str:
    """
    Generate a filename prefix from (lat, lon), mirroring MATLAB getfilename.m.

    Example: (37, -107) -> 'W107N37_dg.dat'
    """
    lon_dir = 'E' if lon >= 0 else 'W'
    if abs(lon) < 100:
        lon_dir += '0'
    if abs(lon) < 10:
        lon_dir += '0'
    lat_dir = 'N' if lat >= 0 else 'S'
    if abs(lat) < 10:
        lat_dir += '0'
    return f"{lon_dir}{abs(int(lon))}{lat_dir}{abs(int(lat))}_dg.dat"


def resample_ref_to_dem(ref: DEMData, dem: DEMData) -> np.ndarray:
    """
    Resample REF onto the DEM grid using cubic spline interpolation.

    Mirrors MATLAB:
        H_REF = interp2(x1_ref, y1_ref, z1_ref, x1, y1, 'cubic');

    Uses scipy.ndimage.map_coordinates with spline order=3 (cubic), which
    is much faster than RegularGridInterpolator(method='cubic') for large
    grids because it avoids solving a global sparse linear system.
    """
    from scipy.ndimage import map_coordinates

    # Ensure REF has no non-finite values (map_coordinates requires finite
    # input for cubic splines).
    z_ref = np.where(np.isfinite(ref.z), ref.z, 0.0)

    # Compute fractional indices into the REF grid for each DEM pixel.
    # REF grid is regular: lats[i] = ref.minlat + i * ref.reslat
    #                     lons[j] = ref.minlon + j * ref.reslon
    # So the fractional index of a DEM pixel at (lat, lon) is:
    #   row = (lat - ref.minlat) / ref.reslat
    #   col = (lon - ref.minlon) / ref.reslon
    rows = (dem.lats - ref.minlat) / ref.reslat
    cols = (dem.lons - ref.minlon) / ref.reslon

    # Build a 2D coordinate array for map_coordinates.
    # map_coordinates expects shape (ndim, ...) with row=axis0, col=axis1.
    lon_grid, lat_grid = np.meshgrid(cols, rows)
    coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=0)

    # order=3 -> cubic spline (matches MATLAB 'cubic')
    # mode='nearest' -> extrapolate boundary (avoids NaN at edges)
    result = map_coordinates(
        z_ref, coords, order=3, mode='nearest'
    )
    return result.reshape(dem.z.shape)


# ============================================================================
# Pipeline
# ============================================================================

def main():
    t_start = time.time()
    t_step = t_start

    # ------------------------------------------------------------------
    # 1. Create output directories
    # ------------------------------------------------------------------
    com_dir       = os.path.join(BASE_OUT_PATH, 'COM')
    out_dir       = os.path.join(BASE_OUT_PATH, 'OUTFILE')
    dem_mass_dir  = os.path.join(BASE_OUT_PATH, 'DEM_MASS')
    for d in (com_dir, out_dir, dem_mass_dir):
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Read computation points  (mirrors MATLAB section 2)
    # ------------------------------------------------------------------
    print(f"[1/7] Reading computation points: {INPUT_POINTS_FILE}")
    compoints = read_compoints_ascii(INPUT_POINTS_FILE, lat_first=False)
    n_total = compoints.n_points
    n_run = n_total if TEST_NUM is None else min(TEST_NUM, n_total)
    print(f"      {n_total} points found, processing {n_run} point(s)")

    lats = compoints.lats[:n_run]
    lons = compoints.lons[:n_run]
    heights = compoints.heights[:n_run]
    ids = np.arange(1, n_run + 1, dtype=np.float64)

    minlon_pt = np.floor(lons.min())
    maxlon_pt = np.ceil(lons.max())
    minlat_pt = np.floor(lats.min())
    maxlat_pt = np.ceil(lats.max())
    print(f"      Point bounds: lon [{minlon_pt}, {maxlon_pt}], "
          f"lat [{minlat_pt}, {maxlat_pt}]")

    # ------------------------------------------------------------------
    # 3. Write ComPoints.bin  (mirrors MATLAB section 3)
    #    MATLAB: com = [N' com_lat(:) com_lon(:) com_h(:)];
    # ------------------------------------------------------------------
    com_points_path = os.path.join(com_dir, 'ComPoints.bin')
    write_com_points(com_points_path, lats=lats, lons=lons,
                     heights=heights, ids=ids)
    print(f"      Wrote {com_points_path}")
    print(f"      [time] {time.time() - t_step:.1f}s")
    t_step = time.time()

    # ------------------------------------------------------------------
    # 4. Read DEM and REF from SIS tile database  (mirrors section 5)
    #    MATLAB:
    #      B1 = max(rzones);
    #      minlon1 = floor(minlon_pt) - B1 - 0.1;
    #      maxlon1 = floor(maxlon_pt) + 1 + B1 + 0.1;
    #      [x1,y1,z1] = dem2017YM_MERIT_R1(dem_name, minlon1,maxlon1,
    #                                      minlat1,maxlat1, facX,facY,
    #                                      pathSIS_data, sea, method);
    # ------------------------------------------------------------------
    B1 = max(RZONES)
    minlon1 = minlon_pt - B1 - 0.1
    maxlon1 = maxlon_pt + 1 + B1 + 0.1
    minlat1 = minlat_pt - B1 - 0.1
    maxlat1 = maxlat_pt + 1 + B1 + 0.1
    print(f"\n[2/7] Extended bounds: lon [{minlon1}, {maxlon1}], "
          f"lat [{minlat1}, {maxlat1}]")

    print(f"      Reading DEM: {DEM_NAME} from SIS database")
    dem = read_merit_dem(
        dem_name=DEM_NAME,
        minlon=minlon1, maxlon=maxlon1,
        minlat=minlat1, maxlat=maxlat1,
        sis_data_path=PATH_SIS_DATA,
        sea=SEA, facX=FAC_X, facY=FAC_Y, method=METHOD,
    )
    print(f"      DEM shape: {dem.z.shape}, "
          f"res: {dem.reslat:.6f} deg ({R1} arc-sec)")

    print(f"      Reading REF: {REF_NAME} from SIS database")
    ref = read_merit_dem(
        dem_name=REF_NAME,
        minlon=minlon1, maxlon=maxlon1,
        minlat=minlat1, maxlat=maxlat1,
        sis_data_path=PATH_SIS_DATA,
        sea=SEA, facX=FAC_X, facY=FAC_Y, method=METHOD,
    )
    print(f"      REF shape: {ref.z.shape}, "
          f"res: {ref.reslat:.6f} deg")
    print(f"      [time] {time.time() - t_step:.1f}s")
    t_step = time.time()

    # ------------------------------------------------------------------
    # 5. Resample REF onto DEM grid  (mirrors MATLAB section 5)
    #    H_REF = interp2(x1_ref, y1_ref, z1_ref, x1, y1, 'cubic');
    #    z1_ref(z1 == 0) = 0;
    # ------------------------------------------------------------------
    print(f"\n[3/7] Resampling REF onto DEM grid (cubic)")
    z1 = dem.z
    z1_ref = resample_ref_to_dem(ref, dem)
    z1_ref[z1 == 0] = 0.0
    print(f"      Resampled REF shape: {z1_ref.shape}")
    print(f"      [time] {time.time() - t_step:.1f}s")
    t_step = time.time()

    # ------------------------------------------------------------------
    # 6. Build the three DEM layers  (mirrors MATLAB section 6)
    # ------------------------------------------------------------------
    print(f"\n[4/7] Building DEM layers")
    res_lon = R1 / 3600.0
    res_lat = R1 / 3600.0

    # --- Coarse layer (boxmean down-sample by SF) ---
    # MATLAB: x2=boxmean(x1,sf); y2=boxmean(y1,sf);
    #         z2=boxmean(z1,sf); z2_ref=boxmean(z1_ref,sf);
    z2 = boxmean(z1, SF)
    z2_ref = boxmean(z1_ref, SF)
    res_lat2 = SF * res_lat
    res_lon2 = SF * res_lon
    # Coarse grid coordinates: boxmean of coordinate matrix is equivalent
    # to picking every SF-th cell center, starting at offset (SF-1)/2
    coarse_lats = dem.lats[(SF - 1) // 2::SF][:z2.shape[0]]
    coarse_lons = dem.lons[(SF - 1) // 2::SF][:z2.shape[1]]
    coarse_dem = DEMData(
        minlat=float(coarse_lats[0]), maxlat=float(coarse_lats[-1]),
        minlon=float(coarse_lons[0]), maxlon=float(coarse_lons[-1]),
        reslat=res_lat2, reslon=res_lon2, z=z2)
    coarse_ref = DEMData(
        minlat=float(coarse_lats[0]), maxlat=float(coarse_lats[-1]),
        minlon=float(coarse_lons[0]), maxlon=float(coarse_lons[-1]),
        reslat=res_lat2, reslon=res_lon2, z=z2_ref)
    print(f"      Coarse: {z2.shape}, res={res_lat2:.6f} deg")

    # --- Detailed / Tess layer (index-based crop, mirrors MATLAB) ---
    # MATLAB:
    #   r3 = max(rzones(1:3));
    #   n1  = round((B1 - (r3 + 0.1)) / R1 * 3600);
    #   n2p = floor((maxlat_pt + (r3 + 0.1) - min(y1(:))) / R1 * 3600);
    #   n2l = floor((maxlon_pt + (r3 + 0.1) - min(x1(:))) / R1 * 3600);
    #   xd = x1(n1:n2p, n1:n2l);  % 1-based, inclusive
    r3 = max(RZONES[:3])
    n1  = int(round((B1 - (r3 + 0.1)) / R1 * 3600))
    n2p = int(np.floor((maxlat_pt + (r3 + 0.1) - dem.minlat) / R1 * 3600))
    n2l = int(np.floor((maxlon_pt + (r3 + 0.1) - dem.minlon) / R1 * 3600))

    # MATLAB 1-based inclusive n1:n2p  ->  Python 0-based [n1-1 : n2p]
    i0 = max(n1 - 1, 0)
    i1 = min(n2p, dem.z.shape[0])
    j0 = max(n1 - 1, 0)
    j1 = min(n2l, dem.z.shape[1])

    zd      = z1[i0:i1, j0:j1]
    zd_ref  = z1_ref[i0:i1, j0:j1]
    detailed_lats = dem.lats[i0:i1]
    detailed_lons = dem.lons[j0:j1]
    detailed_dem = DEMData(
        minlat=float(detailed_lats[0]), maxlat=float(detailed_lats[-1]),
        minlon=float(detailed_lons[0]), maxlon=float(detailed_lons[-1]),
        reslat=res_lat, reslon=res_lon, z=zd)
    detailed_ref = DEMData(
        minlat=float(detailed_lats[0]), maxlat=float(detailed_lats[-1]),
        minlon=float(detailed_lons[0]), maxlon=float(detailed_lons[-1]),
        reslat=res_lat, reslon=res_lon, z=zd_ref)
    print(f"      Detailed: {zd.shape}, res={res_lat:.6f} deg")

    # Tess layer: same resolution and crop as Detailed (mirrors MATLAB)
    tess_dem = detailed_dem
    tess_ref = detailed_ref
    print(f"      Tess:     {tess_dem.z.shape} (same as Detailed)")
    print(f"      [time] {time.time() - t_step:.1f}s")
    t_step = time.time()

    # ------------------------------------------------------------------
    # 7. Write all binary files  (mirrors MATLAB section 7)
    # ------------------------------------------------------------------
    print(f"\n[5/7] Writing binary DEM files to {dem_mass_dir}")
    detailed_dem_path = os.path.join(dem_mass_dir, 'DetailedDEM.bin')
    detailed_ref_path = os.path.join(dem_mass_dir, 'DetailedREF.bin')
    tess_dem_path     = os.path.join(dem_mass_dir, 'TessDEM.bin')
    tess_ref_path     = os.path.join(dem_mass_dir, 'TessREF.bin')
    coarse_dem_path   = os.path.join(dem_mass_dir, 'CoarseDEM.bin')
    coarse_ref_path   = os.path.join(dem_mass_dir, 'CoarseREF.bin')

    write_dem(detailed_dem_path, detailed_dem)
    write_dem(detailed_ref_path, detailed_ref)
    write_dem(tess_dem_path, tess_dem)
    write_dem(tess_ref_path, tess_ref)
    write_dem(coarse_dem_path, coarse_dem)
    write_dem(coarse_ref_path, coarse_ref)
    print(f"      6 files written")
    print(f"      [time] {time.time() - t_step:.1f}s")
    t_step = time.time()

    # ------------------------------------------------------------------
    # 8. Configure and run TC_GUI
    # ------------------------------------------------------------------
    filename = get_filename(minlat_pt, minlon_pt)
    outname = os.path.join(out_dir, filename)

    print(f"\n[6/7] Configuration")
    print(f"      ikind={IKIND}, itype={ITYPE}, idensity={IDENSITY}, "
          f"flag_earth={FLAG_EARTH}")
    print(f"      rzones={RZONES}, e={E_DENSITY}")
    print(f"      Output: {outname}")

    cfg = TGFConfig(
        vstpar='RUN',
        com_points=com_points_path,
        detailed_dem=detailed_dem_path,
        detailed_ref=detailed_ref_path,
        tess_dem=tess_dem_path,
        tess_ref=tess_ref_path,
        coarse_dem=coarse_dem_path,
        coarse_ref=coarse_ref_path,
        outname=outname,
        ikind=IKIND,
        itype=ITYPE,
        idensity=IDENSITY,
        flag_earth=FLAG_EARTH,
        rzones=RZONES,
        e=E_DENSITY,
    )
    print(f"      [time] {time.time() - t_step:.1f}s")
    t_step = time.time()

    print(f"\n[7/7] Running TC_GUI calculation...")
    result = TC_GUI(cfg)
    elapsed_calc = time.time() - t_step
    print(f"      Calculation done in {elapsed_calc:.1f}s")
    print(f"      Result: {result.get('output_path', outname)}")

    total = time.time() - t_start
    print(f"\nTotal pipeline time: {total:.1f}s")


if __name__ == "__main__":
    main()
