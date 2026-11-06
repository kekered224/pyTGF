"""
Self-contained pyTGF demo: generates a small synthetic terrain and runs the
full RTM pipeline — no external data required.

The script
  1. builds a synthetic mountainous terrain (sum of Gaussian bumps),
  2. creates the three DEM layers (Detailed / Tess / Coarse) and smoothed
     reference surfaces for RTM,
  3. writes them in the TGF binary format under output/demo/,
  4. runs TC_GUI (ikind=2, itype=10) on 3 computation points,
  5. prints the result table.

Usage:
    python examples/make_synthetic_demo.py
"""

import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO_ROOT, 'output', 'demo')

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyTGF import TGFConfig, TC_GUI
from pyTGF.io.dem_io import DEMData, write_dem
from pyTGF.io import write_com_points

# ---------------------------------------------------------------------------
# 1. Synthetic terrain: 40.0-40.4 N, 110.0-110.4 E, heights ~1000-2600 m
# ---------------------------------------------------------------------------
MIN_LAT, MAX_LAT = 40.0, 40.4
MIN_LON, MAX_LON = 110.0, 110.4

bumps = [                     # (lat0, lon0, amplitude [m], sigma [deg])
    (40.16, 110.18, 1200.0, 0.06),
    (40.24, 110.10,  900.0, 0.05),
    (40.12, 110.28,  700.0, 0.04),
    (40.30, 110.28,  500.0, 0.08),
]


def terrain(lat2d, lon2d):
    """Synthetic elevation on 2-D lat/lon meshes."""
    z = 1000.0 + 300.0 * (lat2d - MIN_LAT) + 200.0 * (lon2d - MIN_LON)
    for lat0, lon0, amp, sig in bumps:
        z += amp * np.exp(-((lat2d - lat0) ** 2 + (lon2d - lon0) ** 2)
                          / (2 * sig ** 2))
    return z


def make_layer(res, smooth_sigma):
    """Build one DEM layer and its smoothed RTM reference surface."""
    lats = np.arange(MIN_LAT, MAX_LAT + res / 2, res)
    lons = np.arange(MIN_LON, MAX_LON + res / 2, res)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing='ij')
    z = terrain(lat2d, lon2d)
    ref = gaussian_filter(z, sigma=smooth_sigma, mode='nearest')
    dem = DEMData(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, res, res, z)
    ref_dem = DEMData(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, res, res, ref)
    return dem, ref_dem


os.makedirs(OUT_DIR, exist_ok=True)

# res [deg], REF smoothing sigma [cells]
LAYERS = {
    'detailed': (0.002, 25.0),   # ~50 cells radius = 0.1 deg coverage
    'tess':     (0.010,  5.0),
    'coarse':   (0.020,  2.0),
}

paths = {}
for name, (res, sig) in LAYERS.items():
    dem, ref = make_layer(res, sig)
    paths[f'{name}_dem'] = os.path.join(OUT_DIR, f'{name}DEM.bin')
    paths[f'{name}_ref'] = os.path.join(OUT_DIR, f'{name}REF.bin')
    write_dem(paths[f'{name}_dem'], dem)
    write_dem(paths[f'{name}_ref'], ref)
    print(f'{name:9s} layer: {dem.n_lat} x {dem.n_lon} cells, '
          f'z = {dem.z.min():.0f} .. {dem.z.max():.0f} m  ->  {paths[f"{name}_dem"]}')

# ---------------------------------------------------------------------------
# 2. Computation points near the main summit
# ---------------------------------------------------------------------------
pts = [
    (1, 40.18, 110.20, 2300.0),
    (2, 40.22, 110.16, 2100.0),
    (3, 40.14, 110.24, 1950.0),
]
com_path = os.path.join(OUT_DIR, 'ComPoints_demo.bin')
write_com_points(com_path, ids=[p[0] for p in pts],
                 lats=[p[1] for p in pts], lons=[p[2] for p in pts],
                 heights=[p[3] for p in pts])
print(f'com points: {len(pts)} -> {com_path}')

# ---------------------------------------------------------------------------
# 3. Run the RTM calculation
# ---------------------------------------------------------------------------
cfg = TGFConfig(
    vstpar='RUN',
    com_points=com_path,
    detailed_dem=paths['detailed_dem'],
    detailed_ref=paths['detailed_ref'],
    tess_dem=paths['tess_dem'],
    tess_ref=paths['tess_ref'],
    coarse_dem=paths['coarse_dem'],
    coarse_ref=paths['coarse_ref'],
    outname=os.path.join(OUT_DIR, 'result_demo.dat'),
    ikind=2,
    itype=10,
    idensity=0,
    flag_earth=0,
    rzones=[0.01, 0.02, 0.05, 0.1],
    e=2.67,
)

print(f'\n=== pyTGF demo: ikind=2 (RTM), itype=10, rzones={cfg.rzones} ===')
t0 = time.time()
result = TC_GUI(cfg)
elapsed = time.time() - t0

tensor = result['result']
hdr = ('lon', 'lat', 'H', 'N[m]', 'xi["]', 'eta["]', 'dg[mGal]',
       'Vxx[E]', 'Vxy[E]', 'Vxz[E]', 'Vyy[E]', 'Vyz[E]', 'Vzz[E]')
print(f'\n{" ".join(f"{h:>12s}" for h in hdr)}')
for row in tensor:
    print('  '.join(f'{v:12.5e}' for v in row))

print(f'\nElapsed: {elapsed:.1f} s (first run includes Numba JIT compilation)')
print(f'Results saved to: {cfg.outname}')
