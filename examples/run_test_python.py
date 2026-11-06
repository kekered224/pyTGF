"""
Example: pyTGF calculation and comparison with MATLAB TGF.

This script regenerates the ComPoints_test.bin file (in MATLAB column-major
order), runs the pyTGF calculation, and prints results.

Usage:
    python examples/run_test_python.py

The script assumes the pyTGF package is installed (pip install -e .) or
the repository root is on sys.path.
"""
import os
import sys
import time
import numpy as np

# Resolve paths relative to this script (works from anywhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, 'data')

# Make pyTGF importable when running from source without install
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Regenerate ComPoints_test.bin in MATLAB column-major order:
# MATLAB reshape(data, nrow, 4) fills column-major, so file must be
# [id1,id2,...,idN, lat1,lat2,...,latN, lon1,...,lonN, h1,...,hN]
ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
lats = np.array([37.0, 37.5, 37.2, 36.8, 37.3, 37.6, 36.9, 37.1, 37.4, 37.7], dtype=np.float64)
lons = np.array([-107.0, -106.5, -107.8, -106.2, -107.5, -106.8, -107.3, -106.0, -107.2, -106.6], dtype=np.float64)
heights = np.array([1000.0, 1500.0, 1200.0, 800.0, 1300.0, 1600.0, 900.0, 1100.0, 1400.0, 1700.0], dtype=np.float64)
com_data = np.concatenate([ids, lats, lons, heights])
com_points_path = os.path.join(DATA_DIR, 'ComPoints_test.bin')
com_data.tofile(com_points_path)
print('Regenerated ComPoints_test.bin (column-major):')
print('  IDs:', ids)
print('  lats:', lats)
print('  lons:', lons)
print('  heights:', heights)

from pyTGF.config import TGFConfig
from pyTGF.core import GravityEngine

cfg = TGFConfig(
    vstpar='RUN',
    com_points=com_points_path,
    detailed_dem=os.path.join(DATA_DIR, 'DetailedDEM.bin'),
    detailed_ref=os.path.join(DATA_DIR, 'DetailedREF.bin'),
    tess_dem=os.path.join(DATA_DIR, 'TessDEM.bin'),
    tess_ref=os.path.join(DATA_DIR, 'TessREF.bin'),
    coarse_dem=os.path.join(DATA_DIR, 'CoarseDEM.bin'),
    coarse_ref=os.path.join(DATA_DIR, 'CoarseREF.bin'),
    outname=os.path.join(DATA_DIR, 'result_python.dat'),
    ikind=2,
    itype=103,
    idensity=0,
    flag_earth=0,
    rzones=[0.03, 0.1, 0.2, 1],
    e=2.67,
)

print('=== Python TC_GUI: ikind=%d itype=%d ===' % (cfg.ikind, cfg.itype))
print('ComPoints:', cfg.com_points)
print('rzones=%s e=%.2f flag_earth=%d' % (cfg.rzones, cfg.e, cfg.flag_earth))
print('Running...')

t0 = time.time()
engine = GravityEngine(cfg)
print('[%.1fs] Engine created' % (time.time() - t0))

t1 = time.time()
result = engine.run_from_files()
print('[%.1fs] Calculation done' % (time.time() - t1))
tensor = result['result']

print('\n=== Python Results ===')
print('%-10s %-10s %-10s %-12s %-12s %-12s %-12s' %
      ('lon', 'lat', 'H', 'N', 'xi', 'eta', 'dg'))
for i in range(tensor.shape[0]):
    print('%-10.4f %-10.4f %-10.2f %-12.6e %-12.6e %-12.6e %-12.6e' %
          (tensor[i, 0], tensor[i, 1], tensor[i, 2],
           tensor[i, 3], tensor[i, 4], tensor[i, 5], tensor[i, 6]))

# Save in MATLAB-compatible ASCII format
np.savetxt(cfg.outname, tensor, fmt='%.18e', delimiter=' ')
print('Done. Result saved to', cfg.outname)
