"""
50-point itype=10 benchmark: pyTGF (Numba) vs MATLAB TC_GUI.m

Generates ComPoints_50.bin, runs pyTGF with itype=10, saves results.
Also generates an equivalent MATLAB script for comparison.

Usage:
    python examples/run_benchmark_50pts.py

Requires the DEM layers (DetailedDEM.bin, ...) in data/ — see data/README.md.
"""
import os, sys, time
import numpy as np

# Path to the original MATLAB TGF source (edit for your installation)
MATLAB_TGF_PATH = r'C:\path\to\TGF_matlab\matlab'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, 'data')

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---- Generate 50 test points spread across DEM coverage ----
np.random.seed(42)
N = 50
ids = np.arange(1, N + 1, dtype=np.float64)
lats = np.random.uniform(36.5, 38.3, N)
lons = np.random.uniform(-108.5, -105.5, N)
heights = np.random.uniform(500.0, 2500.0, N)

# Write ComPoints in MATLAB column-major order:
# [id1..idN, lat1..latN, lon1..lonN, h1..hN]
com_data = np.concatenate([ids, lats, lons, heights])
com_points_path = os.path.join(DATA_DIR, 'ComPoints_50.bin')
com_data.tofile(com_points_path)
print(f'Generated {N} test points -> {com_points_path}')

# ---- Run pyTGF ----
from pyTGF.config import TGFConfig
from pyTGF.core import GravityEngine

result_path = os.path.join(DATA_DIR, 'result_python_50pts.dat')
report_path = os.path.join(DATA_DIR, 'result_python_50pts_report.txt')

cfg = TGFConfig(
    vstpar='RUN',
    com_points=com_points_path,
    detailed_dem=os.path.join(DATA_DIR, 'DetailedDEM.bin'),
    detailed_ref=os.path.join(DATA_DIR, 'DetailedREF.bin'),
    tess_dem=os.path.join(DATA_DIR, 'TessDEM.bin'),
    tess_ref=os.path.join(DATA_DIR, 'TessREF.bin'),
    coarse_dem=os.path.join(DATA_DIR, 'CoarseDEM.bin'),
    coarse_ref=os.path.join(DATA_DIR, 'CoarseREF.bin'),
    outname=result_path,
    ikind=2,
    itype=10,       # ALL 10 components
    idensity=0,
    flag_earth=0,
    rzones=[0.03, 0.1, 0.2, 1],
    e=2.67,
)

print(f'=== Python TC_GUI: ikind={cfg.ikind} itype={cfg.itype} ===')
print(f'rzones={cfg.rzones} e={cfg.e}')
print('Running...')

t0 = time.time()
engine = GravityEngine(cfg)
print(f'[{time.time()-t0:.1f}s] Engine created')

t1 = time.time()
result = engine.run_from_files()
elapsed = time.time() - t1
print(f'[{elapsed:.1f}s] Calculation done')

tensor = result['result']
# itype=10 output: lon, lat, H, N, xi, eta, dg, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz
n_cols = tensor.shape[1]
print(f'\nResult shape: {tensor.shape[0]} x {tensor.shape[1]}')

# Print header
if n_cols == 13:
    hdr = '%-8s %-8s %-7s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s' % \
          ('lon', 'lat', 'H', 'N', 'xi', 'eta', 'dg', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')
else:
    hdr = '%-8s %-8s %-7s %-12s %-12s %-12s %-12s' % ('lon', 'lat', 'H', 'N', 'xi', 'eta', 'dg')
print(hdr)
for i in range(tensor.shape[0]):
    row = tensor[i]
    vals = [f'{v:12.6e}' for v in row[3:]]
    print(f'{row[0]:8.4f} {row[1]:8.4f} {row[2]:7.2f} ' + ' '.join(vals))

np.savetxt(result_path, tensor, fmt='%.18e', delimiter=' ')
print(f'\nSaved to {result_path}')
print(f'Total time: {elapsed:.1f}s ({N} points, itype=10)')

# ---- Generate equivalent MATLAB script ----
matlab_script = f"""%% MATLAB benchmark: 50 points, ikind=2, itype=10
% Run with: matlab -batch "run('run_benchmark_50pts_matlab.m')"
% Then compare with result_python_50pts.dat

clear; clc;
addpath('{MATLAB_TGF_PATH}');

vstpar     = 'RUN';
ComPoints  = '{com_points_path.replace(os.sep, chr(92))}';
DetailedDEM = '{os.path.join(DATA_DIR, 'DetailedDEM.bin').replace(os.sep, chr(92))}';
DetailedREF = '{os.path.join(DATA_DIR, 'DetailedREF.bin').replace(os.sep, chr(92))}';
MassDensity = '';
TessDEM    = '{os.path.join(DATA_DIR, 'TessDEM.bin').replace(os.sep, chr(92))}';
TessREF    = '{os.path.join(DATA_DIR, 'TessREF.bin').replace(os.sep, chr(92))}';
TessDensity = '';
CoarseDEM  = '{os.path.join(DATA_DIR, 'CoarseDEM.bin').replace(os.sep, chr(92))}';
CoarseREF  = '{os.path.join(DATA_DIR, 'CoarseREF.bin').replace(os.sep, chr(92))}';
CoarseDensity = '';
GlobalDEM  = '';
GlobalREF  = '';
outname    = '{os.path.join(DATA_DIR, 'result_matlab_50pts.dat').replace(os.sep, chr(92))}';

ikind      = 2;
itype      = 10;
idensity   = 0;
flag_earth = 0;
rzones     = [0.03, 0.1, 0.2, 1];
e          = 2.67;

fprintf('=== MATLAB TC_GUI: ikind=%d itype=%d ===\\n', ikind, itype);
fprintf('ComPoints: %s\\n', ComPoints);
fprintf('rzones=[%.2f %.2f %.2f %.2f] e=%.2f\\n', rzones, e);

t_start = tic;
TC_GUI(vstpar, ComPoints, DetailedDEM, DetailedREF, MassDensity, ...
       TessDEM, TessREF, TessDensity, CoarseDEM, CoarseREF, CoarseDensity, ...
       GlobalDEM, GlobalREF, outname, ikind, itype, idensity, flag_earth, rzones, e);
elapsed = toc(t_start);
fprintf('TC_GUI total time: %.3f s\\n', elapsed);

% Load and display result
tensor = load(outname);
fprintf('\\n=== MATLAB Results (%d points) ===\\n', size(tensor,1));
fprintf('%-8s %-8s %-7s', 'lon', 'lat', 'H');
for j = 4:size(tensor,2)
    fprintf(' %-12s', sprintf('col%d', j));
end
fprintf('\\n');
for i = 1:size(tensor,1)
    fprintf('%-8.4f %-8.4f %-7.2f', tensor(i,1), tensor(i,2), tensor(i,3));
    for j = 4:size(tensor,2)
        fprintf(' %12.6e', tensor(i,j));
    end
    fprintf('\\n');
end

% Compare with Python
py_file = '{result_path.replace(os.sep, chr(92))}';
if exist(py_file, 'file')
    py_tensor = load(py_file);
    fprintf('\\n=== Python vs MATLAB Comparison ===\\n');
    fprintf('%-6s %-12s %-12s %-12s %-12s\\n', 'Point', 'N_py', 'N_ml', 'diff', 'rel_diff');
    max_diff = 0;
    for i = 1:min(size(tensor,1), size(py_tensor,1))
        for j = 4:size(tensor,2)
            pv = py_tensor(i,j);
            mv = tensor(i,j);
            d = abs(pv - mv);
            rd = d / max(abs(mv), 1e-30);
            if rd > max_diff
                max_diff = rd;
            end
        end
        % Show N comparison
        pv = py_tensor(i,4); mv = tensor(i,4);
        d = abs(pv - mv); rd = d / max(abs(mv), 1e-30);
        fprintf('%-6d %12.6e %12.6e %12.4e %12.4e\\n', i, pv, mv, d, rd);
    end
    fprintf('Max relative diff (N): %.4e\\n', max_diff);
end
fprintf('\\nDone.\\n');
"""

matlab_path = os.path.join(DATA_DIR, 'run_benchmark_50pts_matlab.m')
with open(matlab_path, 'w') as f:
    f.write(matlab_script)
print(f'\nMATLAB script saved to: {matlab_path}')
print('Run in MATLAB: run(\'' + matlab_path.replace(os.sep, '\\\\') + '\')')
