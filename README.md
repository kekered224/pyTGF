# pyTGF

**Python implementation of TGF (Terrain Gravity Field) software**

`pyTGF` is a Python reimplementation of the MATLAB TGF software for calculating
full-scale topographic and RTM (Residual Terrain Model) gravitational fields.
It mirrors the functionality of `TC_GUI.m` from the original TGF software
(Yang, Hirt, Pail, 2019) and produces numerically equivalent results —
with **Numba-accelerated kernels that run several times faster than the
MATLAB original**.

The calculation uses a **four-zone integration scheme** that selects the most
appropriate mass element for each radial distance from the computation point:

| Zone | Radius range | Mass element       | DEM layer   |
|------|--------------|--------------------|-------------|
| 1    | `0   – r1`   | Polyhedron         | Detailed    |
| 2    | `r1  – r2`   | Rectangular prism  | Detailed    |
| 3    | `r2  – r3`   | Tesseroid          | Tesseroid   |
| 4    | `r3  – r4`   | Point mass         | Coarse      |

A leakage-compensation step at the Zone 1 / Zone 2 boundary half-weights the
boundary cells so the two zones tile without overlap or gaps.

## Features

- Full port of the MATLAB `TC_GUI` workflow to Python (numpy / scipy / numba).
- All four mass elements implemented: **polyhedron**, **prism**, **tesseroid**,
  **point mass**.
- Supports `ikind` = 1 (topographic) and 2 (RTM); `itype` ∈
  {0, 1, 2, 4, 10, 103, 104} for gravity disturbances, deflections, and
  gravity gradients.
- Constant density (`idensity=0`) or density-map mode (`idensity=1`).
- Spherical (`flag_earth=0`) or ellipsoidal (`flag_earth=1`) Earth model.
- **Numba JIT-accelerated kernels**:
  - Zone 1 (Polyhedron): full Numba scalar loop.
  - Zone 2 (Prism): Nagy et al. (2000/2002) formulas, Numba scalar loop, **4.5x** speedup.
  - Zone 3 (Tesseroid): Grombein et al. (2013) Taylor expansion, fused geometry+formula
    into single Numba loops. **28-80x** speedup depending on itype.
  - All itype variants (0/1/2/4/10) have dedicated Numba kernels with machine-precision
    accuracy (<3e-14 relative error vs numpy reference).
- **Pre-built cubic interpolators** (tensor-product, grid-native) for the
  tesseroid and coarse DEM layers.
- **Web-based GUI** (`pyTGF.web`): self-contained single-page application with:
  - DEM hillshade/hypsometric visualization with zoom/pan, rendered on `<canvas>`.
  - Computation point management (map placement, file import, grid generation).
  - GeoTIFF import and conversion to binary DEM format.
  - Real-time progress monitoring during calculations.
  - Result display (paged) and CSV export.
  - **Fully offline**: React and ReactDOM are vendored locally, JSX is
    pre-compiled to a static `app.js`, and no CDN, web fonts, or map tiles are
    requested at runtime.
- Binary DEM / reference / computation-point readers compatible with the
  MATLAB file layout (column-major order).
- **Preprocessing toolkit** (`pyTGF.preprocess`):
  - GeoTIFF / ESRI ASCII / NumPy import of DEM and reference surfaces.
  - Block-mean down-sampling (port of MATLAB `boxmean.m`) for building the
    Coarse layer from high-resolution DEM.
  - ASCII / CSV computation-point import with auto-detected column layout.

## Installation

### Prerequisites

- Python >= 3.8
- A C-compatible toolchain is **not** required (numba ships its own).

### From source

```bash
git clone <your-repo-url> pyTGF
cd pyTGF
pip install -e .
```

### Dependencies

```
numpy>=1.20
scipy>=1.7
numba>=0.55
rasterio>=1.3
flask>=2.0
pillow>=9.0
```

## Quick start

### 1. Self-contained demo (no data required)

`pyTGF` ships **without any data**. To verify your installation immediately,
run the synthetic-terrain demo — it generates a small DEM data set in
`output/demo/` and runs the full RTM pipeline:

```bash
python examples/make_synthetic_demo.py
```

### 2. Launch the Web UI (recommended for real data)

**Windows**: double-click `start.bat` in the project root.

**Command line**:
```bash
python -m pyTGF.web
# or after pip install:
pytgf-web
```

Then open http://127.0.0.1:5000 in your browser.

### 3. Run on your own data (command line)

Place your DEM layers and computation points as described in
[`data/README.md`](data/README.md) (TGF binary format, buildable from GeoTIFF
with `pyTGF.preprocess`), then:

```bash
python -m pyTGF.main \
    --com-points data/ComPoints.bin \
    --detailed-dem data/DetailedDEM.bin --detailed-ref data/DetailedREF.bin \
    --tess-dem data/TessDEM.bin --tess-ref data/TessREF.bin \
    --coarse-dem data/CoarseDEM.bin --coarse-ref data/CoarseREF.bin \
    --ikind 2 --itype 104 --rzones 0 0.03 0.8 2.0 --density 2.67 \
    --outname result.dat
```

### 4. Use the API directly

```python
from pyTGF import TGFConfig, TC_GUI

cfg = TGFConfig(
    vstpar="RUN",
    com_points="data/ComPoints.bin",
    detailed_dem="data/DetailedDEM.bin",
    detailed_ref="data/DetailedREF.bin",
    tess_dem="data/TessDEM.bin",
    tess_ref="data/TessREF.bin",
    coarse_dem="data/CoarseDEM.bin",
    coarse_ref="data/CoarseREF.bin",
    outname="result.dat",
    ikind=2,
    itype=10,
    idensity=0,
    flag_earth=0,
    rzones=[0.03, 0.1, 0.2, 1],
    e=2.67,
)

TC_GUI(cfg)
```

`examples/run_colorado.py` shows the complete data pipeline: raw MERIT DEM
tiles → three DEM layers + RTM reference surfaces → `TC_GUI` calculation.

## Web API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET  | `/` | Single-page UI |
| GET  | `/api/health` | Liveness probe with package version |
| GET  | `/api/list-files` | List available DEM / point files under the data roots |
| GET  | `/api/dem-image` | Render a DEM layer as a PNG preview |
| POST | `/api/dem-sample` | Bilinear elevation sampling at arbitrary points |
| POST | `/api/import-points` | Parse computation points from pasted text |
| POST | `/api/import-points-file` | Parse computation points from an uploaded file |
| POST | `/api/run` | Queue a calculation job (returns `202` + `job_id`) |
| GET  | `/api/status/<job_id>` | Job progress and phase |
| GET  | `/api/result/<job_id>` | Job results (paged preview) |
| GET  | `/api/download-result/<job_id>` | Download the result file |
| POST | `/api/upload-tifs` | Upload GeoTIFF files for conversion |
| GET  | `/api/scan-tifs` | List convertible GeoTIFF files |
| POST | `/api/preview-tif-extent` | Read a GeoTIFF's extent and resolution |
| POST | `/api/convert-tif-to-bin` | Convert GeoTIFF to binary DEM |

**Limits and guards**

- At most `MAX_POINTS = 200_000` computation points per request.
- All input parameters are validated server-side before the engine is called
  (point structure, finiteness, latitude/longitude ranges, non-negative zone
  radii, density in `(0, 10] g/cm³`, resolvable file paths).
- `files` must supply `detailedDEM`, `tessDEM` and `coarseDEM`; RTM runs
  (`ikind=2`) additionally require `detailedREF`, `tessREF` and `coarseREF`.
  Missing layers are reported as a `400` instead of failing inside the engine.
- Computation-point text import accepts `lat lon [height]` or `lon lat [height]`
  in whitespace-, comma- or semicolon-separated form; the column order is
  auto-detected and a missing height defaults to `0`.
- Every path supplied by the client is resolved with `realpath` and confined to
  the project tree; escaping paths are rejected.
- Finished jobs are kept for one hour (`_JOB_TTL_SECONDS`) with a hard cap of 64
  live entries; temporary files are removed in a `finally` block.
- GeoTIFF batch operations accept at most 500 files per request.

## Front-end development

The UI lives in `pyTGF/web/static/`:

```
static/
├── pytgf_ui.html                 # Markup + design tokens (styles only)
├── app.js                        # Pre-compiled bundle (committed, served as-is)
├── src/app.jsx                   # JSX source
└── vendor/
    ├── react.production.min.js
    └── react-dom.production.min.js
```

Serving needs nothing beyond the committed files. If you edit `src/app.jsx`,
rebuild the bundle:

```bash
npx esbuild pyTGF/web/static/src/app.jsx \
    --bundle --minify --target=es2019 --jsx=transform \
    --outfile=pyTGF/web/static/app.js
```

Rebuilding is only required for development; the shipped `app.js` is a plain
script with no Babel or module loader involved.

## Validation against MATLAB TGF

pyTGF was validated against the original MATLAB TGF software
(`TC_GUI.m`) on the Colorado region (36.5°-38.3°N, 105.5°-108.5°W),
`ikind=2, e=2.67, flag_earth=0`, over two independent 20-point samples
with large integration zones (`rzones` out to 2° and 2.2°).

### Numerical equivalence (itype=104 / itype=4, 20 points each)

| Quantity | pyTGF − MATLAB (max abs) |
|----------|--------------------------|
| dg (mGal)        | 3.5e-4 (rel. 5.4e-6) |
| N (m)            | 1.3e-4 |
| xi, eta (arcsec) | 1.1e-4 |
| V (itype=0)      | 1.1e-4 |

Residuals originate from floating-point summation order in the tesseroid and
prism kernels; they are negligible against the typical RTM signal
(±10-100 mGal in rugged terrain).

### Computation time (pyTGF vs MATLAB TC_GUI.m)

20-point RTM runs, identical parameters and machine:

| Test | MATLAB (s) | pyTGF (s) | pyTGF advantage |
|------|-----------|-----------|-----------------|
| `rzones=[0, 0.03, 0.8, 2.0]`° | 76.1  | 15.8 | **4.8x** |
| `rzones=[0, 0.03, 1.2, 2.2]`° | 223.8 | 46.2 | **4.8x** |

Zone 3 (tesseroid) Numba kernel speedup vs the NumPy reference implementation:

| itype | Components | NumPy (s) | Numba (s) | Speedup |
|-------|-----------|-----------|-----------|---------|
| 0 | V | 0.665 | 0.008 | **79.6x** |
| 1 | Vx,Vy,Vz | 0.650 | 0.008 | **78.0x** |
| 2 | Vxx..Vzz | 0.678 | 0.016 | **43.9x** |
| 4 | V+gradients | 0.665 | 0.009 | **77.5x** |
| 10 | All 10 | 0.652 | 0.023 | **28.4x** |

All Numba kernels match numpy reference to machine precision (<3e-14 relative error).

## Repository layout

```
pyTGF/
├── pyTGF/                     # Source package
│   ├── __init__.py
│   ├── config.py              # TGFConfig + physical constants
│   ├── core/
│   │   ├── gravity_engine.py  # Main calculation orchestrator
│   │   ├── tc_gui.py          # TC_GUI() Python equivalent
│   │   └── zone_strategy.py   # Zone extent + leakage
│   ├── io/                    # Binary I/O (DEM, ComPoints, output)
│   ├── mass_elements/
│   │   ├── polyhedron.py      # Zone 1 (Numba)
│   │   ├── prism.py           # Zone 2 (NumPy reference)
│   │   ├── prism_numba.py     # Zone 2 (Numba, 4.5x)
│   │   ├── tesseroid.py       # Zone 3 (NumPy reference)
│   │   ├── tesseroid_numba.py # Zone 3 (Numba, 28-80x)
│   │   └── point_mass.py      # Zone 4
│   ├── preprocess/            # DEM/ComPoints import tools
│   ├── utils/                 # Coordinate, grid, normal gravity
│   ├── web/                   # Flask web server + SPA
│   │   ├── server.py
│   │   ├── __main__.py
│   │   └── static/
│   │       ├── pytgf_ui.html
│   │       ├── app.js         # Pre-compiled React bundle
│   │       ├── src/app.jsx    # JSX source (development only)
│   │       └── vendor/        # Local React + ReactDOM
│   └── tests/
├── data/                      # Empty in the release — put your DEM files
│   └── README.md              #   here; format documented in README.md
├── examples/
│   ├── make_synthetic_demo.py # Self-contained demo (no data needed)
│   ├── run_test_python.py     # Basic pipeline example
│   ├── run_colorado.py        # Full pipeline: MERIT DEM -> layers -> TC_GUI
│   └── run_benchmark_50pts.py # 50-point benchmark vs MATLAB
├── output/                    # Calculation results (git-ignored)
├── start.bat                  # Windows launcher (Web UI)
├── server.py                  # Quick web server launcher
├── pyproject.toml
└── README.md
```

## Configuration reference

`TGFConfig` mirrors the `TC_GUI.m` parameter list:

| Field         | Type            | Default              | Description                              |
|---------------|-----------------|----------------------|------------------------------------------|
| `ikind`       | int (1, 2)      | 2                    | 1 = topographic, 2 = RTM                 |
| `itype`       | int             | 104                  | Output type (see below)                  |
| `idensity`    | int (0, 1)      | 0                    | 0 = constant density, 1 = density maps   |
| `flag_earth`  | int (0, 1)      | 0                    | 0 = sphere, 1 = ellipsoid                |
| `rzones`      | list[float] x 4 | [0, 0.03, 0.3, 2.0]  | Zone radii [r1,r2,r3,r4] in degrees      |
| `e`           | float           | 2.67                 | Constant density [g/cm3]                 |

### Supported `itype` values

| `itype` | Output quantities                              |
|---------|------------------------------------------------|
| 0       | Potential V                                    |
| 1       | 1st derivatives Vx, Vy, Vz                    |
| 2       | 2nd derivatives Vxx, Vxy, Vxz, Vyy, Vyz, Vzz |
| 4       | V + 1st derivatives                            |
| 10      | All 10 components                              |
| 103     | dg + deflections + N                           |
| 104     | dg + deflections + N (compact)                 |

## Acknowledgements

pyTGF is a Python reimplementation of the algorithmic flow of the original
MATLAB TGF software. We gratefully acknowledge the authors of the original
software:

> M. Yang, C. Hirt, R. Pail (2019). *TGF: A New MATLAB-based Software for
> Terrain Gravity Field Calculation.* Computers & Geosciences.

The four-zone integration strategy, mass element formulations (polyhedron:
Tsoulis 2012; prism: Nagy et al. 2000/2002; tesseroid: Grombein et al. 2013),
and the overall software architecture follow the original TGF implementation.

This Python version adds Numba JIT acceleration for all mass element kernels
and a web-based graphical user interface, while maintaining numerical
equivalence with the MATLAB reference implementation.

## License

This Python reimplementation is provided for research and educational use.
Please refer to the original TGF software for the algorithmic specification
and citation information.
