# Data directory

pyTGF is shipped **without any data**. Place your own files here (or anywhere)
and pass their paths to `TGFConfig` / the Web UI / the CLI.

## Expected files

| File | Role | Needed for |
|------|------|-----------|
| `DetailedDEM.bin` | High-resolution DEM for Zones 1–2 | all runs |
| `DetailedREF.bin` | Reference surface over the detailed grid | `ikind=2` (RTM) |
| `TessDEM.bin` | Medium-resolution DEM for Zone 3 (tesseroids) | all runs |
| `TessREF.bin` | Reference surface over the tesseroid grid | `ikind=2` (RTM) |
| `CoarseDEM.bin` | Low-resolution DEM for Zone 4 (point masses) | all runs |
| `CoarseREF.bin` | Reference surface over the coarse grid | `ikind=2` (RTM) |
| `ComPoints.bin` | Computation points | all runs |

For `ikind=1` (pure topography) the `*REF*` files are not required.

## Binary DEM format (`.bin`)

Native-endian float64, identical to the MATLAB TGF `read_DEM` layout:

```
Header (6 doubles):  minlat, maxlat, minlon, maxlon, reslat, reslon
Body  (N doubles):   elevation [m], row by row (row = one latitude band)
```

Grid size is derived from the header:

```
numCols = ((maxlon - minlon) / reslon) + 1
numRows = (N_total - 6) / numCols
```

`z[i, j]` is the elevation at latitude `minlat + i*reslat`,
longitude `minlon + j*reslon`. Latitudes increase along rows (axis 0),
longitudes along columns (axis 1).

## Computation points format

Native-endian float64, MATLAB column-major layout with `N` points:

```
[id1, ..., idN, lat1, ..., latN, lon1, ..., lonN, h1, ..., hN]
```

`lat/lon` in decimal degrees (WGS84), `h` in meters. Negative heights are
corrected to 0 by default (`correct_negative=True`).

## Building the files

- `pyTGF.preprocess` imports GeoTIFF / ESRI ASCII / NumPy DEMs, resamples the
  reference surface and block-mean downsamples the Coarse layer.
- `examples/run_colorado.py` is a complete end-to-end pipeline: raw MERIT DEM
  tiles → three DEM layers + REF surfaces → `TC_GUI` calculation.
- `examples/make_synthetic_demo.py` generates a small synthetic data set in
  `output/` and runs the full pipeline — useful as a smoke test without real
  data.
