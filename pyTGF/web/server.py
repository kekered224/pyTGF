"""
pyTGF Web Server — Flask application serving the UI and computation API.

Usage:
    python server.py                      # from project root
    python -m pyTGF.web                   # as a module
    python server.py --port 8080          # custom port
    python server.py --no-browser         # don't auto-open browser
"""

import argparse
import io
import math
import os
import sys
import tempfile
import threading
import time
import uuid
import webbrowser

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

# Ensure pyTGF is importable when running from source
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyTGF.config import TGFConfig
from pyTGF.core import GravityEngine
from pyTGF.io.com_points import write_com_points

# ---------------------------------------------------------------------------
#  JSON helpers — numpy aware, NaN/Inf become null
# ---------------------------------------------------------------------------
def _sanitize(obj):
    """Convert numpy scalars/arrays and non-finite floats into JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
#  Flask application
# ---------------------------------------------------------------------------
from .. import __version__ as _PKG_VERSION

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=_STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 4096 * 1024 * 1024  # GeoTIFF mosaics can be large
# Flask >= 2.2 JSON provider hook (fallback for older versions below)
try:
    app.json.default = lambda o: _sanitize(o)
    app.json.ensure_ascii = False
except Exception:  # pragma: no cover - very old Flask
    app.config["JSON_AS_ASCII"] = False

# ---------------------------------------------------------------------------
#  In-memory job registry (bounded, TTL based)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_jobs: dict = {}
_JOB_TTL_SECONDS = 3600.0
_MAX_JOBS = 64


def _now() -> float:
    return time.time()


def _register_job(job_id: str, total: int) -> None:
    """Create a job entry, evicting expired/old entries when needed."""
    with _lock:
        cutoff = _now() - _JOB_TTL_SECONDS
        for key in [k for k, v in _jobs.items() if v.get("created", 0.0) < cutoff]:
            _jobs.pop(key, None)
        while len(_jobs) >= _MAX_JOBS:
            oldest = min(_jobs, key=lambda k: _jobs[k].get("created", 0.0))
            _jobs.pop(oldest, None)
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": max(int(total), 0),
            "phase": "queued",
            "error": None,
            "result": None,
            "created": _now(),
        }


def _update_job(job_id: str, **kwargs):
    """Thread-safe update of a job entry (no-op if the job was evicted)."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(kwargs)


_DATA_ROOT = _REPO_ROOT  # data/ is at project root


def _resolve(p, *, must_exist: bool = False):
    """Resolve *p* inside the project root and reject path escapes.

    Every path coming from the browser goes through here so that the local
    server never reads or writes outside the project directory.
    """
    if not p:
        return p
    candidate = p if os.path.isabs(p) else os.path.join(_DATA_ROOT, p)
    real = os.path.realpath(candidate)
    root = os.path.realpath(_DATA_ROOT)
    try:
        inside = os.path.commonpath([real, root]) == root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise ValueError(f"Path escapes the project directory: {p}")
    if must_exist and not os.path.exists(real):
        raise FileNotFoundError(p)
    return real


def _safe_name(name: str, fallback: str = "output") -> str:
    """Strip directory components and unsafe characters from a user file name."""
    base = os.path.basename(str(name).replace("\\", "/")).strip()
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    return base or fallback


# ---------------------------------------------------------------------------
#  Background worker
# ---------------------------------------------------------------------------
def _run_job(job_id: str, params: dict):
    """Execute a pyTGF calculation in a background thread."""
    tmp_com = None
    try:
        _update_job(job_id, status="running", phase="preparing")

        # -- Write computation points to a temporary binary file --
        comp_points = params["comp_points"]
        n = len(comp_points)
        lats = np.array([p["lat"] for p in comp_points], dtype=np.float64)
        lons = np.array([p["lon"] for p in comp_points], dtype=np.float64)
        hs = np.array([p["h"] for p in comp_points], dtype=np.float64)

        fd, tmp_com = tempfile.mkstemp(prefix=f"pytgf_{job_id}_", suffix=".bin")
        os.close(fd)
        write_com_points(tmp_com, lats, lons, hs)

        # -- Build TGFConfig --
        files = params.get("files") or {}
        outname = _safe_name(params.get("outname", "tgf_result.dat"), "tgf_result.dat")
        output_dir = params.get("output_dir", "output")

        resolved_files = {k: _resolve(v) for k, v in files.items() if v}

        # Write output to user-specified directory (relative to project root)
        out_dir = _resolve(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, outname)

        _update_job(job_id, phase="loading-data")

        config = TGFConfig(
            vstpar="RUN",
            com_points=tmp_com,
            detailed_dem=resolved_files.get("detailedDEM", ""),
            detailed_ref=resolved_files.get("detailedREF", ""),
            tess_dem=resolved_files.get("tessDEM", ""),
            tess_ref=resolved_files.get("tessREF", ""),
            coarse_dem=resolved_files.get("coarseDEM", ""),
            coarse_ref=resolved_files.get("coarseREF", ""),
            global_dem=resolved_files.get("globalDEM", ""),
            global_ref=resolved_files.get("globalREF", ""),
            outname=out_path,
            ikind=int(params.get("ikind", 2)),
            itype=int(params.get("itype", 103)),
            idensity=int(params.get("idensity", 0)),
            flag_earth=int(params.get("flag_earth", 0)),
            rzones=[float(v) for v in params.get("rzones", [0.0, 0.03, 0.3, 2.0])],
            e=float(params.get("density", 2.67)),
        )

        # -- Create engine and run with progress callback --
        engine = GravityEngine(config)

        def progress_cb(current, total, phase):
            _update_job(job_id, progress=current, total=total, phase=phase)

        result = engine.run_from_files(progress_callback=progress_cb)

        # -- Collect results --
        tensor = result["result"]

        data_list = [[_sanitize(float(v)) for v in row] for row in tensor]

        report_text = ""
        report_path = result.get("report_path")
        if report_path and os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                report_text = f.read()

        _update_job(
            job_id,
            status="done",
            phase="complete",
            progress=n,
            total=n,
            result={
                "columns": config.output_columns,
                "data": data_list,
                "report": report_text,
                "elapsed_hours": float(result["elapsed_hours"]),
                "output_path": result["output_path"],
            },
        )

    except Exception as exc:
        _update_job(job_id, status="error", phase="error", error=str(exc))
    finally:
        if tmp_com and os.path.exists(tmp_com):
            try:
                os.remove(tmp_com)
            except OSError:
                pass


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory(_STATIC_DIR, "pytgf_ui.html")


MAX_POINTS = 200_000


def _validate_run_params(params):
    """Validate a run request up front so bad input fails fast with HTTP 400."""
    if not isinstance(params, dict):
        raise ValueError("Request body must be a JSON object.")

    points = params.get("comp_points")
    if not points or not isinstance(points, list):
        raise ValueError("comp_points is required and must be a non-empty list.")
    if len(points) > MAX_POINTS:
        raise ValueError(f"Too many computation points ({len(points)} > {MAX_POINTS}).")

    for i, p in enumerate(points):
        if not isinstance(p, dict):
            raise ValueError(f"comp_points[{i}] must be an object with lat/lon/h.")
        for key in ("lat", "lon", "h"):
            if key not in p:
                raise ValueError(f"comp_points[{i}] is missing '{key}'.")
            try:
                val = float(p[key])
            except (TypeError, ValueError):
                raise ValueError(f"comp_points[{i}].{key} is not a number.") from None
            if not math.isfinite(val):
                raise ValueError(f"comp_points[{i}].{key} is not finite.")
            points[i][key] = val
        if not (-90.0 <= points[i]["lat"] <= 90.0):
            raise ValueError(f"comp_points[{i}].lat out of range [-90, 90].")
        if not (-360.0 <= points[i]["lon"] <= 360.0):
            raise ValueError(f"comp_points[{i}].lon out of range [-360, 360].")

    rzones = params.get("rzones")
    if rzones is not None:
        if not isinstance(rzones, list) or len(rzones) != 4:
            raise ValueError("rzones must be a list of 4 radii (degrees).")
        rz = [float(v) for v in rzones]
        if any(not math.isfinite(v) or v < 0 for v in rz):
            raise ValueError("rzones must be finite and non-negative.")
        params["rzones"] = rz

    density = params.get("density", 2.67)
    try:
        density = float(density)
    except (TypeError, ValueError):
        raise ValueError("density is not a number.") from None
    if not (0.0 < density <= 10.0):
        raise ValueError("density must be in (0, 10] g/cm3.")
    params["density"] = density

    # File paths and the output location must stay inside the project.
    files = params.get("files") or {}
    if not isinstance(files, dict):
        raise ValueError("files must be an object of layer -> path.")
    for key, value in files.items():
        if value:
            _resolve(value)

    # The engine unconditionally reads the three DEM layers; RTM (ikind=2)
    # also needs the matching reference surfaces. Fail here with a clear
    # message instead of an empty-path OSError deep inside the engine.
    required = ["detailedDEM", "tessDEM", "coarseDEM"]
    if int(params.get("ikind", 2)) == 2:
        required += ["detailedREF", "tessREF", "coarseREF"]
    missing = [k for k in required if not files.get(k)]
    if missing:
        raise ValueError(
            "Missing DEM file(s) in 'files': " + ", ".join(missing)
            + ". Provide all of detailedDEM, tessDEM, coarseDEM"
            + (" plus detailedREF, tessREF, coarseREF for RTM (ikind=2)."
               if int(params.get("ikind", 2)) == 2 else "."))

    _resolve(params.get("output_dir", "output"))
    return params


@app.route("/api/run", methods=["POST"])
def api_run():
    """Submit a new computation job.

    Request JSON:
        comp_points, ikind, itype, idensity, flag_earth, rzones,
        density, outname, output_dir, files

    Returns:
        { job_id, status }
    """
    params = request.get_json(silent=True)
    try:
        params = _validate_run_params(params)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = uuid.uuid4().hex[:12]
    _register_job(job_id, len(params["comp_points"]))

    t = threading.Thread(target=_run_job, args=(job_id, params), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/status/<job_id>")
def api_status(job_id):
    """Poll job progress.

    Returns:
        { status, progress, total, phase, percent }
    """
    with _lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return jsonify({"error": "job not found"}), 404

    total = job.get("total", 0)
    progress = job.get("progress", 0)
    percent = (progress / total * 100) if total > 0 else 0

    return jsonify({
        "status": job.get("status", "unknown"),
        "progress": progress,
        "total": total,
        "phase": job.get("phase", ""),
        "percent": round(percent, 1),
        "error": job.get("error"),
    })


@app.route("/api/result/<job_id>")
def api_result(job_id):
    """Fetch completed job results.

    Returns:
        { status, columns, data, report, elapsed_hours }
    """
    with _lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return jsonify({"error": "job not found"}), 404

    if job.get("status") == "error":
        return jsonify({"status": "error", "error": job.get("error", "unknown")})

    if job.get("status") != "done":
        return jsonify({"status": job.get("status", "unknown"),
                        "message": "still running"}), 202

    r = job["result"]
    resp = {
        "status": "done",
        "columns": r["columns"],
        "data": r["data"],
        "report": r["report"],
        "elapsed_hours": r["elapsed_hours"],
    }
    return jsonify(_sanitize(resp))


@app.route("/api/health")
def api_health():
    """Lightweight liveness probe."""
    with _lock:
        active = sum(1 for j in _jobs.values() if j.get("status") in ("queued", "running"))
    return jsonify({"status": "ok", "version": _PKG_VERSION, "active_jobs": active})


def _parse_point_lines(lines, max_points=MAX_POINTS):
    """Parse computation points, auto-detecting the lat/lon column order.

    Accepted per row (whitespace, comma or semicolon separated):
        lat lon [height]  |  lon lat [height]
    A missing height defaults to 0. Comment prefixes (#, %, //) are skipped.
    If the latitude/longitude order is ambiguous, latitude-first is assumed.
    """
    triples = []
    skipped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "%", "//")):
            continue
        parts = stripped.replace(",", " ").replace(";", " ").split()
        if len(parts) < 2:
            skipped += 1
            continue
        try:
            a = float(parts[0])
            b = float(parts[1])
            h = float(parts[2]) if len(parts) >= 3 else 0.0
        except ValueError:
            skipped += 1
            continue
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(h)):
            skipped += 1
            continue
        triples.append((a, b, h))

    if not triples:
        return [], skipped

    def is_lat(v):
        return -90.0 <= v <= 90.0

    def is_lon(v):
        return -360.0 <= v <= 360.0

    first_all_lat = all(is_lat(t[0]) for t in triples)
    second_all_lat = all(is_lat(t[1]) for t in triples)

    if first_all_lat or not second_all_lat:
        lat_first = True
    else:
        lat_first = False

    rows = []
    for a, b, h in triples:
        lat, lon = (a, b) if lat_first else (b, a)
        if not (is_lat(lat) and is_lon(lon)):
            # Fall back to the other order before giving up on this row.
            lat, lon = (b, a) if lat_first else (a, b)
            if not (is_lat(lat) and is_lon(lon)):
                skipped += 1
                continue
        rows.append({"lat": lat, "lon": lon, "h": h})
        if len(rows) > max_points:
            raise ValueError(f"Point file exceeds the {max_points} point limit.")
    return rows, skipped


def _dem_bounds_warning(rows):
    """Return a warning string when points fall outside the detailed DEM."""
    try:
        dem_path = _resolve("data/DetailedREF.bin", must_exist=True)
        header = np.fromfile(dem_path, dtype=np.float64, count=6)
        if header.size < 6:
            return None
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in header[:4])
    except (ValueError, FileNotFoundError, OSError):
        return None

    outside = sum(
        1 for p in rows
        if not (lat_min <= p["lat"] <= lat_max and lon_min <= p["lon"] <= lon_max)
    )
    if not outside:
        return None
    return (f"{outside} point(s) outside DEM bounds "
            f"[lat {lat_min:.4f}..{lat_max:.4f}, lon {lon_min:.4f}..{lon_max:.4f}]")


def _points_response(rows, skipped):
    """Build the JSON payload shared by both import endpoints."""
    warnings = []
    if skipped:
        warnings.append(
            f"{skipped} line(s) skipped (expected 'lat lon [height]' or "
            "'lon lat [height]').")
    dem_warning = _dem_bounds_warning(rows)
    if dem_warning:
        warnings.append(dem_warning)

    # TGF convention: negative heights are replaced by a positive constant.
    clamped = sum(1 for p in rows if p["h"] < 0)
    if clamped:
        for p in rows:
            if p["h"] < 0:
                p["h"] = 1.0
        warnings.append(f"{clamped} negative height(s) set to 1 m (TGF convention).")

    lats = [p["lat"] for p in rows]
    lons = [p["lon"] for p in rows]
    hs = [p["h"] for p in rows]
    return jsonify({
        "points": rows,
        "count": len(rows),
        "bounds": {
            "lat_min": min(lats), "lat_max": max(lats),
            "lon_min": min(lons), "lon_max": max(lons),
            "h_min": min(hs), "h_max": max(hs),
        },
        "warnings": warnings,
    })


@app.route("/api/import-points", methods=["POST"])
def api_import_points():
    """Import computation points from a server-side text file.

    Accepts: { path: "path/to/file.dat" }
    Expected columns: lat lon [height] or lon lat [height]; the order is
    auto-detected. Comments with #, % or // are skipped; height defaults to 0.

    Returns: { points: [{lat, lon, h}, ...], count, bounds, warnings }
    """
    params = request.get_json(silent=True) or {}
    fpath = str(params.get("path", "")).strip()
    if not fpath:
        return jsonify({"error": "path is required"}), 400

    try:
        resolved = _resolve(fpath, must_exist=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {fpath}"}), 404

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            rows, skipped = _parse_point_lines(fh)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": f"Cannot read file: {exc}"}), 400

    if not rows:
        return jsonify({"error": "No valid data rows found. Expected columns: "
                                 "lat lon [height] or lon lat [height]"}), 400
    return _points_response(rows, skipped)


@app.route("/api/import-points-file", methods=["POST"])
def api_import_points_file():
    """Import computation points from an uploaded text file.

    Accepts: multipart form with 'file' field.
    Expected columns: lat lon [height] or lon lat [height]; the order is
    auto-detected. Comments with #, % or // are skipped; height defaults to 0.

    Returns: { points: [{lat, lon, h}, ...], count, bounds, warnings }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        content = upload.read().decode("utf-8", errors="replace")
        rows, skipped = _parse_point_lines(content.splitlines())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not rows:
        return jsonify({"error": "No valid data rows found. Expected columns: "
                                 "lat lon [height] or lon lat [height]"}), 400
    return _points_response(rows, skipped)


# ---------------------------------------------------------------------------
#  DEM image rendering
# ---------------------------------------------------------------------------

# Hypsometric tint stops matching the frontend colormap
_TERRAIN_STOPS = [
    (0.00,  30,  80,  30), (0.08,  50, 120,  40), (0.18,  80, 155,  55),
    (0.30, 140, 175,  75), (0.42, 195, 190, 100), (0.52, 210, 185, 115),
    (0.62, 190, 155, 100), (0.72, 165, 120,  80), (0.82, 140,  95,  70),
    (0.90, 165, 155, 145), (0.96, 210, 210, 205), (1.00, 250, 250, 248),
]


def _terrain_rgb(t):
    """Compute hypsometric tint RGB for t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    S = _TERRAIN_STOPS
    for i in range(len(S) - 1):
        if t <= S[i + 1][0]:
            f = (t - S[i][0]) / (S[i + 1][0] - S[i][0])
            return (
                int(S[i][1] + f * (S[i + 1][1] - S[i][1])),
                int(S[i][2] + f * (S[i + 1][2] - S[i][2])),
                int(S[i][3] + f * (S[i + 1][3] - S[i][3])),
            )
    return (250, 250, 248)


@app.route("/api/dem-image")
def api_dem_image():
    """Render a binary DEM file as a PNG image with hypsometric tint.

    Query params:
        file: relative path to DEM binary file (e.g. data/DetailedDEM.bin)
        max_size: max dimension in pixels (default 600)

    Returns: PNG image.
    """
    try:
        from PIL import Image
    except ImportError:
        return jsonify({"error": "Pillow (PIL) is required for DEM rendering. "
                        "Install with: pip install Pillow"}), 500

    fname = request.args.get("file", "").strip()
    if not fname:
        return jsonify({"error": "file parameter required"}), 400

    try:
        resolved = _resolve(fname, must_exist=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {fname}"}), 404

    try:
        max_size = int(request.args.get("max_size", "600"))
    except (TypeError, ValueError):
        return jsonify({"error": "max_size must be an integer"}), 400
    max_size = max(64, min(max_size, 4096))

    try:
        file_size = os.path.getsize(resolved)
        if file_size < 48:
            return jsonify({"error": f"File too small to be a DEM: {fname}"}), 400

        header = np.fromfile(resolved, dtype=np.float64, count=6)
        if header.size < 6:
            return jsonify({"error": f"Corrupt DEM header: {fname}"}), 400
        minlat, maxlat, minlon, maxlon, reslat, reslon = (float(v) for v in header)

        if not (math.isfinite(reslat) and math.isfinite(reslon)) or reslat <= 0 or reslon <= 0:
            return jsonify({"error": f"Invalid DEM resolution in {fname}"}), 400
        num_cols = int(round((maxlon - minlon) / reslon)) + 1
        if num_cols <= 0:
            return jsonify({"error": f"Invalid DEM extent in {fname}"}), 400

        remaining = file_size // 8 - 6
        num_rows = int(remaining // num_cols)
        if num_rows <= 0:
            return jsonify({"error": f"DEM contains no rows: {fname}"}), 400

        elev = np.fromfile(resolved, dtype=np.float64, offset=48, count=num_rows * num_cols)
        if elev.size != num_rows * num_cols:
            return jsonify({"error": f"Truncated DEM body: {fname}"}), 400
        elev = elev.reshape((num_rows, num_cols))

        # Flip vertically: binary format has row 0 = minlat (south),
        # but map display needs row 0 = maxlat (north) at top.
        elev = np.flipud(elev)

        # Downsample to max_size
        if max(num_rows, num_cols) > max_size:
            step = int(np.ceil(max(num_rows, num_cols) / max_size))
            elev = elev[::step, ::step]
            num_rows, num_cols = elev.shape

        # Mask non-finite cells so they render as "no data" instead of black
        finite = np.isfinite(elev)
        if not finite.any():
            return jsonify({"error": f"DEM contains no finite elevation: {fname}"}), 400
        zmin = float(elev[finite].min())
        zmax = float(elev[finite].max())
        zrange = zmax - zmin if zmax > zmin else 1.0

        norm = np.clip((np.where(finite, elev, zmin) - zmin) / zrange, 0.0, 1.0)

        # Pre-compute colour lookup table (256 levels)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for level in range(256):
            lut[level] = _terrain_rgb(level / 255.0)

        indices = (norm * 255).astype(np.uint8)
        img = Image.fromarray(lut[indices], 'RGB')

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Return actual geographic bounds as custom headers
        resp = send_file(buf, mimetype="image/png")
        resp.headers["X-DEM-MinLat"] = f"{minlat:.6f}"
        resp.headers["X-DEM-MaxLat"] = f"{maxlat:.6f}"
        resp.headers["X-DEM-MinLon"] = f"{minlon:.6f}"
        resp.headers["X-DEM-MaxLon"] = f"{maxlon:.6f}"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    except Exception as exc:
        return jsonify({"error": f"Failed to render {fname}: {exc}"}), 500


@app.route("/api/dem-sample", methods=["POST"])
def api_dem_sample():
    """Sample a binary DEM at geographic points with bilinear interpolation.

    Request JSON:
        file:   relative DEM path (default "data/DetailedDEM.bin")
        points: [{ "lat": ..., "lon": ... }, ...]

    Returns: { values: [float|null, ...], file: str }
    """
    payload = request.get_json(silent=True) or {}
    fname = str(payload.get("file") or "data/DetailedDEM.bin").strip()
    points = payload.get("points")

    if not isinstance(points, list) or not points:
        return jsonify({"error": "points must be a non-empty list"}), 400
    if len(points) > 200_000:
        return jsonify({"error": "Too many points (max 200000)"}), 400

    try:
        resolved = _resolve(fname, must_exist=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {fname}"}), 404

    try:
        file_size = os.path.getsize(resolved)
        if file_size < 56:
            return jsonify({"error": f"File too small to be a DEM: {fname}"}), 400

        header = np.fromfile(resolved, dtype=np.float64, count=6)
        if header.size < 6:
            return jsonify({"error": f"Corrupt DEM header: {fname}"}), 400
        minlat, maxlat, minlon, maxlon, reslat, reslon = (float(v) for v in header)
        if reslat <= 0 or reslon <= 0:
            return jsonify({"error": f"Invalid DEM resolution in {fname}"}), 400

        num_cols = int(round((maxlon - minlon) / reslon)) + 1
        num_rows = int((file_size // 8 - 6) // num_cols)
        if num_cols <= 0 or num_rows <= 0:
            return jsonify({"error": f"Invalid DEM geometry in {fname}"}), 400

        dem = np.memmap(resolved, dtype=np.float64, mode="r", offset=48,
                        shape=(num_rows, num_cols))

        lats = np.empty(len(points), dtype=float)
        lons = np.empty(len(points), dtype=float)
        for i, p in enumerate(points):
            if not isinstance(p, dict) or "lat" not in p or "lon" not in p:
                return jsonify(
                    {"error": f"points[{i}] must be an object with lat/lon."}), 400
            try:
                lats[i] = float(p["lat"])
                lons[i] = float(p["lon"])
            except (TypeError, ValueError):
                return jsonify(
                    {"error": f"points[{i}] lat/lon must be numbers."}), 400
        if not (np.all(np.isfinite(lats)) and np.all(np.isfinite(lons))):
            return jsonify({"error": "Point coordinates must be finite"}), 400

        # Fractional grid coordinates (row 0 = minlat, col 0 = minlon)
        fy = np.clip((lats - minlat) / reslat, 0, num_rows - 1)
        fx = np.clip((lons - minlon) / reslon, 0, num_cols - 1)

        y0 = np.floor(fy).astype(np.int64)
        x0 = np.floor(fx).astype(np.int64)
        y1 = np.minimum(y0 + 1, num_rows - 1)
        x1 = np.minimum(x0 + 1, num_cols - 1)
        dy = fy - y0
        dx = fx - x0

        v00 = np.asarray(dem[y0, x0], dtype=float)
        v10 = np.asarray(dem[y0, x1], dtype=float)
        v01 = np.asarray(dem[y1, x0], dtype=float)
        v11 = np.asarray(dem[y1, x1], dtype=float)

        values = (v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy)
                  + v01 * (1 - dx) * dy + v11 * dx * dy)

        # Points outside the DEM extent are reported as null
        outside = ((lats < minlat) | (lats > maxlat)
                   | (lons < minlon) | (lons > maxlon))
        values = np.where(outside, np.nan, values)
        del dem

        return jsonify({
            "values": _sanitize(values.tolist()),
            "file": fname,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to sample {fname}: {exc}"}), 500


# ---------------------------------------------------------------------------
#  File browser & result download
# ---------------------------------------------------------------------------
@app.route("/api/list-files")
def api_list_files():
    """List files or directories relative to the project root.

    Query params:
        dir:  subdirectory path (default: "data")
        type: "file" (default, list files) or "dir" (list subdirectories)

    Returns: { files: [...] } or { dirs: [...] } depending on type
    """
    subdir = request.args.get("dir", "data").strip() or "data"
    list_type = request.args.get("type", "file").strip()

    try:
        resolved = _resolve(subdir)
    except ValueError as exc:
        return jsonify({"files": [], "dirs": [], "path": subdir, "error": str(exc)}), 400

    if not os.path.isdir(resolved):
        return jsonify({"files": [], "dirs": [], "path": subdir,
                        "error": "directory not found"}), 404

    try:
        names = sorted(os.listdir(resolved))
        if list_type == "dir":
            entries = [
                name for name in names
                if os.path.isdir(os.path.join(resolved, name))
                and not name.startswith(".")
                and name != "__pycache__"
            ]
            return jsonify({"dirs": entries, "path": subdir})

        entries = [name for name in names if os.path.isfile(os.path.join(resolved, name))]
        return jsonify({"files": entries, "path": subdir})
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/download-result/<job_id>")
def api_download_result(job_id):
    """Download the result file of a completed job.

    Returns: the output .dat file as a download.
    """
    with _lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "job not completed"}), 400

    output_path = (job.get("result") or {}).get("output_path", "")
    if not output_path or not os.path.isfile(output_path):
        return jsonify({"error": "result file not found"}), 404

    return send_file(output_path, as_attachment=True,
                     download_name=os.path.basename(output_path))


# ---------------------------------------------------------------------------
#  TIF import endpoints
# ---------------------------------------------------------------------------
TIF_UPLOAD_DIR = "data/tif_uploads"
_ALLOWED_TIF = (".tif", ".tiff")
MAX_TIF_FILES = 500


def _resolve_tif_list(files):
    """Validate and resolve a list of TIF paths from a request body."""
    if not isinstance(files, list) or not files:
        raise ValueError("files must be a non-empty list of paths.")
    if len(files) > MAX_TIF_FILES:
        raise ValueError(f"Too many files ({len(files)} > {MAX_TIF_FILES}).")
    return [_resolve(str(f), must_exist=True) for f in files]


@app.route("/api/upload-tifs", methods=["POST"])
def api_upload_tifs():
    """Upload TIF files from the user's computer.

    Receives multipart/form-data with field 'files'.
    Saves to data/tif_uploads/.
    Returns: { files: [name, ...], rejected: [name, ...] }
    """
    upload_dir = _resolve(TIF_UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)

    incoming = request.files.getlist("files")
    if not incoming:
        return jsonify({"error": "No files uploaded"}), 400

    uploaded, rejected = [], []
    for item in incoming:
        if not item.filename:
            continue
        name = _safe_name(item.filename, "upload.tif")
        if not name.lower().endswith(_ALLOWED_TIF):
            rejected.append(name)
            continue
        base, ext = os.path.splitext(name)
        dest = os.path.join(upload_dir, name)
        counter = 1
        while os.path.exists(dest):
            name = f"{base}_{counter}{ext}"
            dest = os.path.join(upload_dir, name)
            counter += 1
        item.save(dest)
        uploaded.append(name)

    if not uploaded:
        message = "No .tif/.tiff files uploaded"
        if rejected:
            message += f" (rejected: {', '.join(rejected[:5])})"
        return jsonify({"error": message}), 400

    return jsonify({"files": uploaded, "rejected": rejected})


@app.route("/api/scan-tifs")
def api_scan_tifs():
    """Scan a directory for .tif files.

    Query params:
        dir: directory to scan (default: "data"), relative to project root

    Returns: { files: [name, ...], dir: str }
    """
    scan_dir = request.args.get("dir", "data").strip() or "data"
    try:
        resolved = _resolve(scan_dir)
    except ValueError as exc:
        return jsonify({"files": [], "dir": scan_dir, "error": str(exc)}), 400

    if not os.path.isdir(resolved):
        return jsonify({"files": [], "dir": scan_dir})

    try:
        tifs = sorted(
            name for name in os.listdir(resolved)
            if name.lower().endswith(_ALLOWED_TIF)
            and os.path.isfile(os.path.join(resolved, name))
        )
    except OSError as exc:
        return jsonify({"files": [], "dir": scan_dir, "error": str(exc)}), 500
    return jsonify({"files": tifs, "dir": scan_dir})


@app.route("/api/preview-tif-extent", methods=["POST"])
def api_preview_tif_extent():
    """Preview the combined geographic extent of selected TIF files.

    Input JSON: { files: ["data/a.tif", "data/b.tif", ...] }
    Returns: { bounds: {minlat, maxlat, minlon, maxlon}, resolution: {reslat, reslon},
               tiles: [...], total_tiles: N }
    """
    from ..preprocess.tif_mosaic import scan_tif_extent

    data = request.get_json(silent=True) or {}
    try:
        resolved = _resolve_tif_list(data.get("files"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"File not found: {exc}"}), 404

    try:
        return jsonify(scan_tif_extent(resolved))
    except ImportError as exc:
        return jsonify({"error": f"Missing optional dependency: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Failed to scan TIF extent: {exc}"}), 500


def _parse_crop_bounds(value):
    """Validate an optional crop_bounds object."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("crop_bounds must be an object or null.")
    try:
        bounds = {k: float(value[k]) for k in ("minlat", "maxlat", "minlon", "maxlon")}
    except (KeyError, TypeError, ValueError):
        raise ValueError("crop_bounds requires numeric minlat, maxlat, minlon, maxlon.") from None
    if not all(math.isfinite(v) for v in bounds.values()):
        raise ValueError("crop_bounds values must be finite.")
    if not (-90.0 <= bounds["minlat"] < bounds["maxlat"] <= 90.0):
        raise ValueError("crop_bounds latitude range is invalid.")
    if not (-360.0 <= bounds["minlon"] < bounds["maxlon"] <= 360.0):
        raise ValueError("crop_bounds longitude range is invalid.")
    return bounds


@app.route("/api/convert-tif-to-bin", methods=["POST"])
def api_convert_tif_to_bin():
    """Convert selected TIF files to a pyTGF binary .bin file.

    Input JSON:
        files:       list of TIF paths (e.g. ["data/a.tif", "data/b.tif"])
        output_name: base name for output (e.g. "DetailedDEM") -> data/DetailedDEM.bin
        buffer:      degrees to expand crop_bounds (default 0.0)
        crop_bounds: { minlat, maxlat, minlon, maxlon } or null for full extent
        reslat:      target lat resolution (null = finest available)
        reslon:      target lon resolution (null = finest available)

    Returns: { job_id, status: "queued" }  (background job)
    """
    from ..preprocess.tif_mosaic import mosaic_to_bin

    data = request.get_json(silent=True) or {}
    try:
        resolved_files = _resolve_tif_list(data.get("files"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"File not found: {exc}"}), 404

    try:
        output_name = _safe_name(data.get("output_name", "DEM"), "DEM")
        if output_name.lower().endswith(".bin"):
            output_name = output_name[:-4]
        buffer_deg = float(data.get("buffer", 0.0))
        if not (0.0 <= buffer_deg <= 10.0):
            raise ValueError("buffer must be between 0 and 10 degrees.")
        crop_bounds = _parse_crop_bounds(data.get("crop_bounds"))
        reslat = data.get("reslat")
        reslon = data.get("reslon")
        reslat = float(reslat) if reslat not in (None, "") else None
        reslon = float(reslon) if reslon not in (None, "") else None
        for label, value in (("reslat", reslat), ("reslon", reslon)):
            if value is not None and not (0.0 < value <= 10.0):
                raise ValueError(f"{label} must be in (0, 10] degrees.")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    output_path = _resolve(os.path.join("data", f"{output_name}.bin"))

    job_id = uuid.uuid4().hex[:12]
    _register_job(job_id, 3)

    def _run_convert():
        try:
            _update_job(job_id, status="running", phase="converting")

            def progress_cb(current, total, phase):
                _update_job(job_id, progress=current, total=total, phase=phase)

            result = mosaic_to_bin(
                tif_paths=resolved_files,
                output_path=output_path,
                crop_bounds=crop_bounds,
                buffer=buffer_deg,
                target_reslat=reslat,
                target_reslon=reslon,
                progress_callback=progress_cb,
            )
            _update_job(job_id, status="done", phase="complete",
                        progress=3, total=3, result=result)
        except Exception as exc:
            _update_job(job_id, status="error", phase="error", error=str(exc))

    threading.Thread(target=_run_convert, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202




# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="pyTGF Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"pyTGF Web Server starting at {url}")

    if not args.no_browser:
        # Open browser after a short delay so the server is ready
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
