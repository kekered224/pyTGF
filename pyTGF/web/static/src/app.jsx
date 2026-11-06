/* ==========================================================================
   pyTGF — Terrain Gravity Field :: web console
   Plain React (no build-time dependencies beyond JSX transpilation).
   ========================================================================== */
const { useState, useRef, useEffect, useCallback, useMemo } = React;

/* ---------- canvas palette (literal colours, CSS vars are not readable here) */
const P = {
  plot: "#FFFFFF",
  grid: "rgba(14,23,38,0.055)",
  axis: "#8491A6",
  ink: "#0E1726",
  ink2: "#46536B",
  ink3: "#8491A6",
  brand: "#1557A0",
  brandSoft: "rgba(21,87,160,0.10)",
  teal: "#0E7490",
  amber: "#B45309",
  ok: "#0F7B5F",
  danger: "#B42318",
  zones: [
    { fill: "rgba(21,87,160,0.10)", line: "#1557A0", name: "Polyhedron" },
    { fill: "rgba(14,116,144,0.09)", line: "#0E7490", name: "Prism" },
    { fill: "rgba(180,83,9,0.07)", line: "#B45309", name: "Tesseroid" },
    { fill: "rgba(90,105,128,0.07)", line: "#5A6980", name: "Point mass" },
  ],
};

const FF = "ui-monospace,'Cascadia Mono','JetBrains Mono',Consolas,monospace";
const FU = "-apple-system,'Segoe UI','Microsoft YaHei',system-ui,sans-serif";

const ITYPE_OPTS = [
  { v: 0, l: "0 — Geoid height N (cm)" },
  { v: 1, l: "1 — Deflections (ξ, η) + gravity disturbance" },
  { v: 2, l: "2 — Gravity gradient tensor (6 components)" },
  { v: 4, l: "4 — Deflections (ξ, η) + gravity anomaly" },
  { v: 10, l: "10 — All functionals" },
  { v: 103, l: "103 — Geoid height + DOVs + disturbance" },
  { v: 104, l: "104 — Geoid height + DOVs + anomaly" },
];

const COLUMNS_BY_ITYPE = {
  0: "lon  lat  H  N",
  1: "lon  lat  H  ξ  η  δg",
  2: "lon  lat  H  Vxx  Vxy  Vxz  Vyy  Vyz  Vzz",
  4: "lon  lat  H  N  ξ  η  δg",
  103: "lon  lat  H  N  ξ  η  δg",
  104: "lon  lat  H  N  ξ  η  δg",
  10: "lon  lat  H  N  ξ  η  δg  Vxx  Vxy  Vxz  Vyy  Vyz  Vzz",
};

const DEFAULT_BOUNDS = { latMin: 36.0992, latMax: 38.7983, lonMin: -108.9008, lonMax: -105.2017 };
const DEM_FILES = { detailed: "data/DetailedDEM.bin", tess: "data/TessDEM.bin", coarse: "data/CoarseDEM.bin" };
const MAX_GRID_POINTS = 10000;
const RESULT_PREVIEW_ROWS = 200;

/* ---------- coordinate helpers ------------------------------------------- */
const r2nx = (lon, b) => (lon - b.lonMin) / (b.lonMax - b.lonMin);
const r2ny = (lat, b) => (b.latMax - lat) / (b.latMax - b.latMin);
const n2rlon = (nx, b) => b.lonMin + nx * (b.lonMax - b.lonMin);
const n2rlat = (ny, b) => b.latMax - ny * (b.latMax - b.latMin);

const fmt = (v, digits = 4) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(3);
  return v.toFixed(digits);
};

/* ---------- hypsometric ramp (shared with the server) -------------------- */
const TERRAIN_STOPS = [
  [0.00, 30, 80, 30], [0.08, 50, 120, 40], [0.18, 80, 155, 55],
  [0.30, 140, 175, 75], [0.42, 195, 190, 100], [0.52, 210, 185, 115],
  [0.62, 190, 155, 100], [0.72, 165, 120, 80], [0.82, 140, 95, 70],
  [0.90, 165, 155, 145], [0.96, 210, 210, 205], [1.00, 250, 250, 248],
];

function terrainRGB(t) {
  const c = Math.max(0, Math.min(1, t));
  for (let i = 0; i < TERRAIN_STOPS.length - 1; i++) {
    const a = TERRAIN_STOPS[i];
    const b = TERRAIN_STOPS[i + 1];
    if (c <= b[0]) {
      const f = (c - a[0]) / (b[0] - a[0]);
      return [
        (a[1] + f * (b[1] - a[1])) | 0,
        (a[2] + f * (b[2] - a[2])) | 0,
        (a[3] + f * (b[3] - a[3])) | 0,
      ];
    }
  }
  return [250, 250, 248];
}

/* ---------- synthetic DEM (fallback when no binary DEM is available) ----- */
function hash(a, b) { const n = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return n - Math.floor(n); }
function snoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}
function fbm(x, y) {
  let v = 0, amp = 1, tot = 0;
  for (let i = 0; i < 6; i++) { v += amp * snoise(x, y); tot += amp; amp *= 0.48; x *= 2.1; y *= 2.1; }
  return v / tot;
}
function generateDEM(rows, cols) {
  const data = new Float64Array(rows * cols);
  let mn = 1e9, mx = -1e9;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const nx = c / cols, ny = r / rows;
      let h = fbm(nx * 5 + 2.7, ny * 5 + 1.3);
      h += 0.3 * fbm(nx * 3 + 10, ny * 3 + 5);
      const r1 = Math.sqrt((nx - 0.32) ** 2 + (ny - 0.42) ** 2);
      h += 0.55 * Math.exp(-r1 * r1 * 18);
      const r2 = Math.sqrt((nx - 0.72) ** 2 + (ny - 0.28) ** 2);
      h += 0.35 * Math.exp(-r2 * r2 * 28);
      const r3 = Math.sqrt((nx - 0.5) ** 2 + (ny - 0.75) ** 2);
      h -= 0.25 * Math.exp(-r3 * r3 * 12);
      data[r * cols + c] = h;
      if (h < mn) mn = h;
      if (h > mx) mx = h;
    }
  }
  const span = mx - mn || 1;
  for (let i = 0; i < data.length; i++) data[i] = 150 + ((data[i] - mn) / span) * 3650;
  return { data, rows, cols, min: 150, max: 3800 };
}

function interpHeight(dem, nx, ny) {
  const fx = Math.max(0, Math.min(1, nx)) * (dem.cols - 1);
  const fy = Math.max(0, Math.min(1, ny)) * (dem.rows - 1);
  const ix = Math.floor(fx), iy = Math.floor(fy);
  const dx = fx - ix, dy = fy - iy;
  const ix1 = Math.min(ix + 1, dem.cols - 1);
  const iy1 = Math.min(iy + 1, dem.rows - 1);
  const h00 = dem.data[iy * dem.cols + ix];
  const h10 = dem.data[iy * dem.cols + ix1];
  const h01 = dem.data[iy1 * dem.cols + ix];
  const h11 = dem.data[iy1 * dem.cols + ix1];
  return h00 * (1 - dx) * (1 - dy) + h10 * dx * (1 - dy) + h01 * (1 - dx) * dy + h11 * dx * dy;
}

/* ==========================================================================
   Small UI primitives
   ========================================================================== */
const Chevron = () => (
  <svg className="chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
    <path d="M6 3.5 10.5 8 6 12.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

function Card({ title, badge, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={"card" + (open ? " is-open" : "")}>
      <header className="card-head" onClick={() => setOpen(!open)}>
        <Chevron />
        <span className="card-title">{title}</span>
        {badge ? <span className="card-badge">{badge}</span> : null}
      </header>
      {open ? <div className="card-body">{children}</div> : null}
    </section>
  );
}

function Field({ label, unit, hint, children }) {
  return (
    <div className="field">
      {label ? (
        <label className="label">
          {label}
          {unit ? <span className="unit">[{unit}]</span> : null}
        </label>
      ) : null}
      {children}
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

function NumberField({ label, unit, hint, value, onChange, step = 1, min, max }) {
  return (
    <Field label={label} unit={unit} hint={hint}>
      <input
        className="input"
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") { onChange(""); return; }
          const v = parseFloat(raw);
          if (!Number.isNaN(v)) onChange(v);
        }}
      />
    </Field>
  );
}

function SelectField({ label, value, options, onChange }) {
  return (
    <Field label={label}>
      <select className="select" value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {options.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </Field>
  );
}

function Segmented({ value, options, onChange, ariaLabel }) {
  return (
    <div className="segmented" role="tablist" aria-label={ariaLabel}>
      {options.map((o) => (
        <button
          key={String(o.v)}
          role="tab"
          aria-selected={value === o.v}
          className={value === o.v ? "is-active" : ""}
          onClick={() => onChange(o.v)}
        >
          {o.l}
        </button>
      ))}
    </div>
  );
}

function FileSelect({ label, value, onChange, fileDir = "data", mode = "file" }) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState([]);
  const [status, setStatus] = useState("");
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setStatus("loading");
    fetch(`/api/list-files?dir=${encodeURIComponent(fileDir || "data")}&type=${mode}`)
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        setEntries(mode === "dir" ? (d.dirs || []) : (d.files || []));
        setStatus(d.error ? "error" : "");
      })
      .catch(() => { if (!cancelled) { setEntries([]); setStatus("error"); } });
    return () => { cancelled = true; };
  }, [open, fileDir, mode]);

  useEffect(() => {
    if (!open) return;
    const close = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const prefix = mode === "dir" ? "" : `${fileDir || "data"}/`;

  return (
    <Field label={label}>
      <div ref={wrapRef}>
        <div className="combo">
          <input
            className="input"
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            spellCheck={false}
          />
          <button className="btn btn-ghost btn-sm" type="button" onClick={() => setOpen(!open)}>
            Browse
          </button>
        </div>
        {open ? (
          <div className="dropdown">
            {status === "loading" ? <div className="dropdown-empty">Loading…</div> : null}
            {status !== "loading" && entries.length === 0 ? (
              <div className="dropdown-empty">
                {status === "error" ? "Cannot list this directory" :
                  mode === "dir" ? "No sub-directories" : "No files"}
              </div>
            ) : null}
            {entries.map((name) => {
              const full = prefix + name;
              return (
                <div
                  key={name}
                  className={"dropdown-item" + (value === full ? " is-active" : "")}
                  onClick={() => { onChange(full); setOpen(false); }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                       stroke="currentColor" strokeWidth="1.8">
                    {mode === "dir"
                      ? <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                      : <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>}
                  </svg>
                  {name}
                </div>
              );
            })}
          </div>
        ) : null}
      </div>
    </Field>
  );
}

function ZoneBar({ rzones }) {
  const maxR = Math.max(...rzones, 0.01);
  const W = 100;
  return (
    <div className="zonebar">
      <svg viewBox={`0 0 ${W + 4} 40`} style={{ width: "100%", height: 46 }} preserveAspectRatio="none">
        <rect x="2" y="14" width={W} height="16" rx="2.5" fill="#EEF2F8" />
        {rzones.map((r, i) => (
          <rect key={i} x="2" y="14" width={Math.max((r / maxR) * W, 0.5)} height="16" rx="2.5"
                fill={P.zones[i].fill} stroke={P.zones[i].line} strokeWidth="0.7" />
        ))}
        {rzones.map((r, i) => {
          const px = 2 + (r / maxR) * W;
          return (
            <g key={"t" + i}>
              <line x1={px} y1="11" x2={px} y2="33" stroke={P.zones[i].line}
                    strokeWidth="0.7" strokeDasharray="2,1.4" />
              <text x={px} y="9" fill={P.zones[i].line} fontSize="4.6"
                    fontFamily={FF} textAnchor="middle">{r}°</text>
            </g>
          );
        })}
      </svg>
      <div className="zone-legend">
        {P.zones.map((z, i) => (
          <span key={i} style={{ color: z.line }}>Z{i + 1}</span>
        ))}
      </div>
    </div>
  );
}

function RadiusRow({ index, value, onChange, minR, maxR }) {
  const lo = minR ?? 0;
  const hi = maxR ?? 10;
  const zone = P.zones[index];
  return (
    <div className="radius-row">
      <div className="radius-top">
        <span className="swatch" style={{ background: zone.line }} />
        <span className="radius-name">r{index + 1} · {zone.name}</span>
        <input
          className="radius-num"
          type="number"
          value={value}
          step={0.01}
          min={lo}
          max={hi}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") return;
            const v = parseFloat(raw);
            if (Number.isNaN(v)) return;
            onChange(Math.max(lo, Math.min(hi, v)));
          }}
        />
      </div>
      <input
        className="range"
        type="range"
        min={lo}
        max={Math.max(hi, lo + 0.01)}
        step={0.005}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          onChange(Math.max(lo, Math.min(hi, v)));
        }}
      />
      <div className="range-ends">
        <span>≥ {lo.toFixed(3)}°</span>
        <span>≤ {hi.toFixed(3)}°</span>
      </div>
    </div>
  );
}

/* ==========================================================================
   App
   ========================================================================== */
function App() {
  const DEM_ROWS = 160, DEM_COLS = 220;
  const dem = useMemo(() => generateDEM(DEM_ROWS, DEM_COLS), []);
  const demImage = useMemo(() => {
    const cv = document.createElement("canvas");
    cv.width = DEM_COLS;
    cv.height = DEM_ROWS;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(DEM_COLS, DEM_ROWS);
    for (let i = 0; i < dem.data.length; i++) {
      const t = (dem.data[i] - dem.min) / (dem.max - dem.min);
      const [r, g, b] = terrainRGB(t);
      const j = i * 4;
      img.data[j] = r; img.data[j + 1] = g; img.data[j + 2] = b; img.data[j + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }, [dem]);

  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const pollRef = useRef(null);
  const panRef = useRef(null);
  const demUrlRef = useRef(null);
  const tifPollRef = useRef(null);

  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 520 });
  const [compPts, setCompPts] = useState([]);
  const [selIdx, setSelIdx] = useState(0);
  const [mouseNorm, setMouseNorm] = useState(null);
  const [dragIdx, setDragIdx] = useState(-1);
  const [hoverZone, setHoverZone] = useState(-1);

  const [rectMode, setRectMode] = useState(false);
  const [rectDragging, setRectDragging] = useState(false);

  const [gridLatMin, setGridLatMin] = useState(37.0);
  const [gridLatMax, setGridLatMax] = useState(38.0);
  const [gridLonMin, setGridLonMin] = useState(-108.0);
  const [gridLonMax, setGridLonMax] = useState(-106.0);
  const [gridRes, setGridRes] = useState(0.05);

  const [ikind, setIkind] = useState(2);
  const [itype, setItype] = useState(103);
  const [flagEarth, setFlagEarth] = useState(0);
  const [rzones, setRzones] = useState([0.03, 0.1, 0.2, 1.0]);
  const [density, setDensity] = useState(2.67);
  const [outname, setOutname] = useState("tgf_result.dat");
  const [outputDir, setOutputDir] = useState("output");
  const [files, setFiles] = useState({
    detailedDEM: "data/DetailedDEM.bin",
    detailedREF: "data/DetailedREF.bin",
    tessDEM: "data/TessDEM.bin",
    tessREF: "data/TessREF.bin",
    coarseDEM: "data/CoarseDEM.bin",
    coarseREF: "data/CoarseREF.bin",
  });

  const [importStatus, setImportStatus] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState("");
  const [result, setResult] = useState(null);
  const [resultJobId, setResultJobId] = useState(null);
  const [runError, setRunError] = useState(null);
  const [sidebarTab, setSidebarTab] = useState(0);

  const [demLayer, setDemLayer] = useState("detailed");
  const [realDemImage, setRealDemImage] = useState(null);
  const [demLoading, setDemLoading] = useState(false);
  const [demBounds, setDemBounds] = useState(DEFAULT_BOUNDS);
  const B = demBounds;

  const [viewZoom, setViewZoom] = useState(1);
  const [viewOffset, setViewOffset] = useState({ x: 0, y: 0 });

  const TIF_UPLOAD = "data/tif_uploads";
  const [tifFiles, setTifFiles] = useState([]);
  const [tifExtent, setTifExtent] = useState(null);
  const [tifTarget, setTifTarget] = useState("Detailed");
  const [tifCustomName, setTifCustomName] = useState("");
  const [tifBuffer, setTifBuffer] = useState(0.3);
  const [tifCropBounds, setTifCropBounds] = useState(null);
  const [tifStatus, setTifStatus] = useState(null);
  const [tifConverting, setTifConverting] = useState(false);
  const [tifUploading, setTifUploading] = useState(false);
  const [tifProgress, setTifProgress] = useState({ cur: 0, tot: 3, phase: "" });

  const sel = compPts[selIdx] || compPts[0] || { lon: 0.5, lat: 0.5, h: 1800 };

  /* ---------- layout observer ------------------------------------------- */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ w: Math.max(320, Math.floor(width)), h: Math.max(240, Math.floor(height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ---------- DEM raster fetch ------------------------------------------ */
  useEffect(() => {
    const fname = DEM_FILES[demLayer];
    if (!fname) { setRealDemImage(null); return; }

    let cancelled = false;
    setDemLoading(true);

    fetch(`/api/dem-image?file=${encodeURIComponent(fname)}`)
      .then((r) => {
        if (!r.ok) throw new Error("DEM unavailable");
        const minLat = parseFloat(r.headers.get("X-DEM-MinLat"));
        const maxLat = parseFloat(r.headers.get("X-DEM-MaxLat"));
        const minLon = parseFloat(r.headers.get("X-DEM-MinLon"));
        const maxLon = parseFloat(r.headers.get("X-DEM-MaxLon"));
        if (!Number.isNaN(minLat) && !Number.isNaN(maxLat) && !Number.isNaN(minLon) && !Number.isNaN(maxLon)
            && maxLat > minLat && maxLon > minLon) {
          setDemBounds({ latMin: minLat, latMax: maxLat, lonMin: minLon, lonMax: maxLon });
        }
        return r.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        if (demUrlRef.current) URL.revokeObjectURL(demUrlRef.current);
        const url = URL.createObjectURL(blob);
        demUrlRef.current = url;
        const img = new Image();
        img.onload = () => { if (!cancelled) { setRealDemImage(img); setDemLoading(false); } };
        img.onerror = () => { if (!cancelled) { setRealDemImage(null); setDemLoading(false); } };
        img.src = url;
      })
      .catch(() => { if (!cancelled) { setRealDemImage(null); setDemLoading(false); } });

    setViewZoom(1);
    setViewOffset({ x: 0, y: 0 });
    return () => { cancelled = true; };
  }, [demLayer]);

  useEffect(() => () => {
    if (demUrlRef.current) URL.revokeObjectURL(demUrlRef.current);
    if (pollRef.current) clearInterval(pollRef.current);
    if (tifPollRef.current) clearInterval(tifPollRef.current);
  }, []);

  /* ---------- real DEM sampling ----------------------------------------- */
  const sampleHeights = useCallback(async (normPoints) => {
    const file = DEM_FILES[demLayer];
    if (!file || normPoints.length === 0) return null;
    const points = normPoints.map((p) => ({
      lat: n2rlat(p.lat, B),
      lon: n2rlon(p.lon, B),
    }));
    try {
      const res = await fetch("/api/dem-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file, points }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      return Array.isArray(data.values) ? data.values : null;
    } catch (err) {
      return null;
    }
  }, [demLayer, B]);

  /* ---------- points: import ------------------------------------------- */
  const handleImport = useCallback(async (file) => {
    if (!file) return;
    setImportStatus({ type: "loading", msg: "Reading file…" });
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/import-points-file", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Import failed");
      const norm = data.points.map((p) => ({ lon: r2nx(p.lon, B), lat: r2ny(p.lat, B), h: p.h }));
      setCompPts(norm);
      setSelIdx(0);
      setImportStatus({
        type: "success",
        msg: `Imported ${data.count} point${data.count === 1 ? "" : "s"}`,
        warnings: data.warnings || [],
      });
    } catch (e) {
      setImportStatus({ type: "error", msg: e.message });
    }
  }, [B]);

  /* ---------- points: grid --------------------------------------------- */
  const generateGrid = useCallback(async () => {
    const lat1 = Math.min(gridLatMin, gridLatMax);
    const lat2 = Math.max(gridLatMin, gridLatMax);
    const lon1 = Math.min(gridLonMin, gridLonMax);
    const lon2 = Math.max(gridLonMin, gridLonMax);
    const step = Number(gridRes);
    if (!(step > 0)) { setRunError("Grid resolution must be greater than zero."); return; }

    const nLat = Math.max(1, Math.floor((lat2 - lat1) / step) + 1);
    const nLon = Math.max(1, Math.floor((lon2 - lon1) / step) + 1);
    if (nLat * nLon > MAX_GRID_POINTS) {
      setRunError(`Grid too large: ${nLat} × ${nLon} = ${nLat * nLon} points (limit ${MAX_GRID_POINTS}).`);
      return;
    }

    const pts = [];
    for (let i = 0; i < nLat; i++) {
      const lat = lat1 + i * step;
      for (let j = 0; j < nLon; j++) {
        const lon = lon1 + j * step;
        pts.push({ lon: r2nx(lon, B), lat: r2ny(lat, B), h: 0 });
      }
    }

    const sampled = await sampleHeights(pts);
    const withH = pts.map((p, i) => ({
      ...p,
      h: sampled && sampled[i] !== null && Number.isFinite(sampled[i])
        ? sampled[i]
        : interpHeight(dem, p.lon, p.lat),
    }));

    setCompPts(withH);
    setSelIdx(0);
    setRectMode(false);
    if (!sampled) {
      setRunError("DEM sampling unavailable — heights fall back to the synthetic preview surface.");
    } else {
      setRunError(null);
    }
  }, [gridLatMin, gridLatMax, gridLonMin, gridLonMax, gridRes, dem, B, sampleHeights]);

  /* ---------- plot geometry -------------------------------------------- */
  const getPlotDims = useCallback(() => {
    const pad = { l: 54, r: 22, t: 18, b: 40 };
    return { pad, pw: Math.max(10, canvasSize.w - pad.l - pad.r), ph: Math.max(10, canvasSize.h - pad.t - pad.b) };
  }, [canvasSize]);

  const screenToNorm = useCallback((sx, sy) => {
    const { pad, pw, ph } = getPlotDims();
    return {
      x: ((sx - pad.l) / pw - viewOffset.x) / viewZoom,
      y: ((sy - pad.t) / ph - viewOffset.y) / viewZoom,
    };
  }, [getPlotDims, viewZoom, viewOffset]);

  const normToScreen = useCallback((nx, ny) => {
    const { pad, pw, ph } = getPlotDims();
    return {
      x: pad.l + (nx * viewZoom + viewOffset.x) * pw,
      y: pad.t + (ny * viewZoom + viewOffset.y) * ph,
    };
  }, [getPlotDims, viewZoom, viewOffset]);

  /* ---------- canvas paint --------------------------------------------- */
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = canvasSize.w, H = canvasSize.h;
    const { pad, pw, ph } = getPlotDims();
    const dpr = window.devicePixelRatio || 1;

    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = P.plot;
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, pw, ph);
    ctx.clip();

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    const active = realDemImage || demImage;
    ctx.drawImage(active, pad.l + viewOffset.x * pw, pad.t + viewOffset.y * ph,
                  pw * viewZoom, ph * viewZoom);

    // graticule
    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      const sx = pad.l + (t * viewZoom + viewOffset.x) * pw;
      const sy = pad.t + (t * viewZoom + viewOffset.y) * ph;
      ctx.beginPath(); ctx.moveTo(sx, pad.t); ctx.lineTo(sx, pad.t + ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad.l, sy); ctx.lineTo(pad.l + pw, sy); ctx.stroke();
    }

    // integration zones
    const cx = sel.lon, cy = sel.lat;
    const [pcx, pcy] = [normToScreen(cx, cy).x, normToScreen(cx, cy).y];
    for (let zi = 3; zi >= 0; zi--) {
      const r = rzones[zi];
      if (!(r > 0)) continue;
      const halfW = (r / (B.lonMax - B.lonMin)) * viewZoom * pw;
      const halfH = (r / (B.latMax - B.latMin)) * viewZoom * ph;
      const z = P.zones[zi];
      ctx.fillStyle = z.fill;
      ctx.fillRect(pcx - halfW, pcy - halfH, halfW * 2, halfH * 2);
      ctx.strokeStyle = z.line;
      ctx.lineWidth = hoverZone === zi ? 2.2 : 1.1;
      ctx.setLineDash(zi === 0 ? [] : [5, 4]);
      ctx.strokeRect(pcx - halfW, pcy - halfH, halfW * 2, halfH * 2);
      ctx.setLineDash([]);
      ctx.font = `600 10px ${FF}`;
      ctx.fillStyle = z.line;
      ctx.textAlign = "left";
      ctx.globalAlpha = 0.95;
      ctx.fillText(`Z${zi + 1}`, pcx - halfW + 5, pcy - halfH + 13);
      ctx.globalAlpha = 0.7;
      ctx.font = `10px ${FF}`;
      ctx.fillText(`${r}°`, pcx - halfW + 5, pcy - halfH + 25);
      ctx.globalAlpha = 1;
    }

    // crosshair
    ctx.strokeStyle = "rgba(21,87,160,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pcx, pad.t); ctx.lineTo(pcx, pad.t + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, pcy); ctx.lineTo(pad.l + pw, pcy); ctx.stroke();
    ctx.setLineDash([]);

    // rectangle preview
    const rectCorners = () => {
      const s1 = normToScreen(
        r2nx(Math.min(gridLonMin, gridLonMax), B),
        r2ny(Math.max(gridLatMin, gridLatMax), B));
      const s2 = normToScreen(
        r2nx(Math.max(gridLonMin, gridLonMax), B),
        r2ny(Math.min(gridLatMin, gridLatMax), B));
      return { s1, s2, w: s2.x - s1.x, h: s2.y - s1.y };
    };
    if (rectMode) {
      const { s1, w, h } = rectCorners();
      ctx.fillStyle = rectDragging ? "rgba(21,87,160,0.10)" : "rgba(21,87,160,0.05)";
      ctx.fillRect(s1.x, s1.y, w, h);
      ctx.strokeStyle = rectDragging ? P.brand : "rgba(21,87,160,0.5)";
      ctx.lineWidth = rectDragging ? 1.8 : 1.2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(s1.x, s1.y, w, h);
      ctx.setLineDash([]);

      const lat1 = Math.min(gridLatMin, gridLatMax);
      const lat2 = Math.max(gridLatMin, gridLatMax);
      const lon1 = Math.min(gridLonMin, gridLonMax);
      const lon2 = Math.max(gridLonMin, gridLonMax);
      const nLat = Math.max(1, Math.floor((lat2 - lat1) / gridRes) + 1);
      const nLon = Math.max(1, Math.floor((lon2 - lon1) / gridRes) + 1);
      ctx.font = `600 11px ${FF}`;
      ctx.fillStyle = P.brand;
      ctx.textAlign = "center";
      ctx.fillText(`${nLat} × ${nLon} = ${nLat * nLon} pts`, s1.x + w / 2, s1.y - 7);
    }

    // computation points
    const showLabels = compPts.length <= 40;
    compPts.forEach((pt, i) => {
      const s = normToScreen(pt.lon, pt.lat);
      const isSel = i === selIdx;
      const rad = isSel ? 5.5 : 4;
      if (isSel) {
        const grd = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, 18);
        grd.addColorStop(0, "rgba(21,87,160,0.22)");
        grd.addColorStop(1, "rgba(21,87,160,0)");
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(s.x, s.y, 18, 0, Math.PI * 2); ctx.fill();
      }
      ctx.beginPath(); ctx.arc(s.x, s.y, rad, 0, Math.PI * 2);
      ctx.fillStyle = isSel ? P.brand : P.teal;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = isSel ? 2 : 1.4;
      ctx.stroke();
      if (showLabels) {
        ctx.font = `600 10px ${FU}`;
        ctx.textAlign = "center";
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 3;
        ctx.strokeText(`P${i + 1}`, s.x, s.y - rad - 5);
        ctx.fillStyle = isSel ? P.brand : P.ink2;
        ctx.fillText(`P${i + 1}`, s.x, s.y - rad - 5);
      }
    });

    // hover readout
    if (mouseNorm) {
      const s = normToScreen(mouseNorm.x, mouseNorm.y);
      const lon = n2rlon(mouseNorm.x, B).toFixed(4);
      const lat = n2rlat(mouseNorm.y, B).toFixed(4);
      const text = `${lat}°N  ${lon}°E`;
      ctx.font = `11px ${FF}`;
      const w = ctx.measureText(text).width + 16;
      const bx = Math.min(Math.max(s.x + 14, pad.l + 2), pad.l + pw - w - 2);
      const by = Math.max(s.y - 30, pad.t + 2);
      ctx.fillStyle = "rgba(255,255,255,0.96)";
      ctx.strokeStyle = "#D3DAE6";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(bx, by, w, 22, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = P.ink;
      ctx.textAlign = "left";
      ctx.fillText(text, bx + 8, by + 15);
    }

    if (rectMode && !rectDragging) {
      ctx.font = `500 12px ${FU}`;
      ctx.fillStyle = P.brand;
      ctx.textAlign = "center";
      ctx.globalAlpha = 0.8;
      ctx.fillText("Drag on the map to select the grid extent", pad.l + pw / 2, pad.t + 26);
      ctx.globalAlpha = 1;
    }

    if (demLoading) {
      ctx.fillStyle = "rgba(255,255,255,0.78)";
      ctx.fillRect(pad.l, pad.t, pw, ph);
      ctx.font = `500 12px ${FU}`;
      ctx.fillStyle = P.ink2;
      ctx.textAlign = "center";
      ctx.fillText("Loading DEM…", pad.l + pw / 2, pad.t + ph / 2);
    }

    ctx.restore();

    // axes
    ctx.font = `10px ${FF}`;
    ctx.fillStyle = P.axis;
    ctx.textAlign = "center";
    for (let i = 0; i <= 10; i += 2) {
      const t = i / 10;
      const sx = pad.l + (t * viewZoom + viewOffset.x) * pw;
      if (sx >= pad.l && sx <= pad.l + pw) {
        const lon = B.lonMin + t * (B.lonMax - B.lonMin);
        ctx.fillText(`${lon.toFixed(2)}°`, sx, H - pad.b + 18);
      }
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= 10; i += 2) {
      const t = i / 10;
      const sy = pad.t + (t * viewZoom + viewOffset.y) * ph;
      if (sy >= pad.t && sy <= pad.t + ph) {
        const lat = B.latMax - t * (B.latMax - B.latMin);
        ctx.fillText(`${lat.toFixed(2)}°`, pad.l - 9, sy + 3.5);
      }
    }
    ctx.strokeStyle = "#D3DAE6";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1);
  }, [canvasSize, dem, demImage, realDemImage, demLoading, compPts, selIdx, mouseNorm,
      rzones, hoverZone, rectMode, rectDragging, gridRes, gridLatMin, gridLatMax,
      gridLonMin, gridLonMax, viewZoom, viewOffset, getPlotDims, normToScreen, B]);

  /* ---------- pointer interaction -------------------------------------- */
  const getMousePos = useCallback((e) => {
    const cv = canvasRef.current;
    if (!cv) return null;
    const rect = cv.getBoundingClientRect();
    return { sx: e.clientX - rect.left, sy: e.clientY - rect.top };
  }, []);

  const isInPlot = useCallback((sx, sy) => {
    const { pad, pw, ph } = getPlotDims();
    return sx >= pad.l && sx <= pad.l + pw && sy >= pad.t && sy <= pad.t + ph;
  }, [getPlotDims]);

  const onWheel = useCallback((e) => {
    e.preventDefault();
    const pos = getMousePos(e);
    if (!pos || !isInPlot(pos.sx, pos.sy)) return;
    const { pad, pw, ph } = getPlotDims();
    const px = (pos.sx - pad.l) / pw;
    const py = (pos.sy - pad.t) / ph;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const next = Math.max(1, Math.min(20, viewZoom * factor));
    const nx = (px - viewOffset.x) / viewZoom;
    const ny = (py - viewOffset.y) / viewZoom;
    setViewZoom(next);
    setViewOffset({ x: px - nx * next, y: py - ny * next });
  }, [getMousePos, isInPlot, getPlotDims, viewZoom, viewOffset]);

  const onMouseMove = useCallback((e) => {
    const pos = getMousePos(e);
    if (!pos) { setMouseNorm(null); return; }
    const n = screenToNorm(pos.sx, pos.sy);
    const inside = n.x >= 0 && n.x <= 1 && n.y >= 0 && n.y <= 1;
    setMouseNorm(inside ? n : null);

    if (panRef.current) {
      const { pw, ph } = getPlotDims();
      setViewOffset({
        x: panRef.current.startOx + (pos.sx - panRef.current.startX) / pw,
        y: panRef.current.startOy + (pos.sy - panRef.current.startY) / ph,
      });
      return;
    }
    if (rectDragging && inside) {
      setGridLonMax(n2rlon(n.x, B));
      setGridLatMin(n2rlat(n.y, B));
      return;
    }
    if (dragIdx >= 0 && inside) {
      setCompPts((prev) => prev.map((p, i) => (i === dragIdx ? { ...p, lon: n.x, lat: n.y } : p)));
    }
  }, [getMousePos, screenToNorm, getPlotDims, dragIdx, rectDragging, B]);

  const onMouseDown = useCallback((e) => {
    const pos = getMousePos(e);
    if (!pos) return;
    if (e.shiftKey || e.button === 1) {
      panRef.current = { startX: pos.sx, startY: pos.sy, startOx: viewOffset.x, startOy: viewOffset.y };
      e.preventDefault();
      return;
    }
    const n = screenToNorm(pos.sx, pos.sy);
    if (n.x < 0 || n.x > 1 || n.y < 0 || n.y > 1) return;
    if (rectMode) {
      setGridLonMin(n2rlon(n.x, B));
      setGridLatMax(n2rlat(n.y, B));
      setGridLonMax(n2rlon(n.x, B));
      setGridLatMin(n2rlat(n.y, B));
      setRectDragging(true);
      return;
    }
    const tol = 0.028 / viewZoom;
    const hit = compPts.findIndex((p) => Math.abs(p.lon - n.x) < tol && Math.abs(p.lat - n.y) < tol);
    if (hit >= 0) setDragIdx(hit);
  }, [getMousePos, screenToNorm, compPts, rectMode, viewOffset, viewZoom, B]);

  const onMouseUp = useCallback(() => {
    if (panRef.current) { panRef.current = null; return; }
    if (rectDragging) { setRectDragging(false); return; }
    setDragIdx(-1);
  }, [rectDragging]);

  const onCanvasClick = useCallback(async (e) => {
    if (rectMode || e.shiftKey) return;
    const pos = getMousePos(e);
    if (!pos) return;
    const n = screenToNorm(pos.sx, pos.sy);
    if (n.x < 0 || n.x > 1 || n.y < 0 || n.y > 1) return;

    const tol = 0.028 / viewZoom;
    const hit = compPts.findIndex((p) => Math.abs(p.lon - n.x) < tol && Math.abs(p.lat - n.y) < tol);
    if (hit >= 0) { setSelIdx(hit); return; }

    const sampled = await sampleHeights([{ lon: n.x, lat: n.y }]);
    const h = sampled && Number.isFinite(sampled[0]) ? sampled[0] : interpHeight(dem, n.x, n.y);
    setCompPts((prev) => [...prev, { lon: n.x, lat: n.y, h }]);
    setSelIdx(compPts.length);
  }, [getMousePos, screenToNorm, compPts, dem, rectMode, viewZoom, sampleHeights]);

  const resetView = useCallback(() => {
    setViewZoom(1);
    setViewOffset({ x: 0, y: 0 });
  }, []);

  const zoomBy = useCallback((factor) => {
    const next = Math.max(1, Math.min(20, viewZoom * factor));
    if (next <= 1) { setViewZoom(1); setViewOffset({ x: 0, y: 0 }); return; }
    setViewZoom(next);
    setViewOffset({ x: 0.5 - 0.5 * next, y: 0.5 - 0.5 * next });
  }, [viewZoom]);

  /* ---------- run ------------------------------------------------------- */
  const handleRun = useCallback(async () => {
    if (isRunning) return;
    if (compPts.length === 0) { setRunError("No computation points. Add points on the map, import a file or generate a grid."); return; }

    setIsRunning(true);
    setProgress(0);
    setPhase("submitting");
    setResult(null);
    setRunError(null);
    setResultJobId(null);

    const realPoints = compPts.map((p) => ({
      lon: n2rlon(p.lon, B), lat: n2rlat(p.lat, B), h: p.h,
    }));

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          comp_points: realPoints, ikind, itype, flag_earth: flagEarth,
          rzones, density, outname, output_dir: outputDir, files,
        }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.error || `Server error ${res.status}`);

      const jobId = payload.job_id;
      setResultJobId(jobId);
      setPhase("queued");

      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`/api/status/${jobId}`);
          const st = await sr.json();
          if (!sr.ok) throw new Error(st.error || "Job lost");
          if (st.total > 0) setProgress(Math.min(100, (st.progress / st.total) * 100));
          setPhase(st.phase || st.status);
          if (st.status === "done") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setProgress(100);
            setPhase("done");
            const rr = await fetch(`/api/result/${jobId}`);
            const data = await rr.json();
            if (data.status === "error") throw new Error(data.error || "Calculation failed");
            setResult(data);
            setIsRunning(false);
          } else if (st.status === "error") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setRunError(st.error || "Calculation failed");
            setIsRunning(false);
          }
        } catch (err) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setRunError(err.message);
          setIsRunning(false);
        }
      }, 700);
    } catch (err) {
      setRunError(err.message);
      setIsRunning(false);
    }
  }, [isRunning, compPts, ikind, itype, flagEarth, rzones, density, outname, outputDir, files, B]);

  const handleExport = useCallback(() => {
    if (!resultJobId) { setRunError("No result file available yet."); return; }
    window.open(`/api/download-result/${resultJobId}`, "_blank");
  }, [resultJobId]);

  /* ---------- TIF workflow --------------------------------------------- */
  const handleUploadTifs = useCallback(async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setTifUploading(true);
    setTifExtent(null);
    setTifCropBounds(null);
    setTifStatus({ type: "loading", msg: `Uploading ${fileList.length} file${fileList.length > 1 ? "s" : ""}…` });
    try {
      const fd = new FormData();
      for (const f of fileList) fd.append("files", f);
      const r = await fetch("/api/upload-tifs", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || "Upload failed");
      setTifFiles(d.files || []);
      setTifStatus({ type: "success", msg: `Uploaded ${(d.files || []).length} file(s)` });
    } catch (err) {
      setTifStatus({ type: "error", msg: err.message });
      setTifFiles([]);
    } finally {
      setTifUploading(false);
    }
  }, []);

  useEffect(() => {
    if (tifFiles.length === 0) { setTifExtent(null); return; }
    let cancelled = false;
    const paths = tifFiles.map((f) => `${TIF_UPLOAD}/${f}`);
    fetch("/api/preview-tif-extent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: paths }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (cancelled || d.error) return;
        setTifExtent(d);
        const maxRz = Math.max(...rzones);
        setTifBuffer((prev) => Math.max(prev, +(maxRz + 0.1).toFixed(2)));
        if (compPts.length > 0) {
          const lats = compPts.map((p) => n2rlat(p.lat, B));
          const lons = compPts.map((p) => n2rlon(p.lon, B));
          setTifCropBounds({
            minlat: +(Math.min(...lats) - maxRz).toFixed(4),
            maxlat: +(Math.max(...lats) + maxRz).toFixed(4),
            minlon: +(Math.min(...lons) - maxRz).toFixed(4),
            maxlon: +(Math.max(...lons) + maxRz).toFixed(4),
          });
        }
      })
      .catch(() => { if (!cancelled) setTifExtent(null); });
    return () => { cancelled = true; };
  }, [tifFiles]);   // eslint-disable-line react-hooks/exhaustive-deps

  const handleConvertTif = useCallback(async () => {
    if (tifFiles.length === 0) { setTifStatus({ type: "error", msg: "No TIF files uploaded" }); return; }
    const paths = tifFiles.map((f) => `${TIF_UPLOAD}/${f}`);
    const outName = tifTarget === "Custom" ? (tifCustomName || "DEM") : `${tifTarget}DEM`;
    const crop = tifCropBounds || (tifExtent ? tifExtent.bounds : null);

    setTifConverting(true);
    setTifStatus({ type: "loading", msg: "Starting conversion…" });
    setTifProgress({ cur: 0, tot: 3, phase: "queued" });

    try {
      const r = await fetch("/api/convert-tif-to-bin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: paths, output_name: outName, buffer: tifBuffer, crop_bounds: crop }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || "Conversion failed");

      if (tifPollRef.current) clearInterval(tifPollRef.current);
      tifPollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`/api/status/${d.job_id}`);
          const sd = await sr.json();
          setTifProgress({ cur: sd.progress || 0, tot: sd.total || 3, phase: sd.phase || "" });
          if (sd.status === "done") {
            clearInterval(tifPollRef.current);
            tifPollRef.current = null;
            setTifConverting(false);
            const shape = (sd.result && sd.result.shape) || [];
            setTifStatus({
              type: "success",
              msg: `Saved ${outName}.bin${shape.length ? ` (${shape[0]} × ${shape[1]})` : ""}`,
            });
            const key = tifTarget === "Detailed" ? "detailedDEM"
                      : tifTarget === "Tess" ? "tessDEM"
                      : tifTarget === "Coarse" ? "coarseDEM" : null;
            if (key) setFiles((prev) => ({ ...prev, [key]: `data/${outName}.bin` }));
          } else if (sd.status === "error") {
            clearInterval(tifPollRef.current);
            tifPollRef.current = null;
            setTifConverting(false);
            setTifStatus({ type: "error", msg: sd.error || "Conversion failed" });
          }
        } catch (err) {
          if (tifPollRef.current) clearInterval(tifPollRef.current);
          tifPollRef.current = null;
          setTifConverting(false);
          setTifStatus({ type: "error", msg: err.message });
        }
      }, 600);
    } catch (err) {
      setTifConverting(false);
      setTifStatus({ type: "error", msg: err.message });
    }
  }, [tifFiles, tifTarget, tifCustomName, tifBuffer, tifCropBounds, tifExtent]);

  const updateFile = (k, v) => setFiles((prev) => ({ ...prev, [k]: v }));

  /* ---------- derived --------------------------------------------------- */
  const pointSummary = useMemo(() => {
    if (compPts.length === 0) return null;
    const lats = compPts.map((p) => n2rlat(p.lat, B));
    const lons = compPts.map((p) => n2rlon(p.lon, B));
    const hs = compPts.map((p) => p.h);
    return {
      lat: [Math.min(...lats), Math.max(...lats)],
      lon: [Math.min(...lons), Math.max(...lons)],
      h: [Math.min(...hs), Math.max(...hs)],
    };
  }, [compPts, B]);

  /* ======================================================================
     render
     ====================================================================== */
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.6">
              <circle cx="12" cy="12" r="9" />
              <path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" />
            </svg>
          </div>
          <span className="brand-name">py<em>TGF</em></span>
          <span className="brand-sub">Terrain Gravity Field</span>
        </div>
        <div className="spacer" />
        {isRunning ? (
          <div className="run-status">
            <span className="run-phase">{phase || "running"}</span>
            <div className="progress"><span style={{ width: `${Math.min(progress, 100)}%` }} /></div>
            <span className="progress-num">{Math.min(progress, 100).toFixed(0)}%</span>
          </div>
        ) : null}
        <button className="btn btn-primary" onClick={handleRun} disabled={isRunning}>
          {isRunning ? "Running…" : "▶  Run calculation"}
        </button>
      </header>

      <div className="body">
        {/* ---------------- sidebar ---------------- */}
        <aside className="sidebar">
          <nav className="tabs">
            {["Parameters", "Data & DEM"].map((t, i) => (
              <button key={t} className={"tab" + (sidebarTab === i ? " is-active" : "")}
                      onClick={() => setSidebarTab(i)}>{t}</button>
            ))}
          </nav>

          <div className="scroll">
            {sidebarTab === 0 ? (
              <>
                <Card title="Calculation mode">
                  <SelectField
                    label="Field type"
                    value={ikind}
                    onChange={setIkind}
                    options={[
                      { v: 1, l: "TC — full-scale topographic" },
                      { v: 2, l: "RTM — residual terrain model" },
                    ]}
                  />
                  <SelectField label="Output quantities (itype)" value={itype}
                               options={ITYPE_OPTS} onChange={setItype} />
                  <Field label="Reference surface">
                    <Segmented
                      ariaLabel="Reference surface"
                      value={flagEarth}
                      onChange={setFlagEarth}
                      options={[{ v: 0, l: "Sphere" }, { v: 1, l: "Ellipsoid" }]}
                    />
                  </Field>
                </Card>

                <Card title="Integration radii" badge="r1 ≤ r2 ≤ r3 ≤ r4">
                  <ZoneBar rzones={rzones} />
                  {rzones.map((r, i) => (
                    <RadiusRow
                      key={i}
                      index={i}
                      value={r}
                      minR={i > 0 ? rzones[i - 1] : 0}
                      maxR={i < 3 ? rzones[i + 1] : 10}
                      onChange={(v) => {
                        const next = [...rzones];
                        next[i] = v;
                        for (let j = i + 1; j < 4; j++) if (next[j] < next[j - 1]) next[j] = next[j - 1];
                        for (let j = i - 1; j >= 0; j--) if (next[j] > next[j + 1]) next[j] = next[j + 1];
                        setRzones(next);
                      }}
                    />
                  ))}
                </Card>

                <Card title="Density">
                  <NumberField label="Crustal density ρ" unit="g/cm³" value={density}
                               onChange={setDensity} step={0.01} min={0} max={5} />
                </Card>

                <Card title="Computation points" badge={compPts.length ? `${compPts.length} pts` : ""}>
                  <Field label="Import from text file" unit="lat lon height">
                    <label className="dropzone">
                      <input type="file" accept=".dat,.txt,.csv,.xyz" style={{ display: "none" }}
                             onChange={(e) => { handleImport(e.target.files[0]); e.target.value = ""; }} />
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                           stroke={P.brand} strokeWidth="1.7" strokeLinecap="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <path d="M12 3v12M7 8l5-5 5 5" />
                      </svg>
                      <div>
                        <div className="dropzone-title">Choose a file…</div>
                        <div className="dropzone-sub">.dat · .txt · .csv · .xyz</div>
                      </div>
                    </label>
                  </Field>

                  {importStatus ? (
                    <div className={"notice notice-" + (
                      importStatus.type === "error" ? "error" :
                      importStatus.type === "success" ? "ok" : "info")}>
                      <span>{importStatus.type === "error" ? "✕" : importStatus.type === "success" ? "✓" : "…"}</span>
                      <span>
                        {importStatus.msg}
                        {(importStatus.warnings || []).map((w, i) => (
                          <div key={i} style={{ marginTop: 3, opacity: .85 }}>⚠ {w}</div>
                        ))}
                      </span>
                    </div>
                  ) : null}

                  <div style={{ height: 12 }} />

                  <Field label="Generate grid from extent">
                    <div className="grid-2" style={{ marginBottom: 9 }}>
                      <NumberField label="Lat min" value={gridLatMin} onChange={setGridLatMin} step={0.01} />
                      <NumberField label="Lat max" value={gridLatMax} onChange={setGridLatMax} step={0.01} />
                      <NumberField label="Lon min" value={gridLonMin} onChange={setGridLonMin} step={0.01} />
                      <NumberField label="Lon max" value={gridLonMax} onChange={setGridLonMax} step={0.01} />
                    </div>
                    <div style={{ display: "flex", alignItems: "flex-end", gap: 9 }}>
                      <div style={{ width: 110 }}>
                        <NumberField label="Step" unit="°" value={gridRes} onChange={setGridRes} step={0.001} min={0.001} max={1} />
                      </div>
                      <div style={{ flex: 1, fontSize: 11, color: "var(--ink-3)", paddingBottom: 9 }}>
                        {(() => {
                          const nLat = Math.max(1, Math.floor(Math.abs(gridLatMax - gridLatMin) / (gridRes || 1)) + 1);
                          const nLon = Math.max(1, Math.floor(Math.abs(gridLonMax - gridLonMin) / (gridRes || 1)) + 1);
                          return `${nLat} × ${nLon} = ${nLat * nLon} pts`;
                        })()}
                      </div>
                    </div>
                  </Field>

                  <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                    <button className="btn btn-ghost btn-sm" style={{ flex: 1 }}
                            onClick={() => setRectMode(!rectMode)}>
                      {rectMode ? "Cancel selection" : "⬚ Select on map"}
                    </button>
                    <button className="btn btn-ok btn-sm" style={{ flex: 1 }} onClick={generateGrid}
                            disabled={rectMode}>
                      ✓ Generate grid
                    </button>
                  </div>

                  <div className="hint">
                    Click the map to add a point · drag to move · Shift+drag to pan · scroll to zoom.
                  </div>

                  <div style={{ height: 10 }} />

                  {compPts.length === 0 ? (
                    <div className="points-empty">No computation points yet.</div>
                  ) : compPts.length > 40 ? (
                    <div className="point-row" style={{ cursor: "default" }}>
                      <span className="point-idx">…</span>
                      <span className="point-coord">
                        {compPts.length} points · lat {pointSummary.lat[0].toFixed(3)}…{pointSummary.lat[1].toFixed(3)}°
                      </span>
                      <button className="icon-btn" title="Clear all"
                              onClick={() => { setCompPts([]); setSelIdx(0); }}>✕</button>
                    </div>
                  ) : (
                    compPts.map((pt, i) => (
                      <div key={i}
                           className={"point-row" + (selIdx === i ? " is-active" : "")}
                           onClick={() => setSelIdx(i)}>
                        <span className="point-idx">{i + 1}</span>
                        <span className="point-coord">
                          {n2rlat(pt.lat, B).toFixed(4)}° {n2rlon(pt.lon, B).toFixed(4)}°
                        </span>
                        <span className="point-h">{pt.h.toFixed(0)} m</span>
                        <button className="icon-btn" title="Remove"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setCompPts((prev) => prev.filter((_, j) => j !== i));
                                  if (selIdx >= i && selIdx > 0) setSelIdx(selIdx - 1);
                                }}>✕</button>
                      </div>
                    ))
                  )}
                </Card>

                <Card title="Output">
                  <FileSelect label="Output directory" value={outputDir} onChange={setOutputDir}
                              fileDir="." mode="dir" />
                  <Field label="File name">
                    <input className="input" value={outname} spellCheck={false}
                           onChange={(e) => setOutname(e.target.value)} />
                  </Field>
                  <div className="notice notice-info" style={{ marginTop: 0 }}>
                    <span>↳</span>
                    <span style={{ fontFamily: "var(--mono)" }}>{outputDir}/{outname}</span>
                  </div>
                  <div style={{ height: 10 }} />
                  <Field label="Output columns">
                    <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--brand-ink)",
                                  background: "var(--brand-soft)", padding: "7px 9px", borderRadius: 6 }}>
                      {COLUMNS_BY_ITYPE[itype] || "—"}
                    </div>
                  </Field>
                </Card>
              </>
            ) : (
              <>
                <Card title="Import GeoTIFF → BIN">
                  <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>
                    Mosaic one or more GeoTIFF tiles and convert them to the binary DEM layout used by the solver.
                  </div>

                  <label className={"dropzone" + (tifUploading ? " is-busy" : "")}>
                    <input type="file" accept=".tif,.tiff" multiple style={{ display: "none" }}
                           onChange={(e) => { handleUploadTifs(e.target.files); e.target.value = ""; }} />
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                         stroke={tifUploading ? P.amber : P.brand} strokeWidth="1.7" strokeLinecap="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <path d="M12 3v12M7 8l5-5 5 5" />
                    </svg>
                    <div>
                      <div className="dropzone-title">{tifUploading ? "Uploading…" : "Choose .tif files"}</div>
                      <div className="dropzone-sub">multiple tiles supported</div>
                    </div>
                  </label>

                  {tifFiles.length > 0 ? (
                    <div style={{ marginTop: 10 }}>
                      <div className="chips">
                        {tifFiles.map((f) => <span key={f} className="chip">{f}</span>)}
                      </div>
                      <div style={{ marginTop: 7 }}>
                        <button className="btn btn-quiet btn-sm"
                                onClick={() => { setTifFiles([]); setTifExtent(null); setTifStatus(null); }}>
                          Clear list
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {tifExtent ? (
                    <div className="notice notice-info" style={{ marginTop: 10, display: "block" }}>
                      <div style={{ fontWeight: 600, marginBottom: 3 }}>
                        Combined extent · {tifExtent.total_tiles} tile{tifExtent.total_tiles > 1 ? "s" : ""}
                      </div>
                      <div style={{ fontFamily: "var(--mono)", lineHeight: 1.7 }}>
                        lat {tifExtent.bounds.minlat.toFixed(4)} … {tifExtent.bounds.maxlat.toFixed(4)}<br />
                        lon {tifExtent.bounds.minlon.toFixed(4)} … {tifExtent.bounds.maxlon.toFixed(4)}<br />
                        res {tifExtent.resolution.reslat.toFixed(6)} × {tifExtent.resolution.reslon.toFixed(6)}
                      </div>
                    </div>
                  ) : null}

                  <div style={{ height: 12 }} />

                  <div style={{ display: "flex", gap: 9, alignItems: "flex-end" }}>
                    <div style={{ flex: 1 }}>
                      <Field label="Target layer">
                        <select className="select" value={tifTarget} onChange={(e) => setTifTarget(e.target.value)}>
                          <option value="Detailed">Detailed DEM</option>
                          <option value="Tess">Tesseroid DEM</option>
                          <option value="Coarse">Coarse DEM</option>
                          <option value="Custom">Custom name</option>
                        </select>
                      </Field>
                    </div>
                    {tifTarget === "Custom" ? (
                      <div style={{ flex: 1 }}>
                        <Field label="Output name">
                          <input className="input" value={tifCustomName} placeholder="MyDEM"
                                 onChange={(e) => setTifCustomName(e.target.value)} />
                        </Field>
                      </div>
                    ) : null}
                  </div>

                  <div style={{ fontSize: 11.5, color: "var(--ink-2)", fontFamily: "var(--mono)", marginBottom: 10 }}>
                    → data/{tifTarget === "Custom" ? (tifCustomName || "DEM") : `${tifTarget}DEM`}.bin
                  </div>

                  <div className="notice notice-info" style={{ marginTop: 0 }}>
                    <span>ⓘ</span>
                    <span>
                      Cropped to the point bounding box + max zone radius
                      ({Math.max(...rzones).toFixed(2)}°) + buffer ({Number(tifBuffer).toFixed(2)}°).
                      {compPts.length === 0 ? <b> No points set — add points first.</b> : null}
                    </span>
                  </div>

                  <div style={{ height: 10 }} />

                  <NumberField label="Buffer" unit="°" value={tifBuffer} step={0.05} min={0} max={5}
                               onChange={(v) => setTifBuffer(v === "" ? 0 : v)}
                               hint={`Recommended: max zone + 0.1° (${(Math.max(...rzones) + 0.1).toFixed(2)}°)`} />

                  {tifCropBounds || tifExtent ? (
                    <div className="grid-2" style={{ marginTop: 8 }}>
                      {["minlat", "maxlat", "minlon", "maxlon"].map((k) => (
                        <NumberField
                          key={k}
                          label={k.replace("lat", " lat").replace("lon", " lon")}
                          value={tifCropBounds?.[k] ?? tifExtent?.bounds?.[k] ?? 0}
                          step={0.01}
                          onChange={(v) => setTifCropBounds((prev) => ({
                            ...(prev || tifExtent?.bounds || {}), [k]: Number(v) || 0,
                          }))}
                        />
                      ))}
                    </div>
                  ) : null}

                  <button className="btn btn-primary btn-block" style={{ marginTop: 10 }}
                          onClick={handleConvertTif}
                          disabled={tifConverting || tifFiles.length === 0}>
                    {tifConverting ? `Converting… (${tifProgress.phase})` : "Convert to .bin"}
                  </button>

                  {tifConverting ? (
                    <div className="progress" style={{ width: "100%", marginTop: 8 }}>
                      <span style={{ width: `${(tifProgress.cur / Math.max(tifProgress.tot, 1)) * 100}%` }} />
                    </div>
                  ) : null}

                  {tifStatus ? (
                    <div className={"notice notice-" + (
                      tifStatus.type === "error" ? "error" :
                      tifStatus.type === "success" ? "ok" : "info")}>
                      <span>{tifStatus.type === "error" ? "✕" : tifStatus.type === "success" ? "✓" : "…"}</span>
                      <span>{tifStatus.msg}</span>
                    </div>
                  ) : null}
                </Card>

                <Card title="DEM & reference files">
                  <div className="hint" style={{ marginTop: 0, marginBottom: 11 }}>
                    Paths are relative to the project root.
                  </div>
                  {[
                    ["detailedDEM", "Detailed DEM"],
                    ["detailedREF", "Detailed REF"],
                    ["tessDEM", "Tesseroid DEM"],
                    ["tessREF", "Tesseroid REF"],
                    ["coarseDEM", "Coarse DEM"],
                    ["coarseREF", "Coarse REF"],
                  ].map(([key, label]) => (
                    <FileSelect key={key} label={label} value={files[key]}
                                onChange={(v) => updateFile(key, v)} fileDir="data" />
                  ))}

                  <div style={{
                    marginTop: 6, padding: "10px 11px", borderRadius: 8,
                    border: "1px solid " + (ikind === 1 ? "#F0D5B4" : "var(--line)"),
                    background: ikind === 1 ? "var(--amber-soft)" : "var(--panel-soft)",
                    opacity: ikind === 1 ? 1 : 0.62,
                  }}>
                    <div className="label" style={{ marginBottom: 8 }}>
                      Global DEM / REF
                      <span className="unit">{ikind === 1 ? "TC mode" : "TC only"}</span>
                    </div>
                    {ikind === 1 ? (
                      <>
                        <FileSelect label="Global DEM" value={files.globalDEM || ""}
                                    onChange={(v) => updateFile("globalDEM", v)} fileDir="data" />
                        <FileSelect label="Global REF" value={files.globalREF || ""}
                                    onChange={(v) => updateFile("globalREF", v)} fileDir="data" />
                      </>
                    ) : (
                      <div className="hint" style={{ marginTop: 0 }}>
                        Switch to TC (full-scale topographic) mode to configure the global layer.
                      </div>
                    )}
                  </div>
                </Card>
              </>
            )}
          </div>
        </aside>

        {/* ---------------- stage ---------------- */}
        <main className="stage">
          <div className="map-wrap" ref={containerRef}>
            <div className="map-toolbar">
              <div className="segmented">
                {[["detailed", "Detailed"], ["tess", "Tesseroid"], ["coarse", "Coarse"]].map(([v, l]) => (
                  <button key={v} className={demLayer === v ? "is-active" : ""}
                          onClick={() => setDemLayer(v)}>{l}</button>
                ))}
              </div>
              <div className="toolbar-sep" />
              <button className="btn btn-quiet" onClick={() => zoomBy(1.3)} title="Zoom in">＋</button>
              <button className="btn btn-quiet" onClick={() => zoomBy(1 / 1.3)} title="Zoom out">−</button>
              <button className="btn btn-quiet" onClick={resetView} title="Reset view">Reset</button>
              <span className="zoom-badge">{viewZoom.toFixed(1)}×</span>
            </div>

            <div className={"map-hint" + (rectMode ? " is-mode" : "")}>
              {rectMode ? "SELECT EXTENT" : `${B.latMin.toFixed(2)}…${B.latMax.toFixed(2)}°N · ${B.lonMin.toFixed(2)}…${B.lonMax.toFixed(2)}°E`}
            </div>

            <canvas
              ref={canvasRef}
              className="map"
              style={{ cursor: rectMode ? "crosshair" : dragIdx >= 0 ? "grabbing" : "crosshair" }}
              onWheel={onWheel}
              onMouseMove={onMouseMove}
              onClick={onCanvasClick}
              onMouseDown={onMouseDown}
              onMouseUp={onMouseUp}
              onMouseLeave={() => {
                setMouseNorm(null);
                setDragIdx(-1);
                if (rectDragging) setRectDragging(false);
                panRef.current = null;
              }}
            />
          </div>

          {result && result.status === "done" ? (
            <div className="drawer">
              <div className="drawer-head">
                <span className="drawer-title">Results</span>
                <span className="drawer-meta">
                  {result.data.length} pts · {(result.elapsed_hours * 3600).toFixed(1)} s
                  {result.data.length > RESULT_PREVIEW_ROWS
                    ? ` · preview ${RESULT_PREVIEW_ROWS} rows` : ""}
                </span>
                <div className="spacer" />
                <button className="btn btn-ghost btn-sm" onClick={handleExport}>↓ Export file</button>
                <button className="btn btn-quiet btn-sm" onClick={() => setResult(null)}>Hide</button>
              </div>
              <div className="drawer-body">
                <table className="results">
                  <thead>
                    <tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {result.data.slice(0, RESULT_PREVIEW_ROWS).map((row, i) => (
                      <tr key={i}>
                        {row.map((v, j) => <td key={j}>{typeof v === "number" ? fmt(v, 6) : (v ?? "—")}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          <div className="statusbar">
            {mouseNorm ? (
              <>
                <span><span className="k">LAT</span> <span className="v">{n2rlat(mouseNorm.y, B).toFixed(4)}°</span></span>
                <span><span className="k">LON</span> <span className="v">{n2rlon(mouseNorm.x, B).toFixed(4)}°</span></span>
                <span><span className="k">H</span> <span className="v-accent">{sel.h.toFixed(1)} m</span></span>
              </>
            ) : (
              <span className="muted">Move over the map to read coordinates</span>
            )}
            <div className="spacer" />
            <span><span className="k">mode</span> <span className="v">{ikind === 1 ? "TC" : "RTM"}</span></span>
            <span><span className="k">itype</span> <span className="v">{itype}</span></span>
            <span><span className="k">ρ</span> <span className="v">{density} g/cm³</span></span>
            <span><span className="k">earth</span> <span className="v">{flagEarth === 0 ? "sphere" : "ellipsoid"}</span></span>
          </div>
        </main>
      </div>

      {runError ? (
        <div className="toast-layer">
          <div className="toast is-error">
            <span>⚠</span>
            <span style={{ flex: 1 }}>{runError}</span>
            <button className="icon-btn" onClick={() => setRunError(null)}>✕</button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
