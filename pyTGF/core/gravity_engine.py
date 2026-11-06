"""
Main gravity field calculation engine

Mirrors the RUN branch of TC_GUI.m (L677-L2035).

Orchestrates the full calculation:
    1. Load computation points and DEM layers
    2. Add Earth radius to convert heights to geocentric radii
    3. Compute zone extents via ZoneStrategy
    4. For each computation point:
        a. Zone 1: Polyhedron (or skip if r1=0)
        b. Leakage boundary: Prism at half weight
        c. Zone 2: Prism
        d. Zone 3: Tesseroid (with cubic interpolation from TessDEM)
        e. Zone 4: Point mass (with cubic interpolation from CoarseDEM)
        f. Zone 5: Point mass on global grid (only ikind=1)
    5. Sum contributions, apply RTM harmonic correction if ikind=2
    6. Convert to physical quantities (N, xi, eta, dg) via normal gravity
    7. Write report and result files
"""

import time
from typing import Dict, Optional, Tuple, List
import numpy as np

from ..config import TGFConfig, PhysicalConstants
from ..io.com_points import ComPointData, read_com_points
from ..io.dem_io import DEMData, read_dem
from ..io.output import (write_report_header, write_report_dem_info,
                          write_report_integration, write_result)
from ..mass_elements import (polyhedron_full, polyhedron_g, polyhedron_g_gx,
                              polyhedron_gx, polyhedron_gxx,
                              prism, prism_g, prism_g_gx, prism_gx, prism_gxx,
                              prism_numba,
                              tesseroid, tesseroid_g, tesseroid_g_gx,
                              tesseroid_gx, tesseroid_gxx,
                              tess_kernel_10, tess_kernel_g, tess_kernel_gx,
                              tess_kernel_g_gx, tess_kernel_gxx,
                              point_g, point_gx, point_g_gx, point_gxx, point_mass,
                              pm_kernel_10, pm_kernel_g, pm_kernel_gx,
                              pm_kernel_g_gx, pm_kernel_gxx)
from ..utils.grid import build_grid, build_grid_layout, matrix_preallocating
from ..utils.normal_gravity import normal_gravity_generic
from .zone_strategy import ZoneStrategy, ZoneExtents


# ---------------------------------------------------------------------------
# itype -> (polyhedron_fn, prism_fn, tesseroid_fn, point_mass_fn) dispatch
# ---------------------------------------------------------------------------
_MASS_DISPATCH = {
    0:   (polyhedron_g,    prism_g,    tess_kernel_g,    pm_kernel_g),
    1:   (polyhedron_gx,   prism_gx,   tess_kernel_gx,   pm_kernel_gx),
    2:   (polyhedron_gxx,  prism_gxx,  tess_kernel_gxx,  pm_kernel_gxx),
    4:   (polyhedron_g_gx, prism_g_gx, tess_kernel_g_gx, pm_kernel_g_gx),
    10:  (polyhedron_full, prism_numba, tess_kernel_10,  pm_kernel_10),
    103: (polyhedron_g_gx, prism_g_gx, tess_kernel_g_gx, pm_kernel_g_gx),
    104: (polyhedron_g_gx, prism_g_gx, tess_kernel_g_gx, pm_kernel_g_gx),
}


def _cubic_interpolator(x_grid_1d: np.ndarray, y_grid_1d: np.ndarray,
                        values_2d: np.ndarray):
    """
    Build a cubic interpolator matching MATLAB griddedInterpolant(x,y,v,'cubic').

    MATLAB's griddedInterpolant expects column-vector x and row-vector y when
    called as griddedInterpolant(xt', yt', zt', 'cubic'), where zt is (n_lat, n_lon).

    Uses scipy's RectBivariateSpline (kx=ky=3, tensor-product cubic spline) which
    is much faster to build than RegularGridInterpolator(method='cubic') on large
    grids, while matching MATLAB's cubic spline behavior.

    Args:
        x_grid_1d: 1D longitude axis (ascending).
        y_grid_1d: 1D latitude axis (ascending).
        values_2d: 2D array shape (n_lat, n_lon) matching meshgrid(y, x).

    Returns:
        Callable accepting (lon_pts, lat_pts) and returning interpolated values.
    """
    from scipy.interpolate import RectBivariateSpline
    # RectBivariateSpline requires ascending axes
    if x_grid_1d[0] > x_grid_1d[-1]:
        x_grid_1d = x_grid_1d[::-1]
        values_2d = values_2d[:, ::-1]
    if y_grid_1d[0] > y_grid_1d[-1]:
        y_grid_1d = y_grid_1d[::-1]
        values_2d = values_2d[::-1, :]

    # Defensive NaN fill: RectBivariateSpline cannot handle NaN input.
    # Fill NaN cells with nearest valid values (border-fill strategy).
    # The copy is only made when NaNs are actually present - on a 46e6-cell
    # tessellation grid an unconditional copy costs ~370 MB per layer.
    if np.any(np.isnan(values_2d)):
        values_2d = np.array(values_2d, dtype=np.float64, copy=True)
        # Fill from interior outward: top/bottom rows, left/right columns
        for _ in range(max(values_2d.shape)):
            still_nan = np.isnan(values_2d)
            if not still_nan.any():
                break
            # Fill from row below / above
            rows_nan = np.where(still_nan.any(axis=1))[0]
            for r in rows_nan:
                if r + 1 < values_2d.shape[0]:
                    fill = values_2d[r + 1]
                    mask = np.isnan(values_2d[r])
                    values_2d[r, mask] = fill[mask]
                elif r - 1 >= 0:
                    fill = values_2d[r - 1]
                    mask = np.isnan(values_2d[r])
                    values_2d[r, mask] = fill[mask]
            # Fill from column right / left
            cols_nan = np.where(still_nan.any(axis=0))[0]
            for c in cols_nan:
                if c + 1 < values_2d.shape[1]:
                    fill = values_2d[:, c + 1]
                    mask = np.isnan(values_2d[:, c])
                    values_2d[mask, c] = fill[mask]
                elif c - 1 >= 0:
                    fill = values_2d[:, c - 1]
                    mask = np.isnan(values_2d[:, c])
                    values_2d[mask, c] = fill[mask]
        # Last resort: replace any remaining NaN with global mean
        if np.any(np.isnan(values_2d)):
            values_2d[np.isnan(values_2d)] = np.nanmean(values_2d)

    # s=0 means exact interpolation (no smoothing); kx=ky=3 is cubic
    spline = RectBivariateSpline(y_grid_1d, x_grid_1d,
                                 np.ascontiguousarray(values_2d, dtype=np.float64),
                                 kx=3, ky=3, s=0)

    # .ev(y, x) evaluates at scattered points (faster than __call__)
    def ev(lon_pts, lat_pts):
        return spline.ev(np.asarray(lat_pts), np.asarray(lon_pts))

    # .grid(x1d, y1d) evaluates on the tensor product of two 1-D axes.
    # Returns shape (len(lat_1d), len(lon_1d)). On a 1921x1920 ring this is
    # ~240x faster than the scattered path (bispev vs. bispeu) and gives
    # bit-identical values.
    def grid(lon_1d, lat_1d):
        return spline(np.asarray(lat_1d), np.asarray(lon_1d), grid=True)

    ev.grid = grid
    ev.spline = spline
    return ev


def _uniform_value(values_2d) -> Optional[float]:
    """Return the constant value of a uniform 2-D field, or None if it varies."""
    a = np.asarray(values_2d)
    if a.size == 0 or np.isnan(a).any():
        return None
    v0 = float(a.flat[0])
    return v0 if bool(np.all(a == v0)) else None


class GravityEngine:
    """
    Main engine that performs the zone-based gravity field integration.

    The engine is configured once with a TGFConfig and a set of DEM layers,
    then can compute gravity for arbitrary computation points.
    """

    def __init__(self, config: TGFConfig):
        """
        Args:
            config: Validated TGFConfig instance.
        """
        self.config = config
        self.dem_layers: Dict[str, DEMData] = {}
        self.ref_layers: Dict[str, np.ndarray] = {}
        self.density_layers: Dict[str, np.ndarray] = {}
        self.zone_strategy: Optional[ZoneStrategy] = None
        self.zone_extents: Optional[ZoneExtents] = None
        # Pre-computed geocentric radius arrays (added in prepare_geometry)
        self._re_layers: Dict[str, np.ndarray] = {}
        self._re_com: Optional[np.ndarray] = None
        # Guard against adding the Earth radius twice
        self._geometry_prepared = False
        self._cached_x1_grid: Optional[np.ndarray] = None
        self._cached_y1_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def load_dem_layers(self,
                        detailed: DEMData, tess: DEMData, coarse: DEMData,
                        detailed_ref: Optional[DEMData] = None,
                        tess_ref: Optional[DEMData] = None,
                        coarse_ref: Optional[DEMData] = None,
                        global_dem: Optional[DEMData] = None,
                        global_ref: Optional[DEMData] = None) -> None:
        """
        Load all DEM layers and reference surfaces.

        Corresponds to TC_GUI.m L874-L930.

        For ikind=1 (topographic), reference surfaces are zeroed (mean sea level).
        For ikind=2 (RTM), reference surfaces are read from provided files.

        Args:
            detailed, tess, coarse: DEMData for the three primary DEM grids.
            detailed_ref, tess_ref, coarse_ref: Reference surfaces (RTM only).
            global_dem, global_ref: Global DEM (topographic only).
        """
        self.dem_layers['detailed'] = detailed
        self.dem_layers['tess'] = tess
        self.dem_layers['coarse'] = coarse
        self._geometry_prepared = False
        self._cached_x1_grid = None
        self._cached_y1_grid = None

        if self.config.is_topographic:
            # ikind==1: reference surfaces are zero (mean sea level)
            self.ref_layers['detailed'] = np.zeros_like(detailed.z)
            self.ref_layers['tess'] = np.zeros_like(tess.z)
            self.ref_layers['coarse'] = np.zeros_like(coarse.z)
            if global_dem is not None:
                self.dem_layers['global'] = global_dem
                self.ref_layers['global'] = (np.zeros_like(global_dem.z)
                                             if global_ref is None else global_ref.z)
        else:
            # ikind==2: RTM reference surfaces from files
            self.ref_layers['detailed'] = (detailed_ref.z if detailed_ref is not None
                                           else np.zeros_like(detailed.z))
            self.ref_layers['tess'] = (tess_ref.z if tess_ref is not None
                                       else np.zeros_like(tess.z))
            self.ref_layers['coarse'] = (coarse_ref.z if coarse_ref is not None
                                         else np.zeros_like(coarse.z))

    def load_density_layers(self,
                            detailed_e: Optional[np.ndarray] = None,
                            tess_e: Optional[np.ndarray] = None,
                            coarse_e: Optional[np.ndarray] = None) -> None:
        """
        Load density maps for each zone, or fill with constant density.

        Corresponds to TC_GUI.m L932-L962.

        Args:
            detailed_e, tess_e, coarse_e: Density maps matching each DEM grid.
                If None and idensity=0, filled with config.e.
        """
        e_const = self.config.e
        if self.config.idensity == 1:
            self.density_layers['detailed'] = (detailed_e
                                               if detailed_e is not None
                                               else np.full_like(self.dem_layers['detailed'].z, e_const))
            self.density_layers['tess'] = (tess_e
                                           if tess_e is not None
                                           else np.full_like(self.dem_layers['tess'].z, e_const))
            self.density_layers['coarse'] = (coarse_e
                                             if coarse_e is not None
                                             else np.full_like(self.dem_layers['coarse'].z, e_const))
        else:
            # constant density mode
            self.density_layers['detailed'] = np.full_like(self.dem_layers['detailed'].z, e_const)
            self.density_layers['tess'] = np.full_like(self.dem_layers['tess'].z, e_const)
            self.density_layers['coarse'] = np.full_like(self.dem_layers['coarse'].z, e_const)

    def prepare_geometry(self) -> None:
        """
        Add Earth radius to all elevations to convert to geocentric radii.

        Corresponds to TC_GUI.m L985-L1015.

        For flag_earth=1, uses ellipsoidal radius:
            re = a / sqrt(1 + e2 * sin(lat)^2)
        For flag_earth=0, uses constant spherical radius a.

        Mutates the DEMData.z arrays in place (mirrors MATLAB z1=z1+re1).
        """
        if self._geometry_prepared:
            return

        a = PhysicalConstants.A
        e2 = PhysicalConstants.E2
        d2r = PhysicalConstants.d2r()
        flag_earth = self.config.flag_earth

        for key in ('detailed', 'tess', 'coarse'):
            dem = self.dem_layers[key]
            # Build latitude meshgrid matching DEM shape
            lats_1d = dem.lats
            lon_grid, lat_grid = np.meshgrid(dem.lons, lats_1d)
            if flag_earth == 1:
                re = a / np.sqrt(1.0 + e2 * (np.sin(lat_grid * d2r)) ** 2)
            else:
                re = np.full_like(lat_grid, a)
            self._re_layers[key] = re
            # Add Earth radius to elevations and reference surfaces
            dem.z = dem.z + re
            self.ref_layers[key] = self.ref_layers[key] + re

        if 'global' in self.dem_layers:
            dem = self.dem_layers['global']
            lats_1d = dem.lats
            lon_grid, lat_grid = np.meshgrid(dem.lons, lats_1d)
            if flag_earth == 1:
                re = a / np.sqrt(1.0 + e2 * (np.sin(lat_grid * d2r)) ** 2)
            else:
                re = np.full_like(lat_grid, a)
            self._re_layers['global'] = re
            dem.z = dem.z + re
            self.ref_layers['global'] = self.ref_layers['global'] + re

        self._geometry_prepared = True

    def setup_zone_strategy(self) -> ZoneExtents:
        """
        Build the ZoneStrategy and compute zone extents.

        Corresponds to TC_GUI.m L920-L942.
        """
        detailed = self.dem_layers['detailed']
        tess = self.dem_layers['tess']
        coarse = self.dem_layers['coarse']

        self.zone_strategy = ZoneStrategy(
            rzones=self.config.rzones,
            res_x1=detailed.reslon, res_y1=detailed.reslat,
            res_x3=tess.reslon, res_y3=tess.reslat,
            res_x4=coarse.reslon, res_y4=coarse.reslat,
        )
        self.zone_extents = self.zone_strategy.compute_extents()
        return self.zone_extents

    # ------------------------------------------------------------------
    # Per-point computation
    # ------------------------------------------------------------------
    def compute_point(self, lat: float, lon: float, h: float,
                      ref_station: Optional[float] = None,
                      polyhedron_result: Optional[Dict[str, float]] = None,
                      zone1_center: Optional[Tuple[float, float]] = None,
                      interp_cache: Optional[Dict[str, object]] = None
                      ) -> Dict[str, float]:
        """
        Compute all gravity field components for a single computation point.

        Corresponds to the inner loop body of TC_GUI.m RUN branch
        (e.g. L870-L1210 for itype=10).

        Args:
            lat, lon: Station coordinates [deg].
            h: Station geocentric radius [m] (already +Earth radius).
            ref_station: Reference surface height at station (RTM only).
            polyhedron_result: Pre-computed Zone 1 polyhedron result for this
                point (dict with V/Vx/.../Vzz). If None, polyhedron is skipped.
            zone1_center: (lon_center, lat_center) of Zone 1 grid used as the
                reference center for Build_grid in outer zones. If None, the
                station (lon, lat) is used.

        Returns:
            dict with keys 'V','Vx','Vy','Vz','Vxx','Vxy','Vxz','Vyy','Vyz','Vzz'.
        """
        if self.zone_extents is None:
            raise RuntimeError("setup_zone_strategy() must be called before compute_point().")

        # Initialize accumulators with polyhedron (Zone 1) result
        result = {k: 0.0 for k in
                  ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')}
        if polyhedron_result is not None:
            for k in result:
                result[k] += polyhedron_result[k]

        extents = self.zone_extents
        r1, r2, r3, r4 = self.config.rzones

        # Determine center for Build_grid (mean of Zone 1 cells)
        if zone1_center is not None:
            center_lon, center_lat = zone1_center
        else:
            center_lon, center_lat = lon, lat

        # --- Build Zone 2 candidate grid (extract from detailed DEM) ---
        detailed = self.dem_layers['detailed']
        # Use cached grids to avoid per-point meshgrid rebuild (400MB each)
        if self._cached_x1_grid is not None:
            x1_grid, y1_grid = self._cached_x1_grid, self._cached_y1_grid
        else:
            x1_grid, y1_grid = detailed.meshgrid()
        z1 = detailed.z
        z1_ref = self.ref_layers['detailed']
        e_d = self.density_layers['detailed']

        # Use integer index slicing instead of full-grid boolean masking
        # DEM is a regular grid: lats (rows) x lons (cols)
        lon_lo = lon - extents.extend_x2
        lon_hi = lon + extents.extend_x2
        lat_lo = lat - extents.extend_y2
        lat_hi = lat + extents.extend_y2
        n_lon = len(detailed.lons)
        n_lat = len(detailed.lats)
        col_lo = max(0, int(np.floor((lon_lo - detailed.lons[0]) / detailed.reslon)))
        col_hi = min(n_lon - 1, int(np.ceil((lon_hi - detailed.lons[0]) / detailed.reslon)))
        row_lo = max(0, int(np.floor((lat_lo - detailed.lats[0]) / detailed.reslat)))
        row_hi = min(n_lat - 1, int(np.ceil((lat_hi - detailed.lats[0]) / detailed.reslat)))
        # A station outside the DEM can yield negative upper indices; Python
        # would interpret them as offsets from the end, silently selecting the
        # wrong part of the grid.
        col_hi = max(-1, col_hi)
        row_hi = max(-1, row_hi)

        xm = x1_grid[row_lo:row_hi+1, col_lo:col_hi+1].ravel()
        ym = y1_grid[row_lo:row_hi+1, col_lo:col_hi+1].ravel()
        zm = z1[row_lo:row_hi+1, col_lo:col_hi+1].ravel()
        ref_m = z1_ref[row_lo:row_hi+1, col_lo:col_hi+1].ravel()
        e_m = e_d[row_lo:row_hi+1, col_lo:col_hi+1].ravel()

        # The bounding box may include one extra row/column beyond the square
        # zone limit; drop those cells so Zone 2 tiles exactly.
        keep = ((np.abs(xm - lon) <= extents.extend_x2) &
                (np.abs(ym - lat) <= extents.extend_y2))
        xm, ym, zm, ref_m, e_m = (a[keep] for a in (xm, ym, zm, ref_m, e_m))

        # --- Leakage boundary (Zone 1 outer ring at half weight) ---
        leakage = self._compute_leakage_zone(
            lat, lon, h, xm, ym, zm, ref_m, e_m,
            extents.extend_x1, extents.extend_y1,
            detailed.reslon, detailed.reslat)
        # Only add gravity field keys (skip metadata like 'remaining_x')
        field_keys = ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')
        for k in field_keys:
            result[k] += leakage[k]
        # Update xm/ym/... to remaining cells (Zone 1 cells removed)
        xm = leakage['remaining_x']
        ym = leakage['remaining_y']
        zm = leakage['remaining_z']
        ref_m = leakage['remaining_ref']
        e_m = leakage['remaining_e']
        # lon1/lat1 for Build_grid center (TC_GUI.m L1470/L1520/L1560):
        #   - with an active Zone 1 (extend_x1>0): MATLAB reduces lon1/lat1 to
        #     the outermost leakage ring before averaging -> boundary centroid
        #   - without Zone 1 (extend_x1==0, the usual rzones=[0 r2 r3 r4]):
        #     MATLAB keeps lon1 = every DetailedDEM cell within ±extend_x2 and
        #     averages that set.
        # Using the station itself instead of the cell-centre mean shifts the
        # Zone-3/4 grids by up to half a cell (~46 m) and changes dg by up to
        # ~0.03 mGal, so mirror MATLAB here.
        if leakage['boundary_lon'].size > 0:
            center_lon = float(np.mean(leakage['boundary_lon']))
            center_lat = float(np.mean(leakage['boundary_lat']))
        else:
            # MATLAB takes mean(lon1) of an empty vector when extend_x2 == 0
            # and no cell centre falls exactly on the station, which yields
            # NaN there as well. Fall back to the station instead of
            # propagating NaN into the spline evaluation.
            if xm.size > 0:
                center_lon = float(np.mean(xm))
                center_lat = float(np.mean(ym))
            else:
                center_lon = float(lon)
                center_lat = float(lat)

        # --- Zone 2: Prism ---
        zone2 = self._compute_zone2_prism(lat, lon, h, xm, ym, zm, ref_m, e_m,
                                          extents, r1, r2)
        for k, v in zone2.items():
            result[k] += v

        # --- Zone 3: Tesseroid ---
        zone3 = self._compute_zone3_tesseroid(
            lat, lon, h, center_lon, center_lat, extents, r2, r3,
            interp_cache=interp_cache)
        for k, v in zone3.items():
            result[k] += v

        # --- Zone 4: Point mass ---
        zone4 = self._compute_zone4_pointmass(
            lat, lon, h, center_lon, center_lat, extents,
            interp_cache=interp_cache)
        for k, v in zone4.items():
            result[k] += v

        # --- Zone 5: Global point mass (topographic only) ---
        if self.config.is_topographic and 'global' in self.dem_layers:
            zone5 = self._compute_zone5_global(
                lat, lon, h, center_lon, center_lat, extents, interp_cache)
            for k, v in zone5.items():
                result[k] += v

        return result

    def _compute_leakage_zone(self, lat: float, lon: float, h: float,
                              xm: np.ndarray, ym: np.ndarray,
                              zm: np.ndarray, ref_m: np.ndarray,
                              e_m: np.ndarray,
                              extend_x1: float, extend_y1: float,
                              resolution_x: float, resolution_y: float
                              ) -> Dict[str, float]:
        """
        Leakage boundary: prism at half weight.

        Corresponds to TC_GUI.m L884-L905 (inside itype==10 branch).

        Returns dict with V/Vx/.../Vzz contributions (already halved),
        plus 'remaining_x', etc. for downstream Zone 2 use, and
        'boundary_lon', 'boundary_lat' for Build_grid center.
        """
        result = {k: 0.0 for k in
                  ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')}

        leakage = self.zone_strategy.extract_leakage_boundary(
            xm, ym, zm, ref_m, e_m, lon, lat,
            extend_x1, extend_y1, resolution_x, resolution_y)

        bdy_lon = leakage['boundary_lon']
        bdy_lat = leakage['boundary_lat']
        bdy_h = leakage['boundary_h']
        bdy_ref = leakage['boundary_ref']
        bdy_e = leakage['boundary_e']

        if bdy_lon.size > 0:
            xx1 = bdy_lon - resolution_x / 2.0
            xx2 = bdy_lon + resolution_x / 2.0
            yy1 = bdy_lat - resolution_y / 2.0
            yy2 = bdy_lat + resolution_y / 2.0
            zz1 = bdy_h
            zz2 = bdy_ref

            _, prism_fn, _, _ = _MASS_DISPATCH[self.config.itype]
            prism_out = prism_fn(xx1, xx2, yy1, yy2, zz1, zz2,
                                 lon, lat, h, bdy_e,
                                 resolution_x, resolution_y)
            # prism_out is a tuple; map to result keys based on itype
            keys = self._prism_output_keys()
            for k, v in zip(keys, prism_out):
                result[k] += float(v) / 2.0  # half weight

        # Pass through remaining arrays for Zone 2
        result['remaining_x'] = leakage['remaining_x']
        result['remaining_y'] = leakage['remaining_y']
        result['remaining_z'] = leakage['remaining_z']
        result['remaining_ref'] = leakage['remaining_ref']
        result['remaining_e'] = leakage['remaining_e']
        result['boundary_lon'] = bdy_lon
        result['boundary_lat'] = bdy_lat
        return result

    def _compute_zone2_prism(self, lat: float, lon: float, h: float,
                             xm: np.ndarray, ym: np.ndarray,
                             zm: np.ndarray, ref_m: np.ndarray,
                             e_m: np.ndarray,
                             extents: ZoneExtents,
                             r1: float, r2: float) -> Dict[str, float]:
        """Zone 2: prism contribution. Mirrors L906-L922."""
        result = {k: 0.0 for k in
                  ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')}
        if not (extents.extend_x2 > extents.extend_x1 and
                extents.extend_y2 > extents.extend_y1) or xm.size == 0:
            return result

        detailed = self.dem_layers['detailed']
        res_x2, res_y2 = detailed.reslon, detailed.reslat

        xx1 = xm - res_x2 / 2.0
        xx2 = xm + res_x2 / 2.0
        yy1 = ym - res_y2 / 2.0
        yy2 = ym + res_y2 / 2.0
        zz1 = zm
        zz2 = ref_m

        _, prism_fn, _, _ = _MASS_DISPATCH[self.config.itype]
        prism_out = prism_fn(xx1, xx2, yy1, yy2, zz1, zz2,
                             lon, lat, h, e_m, res_x2, res_y2)
        keys = self._prism_output_keys()
        for k, v in zip(keys, prism_out):
            result[k] += float(v)
        return result

    def _compute_zone3_tesseroid(self, lat: float, lon: float, h: float,
                                 center_lon: float, center_lat: float,
                                 extents: ZoneExtents,
                                 r2: float, r3: float,
                                 interp_cache: Optional[Dict[str, object]] = None
                                 ) -> Dict[str, float]:
        """Zone 3: tesseroid contribution. Mirrors L923-L957."""
        result = {k: 0.0 for k in
                  ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')}
        if not (extents.extend_x3 > extents.extend_x2 and
                extents.extend_y3 > extents.extend_y2):
            return result

        tess = self.dem_layers['tess']
        zt = tess.z
        zt_ref = self.ref_layers['tess']
        e_t = self.density_layers['tess']
        res_x3, res_y3 = tess.reslon, tess.reslat

        layout = (interp_cache or {}).get('zone3_layout')
        cached = (interp_cache is not None and 'tess_h' in interp_cache)

        if layout is not None and cached:
            # Fast path: the ring is a regular tensor grid, so sample the
            # splines with bispev (~0.05 s) instead of 3.7e6 scattered lookups
            # (~12 s per field). Values are bit-identical to the scattered path.
            mask_f = layout['mask_f']
            if mask_f.size == 0 or not mask_f.any():
                return result
            lon2 = layout['lon_f'] + center_lon
            lat2 = layout['lat_f'] + center_lat
            lonv = layout['lon_1d'] + center_lon
            latv = layout['lat_1d'] + center_lat
            h2 = interp_cache['tess_h'].grid(lonv, latv).flatten(order='F')[mask_f]
            ref_2 = interp_cache['tess_ref'].grid(lonv, latv).flatten(order='F')[mask_f]
            e_const = interp_cache.get('tess_e_const')
            if e_const is not None:
                e_2 = np.full(h2.shape, e_const, dtype=np.float64)
            else:
                e_2 = interp_cache['tess_e'].grid(lonv, latv).flatten(order='F')[mask_f]
        else:
            # Build ring-zone grid (use precomputed offsets if available)
            if interp_cache is not None and 'zone3_lon_off' in interp_cache:
                lon2 = interp_cache['zone3_lon_off'] + center_lon
                lat2 = interp_cache['zone3_lat_off'] + center_lat
            else:
                lon2, lat2 = build_grid(center_lon, center_lat,
                                        extents.extend_x2, extents.extend_y2,
                                        extents.extend_x3, extents.extend_y3,
                                        res_x3, res_y3)
            if lon2.size == 0:
                return result

            # Use cached interpolators if available (major perf win); otherwise build
            if cached:
                h2 = interp_cache['tess_h'](lon2, lat2)
                ref_2 = interp_cache['tess_ref'](lon2, lat2)
                if 'tess_e_const' in interp_cache:
                    e_2 = np.full(lon2.shape, interp_cache['tess_e_const'],
                                  dtype=np.float64)
                else:
                    e_2 = interp_cache['tess_e'](lon2, lat2)
            else:
                F_h = _cubic_interpolator(tess.lons, tess.lats, zt)
                F_ref = _cubic_interpolator(tess.lons, tess.lats, zt_ref)
                F_e = _cubic_interpolator(tess.lons, tess.lats, e_t)
                h2 = F_h(lon2, lat2)
                ref_2 = F_ref(lon2, lat2)
                e_2 = F_e(lon2, lat2)

        _, _, tess_fn, _ = _MASS_DISPATCH[self.config.itype]
        tess_out = tess_fn(lon2, lat2, h2, ref_2, lon, lat, h,
                           res_x3, res_y3, e_2)
        keys = self._tess_output_keys()
        for k, v in zip(keys, tess_out):
            result[k] += float(v)
        return result

    def _compute_zone4_pointmass(self, lat: float, lon: float, h: float,
                                 center_lon: float, center_lat: float,
                                 extents: ZoneExtents,
                                 interp_cache: Optional[Dict[str, object]] = None
                                 ) -> Dict[str, float]:
        """Zone 4: point mass contribution. Mirrors L958-L993."""
        result = {k: 0.0 for k in
                  ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')}
        if not (extents.extend_x4 > extents.extend_x3 and
                extents.extend_y4 > extents.extend_y3):
            return result

        coarse = self.dem_layers['coarse']
        z2 = coarse.z
        z2_ref = self.ref_layers['coarse']
        e_c = self.density_layers['coarse']
        res_x4, res_y4 = coarse.reslon, coarse.reslat

        layout = (interp_cache or {}).get('zone4_layout')
        cached = (interp_cache is not None and 'coarse_h' in interp_cache)

        if layout is not None and cached:
            # Fast path: same tensor-grid trick as Zone 3.
            mask_f = layout['mask_f']
            if mask_f.size == 0 or not mask_f.any():
                return result
            lon3 = layout['lon_f'] + center_lon
            lat3 = layout['lat_f'] + center_lat
            lonv = layout['lon_1d'] + center_lon
            latv = layout['lat_1d'] + center_lat
            h3 = interp_cache['coarse_h'].grid(lonv, latv).flatten(order='F')[mask_f]
            ref3 = interp_cache['coarse_ref'].grid(lonv, latv).flatten(order='F')[mask_f]
            e_const = interp_cache.get('coarse_e_const')
            if e_const is not None:
                e3 = np.full(h3.shape, e_const, dtype=np.float64)
            else:
                e3 = interp_cache['coarse_e'].grid(lonv, latv).flatten(order='F')[mask_f]
        else:
            # Use precomputed offsets if available
            if interp_cache is not None and 'zone4_lon_off' in interp_cache:
                lon3 = interp_cache['zone4_lon_off'] + center_lon
                lat3 = interp_cache['zone4_lat_off'] + center_lat
            else:
                lon3, lat3 = build_grid(center_lon, center_lat,
                                        extents.extend_x3, extents.extend_y3,
                                        extents.extend_x4, extents.extend_y4,
                                        res_x4, res_y4)
            if lon3.size == 0:
                return result

            if cached:
                h3 = interp_cache['coarse_h'](lon3, lat3)
                ref3 = interp_cache['coarse_ref'](lon3, lat3)
                if 'coarse_e_const' in interp_cache:
                    e3 = np.full(lon3.shape, interp_cache['coarse_e_const'],
                                 dtype=np.float64)
                else:
                    e3 = interp_cache['coarse_e'](lon3, lat3)
            else:
                F_h = _cubic_interpolator(coarse.lons, coarse.lats, z2)
                F_ref = _cubic_interpolator(coarse.lons, coarse.lats, z2_ref)
                F_e = _cubic_interpolator(coarse.lons, coarse.lats, e_c)
                h3 = F_h(lon3, lat3)
                ref3 = F_ref(lon3, lat3)
                e3 = F_e(lon3, lat3)

        xx1 = lon3 - res_x4 / 2.0
        xx2 = lon3 + res_x4 / 2.0
        yy1 = lat3 - res_y4 / 2.0
        yy2 = lat3 + res_y4 / 2.0
        zz1 = h3
        zz2 = ref3

        _, _, _, pm_fn = _MASS_DISPATCH[self.config.itype]
        # Point mass functions take (xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)
        # where r1=zz1 (upper), r2=zz2 (lower)
        pm_out = pm_fn(xx1, xx2, yy1, yy2, zz1, zz2, lon, lat, h, e3)
        keys = self._pm_output_keys()
        for k, v in zip(keys, pm_out):
            result[k] += float(v)
        return result

    def _compute_zone5_global(self, lat: float, lon: float, h: float,
                              center_lon: float, center_lat: float,
                              extents: ZoneExtents,
                              interp_cache: Dict) -> Dict[str, float]:
        """Zone 5: global point mass (topographic only). Mirrors L994-L1024."""
        result = {k: 0.0 for k in
                  ('V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz')}
        if 'global' not in self.dem_layers:
            return result

        gdem = self.dem_layers['global']
        res_g = gdem.reslon  # MATLAB uses resolution_g for both x and y

        # Build global ring grid (use precomputed offsets if available)
        if interp_cache is not None and 'zone5_lon_off' in interp_cache:
            lon4 = interp_cache['zone5_lon_off'] + center_lon
            lat4 = interp_cache['zone5_lat_off'] + center_lat
        else:
            lon4, lat4 = build_grid(center_lon, center_lat,
                                    extents.extend_x4, extents.extend_y4,
                                    360.0, 180.0, res_g, res_g)
        if lon4.size == 0:
            return result

        # Restrict to global DEM coverage
        minxg, maxxg = gdem.minlon, gdem.maxlon
        minyg, maxyg = gdem.minlat, gdem.maxlat
        mask = ((lon4 < maxxg) & (lon4 > minxg) &
                (lat4 < maxyg) & (lat4 > minyg))
        lon4 = lon4[mask]
        lat4 = lat4[mask]
        if lon4.size == 0:
            return result

        # Use cached interpolators (built once in run(), not per point)
        if interp_cache is not None and 'global_h' in interp_cache:
            F_h = interp_cache['global_h']
            F_ref = interp_cache['global_ref']
        else:
            F_h = _cubic_interpolator(gdem.lons, gdem.lats, gdem.z)
            F_ref = _cubic_interpolator(gdem.lons, gdem.lats,
                                        self.ref_layers['global'])
        h4 = F_h(lon4, lat4)
        ref4 = F_ref(lon4, lat4)

        e4 = np.full_like(h4, self.config.e)
        xx1 = lon4 - res_g / 2.0
        xx2 = lon4 + res_g / 2.0
        yy1 = lat4 - res_g / 2.0
        yy2 = lat4 + res_g / 2.0
        zz1 = h4
        zz2 = ref4

        _, _, _, pm_fn = _MASS_DISPATCH[self.config.itype]
        pm_out = pm_fn(xx1, xx2, yy1, yy2, zz1, zz2, lon, lat, h, e4)
        keys = self._pm_output_keys()
        for k, v in zip(keys, pm_out):
            result[k] += float(v)
        return result

    # ------------------------------------------------------------------
    # Output key mapping per itype
    # ------------------------------------------------------------------
    def _prism_output_keys(self) -> List[str]:
        """Return list of result keys produced by the active prism variant."""
        itype = self.config.itype
        if itype == 0:
            return ['V']
        if itype == 1:
            return ['Vx', 'Vy', 'Vz']
        if itype == 2:
            return ['Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz']
        if itype in (4, 103, 104):
            return ['V', 'Vx', 'Vy', 'Vz']
        if itype == 10:
            return ['V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz']
        raise ValueError(f"Unknown itype {itype}")

    def _tess_output_keys(self) -> List[str]:
        """Return list of result keys produced by the active tesseroid variant."""
        return self._prism_output_keys()

    def _pm_output_keys(self) -> List[str]:
        """Return list of result keys produced by the active point mass variant."""
        return self._prism_output_keys()

    def _polyhedron_output_keys(self) -> List[str]:
        """Return list of result keys produced by the active polyhedron variant."""
        return ['V', 'Vx', 'Vy', 'Vz', 'Vxx', 'Vxy', 'Vxz', 'Vyy', 'Vyz', 'Vzz']

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def apply_harmonic_correction(self, Vz: np.ndarray,
                                  ref_station: np.ndarray,
                                  com_h: np.ndarray) -> np.ndarray:
        """
        Apply RTM harmonic correction to vertical gravity component.

        Corresponds to TC_GUI.m L1224-L1228:
            harmonic = 4*pi*G*2.67*(ref_station - com_h)  where ref > com_h
            Vz = Vz + harmonic

        Args:
            Vz: Vertical gravity component [mGal].
            ref_station: Reference surface height at each station [m].
            com_h: Station geocentric radius [m].

        Returns:
            Corrected Vz [mGal].
        """
        Pi = PhysicalConstants.PI
        G = PhysicalConstants.G
        harmonic = np.zeros_like(Vz, dtype=np.float64)
        mask = (ref_station - com_h) > 0
        harmonic[mask] = 4.0 * Pi * G * self.config.e * (ref_station[mask] - com_h[mask])
        return Vz + harmonic

    def convert_to_physical_quantities(self, V, Vx, Vy, Vz,
                                       com_lat, com_h) -> dict:
        """
        Convert potential derivatives to physical quantities.

        Corresponds to TC_GUI.m L1230-L1240:
            gama = normal_gravity_generic(com_lat, H_com)
            Dov_const = 3600*180/pi / gama * 1e-5
            xi   = -Vx * Dov_const        [arc-sec]
            eta  = -Vy * Dov_const        [arc-sec]
            dg   = -Vz                    [mGal]
            N    = V / gama               [m]
            (itype=4/104) dg = dg - 0.3086 * N   [mGal]  gravity anomaly

        Args:
            V, Vx, Vy, Vz: Potential and 1st derivatives (arrays).
            com_lat: Station latitudes [deg].
            com_h: Station original heights (above ellipsoid) [m].

        Returns:
            dict with keys 'N','xi','eta','dg' (and 'dg_anomaly' if applicable).
        """
        Pi = PhysicalConstants.PI
        gama = normal_gravity_generic(np.asarray(com_lat), np.asarray(com_h))
        Dov_const = 3600.0 * 180.0 / Pi / gama * 1e-5

        xi = -np.asarray(Vx) * Dov_const
        eta = -np.asarray(Vy) * Dov_const
        dg = -np.asarray(Vz)
        N = np.asarray(V) / gama

        out = {'N': N, 'xi': xi, 'eta': eta, 'dg': dg}
        if self.config.itype in (4, 104):
            # gravity anomaly: dg = dg - 0.3086 * N
            out['dg'] = dg - 0.3086 * N
        return out

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------
    def run_from_files(self, progress_callback=None) -> Dict[str, object]:
        """
        Load all inputs from configured file paths, run calculation, write outputs.

        Corresponds to TC_GUI.m RUN branch top-level flow (L776-L2035):
            1. Read computation points (L786-L800)
            2. Read DEM and density files (L874-L962)
            3. Prepare geometry (add Earth radius) (L985-L1015)
            4. Setup zone strategy (L920-L942)
            5. Loop over points calling compute_point (L870-L1210)
            6. Apply RTM harmonic correction (L1224-L1228)
            7. Convert to physical quantities (L1230-L1240)
            8. Write report and result files (L1255-L1290)

        Args:
            progress_callback: Optional callable(current, total, phase) for
                reporting progress to the web UI.  When None (default) the
                method behaves exactly as before.

        Returns:
            dict with keys 'result', 'report_path', 'output_path',
            'config', 'elapsed_hours'.
        """
        cfg = self.config

        # 1. Read computation points
        com_data = read_com_points(cfg.com_points)

        # 2. Read DEM files
        detailed = read_dem(cfg.detailed_dem)
        tess = read_dem(cfg.tess_dem)
        coarse = read_dem(cfg.coarse_dem)

        detailed_ref = read_dem(cfg.detailed_ref) if cfg.detailed_ref else None
        tess_ref = read_dem(cfg.tess_ref) if cfg.tess_ref else None
        coarse_ref = read_dem(cfg.coarse_ref) if cfg.coarse_ref else None

        global_dem = read_dem(cfg.global_dem) if cfg.global_dem else None
        global_ref = read_dem(cfg.global_ref) if cfg.global_ref else None

        self.load_dem_layers(detailed, tess, coarse,
                             detailed_ref, tess_ref, coarse_ref,
                             global_dem, global_ref)

        # Read density maps if idensity=1
        detailed_e = tess_e = coarse_e = None
        if cfg.idensity == 1:
            if cfg.mass_density:
                detailed_e = read_dem(cfg.mass_density).z
            if cfg.tess_density:
                tess_e = read_dem(cfg.tess_density).z
            if cfg.coarse_density:
                coarse_e = read_dem(cfg.coarse_density).z
        self.load_density_layers(detailed_e, tess_e, coarse_e)

        # 3. Prepare geometry (add Earth radius)
        self.prepare_geometry()

        # 4. Setup zone strategy
        self.setup_zone_strategy()

        # Write report header + DEM info
        with open(cfg.report_path, 'w') as fid:
            write_report_header(fid, cfg, com_data)
            write_report_dem_info(fid, cfg, self.dem_layers)

        # 5. Run calculation
        t_start = time.time()
        n_points = com_data.n_points
        if progress_callback:
            progress_callback(0, n_points, "loading")
        result = self.run(com_data, progress_callback=progress_callback)
        elapsed_hours = (time.time() - t_start) / 3600.0

        # 8. Write integration info + result file
        with open(cfg.report_path, 'a') as fid:
            write_report_integration(fid, elapsed_hours, cfg.outname,
                                     cfg.output_columns)
        write_result(cfg.outname, result)

        return {
            'result': result,
            'report_path': cfg.report_path,
            'output_path': cfg.outname,
            'config': cfg,
            'elapsed_hours': elapsed_hours,
        }

    def run(self, com_points: ComPointData, progress_callback=None) -> np.ndarray:
        """
        Execute the full gravity field calculation for all computation points.

        Assumes DEM layers and density layers have already been loaded via
        load_dem_layers() and load_density_layers(), and geometry prepared
        via prepare_geometry().

        Args:
            com_points: ComPointData with all computation points.
            progress_callback: Optional callable(current, total, phase)
                invoked after each point.  None = no reporting (default).

        Returns:
            Result tensor of shape (n_points, n_columns) where n_columns
            depends on itype (see TGFConfig.output_columns).
        """
        cfg = self.config
        n_points = com_points.n_points
        com_lat = com_points.lats
        com_lon = com_points.lons
        com_h = com_points.heights.copy()
        H_com = com_points.heights.copy()  # original heights (above ellipsoid)

        # Add Earth radius to computation heights
        a = PhysicalConstants.A
        e2 = PhysicalConstants.E2
        d2r = PhysicalConstants.d2r()
        if cfg.flag_earth == 1:
            re_com = a / np.sqrt(1.0 + e2 * (np.sin(com_lat * d2r)) ** 2)
        else:
            re_com = np.full_like(com_lat, a)
        com_h = com_h + re_com

        # RTM: interpolate reference surface at computation points
        ref_station = None
        if cfg.is_rtm:
            detailed = self.dem_layers['detailed']
            z1_ref = self.ref_layers['detailed']
            F_ref = _cubic_interpolator(detailed.lons, detailed.lats, z1_ref)
            ref_station = F_ref(com_lon, com_lat)

        # --- Pre-build cubic interpolators ONCE (biggest perf win) ---
        # Zone 3 (TessDEM) and Zone 4 (CoarseDEM) interpolators are reused
        # across all computation points; building them inside the per-point
        # loop was the dominant cost (~15s per point on large grids).
        interp_cache = {}
        if 'tess' in self.dem_layers:
            tess = self.dem_layers['tess']
            interp_cache['tess_h'] = _cubic_interpolator(
                tess.lons, tess.lats, tess.z)
            interp_cache['tess_ref'] = _cubic_interpolator(
                tess.lons, tess.lats, self.ref_layers['tess'])
            # Constant density (idensity=0) needs no spline at all - the cubic
            # spline of a constant field is that constant, exactly.
            e_tess = self.density_layers['tess']
            interp_cache['tess_e_const'] = _uniform_value(e_tess)
            if interp_cache['tess_e_const'] is None:
                interp_cache['tess_e'] = _cubic_interpolator(
                    tess.lons, tess.lats, e_tess)
        if 'coarse' in self.dem_layers:
            coarse = self.dem_layers['coarse']
            interp_cache['coarse_h'] = _cubic_interpolator(
                coarse.lons, coarse.lats, coarse.z)
            interp_cache['coarse_ref'] = _cubic_interpolator(
                coarse.lons, coarse.lats, self.ref_layers['coarse'])
            e_coarse = self.density_layers['coarse']
            interp_cache['coarse_e_const'] = _uniform_value(e_coarse)
            if interp_cache['coarse_e_const'] is None:
                interp_cache['coarse_e'] = _cubic_interpolator(
                    coarse.lons, coarse.lats, e_coarse)
        if 'global' in self.dem_layers:
            gdem = self.dem_layers['global']
            interp_cache['global_h'] = _cubic_interpolator(
                gdem.lons, gdem.lats, gdem.z)
            interp_cache['global_ref'] = _cubic_interpolator(
                gdem.lons, gdem.lats, self.ref_layers['global'])

        # --- Pre-compute relative grid offsets for Zones 3/4/5 ---
        # build_grid(lon, lat, ...) = build_grid(0, 0, ...) + (lon, lat)
        # So we compute the relative grid ONCE and shift per-point.
        ext = self.zone_extents
        if 'tess' in self.dem_layers:
            tess = self.dem_layers['tess']
            lon3_off, lat3_off = build_grid(
                0.0, 0.0, ext.extend_x2, ext.extend_y2,
                ext.extend_x3, ext.extend_y3,
                tess.reslon, tess.reslat)
            interp_cache['zone3_lon_off'] = lon3_off
            interp_cache['zone3_lat_off'] = lat3_off
            # Tensor layout of the same ring: lets Zone 3 sample the three
            # TessDEM fields with bispev instead of 3.7e6 scattered lookups.
            lon3v, lat3v, mask3 = build_grid_layout(
                0.0, 0.0, ext.extend_x2, ext.extend_y2,
                ext.extend_x3, ext.extend_y3,
                tess.reslon, tess.reslat)
            mf3 = mask3.flatten(order='F')
            lon3g, lat3g = np.meshgrid(lon3v, lat3v)
            interp_cache['zone3_layout'] = {
                'lon_1d': lon3v, 'lat_1d': lat3v, 'mask_f': mf3,
                'lon_f': lon3g.flatten(order='F')[mf3],
                'lat_f': lat3g.flatten(order='F')[mf3],
            }
        if 'coarse' in self.dem_layers:
            coarse = self.dem_layers['coarse']
            lon4_off, lat4_off = build_grid(
                0.0, 0.0, ext.extend_x3, ext.extend_y3,
                ext.extend_x4, ext.extend_y4,
                coarse.reslon, coarse.reslat)
            interp_cache['zone4_lon_off'] = lon4_off
            interp_cache['zone4_lat_off'] = lat4_off
            lon4v, lat4v, mask4 = build_grid_layout(
                0.0, 0.0, ext.extend_x3, ext.extend_y3,
                ext.extend_x4, ext.extend_y4,
                coarse.reslon, coarse.reslat)
            mf4 = mask4.flatten(order='F')
            lon4g, lat4g = np.meshgrid(lon4v, lat4v)
            interp_cache['zone4_layout'] = {
                'lon_1d': lon4v, 'lat_1d': lat4v, 'mask_f': mf4,
                'lon_f': lon4g.flatten(order='F')[mf4],
                'lat_f': lat4g.flatten(order='F')[mf4],
            }
        if 'global' in self.dem_layers:
            global_dem = self.dem_layers['global']
            lon5_off, lat5_off = build_grid(
                0.0, 0.0, ext.extend_x4, ext.extend_y4,
                360.0, 180.0,
                global_dem.reslon, global_dem.reslat)
            interp_cache['zone5_lon_off'] = lon5_off
            interp_cache['zone5_lat_off'] = lat5_off

        # --- Pre-compute Zone 1 polyhedron for all points (batch) ---
        poly_fn, _, _, _ = _MASS_DISPATCH[cfg.itype]
        detailed = self.dem_layers['detailed']
        x1_grid, y1_grid = detailed.meshgrid()
        # Cache grids for compute_point to avoid per-point meshgrid rebuild
        self._cached_x1_grid = x1_grid
        self._cached_y1_grid = y1_grid
        z1 = detailed.z
        z1_ref = self.ref_layers['detailed']
        e_d = self.density_layers['detailed']

        poly_out = poly_fn(x1_grid, y1_grid, z1, z1_ref,
                           com_lat, com_lon, com_h,
                           self.zone_extents.extend_x1,
                           self.zone_extents.extend_y1,
                           e_d, detailed.reslon, detailed.reslat)
        # poly_out is a tuple of arrays of length n_points
        poly_keys = self._polyhedron_output_keys()
        poly_results = {k: np.asarray(v) for k, v in zip(poly_keys, poly_out)}

        # --- Loop over points for Zones 2-5 ---
        V = np.zeros(n_points)
        Vx = np.zeros(n_points)
        Vy = np.zeros(n_points)
        Vz = np.zeros(n_points)
        Vxx = np.zeros(n_points)
        Vxy = np.zeros(n_points)
        Vxz = np.zeros(n_points)
        Vyy = np.zeros(n_points)
        Vyz = np.zeros(n_points)
        Vzz = np.zeros(n_points)

        for i in range(n_points):
            poly_pt = {k: float(poly_results[k][i]) for k in poly_keys}
            pt_result = self.compute_point(
                float(com_lat[i]), float(com_lon[i]), float(com_h[i]),
                ref_station=(float(ref_station[i]) if ref_station is not None else None),
                polyhedron_result=poly_pt,
                interp_cache=interp_cache)
            V[i] = pt_result['V']
            Vx[i] = pt_result['Vx']
            Vy[i] = pt_result['Vy']
            Vz[i] = pt_result['Vz']
            Vxx[i] = pt_result['Vxx']
            Vxy[i] = pt_result['Vxy']
            Vxz[i] = pt_result['Vxz']
            Vyy[i] = pt_result['Vyy']
            Vyz[i] = pt_result['Vyz']
            Vzz[i] = pt_result['Vzz']

            if progress_callback:
                progress_callback(i + 1, n_points, "computing")

        # --- Apply RTM harmonic correction ---
        if cfg.is_rtm and ref_station is not None:
            Vz = self.apply_harmonic_correction(Vz, ref_station, com_h)

        # --- Convert to physical quantities ---
        # Only compute quantities requested by itype
        columns = cfg.output_columns
        n_cols = len(columns)
        tensor = np.zeros((n_points, n_cols))
        tensor[:, 0] = com_lon
        tensor[:, 1] = com_lat
        tensor[:, 2] = H_com

        if cfg.itype == 0:
            # N only
            gama = normal_gravity_generic(com_lat, H_com)
            N = V / gama
            tensor[:, 3] = N
        elif cfg.itype == 1:
            # xi, eta, dg
            gama = normal_gravity_generic(com_lat, H_com)
            Dov_const = 3600.0 * 180.0 / PhysicalConstants.PI / gama * 1e-5
            tensor[:, 3] = -Vx * Dov_const
            tensor[:, 4] = -Vy * Dov_const
            tensor[:, 5] = -Vz
        elif cfg.itype == 2:
            # 6 tensors
            tensor[:, 3] = Vxx
            tensor[:, 4] = Vxy
            tensor[:, 5] = Vxz
            tensor[:, 6] = Vyy
            tensor[:, 7] = Vyz
            tensor[:, 8] = Vzz
        elif cfg.itype in (4, 103, 104):
            # N, xi, eta, dg
            gama = normal_gravity_generic(com_lat, H_com)
            Dov_const = 3600.0 * 180.0 / PhysicalConstants.PI / gama * 1e-5
            N = V / gama
            xi = -Vx * Dov_const
            eta = -Vy * Dov_const
            dg = -Vz
            if cfg.itype in (4, 104):
                dg = dg - 0.3086 * N  # gravity anomaly
            tensor[:, 3] = N
            tensor[:, 4] = xi
            tensor[:, 5] = eta
            tensor[:, 6] = dg
        elif cfg.itype == 10:
            # N, xi, eta, dg, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz
            gama = normal_gravity_generic(com_lat, H_com)
            Dov_const = 3600.0 * 180.0 / PhysicalConstants.PI / gama * 1e-5
            N = V / gama
            xi = -Vx * Dov_const
            eta = -Vy * Dov_const
            dg = -Vz
            tensor[:, 3] = N
            tensor[:, 4] = xi
            tensor[:, 5] = eta
            tensor[:, 6] = dg
            tensor[:, 7] = Vxx
            tensor[:, 8] = Vxy
            tensor[:, 9] = Vxz
            tensor[:, 10] = Vyy
            tensor[:, 11] = Vyz
            tensor[:, 12] = Vzz

        return tensor
