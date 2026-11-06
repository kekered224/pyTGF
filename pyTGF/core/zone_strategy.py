"""
Zone division strategy

Mirrors TC_GUI.m L920-L942 (extend_x/y computation) and the per-point
zone assembly logic in the RUN branch (L870-L1210).

Responsibilities:
    - Compute integration half-extents for each zone aligned to grid resolution
    - Build ring-zone grids avoiding overlap/leakage between adjacent zones
    - Handle the leakage zone between polyhedron (Zone 1) and prism (Zone 2)
      via boundary-cell half-weighting (L870-L893)
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ZoneExtents:
    """Half-extents [deg] for the four integration zones around a station"""

    extend_x1: float   # Zone 1 (polyhedron) inner half-extent
    extend_y1: float
    extend_x2: float   # Zone 2 (prism) outer half-extent
    extend_y2: float
    extend_x3: float   # Zone 3 (tesseroid) outer half-extent
    extend_y3: float
    extend_x4: float   # Zone 4 (point mass) outer half-extent
    extend_y4: float

    @property
    def r1_active(self) -> bool:
        return self.extend_x1 > 0 and self.extend_y1 > 0

    @property
    def r2_active(self) -> bool:
        return self.extend_x2 > self.extend_x1

    @property
    def r3_active(self) -> bool:
        return self.extend_x3 > self.extend_x2

    @property
    def r4_active(self) -> bool:
        return self.extend_x4 > self.extend_x3


class ZoneStrategy:
    """
    Computes zone extents for the four-zone integration scheme.

    Mirrors TC_GUI.m L920-L942:
        extend_x1 = floor(r1/res_x1) * res_x1
        extend_x2 = floor(r2/res_x2) * res_x2
        extend_x3 = n1 * floor(r3/res_x3/n1) * res_x3   (n1 = res_x4/res_x3)
        extend_x4 = 4 * floor(r4/res_x4/4) * res_x4

    The integer-multiple alignment ensures that coarse-zone cells tile
    exactly onto fine-zone cells, avoiding gaps or overlaps.
    """

    def __init__(self, rzones: List[float],
                 res_x1: float, res_y1: float,
                 res_x3: float, res_y3: float,
                 res_x4: float, res_y4: float):
        """
        Args:
            rzones: [r1, r2, r3, r4] integration radii [deg].
            res_x1, res_y1: Detailed DEM resolution [deg] (also used for Zone 2).
            res_x3, res_y3: Tesseroid DEM resolution [deg].
            res_x4, res_y4: Coarse DEM resolution [deg].
        """
        self.rzones = rzones
        self.res_x1, self.res_y1 = res_x1, res_y1
        self.res_x3, self.res_y3 = res_x3, res_y3
        self.res_x4, self.res_y4 = res_x4, res_y4

    def compute_extents(self) -> ZoneExtents:
        """
        Compute aligned half-extents for all four zones.

        Corresponds to TC_GUI.m L920-L942.

        Returns:
            ZoneExtents with all extend_x/y values populated.
        """
        r1, r2, r3, r4 = self.rzones

        # Zone 2 resolution equals Zone 1 (detailed DEM)
        res_x2, res_y2 = self.res_x1, self.res_y1

        # Zone 1: floor(r1/res) * res
        extend_x1 = (np.floor(r1 / res_x2)) * res_x2
        extend_y1 = (np.floor(r1 / res_y2)) * res_y2

        # Zone 2: floor(r2/res) * res
        extend_x2 = (np.floor(r2 / res_x2)) * res_x2
        extend_y2 = (np.floor(r2 / res_y2)) * res_y2

        # Zone 3: n1 * floor(r3/res_x3/n1) * res_x3, where n1 = res_x4/res_x3
        n1 = self.res_x4 / self.res_x3
        n2 = self.res_y4 / self.res_y3
        extend_x3 = (n1 * (np.floor(r3 / self.res_x3 / n1))) * self.res_x3
        extend_y3 = (n2 * (np.floor(r3 / self.res_y3 / n2))) * self.res_y3

        # Zone 4: 4 * floor(r4/res_x4/4) * res_x4
        extend_x4 = (4 * (np.floor(r4 / self.res_x4 / 4))) * self.res_x4
        extend_y4 = (4 * (np.floor(r4 / self.res_y4 / 4))) * self.res_y4

        return ZoneExtents(
            extend_x1=float(extend_x1), extend_y1=float(extend_y1),
            extend_x2=float(extend_x2), extend_y2=float(extend_y2),
            extend_x3=float(extend_x3), extend_y3=float(extend_y3),
            extend_x4=float(extend_x4), extend_y4=float(extend_y4),
        )

    def extract_leakage_boundary(self, xm: np.ndarray, ym: np.ndarray,
                                 zm: np.ndarray, ref_m: np.ndarray,
                                 e_m: np.ndarray,
                                 lon: float, lat: float,
                                 extend_x1: float, extend_y1: float,
                                 resolution_x1: float, resolution_y1: float
                                 ) -> dict:
        """
        Extract boundary cells of Zone 1 for leakage compensation.

        Corresponds to TC_GUI.m L870-L893 (inside the itype==10 branch).

        The boundary cells (those on the outer edge of Zone 1) are removed
        from the Zone 2 mass list and re-evaluated with Prism at half weight
        to smoothly transition between polyhedron and prism zones.

        MATLAB:
            % find Zone 1 cells (inside extend_x1/y1)
            n = find(xm<=lon+extend_x1 & xm>=lon-extend_x1 &
                     ym<=lat+extend_y1 & ym>=lat-extend_y1);
            lon1=xm(n); lat1=ym(n); h1=zm(n); ref1=ref_m(n); e_zone1=e_m(n);
            % remove from Zone 2
            xm(n)=[]; ym(n)=[]; zm(n)=[]; ref_m(n)=[]; e_m(n)=[];
            % keep only boundary cells (outermost ring)
            lon_r=max(lon1); lon_l=min(lon1);
            lat_b=min(lat1); lat_t=max(lat1);
            n = find(lat1==lat_t | lat1==lat_b | lon1==lon_r | lon1==lon_l);
            lon1=lon1(n); lat1=lat1(n); h1=h1(n); ref1=ref1(n); e_zone1=e_zone1(n);

        Args:
            xm, ym, zm, ref_m, e_m: Flattened arrays of Zone 2 candidate cells.
            lon, lat: Station coordinates.
            extend_x1, extend_y1: Zone 1 half-extent.
            resolution_x1, resolution_y1: Zone 1 grid resolution.

        Returns:
            dict with keys:
                'boundary_lon','boundary_lat','boundary_h','boundary_ref','boundary_e':
                    boundary cell arrays (for half-weighted Prism)
                'remaining_x','remaining_y','remaining_z','remaining_ref','remaining_e':
                    Zone 2 cells minus Zone 1 cells
        """
        xm = np.asarray(xm)
        ym = np.asarray(ym)
        zm = np.asarray(zm)
        ref_m = np.asarray(ref_m)
        e_m = np.asarray(e_m)

        if extend_x1 > 0 and extend_y1 > 0:
            # Find Zone 1 cells (inside extend_x1/y1)
            mask_z1 = ((xm <= lon + extend_x1) & (xm >= lon - extend_x1) &
                       (ym <= lat + extend_y1) & (ym >= lat - extend_y1))

            lon1 = xm[mask_z1]
            lat1 = ym[mask_z1]
            h1 = zm[mask_z1]
            ref1 = ref_m[mask_z1]
            e_zone1 = e_m[mask_z1]

            # Remove Zone 1 cells from Zone 2 list
            xm = xm[~mask_z1]
            ym = ym[~mask_z1]
            zm = zm[~mask_z1]
            ref_m = ref_m[~mask_z1]
            e_m = e_m[~mask_z1]

            if lon1.size == 0:
                # No Zone 1 cells found (point near DEM edge)
                boundary_lon = np.array([])
                boundary_lat = np.array([])
                boundary_h = np.array([])
                boundary_ref = np.array([])
                boundary_e = np.array([])
            else:
                # Keep only boundary cells (outermost ring of Zone 1)
                lon_r = np.max(lon1)
                lon_l = np.min(lon1)
                lat_b = np.min(lat1)
                lat_t = np.max(lat1)
                mask_bdy = ((lat1 == lat_t) | (lat1 == lat_b) |
                            (lon1 == lon_r) | (lon1 == lon_l))

                boundary_lon = lon1[mask_bdy]
                boundary_lat = lat1[mask_bdy]
                boundary_h = h1[mask_bdy]
                boundary_ref = ref1[mask_bdy]
                boundary_e = e_zone1[mask_bdy]
        else:
            # No Zone 1 active: no leakage boundary
            boundary_lon = np.array([])
            boundary_lat = np.array([])
            boundary_h = np.array([])
            boundary_ref = np.array([])
            boundary_e = np.array([])

        return {
            'boundary_lon': boundary_lon,
            'boundary_lat': boundary_lat,
            'boundary_h': boundary_h,
            'boundary_ref': boundary_ref,
            'boundary_e': boundary_e,
            'remaining_x': xm,
            'remaining_y': ym,
            'remaining_z': zm,
            'remaining_ref': ref_m,
            'remaining_e': e_m,
        }
