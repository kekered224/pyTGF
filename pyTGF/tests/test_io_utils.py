"""
Smoke test for io + utils modules of pyTGF.

Verifies that the implementations match MATLAB logic for representative inputs.
Run: python -m pyTGF.tests.test_io_utils
"""

import os
import tempfile
import numpy as np

from pyTGF.io.dem_io import read_dem, write_dem, DEMData
from pyTGF.io.com_points import read_com_points
from pyTGF.io.output import write_report_header, write_report_dem_info, write_report_integration, write_result
from pyTGF.utils.coordinate import global2local, g_prism2point, ggt_prism2point
from pyTGF.utils.grid import build_grid, matrix_preallocating
from pyTGF.utils.normal_gravity import normal_gravity_generic
from pyTGF.utils.prism_helpers import gz3term, g6term
from pyTGF.config import TGFConfig, PhysicalConstants


def test_dem_roundtrip():
    """DEM write -> read should preserve data and metadata."""
    z = np.array([[100.0, 200.0, 300.0],
                  [110.0, 210.0, 310.0],
                  [120.0, 220.0, 320.0]])
    dem = DEMData(minlat=30.0, maxlat=30.02, minlon=110.0, maxlon=110.02,
                  reslat=0.01, reslon=0.01, z=z)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test_dem.bin')
        write_dem(path, dem)
        dem2 = read_dem(path)
    assert dem2.minlat == dem.minlat
    assert dem2.maxlat == dem.maxlat
    assert dem2.minlon == dem.minlon
    assert dem2.maxlon == dem.maxlon
    assert dem2.reslat == dem.reslat
    assert dem2.reslon == dem.reslon
    assert np.array_equal(dem2.z, dem.z), f"z mismatch:\n{dem2.z}\n{dem.z}"
    # lats / lons
    assert len(dem2.lats) == 3
    assert len(dem2.lons) == 3
    print("[OK] dem roundtrip")


def test_com_points():
    """Computation points file: [N, lat, lon, H] per row, negative H -> 1."""
    raw = np.array([
        [1, 40.0, 116.0, 100.0],
        [2, 41.0, 117.0, -5.0],   # negative -> corrected to 1
        [3, 42.0, 118.0, 200.0],
    ], dtype=np.float64)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'com.bin')
        # MATLAB writes the (N, 4) matrix column-major: id1..idN, lat1..latN,
        # lon1..lonN, h1..hN.
        raw.ravel(order='F').tofile(path)
        com = read_com_points(path)
    assert com.n_points == 3
    assert com.num == 3.0
    assert np.allclose(com.lats, [40.0, 41.0, 42.0])
    assert np.allclose(com.lons, [116.0, 117.0, 118.0])
    assert np.allclose(com.heights, [100.0, 1.0, 200.0])  # -5 -> 1
    print("[OK] com_points (negative height correction)")


def test_normal_gravity():
    """Normal gravity at equator/pole should match GRS80 values."""
    gamma_equator = normal_gravity_generic(np.array([0.0]), np.array([0.0]))
    gamma_pole = normal_gravity_generic(np.array([90.0]), np.array([0.0]))
    assert abs(gamma_equator[0] - 9.7803267715) < 1e-9, gamma_equator
    assert abs(gamma_pole[0] - 9.8321863685) < 1e-9, gamma_pole
    # At height 0, gamma_h == gamma_0
    gamma_45 = normal_gravity_generic(np.array([45.0]), np.array([0.0]))
    assert 9.806 < gamma_45[0] < 9.808, gamma_45
    print("[OK] normal_gravity (equator/pole/45deg)")


def test_global2local_origin():
    """Station at prism center -> local (0,0,0)."""
    lon = np.array([116.0])
    lat = np.array([40.0])
    r = np.array([6378137.0])
    x, y, z = global2local(lon, lat, r, lon, lat, r)
    assert np.allclose(x, 0.0, atol=1e-6), x
    assert np.allclose(y, 0.0, atol=1e-6), y
    assert np.allclose(z, 0.0, atol=1e-6), z
    print("[OK] global2local (origin)")


def test_global2local_offset():
    """Station 1 deg north of prism at same radius -> exact local coords.

    For prism at (lon0, lat0, a) and station at (lon0, lat0+1deg, a) on the
    same sphere of radius a, the prism-local frame (x=north, y=east, z=down)
    gives exactly:
        x = a * sin(1deg)           (north chord projection)
        y = 0                       (no east component)
        z = a * (1 - cos(1deg))     (station lies below tangent plane, z>0)
    """
    d2r = PhysicalConstants.d2r()
    a = PhysicalConstants.A
    lon_s = np.array([116.0])
    lat_s = np.array([41.0])
    r_s = np.array([a])
    x, y, z = global2local(lon_s, lat_s, r_s,
                           np.array([116.0]), np.array([40.0]), np.array([a]))
    expected_x = a * np.sin(d2r)
    expected_y = 0.0
    expected_z = a * (1.0 - np.cos(d2r))  # positive: z points inward/down
    assert abs(x[0] - expected_x) < 1e-6, f"x={x[0]}, expected={expected_x}"
    assert abs(y[0] - expected_y) < 1e-6, f"y={y[0]}, expected={expected_y}"
    assert abs(z[0] - expected_z) < 1e-6, f"z={z[0]}, expected={expected_z}"
    print(f"[OK] global2local (1deg north: x={x[0]:.6f}, y={y[0]:.2e}, z={z[0]:.6f})")


def test_g_prism2point_identity():
    """Prism at station location -> rotation is identity (R = I)."""
    gx = np.array([1.0, 2.0])
    gy = np.array([3.0, 4.0])
    gz = np.array([5.0, 6.0])
    lon, lat = 116.0, 40.0
    prism_lon = np.array([116.0, 116.0])
    prism_lat = np.array([40.0, 40.0])
    Vx, Vy, Vz = g_prism2point(gx, gy, gz, lon, lat, prism_lon, prism_lat)
    assert np.allclose(Vx, gx), Vx
    assert np.allclose(Vy, gy), Vy
    assert np.allclose(Vz, gz), Vz
    print("[OK] g_prism2point (identity at station)")


def test_ggt_prism2point_identity():
    """Tensor rotation identity when prism == station."""
    gxx = np.array([1.0]); gxy = np.array([2.0]); gxz = np.array([3.0])
    gyy = np.array([4.0]); gyz = np.array([5.0]); gzz = np.array([6.0])
    lon, lat = 116.0, 40.0
    plon = np.array([116.0]); plat = np.array([40.0])
    Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = ggt_prism2point(
        gxx, gxy, gxz, gyy, gyz, gzz, lon, lat, plon, plat)
    assert np.allclose(Vxx, gxx)
    assert np.allclose(Vxy, gxy)
    assert np.allclose(Vxz, gxz)
    assert np.allclose(Vyy, gyy)
    assert np.allclose(Vyz, gyz)
    assert np.allclose(Vzz, gzz)
    print("[OK] ggt_prism2point (identity at station)")


def test_build_grid_shape():
    """Ring grid: outer 0.04, inner 0.02, res 0.01 -> ring cells only."""
    lon, lat = 116.0, 40.0
    ex1, ey1 = 0.02, 0.02   # inner half-extent
    ex2, ey2 = 0.04, 0.04   # outer half-extent
    rx, ry = 0.01, 0.01
    lon21, lat21 = build_grid(lon, lat, ex1, ey1, ex2, ey2, rx, ry)
    # Outer grid: 8x8 = 64 cells; inner 4x4 = 16 cells; ring = 64-16 = 48
    assert lon21.size == 48, f"expected 48 ring cells, got {lon21.size}"
    assert lat21.size == 48
    # All ring cells should be outside the inner rectangle
    # (cell centers, so |dx| > ex1 or |dy| > ey1)
    dx = lon21 - lon
    dy = lat21 - lat
    inner_mask = (np.abs(dx) <= ex1) & (np.abs(dy) <= ey1)
    assert not np.any(inner_mask), "some ring cells fall inside inner zone"
    print(f"[OK] build_grid (ring cells={lon21.size})")


def test_matrix_preallocating():
    """10 zero arrays of shape (m, n)."""
    mats = matrix_preallocating(2, 3)
    assert len(mats) == 10
    for m in mats:
        assert m.shape == (2, 3)
        assert np.all(m == 0.0)
    print("[OK] matrix_preallocating (10 x (2,3))")


def test_gz3term_singular():
    """gz3term zeros out singular points where zpos1 == 0."""
    r = np.array([1.0, 1.0, 1.0])
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 1.0])
    z = np.array([1.0, 0.0, 1.0])
    nz1 = np.array([1])  # index 1 is singular (z=0)
    t3 = gz3term(r, x, y, z, nz1)
    assert t3[1] == 0.0, f"singular point not zeroed: {t3[1]}"
    assert t3[0] != 0.0
    print(f"[OK] gz3term (singular zeroed: t3={t3})")


def test_g6term_singular():
    """g6term zeros out singular points per term."""
    r = np.array([1.0, 1.0, 1.0])
    x = np.array([0.0, 1.0, 1.0])  # index 0: x=0
    y = np.array([1.0, 0.0, 1.0])  # index 1: y=0
    z = np.array([1.0, 1.0, 0.0])  # index 2: z=0
    nx1 = np.array([0])
    ny1 = np.array([1])
    nz1 = np.array([2])
    t1, t2, t3, t4, t5, t6 = g6term(r, x, y, z, nx1, ny1, nz1)
    # t1 depends on x and y -> zeroed at nx1 and ny1
    assert t1[0] == 0.0 and t1[1] == 0.0
    # t2 depends on y and z -> zeroed at ny1 and nz1
    assert t2[1] == 0.0 and t2[2] == 0.0
    # t3 depends on x and z -> zeroed at nx1 and nz1
    assert t3[0] == 0.0 and t3[2] == 0.0
    # t4 depends on x -> zeroed at nx1
    assert t4[0] == 0.0
    # t5 depends on y -> zeroed at ny1
    assert t5[1] == 0.0
    # t6 depends on z -> zeroed at nz1
    assert t6[2] == 0.0
    print("[OK] g6term (singular points zeroed per term)")


def test_report_writers():
    """Report writers produce non-empty files without error."""
    config = TGFConfig(ikind=2, itype=104, rzones=[0.0, 0.03, 0.3, 2.0])
    from pyTGF.io.com_points import ComPointData
    com = ComPointData(
        ids=np.array([1.0, 2.0]),
        lats=np.array([40.0, 41.0]),
        lons=np.array([116.0, 117.0]),
        heights=np.array([100.0, 200.0]),
    )
    dem_layers = {
        'detailed': DEMData(30.0, 45.0, 110.0, 125.0, 0.01, 0.01,
                            np.zeros((1501, 1501))),
        'coarse': DEMData(0.0, 60.0, 80.0, 140.0, 0.1, 0.1,
                          np.zeros((601, 601))),
    }
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, 'report.txt')
        with open(rp, 'w') as f:
            write_report_header(f, config, com)
            write_report_dem_info(f, config, dem_layers)
            write_report_integration(f, 0.123, 'out.dat',
                                      config.output_columns)
        assert os.path.getsize(rp) > 0
        # result writer
        op = os.path.join(d, 'out.dat')
        tensor = np.array([[116.0, 40.0, 100.0, 5.0, 1.0, 2.0, 3.0]])
        write_result(op, tensor)
        assert os.path.getsize(op) > 0
    print("[OK] report writers (header/dem_info/integration/result)")


def main():
    test_dem_roundtrip()
    test_com_points()
    test_normal_gravity()
    test_global2local_origin()
    test_global2local_offset()
    test_g_prism2point_identity()
    test_ggt_prism2point_identity()
    test_build_grid_shape()
    test_matrix_preallocating()
    test_gz3term_singular()
    test_g6term_singular()
    test_report_writers()
    print("\nAll io+utils tests passed.")


if __name__ == '__main__':
    main()
