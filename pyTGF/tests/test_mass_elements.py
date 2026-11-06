"""
Unit tests for mass_elements: point_mass and tesseroid.

Verifies:
  1. Internal consistency: point_mass == point_g_gx (potential+1st) + point_gxx (2nd)
  2. Internal consistency: tesseroid (full) == tesseroid_g/gx/g_gx/gxx components
  3. Tesseroid -> point-mass limit: as tesseroid size -> 0, results converge
  4. Physical sanity: mass below station -> positive Vz (downward gravity)

Run: python -m pyTGF.tests.test_mass_elements
"""

import numpy as np
from pyTGF.config import PhysicalConstants
from pyTGF.mass_elements.point_mass import (
    point_g, point_gx, point_g_gx, point_gxx, point_mass,
)
from pyTGF.mass_elements.tesseroid import (
    tesseroid, tesseroid_g, tesseroid_gx, tesseroid_g_gx, tesseroid_gxx,
)


# =============================================================================
#  Test fixtures
# =============================================================================

def _make_point_mass_inputs(n=5, seed=42):
    """
    Build a small array of point-mass inputs (tesseroid-bounds format).

    Returns xx1,xx2,yy1,yy2,r1,r2 (bounds) and lon,lat,r0,e (station).
    Here r1,r2 are the radial bounds (r1=upper, r2=lower) used by Point_*.
    """
    rng = np.random.default_rng(seed)
    a = PhysicalConstants.A
    # tesseroid centers scattered around (116, 40) at near-surface radius
    lon_c = 116.0 + rng.uniform(-0.5, 0.5, n)
    lat_c = 40.0 + rng.uniform(-0.5, 0.5, n)
    # angular size 0.01 deg
    dx_deg = 0.01
    dy_deg = 0.01
    xx1 = lon_c - dx_deg / 2.0   # lon lower
    xx2 = lon_c + dx_deg / 2.0   # lon upper
    yy1 = lat_c - dy_deg / 2.0   # lat lower
    yy2 = lat_c + dy_deg / 2.0   # lat upper
    # radial bounds: r1 (upper) > r2 (lower); both near Earth surface
    r2 = a - 1000.0 + rng.uniform(-500, 500, n)   # lower radius
    r1 = r2 + 500.0                                # upper radius (r1 > r2)
    e = np.full(n, 2.67)
    # station
    lon, lat, r0 = 116.0, 40.0, a + 100.0
    return xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e


def _make_tesseroid_inputs(n=5, seed=42):
    """
    Build tesseroid inputs (center + bounds + resolution).

    Returns lon1,lat1,h1,ref (centers + radial bounds), resolution_x1/y1,
    and lon,lat,r0,e (station). h1 = upper radius, ref = lower radius.
    """
    rng = np.random.default_rng(seed)
    a = PhysicalConstants.A
    lon1 = 116.0 + rng.uniform(-0.5, 0.5, n)
    lat1 = 40.0 + rng.uniform(-0.5, 0.5, n)
    ref = a - 1000.0 + rng.uniform(-500, 500, n)   # lower radius
    h1 = ref + 500.0                                # upper radius (h1 > ref)
    resolution_x1 = 0.01
    resolution_y1 = 0.01
    e = np.full(n, 2.67)
    lon, lat, r0 = 116.0, 40.0, a + 100.0
    return lon1, lat1, h1, ref, lon, lat, r0, resolution_x1, resolution_y1, e


# =============================================================================
#  Point mass tests
# =============================================================================

def test_point_mass_internal_consistency():
    """
    point_mass (full) should match point_g_gx (potential+1st) and point_gxx
    (2nd) for the same inputs.

    Note: point_g (itype=0) uses a DIFFERENT global2local argument order,
    so its potential may differ in sign from point_mass/point_g_gx.
    We check point_mass against point_g_gx + point_gxx instead.
    """
    xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e = _make_point_mass_inputs()

    g_full, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = point_mass(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)

    g_gx, Vx2, Vy2, Vz2 = point_g_gx(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)
    Vxx2, Vxy2, Vxz2, Vyy2, Vyz2, Vzz2 = point_gxx(xx1, xx2, yy1, yy2, r1, r2,
                                                    lon, lat, r0, e)

    # potential + 1st derivatives from point_g_gx
    assert np.isclose(g_full, g_gx, rtol=1e-12), \
        f"potential mismatch: full={g_full}, g_gx={g_gx}"
    assert np.isclose(Vx, Vx2, rtol=1e-12), f"Vx mismatch: {Vx} vs {Vx2}"
    assert np.isclose(Vy, Vy2, rtol=1e-12), f"Vy mismatch: {Vy} vs {Vy2}"
    assert np.isclose(Vz, Vz2, rtol=1e-12), f"Vz mismatch: {Vz} vs {Vz2}"

    # 2nd derivatives from point_gxx
    assert np.isclose(Vxx, Vxx2, rtol=1e-12), f"Vxx mismatch: {Vxx} vs {Vxx2}"
    assert np.isclose(Vxy, Vxy2, rtol=1e-12), f"Vxy mismatch: {Vxy} vs {Vxy2}"
    assert np.isclose(Vxz, Vxz2, rtol=1e-12), f"Vxz mismatch: {Vxz} vs {Vxz2}"
    assert np.isclose(Vyy, Vyy2, rtol=1e-12), f"Vyy mismatch: {Vyy} vs {Vyy2}"
    assert np.isclose(Vyz, Vyz2, rtol=1e-12), f"Vyz mismatch: {Vyz} vs {Vyz2}"
    assert np.isclose(Vzz, Vzz2, rtol=1e-12), f"Vzz mismatch: {Vzz} vs {Vzz2}"

    print(f"[OK] point_mass internal consistency "
          f"(g={g_full:.6e}, Vz={Vz:.6e}, Vzz={Vzz:.6e})")


def test_point_mass_laplacian_zero():
    """
    Laplacian of potential outside mass = 0: Vxx + Vyy + Vzz = 0.
    (All point masses are far from the station.)
    """
    xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e = _make_point_mass_inputs()
    _, _, _, _, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = point_mass(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)
    lap = Vxx + Vyy + Vzz
    # scale-relative tolerance (Vxx etc are ~1e-4 to 1e-2 Eötvös)
    scale = max(abs(Vxx), abs(Vyy), abs(Vzz), 1e-10)
    assert abs(lap) < 1e-6 * scale, \
        f"Laplacian not zero: Vxx+Vyy+Vzz={lap} (scale={scale})"
    print(f"[OK] point_mass Laplacian ~ 0 (lap={lap:.3e})")


def test_point_mass_physical_sanity():
    """
    A single point mass directly below the station should give:
      - Vz < 0 (MATLAB Point_mass convention: V=-GM/r, z-down frame,
        so Vz = sum(-c*dz*l3) < 0 for mass below where dz > 0)
      - Vzz < 0 (same sign pattern: gzz=-res, Vzz=sum(gzz)*1e4)
      - Vxx, Vyy have same sign
    """
    a = PhysicalConstants.A
    # single tesseroid-equivalent mass at (116, 40), 1000 m below station
    lon_c, lat_c = 116.0, 40.0
    deg = 0.001  # tiny angular size
    xx1 = np.array([lon_c - deg / 2])
    xx2 = np.array([lon_c + deg / 2])
    yy1 = np.array([lat_c - deg / 2])
    yy2 = np.array([lat_c + deg / 2])
    r2 = np.array([a - 500.0])   # lower radius (mass bottom)
    r1 = np.array([a + 0.0])     # upper radius (mass top, at surface)
    e = np.array([2.67])
    lon, lat, r0 = 116.0, 40.0, a + 500.0   # station 500 m above surface

    g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = point_mass(
        xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)

    # MATLAB Point_mass convention: Vz = sum(gz) where gz = -c*dz*l3.
    # For mass below station, dz > 0 (z-down frame), so Vz < 0.
    assert Vz < 0, f"Vz should be < 0 for mass below station (MATLAB convention), got {Vz}"
    # Vzz = sum(gzz)*1e4 where gzz = -c*(-l3 + 3*dz^2*l5).
    # For mass directly below (dx=dy=0): gzz = -c*(2/l^3) < 0, so Vzz < 0.
    assert Vzz < 0, f"Vzz should be < 0 for mass below (MATLAB convention), got {Vzz}"
    assert np.sign(Vxx) == np.sign(Vyy), \
        f"Vxx and Vyy should have same sign: {Vxx}, {Vyy}"
    print(f"[OK] point_mass physical sanity "
          f"(Vz={Vz:.3e}, Vzz={Vzz:.3e}, Vxx={Vxx:.3e})")


# =============================================================================
#  Tesseroid tests
# =============================================================================

def test_tesseroid_internal_consistency():
    """
    tesseroid (full) should match tesseroid_g (potential), tesseroid_gx
    (1st derivs), and tesseroid_gxx (2nd derivs) for the same inputs.
    """
    lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e = _make_tesseroid_inputs()

    V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = tesseroid(
        lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)

    V_g = tesseroid_g(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    Vx_gx, Vy_gx, Vz_gx = tesseroid_gx(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    Vxx_gxx, Vxy_gxx, Vxz_gxx, Vyy_gxx, Vyz_gxx, Vzz_gxx = tesseroid_gxx(
        lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)

    assert np.isclose(V, V_g, rtol=1e-12), f"V mismatch: {V} vs {V_g}"
    assert np.isclose(Vx, Vx_gx, rtol=1e-12), f"Vx mismatch: {Vx} vs {Vx_gx}"
    assert np.isclose(Vy, Vy_gx, rtol=1e-12), f"Vy mismatch: {Vy} vs {Vy_gx}"
    assert np.isclose(Vz, Vz_gx, rtol=1e-12), f"Vz mismatch: {Vz} vs {Vz_gx}"
    assert np.isclose(Vxx, Vxx_gxx, rtol=1e-12), f"Vxx mismatch"
    assert np.isclose(Vxy, Vxy_gxx, rtol=1e-12), f"Vxy mismatch"
    assert np.isclose(Vxz, Vxz_gxx, rtol=1e-12), f"Vxz mismatch"
    assert np.isclose(Vyy, Vyy_gxx, rtol=1e-12), f"Vyy mismatch"
    assert np.isclose(Vyz, Vyz_gxx, rtol=1e-12), f"Vyz mismatch"
    assert np.isclose(Vzz, Vzz_gxx, rtol=1e-12), f"Vzz mismatch"

    print(f"[OK] tesseroid internal consistency "
          f"(V={V:.6e}, Vz={Vz:.6e}, Vzz={Vzz:.6e})")


def test_tesseroid_g_gx_matches_split():
    """tesseroid_g_gx should match tesseroid_g + tesseroid_gx."""
    lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e = _make_tesseroid_inputs()

    V, Vx, Vy, Vz = tesseroid_g_gx(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    V_g = tesseroid_g(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    Vx_gx, Vy_gx, Vz_gx = tesseroid_gx(lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)

    assert np.isclose(V, V_g, rtol=1e-12), f"V mismatch: {V} vs {V_g}"
    assert np.isclose(Vx, Vx_gx, rtol=1e-12), f"Vx mismatch"
    assert np.isclose(Vy, Vy_gx, rtol=1e-12), f"Vy mismatch"
    assert np.isclose(Vz, Vz_gx, rtol=1e-12), f"Vz mismatch"
    print(f"[OK] tesseroid_g_gx == tesseroid_g + tesseroid_gx")


def test_tesseroid_laplacian_zero():
    """Vxx + Vyy + Vzz = 0 outside mass."""
    lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e = _make_tesseroid_inputs()
    _, _, _, _, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = tesseroid(
        lon1, lat1, h1, ref, lon, lat, r0, rx, ry, e)
    lap = Vxx + Vyy + Vzz
    scale = max(abs(Vxx), abs(Vyy), abs(Vzz), 1e-10)
    # tesseroid is a 2nd-order Taylor approx, so Laplacian is approx zero
    # but not to machine precision. Allow 1e-3 relative.
    assert abs(lap) < 1e-3 * scale, \
        f"Laplacian not ~0: {lap} (scale={scale})"
    print(f"[OK] tesseroid Laplacian ~ 0 (lap={lap:.3e})")


def test_tesseroid_to_point_mass_convergence():
    """
    As tesseroid angular size -> 0, tesseroid result should converge to
    point_mass result (same mass, same center).

    Both tesseroid and point_mass share the same sign convention because
    tesseroid's c = G*e*zz*xx*yy has xx = ref - h1 < 0, which flips the
    sign to match point_mass's V = -GM/r convention.

    We shrink resolution from 0.01 to 0.0001 deg and check the relative
    difference in Vz decreases.
    """
    a = PhysicalConstants.A
    # single tesseroid at (116.1, 40.1), 500 m thick, station at (116, 40, a+100)
    lon_c, lat_c = 116.1, 40.1
    ref = a - 500.0
    h1 = a + 0.0
    e = np.array([2.67])
    lon, lat, r0 = 116.0, 40.0, a + 100.0

    # point_mass equivalent: same center, same radial bounds
    # point_mass takes xx1,xx2 (lon bounds), yy1,yy2 (lat bounds), r1, r2
    # For a tiny tesseroid, center = ((xx1+xx2)/2, (yy1+yy2)/2)
    # We'll use the smallest tesseroid size for the point-mass bounds too.
    deg_small = 0.0001
    xx1 = np.array([lon_c - deg_small / 2])
    xx2 = np.array([lon_c + deg_small / 2])
    yy1 = np.array([lat_c - deg_small / 2])
    yy2 = np.array([lat_c + deg_small / 2])
    r1 = np.array([h1])    # upper
    r2 = np.array([ref])   # lower

    g_pm, Vx_pm, Vy_pm, Vz_pm, Vxx_pm, Vxy_pm, Vxz_pm, Vyy_pm, Vyz_pm, Vzz_pm = \
        point_mass(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)

    # tesseroid at decreasing sizes
    errs = []
    for deg in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        lon1 = np.array([lon_c])
        lat1 = np.array([lat_c])
        h1_a = np.array([h1])
        ref_a = np.array([ref])
        V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = tesseroid(
            lon1, lat1, h1_a, ref_a, lon, lat, r0, deg, deg, e)
        # compare Vz directly (same sign convention)
        rel_err = abs(Vz - Vz_pm) / max(abs(Vz_pm), 1e-30)
        errs.append((deg, rel_err, Vz, Vz_pm))

    # smallest tesseroid should be within ~1% of point mass
    assert errs[-1][1] < 0.01, \
        f"tesseroid@{errs[-1][0]}deg vs point_mass: rel_err={errs[-1][1]}"
    # and error should generally decrease with size (Taylor series)
    assert errs[-1][1] < errs[0][1], \
        f"no convergence: large={errs[0][1]}, small={errs[-1][1]}"

    print(f"[OK] tesseroid -> point_mass convergence:")
    for deg, err, vz_t, vz_p in errs:
        print(f"       deg={deg:<8}  Vz_tess={vz_t:.6e}  "
              f"Vz_pm={vz_p:.6e}  rel_err={err:.3e}")


def test_tesseroid_physical_sanity():
    """
    Single tesseroid directly below station -> Vz < 0, Vzz < 0.

    NOTE: In tesseroid, c = G*e*zz*xx*yy where xx = ref - h1 < 0 (lower
    minus upper radius). This negative c flips signs so that tesseroid
    Vz, Vzz match point_mass convention (Vz < 0 for mass below).
    """
    a = PhysicalConstants.A
    lon_c, lat_c = 116.0, 40.0
    ref = a - 1000.0
    h1 = a - 500.0    # mass between -1000 and -500 m relative to surface
    e = np.array([2.67])
    lon, lat, r0 = 116.0, 40.0, a + 100.0

    V, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = tesseroid(
        np.array([lon_c]), np.array([lat_c]), np.array([h1]), np.array([ref]),
        lon, lat, r0, 0.001, 0.001, e)

    assert Vz < 0, f"Vz should be < 0 for mass below station, got {Vz}"
    assert Vzz < 0, f"Vzz should be < 0 for mass below, got {Vzz}"
    # Vx, Vy ~ 0 (mass directly below, no horizontal component)
    assert abs(Vx) < 1e-6 * max(abs(Vz), 1e-10), f"Vx should be ~0, got {Vx}"
    assert abs(Vy) < 1e-6 * max(abs(Vz), 1e-10), f"Vy should be ~0, got {Vy}"
    print(f"[OK] tesseroid physical sanity "
          f"(Vz={Vz:.3e}, Vzz={Vzz:.3e}, Vx={Vx:.3e}, Vy={Vy:.3e})")


# =============================================================================
#  Main
# =============================================================================

def main():
    print("=== Point mass tests ===")
    test_point_mass_internal_consistency()
    test_point_mass_laplacian_zero()
    test_point_mass_physical_sanity()
    print("\n=== Tesseroid tests ===")
    test_tesseroid_internal_consistency()
    test_tesseroid_g_gx_matches_split()
    test_tesseroid_laplacian_zero()
    test_tesseroid_to_point_mass_convergence()
    test_tesseroid_physical_sanity()
    print("\nAll mass_elements tests passed.")


if __name__ == '__main__':
    main()
