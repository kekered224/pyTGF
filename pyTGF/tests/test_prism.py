"""
Unit tests for mass_elements: prism.

Verifies:
  1. Internal consistency:
       prism (full) == prism_g_gx (potential+1st) + prism_gxx (2nd)
       prism_g_gx  == prism_g (potential) + prism_gx (1st)
  2. Prism -> point-mass limit: as prism size -> 0, results converge to
     point_mass (same mass, same center).
  3. Physical sanity: mass directly below station -> Vz < 0, Vzz < 0,
     Vx ~ 0, Vy ~ 0.
  4. Laplacian ~ 0 outside mass (Vxx + Vyy + Vzz = 0).

Run: python -m pyTGF.tests.test_prism
"""

import numpy as np
from pyTGF.config import PhysicalConstants
from pyTGF.mass_elements.point_mass import point_mass
from pyTGF.mass_elements.prism import (
    prism, prism_g, prism_gx, prism_g_gx, prism_gxx,
)


# =============================================================================
#  Test fixtures
# =============================================================================

def _make_prism_inputs(n=5, seed=42):
    """
    Build prism inputs (tesseroid-bounds format).

    Returns x1,x2,y1,y2,d1,d2 (bounds), lon,lat,r0 (station), e (density),
    resolution_x,resolution_y (prism resolution [deg]).

    Convention (matching MATLAB Prism, called from TC_GUI.m L1113):
        x1,x2: longitude bounds [deg]
        y1,y2: latitude bounds [deg]
        d1:    upper radial bound [m] = h1 (top of mass, LARGER radius,
               higher altitude)
        d2:    lower radial bound [m] = ref (bottom of mass, SMALLER radius,
               lower altitude)
        So d1 > d2 (matching zz1=h1, zz2=ref1 in the calling code).
    """
    rng = np.random.default_rng(seed)
    a = PhysicalConstants.A
    # prism centers scattered around (116, 40) at near-surface radius
    lon_c = 116.0 + rng.uniform(-0.5, 0.5, n)
    lat_c = 40.0 + rng.uniform(-0.5, 0.5, n)
    # angular size 0.01 deg
    dx_deg = 0.01
    dy_deg = 0.01
    x1 = lon_c - dx_deg / 2.0
    x2 = lon_c + dx_deg / 2.0
    y1 = lat_c - dy_deg / 2.0
    y2 = lat_c + dy_deg / 2.0
    # radial bounds: d1 (upper) > d2 (lower) per MATLAB convention
    # d1 = h1 = top of mass (larger r), d2 = ref = bottom (smaller r)
    d2 = a - 1000.0 + rng.uniform(-500, 500, n)   # lower radius (bottom, smaller r)
    d1 = d2 + 500.0                                # upper radius (top, larger r, d1 > d2)
    e = np.full(n, 2.67)
    # station
    lon, lat, r0 = 116.0, 40.0, a + 100.0
    resolution_x = 0.01
    resolution_y = 0.01
    return x1, x2, y1, y2, d1, d2, lon, lat, r0, e, resolution_x, resolution_y


# =============================================================================
#  Internal consistency tests
# =============================================================================

def test_prism_internal_consistency():
    """
    prism (full 10 components) should match prism_g_gx (potential+1st) and
    prism_gxx (2nd) for the same inputs.
    """
    x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry = _make_prism_inputs()

    g_full, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = prism(
        x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)

    g_g_gx, Vx2, Vy2, Vz2 = prism_g_gx(x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)
    Vxx2, Vxy2, Vxz2, Vyy2, Vyz2, Vzz2 = prism_gxx(
        x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)

    assert np.isclose(g_full, g_g_gx, rtol=1e-12), \
        f"potential mismatch: full={g_full}, g_g_gx={g_g_gx}"
    assert np.isclose(Vx, Vx2, rtol=1e-12), f"Vx mismatch: {Vx} vs {Vx2}"
    assert np.isclose(Vy, Vy2, rtol=1e-12), f"Vy mismatch: {Vy} vs {Vy2}"
    assert np.isclose(Vz, Vz2, rtol=1e-12), f"Vz mismatch: {Vz} vs {Vz2}"
    assert np.isclose(Vxx, Vxx2, rtol=1e-12), f"Vxx mismatch: {Vxx} vs {Vxx2}"
    assert np.isclose(Vxy, Vxy2, rtol=1e-12), f"Vxy mismatch: {Vxy} vs {Vxy2}"
    assert np.isclose(Vxz, Vxz2, rtol=1e-12), f"Vxz mismatch: {Vxz} vs {Vxz2}"
    assert np.isclose(Vyy, Vyy2, rtol=1e-12), f"Vyy mismatch: {Vyy} vs {Vyy2}"
    assert np.isclose(Vyz, Vyz2, rtol=1e-12), f"Vyz mismatch: {Vyz} vs {Vyz2}"
    assert np.isclose(Vzz, Vzz2, rtol=1e-12), f"Vzz mismatch: {Vzz} vs {Vzz2}"

    print(f"[OK] prism internal consistency "
          f"(g={g_full:.6e}, Vz={Vz:.6e}, Vzz={Vzz:.6e})")


def test_prism_g_gx_matches_split():
    """
    prism_g_gx (potential + 1st) should match prism_g (potential) +
    prism_gx (1st derivatives).
    """
    x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry = _make_prism_inputs()

    V, Vx, Vy, Vz = prism_g_gx(x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)
    V_g = prism_g(x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)
    Vx_gx, Vy_gx, Vz_gx = prism_gx(x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)

    assert np.isclose(V, V_g, rtol=1e-12), f"V mismatch: {V} vs {V_g}"
    assert np.isclose(Vx, Vx_gx, rtol=1e-12), f"Vx mismatch: {Vx} vs {Vx_gx}"
    assert np.isclose(Vy, Vy_gx, rtol=1e-12), f"Vy mismatch: {Vy} vs {Vy_gx}"
    assert np.isclose(Vz, Vz_gx, rtol=1e-12), f"Vz mismatch: {Vz} vs {Vz_gx}"
    print(f"[OK] prism_g_gx == prism_g + prism_gx")


# =============================================================================
#  Physical sanity tests
# =============================================================================

def test_prism_physical_sanity():
    """
    Single prism directly below station -> g < 0, Vz < 0, Vzz < 0,
    Vx ~ 0, Vy ~ 0.

    Sign convention (matching MATLAB Prism):
        g  = -sum(c*res)*1e-5    (potential, negative for positive mass)
        gz =  c*value            (z-derivative in prism frame)
        After rotation to station frame, Vz < 0 for mass below station
        (z-down convention).
    """
    a = PhysicalConstants.A
    lon_c, lat_c = 116.0, 40.0
    deg = 0.001  # tiny angular size
    x1 = np.array([lon_c - deg / 2])
    x2 = np.array([lon_c + deg / 2])
    y1 = np.array([lat_c - deg / 2])
    y2 = np.array([lat_c + deg / 2])
    # mass between a-1000 and a-500 (below surface)
    # d1 = h1 = upper radius (top of mass, larger r)
    # d2 = ref = lower radius (bottom of mass, smaller r)
    d1 = np.array([a - 500.0])    # upper radius (top of mass)
    d2 = np.array([a - 1000.0])   # lower radius (bottom of mass)
    e = np.array([2.67])
    lon, lat, r0 = 116.0, 40.0, a + 100.0

    g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = prism(
        x1, x2, y1, y2, d1, d2, lon, lat, r0, e, deg, deg)

    # Potential should be negative (attractive, V = -GM/r convention)
    assert g < 0, f"g should be < 0 for positive mass, got {g}"
    # Vz < 0 (z-down frame, mass below)
    assert Vz < 0, f"Vz should be < 0 for mass below station, got {Vz}"
    # Vzz < 0 (mass directly below)
    assert Vzz < 0, f"Vzz should be < 0 for mass below, got {Vzz}"
    # Vx, Vy ~ 0 (mass directly below, no horizontal component)
    assert abs(Vx) < 1e-6 * max(abs(Vz), 1e-10), f"Vx should be ~0, got {Vx}"
    assert abs(Vy) < 1e-6 * max(abs(Vz), 1e-10), f"Vy should be ~0, got {Vy}"
    print(f"[OK] prism physical sanity "
          f"(g={g:.3e}, Vz={Vz:.3e}, Vzz={Vzz:.3e}, Vx={Vx:.3e}, Vy={Vy:.3e})")


def test_prism_laplacian_zero():
    """
    Laplacian of potential outside mass = 0: Vxx + Vyy + Vzz = 0.
    (All prisms are far from the station.)
    """
    x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry = _make_prism_inputs()
    _, _, _, _, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = prism(
        x1, x2, y1, y2, d1, d2, lon, lat, r0, e, rx, ry)
    lap = Vxx + Vyy + Vzz
    scale = max(abs(Vxx), abs(Vyy), abs(Vzz), 1e-10)
    # Prism formulas are exact (not Taylor approx), so Laplacian should be
    # ~machine precision zero.
    assert abs(lap) < 1e-6 * scale, \
        f"Laplacian not zero: Vxx+Vyy+Vzz={lap} (scale={scale})"
    print(f"[OK] prism Laplacian ~ 0 (lap={lap:.3e})")


# =============================================================================
#  Convergence to point mass
# =============================================================================

def test_prism_to_point_mass_convergence():
    """
    As prism angular size -> 0, prism result should converge to point_mass
    result (same mass, same center).

    Both share the same sign convention (V = -GM/r, z-down frame).
    """
    a = PhysicalConstants.A
    # single prism at (116.1, 40.1), 500 m thick, station at (116, 40, a+100)
    lon_c, lat_c = 116.1, 40.1
    # d1 = h1 = upper radius (top, larger r), d2 = ref = lower radius (bottom, smaller r)
    d1 = a + 0.0       # upper radius (top, at surface)
    d2 = a - 500.0     # lower radius (bottom)
    e = np.array([2.67])
    lon, lat, r0 = 116.0, 40.0, a + 100.0

    # point_mass equivalent: same center, same radial bounds
    # point_mass takes xx1,xx2 (lon bounds), yy1,yy2 (lat bounds), r1, r2
    # where r1 = upper (larger r), r2 = lower (smaller r)
    deg_small = 0.0001
    xx1 = np.array([lon_c - deg_small / 2])
    xx2 = np.array([lon_c + deg_small / 2])
    yy1 = np.array([lat_c - deg_small / 2])
    yy2 = np.array([lat_c + deg_small / 2])
    r1 = np.array([d1])    # upper (larger r)
    r2 = np.array([d2])    # lower (smaller r)

    g_pm, Vx_pm, Vy_pm, Vz_pm, Vxx_pm, Vxy_pm, Vxz_pm, Vyy_pm, Vyz_pm, Vzz_pm = \
        point_mass(xx1, xx2, yy1, yy2, r1, r2, lon, lat, r0, e)

    # prism at decreasing sizes
    errs = []
    for deg in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        x1 = np.array([lon_c - deg / 2])
        x2 = np.array([lon_c + deg / 2])
        y1 = np.array([lat_c - deg / 2])
        y2 = np.array([lat_c + deg / 2])
        g, Vx, Vy, Vz, Vxx, Vxy, Vxz, Vyy, Vyz, Vzz = prism(
            x1, x2, y1, y2, np.array([d1]), np.array([d2]),
            lon, lat, r0, e, deg, deg)
        # compare Vz directly (same sign convention)
        rel_err = abs(Vz - Vz_pm) / max(abs(Vz_pm), 1e-30)
        errs.append((deg, rel_err, Vz, Vz_pm))

    # smallest prism should be within ~1% of point mass
    assert errs[-1][1] < 0.01, \
        f"prism@{errs[-1][0]}deg vs point_mass: rel_err={errs[-1][1]}"
    # and error should generally decrease with size
    assert errs[-1][1] < errs[0][1], \
        f"no convergence: large={errs[0][1]}, small={errs[-1][1]}"

    print(f"[OK] prism -> point_mass convergence:")
    for deg, err, vz_p, vz_pm in errs:
        print(f"       deg={deg:<8}  Vz_prism={vz_p:.6e}  "
              f"Vz_pm={vz_pm:.6e}  rel_err={err:.3e}")


# =============================================================================
#  Main
# =============================================================================

def main():
    print("=== Prism tests ===")
    test_prism_internal_consistency()
    test_prism_g_gx_matches_split()
    test_prism_physical_sanity()
    test_prism_laplacian_zero()
    test_prism_to_point_mass_convergence()
    print("\nAll prism tests passed.")


if __name__ == '__main__':
    main()
