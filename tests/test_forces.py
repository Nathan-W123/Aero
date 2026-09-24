"""
Tests for aerodynamic force computation.

Invariants:
  - Empty domain (no solid): Fx = Fy = 0
  - Forces are computed from surface links only
  - Coefficient normalisation is dimensionally correct
"""

import numpy as np
import pytest

from aero.lbm.d2q9 import compute_feq
from aero.forces import (
    compute_forces,
    forces_to_coefficients,
    compute_force_split_2d,
    split_to_coefficients,
    compute_force_moment_2d,
    moment_to_coefficient_2d,
)
from aero.lbm.boundary import build_surface_links


Ny, Nx = 40, 80
U0    = 0.05
RHO0  = 1.0


@pytest.fixture
def feq_uniform():
    rho = np.ones((Ny, Nx))
    ux  = np.full((Ny, Nx), U0)
    uy  = np.zeros((Ny, Nx))
    return compute_feq(rho, ux, uy)


def test_no_force_empty_domain(feq_uniform):
    """Without any solid cells, forces must be exactly zero."""
    solid = np.zeros((Ny, Nx), dtype=bool)
    links, _ = build_surface_links(solid)
    Fx, Fy = compute_forces(feq_uniform, feq_uniform, links, RHO0, U0)
    assert Fx == 0.0
    assert Fy == 0.0


def test_force_coefficients_scale_correctly():
    """Cd = Fx / (0.5 * rho0 * u0^2 * D)."""
    Fx, Fy, D = 1.0, 0.5, 40.0
    Cd, Cl = forces_to_coefficients(Fx, Fy, RHO0, U0, D)
    F_dyn = 0.5 * RHO0 * U0 ** 2 * D
    assert abs(Cd - Fx / F_dyn) < 1e-14
    assert abs(Cl - Fy / F_dyn) < 1e-14


def test_force_zero_dyn_pressure_returns_zero():
    """If u0=0 the dynamic pressure is zero — function should not divide by zero."""
    Cd, Cl = forces_to_coefficients(1.0, 1.0, RHO0, u0=0.0, D=40.0)
    assert Cd == 0.0
    assert Cl == 0.0


def test_symmetry_top_bottom(feq_uniform):
    """
    For a horizontally centred obstacle with symmetric flow,
    the y-force (lift) from the top half and bottom half should cancel.
    """
    from aero.geometry.cylinder import Cylinder
    cyl   = Cylinder(radius=8, cx_frac=0.5, cy_frac=0.5)
    solid = cyl.mark_solid(Ny, Nx)
    links, _ = build_surface_links(solid)
    f     = feq_uniform

    _, Fy = compute_forces(f, f, links, RHO0, U0)
    # Pure equilibrium → no net lift; small floating point residual expected
    assert abs(Fy) < 1e-6


def test_force_split_sums_to_momentum_exchange(feq_uniform):
    """Pressure + viscous split must equal total momentum exchange."""
    from aero.geometry.cylinder import Cylinder
    cyl = Cylinder(radius=8, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(Ny, Nx)
    links, _ = build_surface_links(solid)
    f = feq_uniform
    Fx, Fy = compute_forces(f, f, links, RHO0, U0)
    fx_p, fy_p, fx_v, fy_v = compute_force_split_2d(f, f, links)
    assert abs((fx_p + fx_v) - Fx) < 1e-12
    assert abs((fy_p + fy_v) - Fy) < 1e-12
    cdp, _, cdv, _ = split_to_coefficients(fx_p, fy_p, fx_v, fy_v, RHO0, U0, 16.0)
    cd, _ = forces_to_coefficients(Fx, Fy, RHO0, U0, 16.0)
    assert abs(cdp + cdv - cd) < 1e-12


def test_force_moment_zero_for_uniform_equilibrium(feq_uniform):
    solid = np.zeros((Ny, Nx), dtype=bool)
    links, _ = build_surface_links(solid)
    Fx, Fy, Mz = compute_force_moment_2d(
        feq_uniform, feq_uniform, links, center_x=Nx / 2.0, center_y=Ny / 2.0
    )
    assert Fx == 0.0
    assert Fy == 0.0
    assert Mz == 0.0
    assert moment_to_coefficient_2d(Mz, RHO0, U0, 16.0) == 0.0


def _random_state(rng, shape, q, feq_fn, *fields):
    rho = 1.0 + 0.02 * rng.standard_normal(shape)
    us = [0.05 * rng.standard_normal(shape) for _ in range(fields[0])]
    base = feq_fn(rho, *us)
    return (
        base + 1e-4 * rng.standard_normal((q, *shape)),
        base + 1e-4 * rng.standard_normal((q, *shape)),
    )


def test_surface_diagnostics_2d_matches_the_individual_helpers():
    """The fused single-gather path must agree exactly with the public helpers."""
    from aero.forces import (
        compute_forces, compute_force_moment_2d, force_profile_2d,
        surface_diagnostics_2d,
    )
    from aero.lbm.d2q9 import compute_feq

    rng = np.random.default_rng(17)
    ny, nx = 40, 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - 20) ** 2 + (xx - 25) ** 2) <= 49
    links, _ = build_surface_links(solid)
    f_pre, f_post = _random_state(rng, (ny, nx), 9, compute_feq, 2)

    diag = surface_diagnostics_2d(
        f_pre, f_post, links, center_x=25.0, center_y=20.0, ny=ny
    )
    fx, fy = compute_forces(f_pre, f_post, links, 1.0, 0.05)
    _, _, mz = compute_force_moment_2d(
        f_pre, f_post, links, center_x=25.0, center_y=20.0
    )
    split = compute_force_split_2d(f_pre, f_post, links)
    profile = force_profile_2d(f_pre, f_post, links, ny=ny)

    assert diag["fx"] == fx and diag["fy"] == fy and diag["mz"] == mz
    assert (diag["fx_p"], diag["fy_p"], diag["fx_v"], diag["fy_v"]) == split
    assert diag["profile"] == profile
    # the split must still reconstruct the total momentum exchange
    assert diag["fx_p"] + diag["fx_v"] == pytest.approx(fx, rel=1e-12, abs=1e-14)


def test_surface_diagnostics_3d_matches_the_individual_helpers():
    from aero.forces3d import (
        compute_forces_3d, compute_force_split_3d, compute_force_moment_3d,
        spanwise_force_profile_3d, surface_diagnostics_3d,
    )
    from aero.lbm.boundary3d import build_surface_links_3d
    from aero.lbm.d3q19 import compute_feq_3d

    rng = np.random.default_rng(23)
    nz, ny, nx = 14, 16, 20
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((zz - 7) ** 2 + (yy - 8) ** 2 + (xx - 8) ** 2) <= 16
    links, _ = build_surface_links_3d(solid)
    f_pre, f_post = _random_state(rng, (nz, ny, nx), 19, compute_feq_3d, 3)

    diag = surface_diagnostics_3d(
        f_pre, f_post, links, center_x=8.0, center_y=8.0, center_z=7.0, nz=nz
    )
    fx, fy, fz = compute_forces_3d(f_pre, f_post, links)
    moments = compute_force_moment_3d(
        f_pre, f_post, links, center_x=8.0, center_y=8.0, center_z=7.0
    )
    split = compute_force_split_3d(f_pre, f_post, links)
    profile = spanwise_force_profile_3d(f_pre, f_post, links, nz=nz)

    assert (diag["fx"], diag["fy"], diag["fz"]) == (fx, fy, fz)
    assert (diag["mx"], diag["my"], diag["mz"]) == moments[3:]
    assert (
        diag["fx_p"], diag["fy_p"], diag["fz_p"],
        diag["fx_v"], diag["fy_v"], diag["fz_v"],
    ) == split
    assert diag["profile"] == profile


def test_surface_diagnostics_handles_an_empty_link_table():
    from aero.forces import surface_diagnostics_2d

    f = np.zeros((9, 6, 6))
    diag = surface_diagnostics_2d(
        f, f, np.empty((0, 3), dtype=np.int32), center_x=0.0, center_y=0.0, ny=6
    )
    assert diag["fx"] == 0.0 and diag["mz"] == 0.0
    assert diag["profile"]["fx"] == [0.0] * 6


# ---------------------------------------------------------------------------
# Reference area — the sphere normalisation bug
# ---------------------------------------------------------------------------

def test_sphere_reference_area_is_pi_r_squared():
    """
    A sphere's frontal area is pi r^2, not D^2.  The old code normalised every
    3D coefficient by D^2, so sphere drag read pi/4 = 0.785 of its true value:
    a 21.5% error that looked like a physics problem and was not.
    """
    import math
    from aero.geometry3d.sphere import Sphere
    from aero.geometry3d.box import Box
    from aero.geometry3d.cylinder3d import Cylinder3D

    assert Sphere(radius=10.0).reference_area() == pytest.approx(math.pi * 100.0)
    assert Box(width=5.0, height=10.0, depth=8.0).reference_area() == 80.0
    assert Cylinder3D(radius=4.0, length=24.0).reference_area() == 192.0


def test_projected_area_of_a_voxel_sphere_matches_pi_r_squared():
    """The measured default agrees with the analytic value to voxel accuracy."""
    import math
    from aero.forces3d import projected_frontal_area
    from aero.geometry3d.sphere import Sphere

    r = 10.0
    solid = Sphere(radius=r, cx_frac=0.5).mark_solid(48, 48, 48)
    area = projected_frontal_area(solid)
    assert area == pytest.approx(math.pi * r * r, rel=0.03)


def test_solver3d_normalises_by_the_frontal_area():
    """
    Same body, same force: the coefficient must scale with 1/area, and the
    default area must be the projected one.
    """
    import math
    from aero.forces3d import projected_frontal_area
    from aero.geometry3d.sphere import Sphere
    from aero.lbm.solver3d import Solver3D

    r = 4.0
    solid = Sphere(radius=r, cx_frac=1 / 3).mark_solid(20, 20, 40)
    a = Solver3D(Nz=20, Ny=20, Nx=40, solid=solid, omega=1.5, u0=0.05, D=2 * r,
                 backend="numpy")
    assert a.ref_area == pytest.approx(projected_frontal_area(solid))
    b = Solver3D(Nz=20, Ny=20, Nx=40, solid=solid, omega=1.5, u0=0.05, D=2 * r,
                 backend="numpy", ref_area=math.pi * r * r)
    ra = a.run(steps=30, check_every=10 ** 9, verbose=False)
    rb = b.run(steps=30, check_every=10 ** 9, verbose=False)
    assert ra["Cd_mean"] * a.ref_area == pytest.approx(rb["Cd_mean"] * b.ref_area, rel=1e-9)
    # the analytic-area coefficient is the measured-area one rescaled by the
    # (voxel / exact) area ratio, and nothing else
    assert rb["Cd_mean"] == pytest.approx(ra["Cd_mean"] * a.ref_area / (math.pi * r * r), rel=1e-9)
