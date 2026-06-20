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
