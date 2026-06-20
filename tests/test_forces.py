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
from aero.forces import compute_forces, forces_to_coefficients
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
    links = build_surface_links(solid)
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
    links = build_surface_links(solid)
    f     = feq_uniform

    _, Fy = compute_forces(f, f, links, RHO0, U0)
    # Pure equilibrium → no net lift; small floating point residual expected
    assert abs(Fy) < 1e-6
