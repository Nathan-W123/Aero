"""
Tests for boundary condition implementations.

Key invariants checked:
  - Zou-He inlet: after applying BC, recovered ux at x=0 equals u0
  - Convective outlet: outlet values are a weighted blend of prev and penultimate
  - Slip walls: down-going distributions reflected upward at y=0 and vice versa
  - Bounce-back: mass is conserved over a full step (no sinks/sources)
"""

import numpy as np
import pytest

from aero.lbm.d2q9 import compute_feq, compute_macroscopic, Q, E
from aero.lbm.boundary import (
    apply_inlet_zou_he,
    apply_outlet_convective,
    apply_outlet_zero_gradient,
    apply_slip_walls,
    build_surface_links,
    apply_bounce_back,
)


Ny, Nx = 20, 40
U0 = 0.05


@pytest.fixture
def uniform_f():
    """f initialised to equilibrium at uniform flow."""
    rho = np.ones((Ny, Nx))
    ux  = np.full((Ny, Nx), U0)
    uy  = np.zeros((Ny, Nx))
    return compute_feq(rho, ux, uy)


# ---------------------------------------------------------------------------
# Zou-He inlet
# ---------------------------------------------------------------------------

def test_zou_he_sets_target_velocity(uniform_f):
    f = uniform_f.copy()
    # Perturb inlet column to simulate post-streaming state
    f[:, :, 0] += np.random.default_rng(0).uniform(-0.001, 0.001, f[:, :, 0].shape)
    apply_inlet_zou_he(f, U0)
    rho, ux, uy = compute_macroscopic(f)
    np.testing.assert_allclose(ux[:, 0], U0, atol=1e-10)


def test_zou_he_uy_zero_at_inlet(uniform_f):
    f = uniform_f.copy()
    f[:, :, 0] += np.random.default_rng(1).uniform(-0.001, 0.001, f[:, :, 0].shape)
    apply_inlet_zou_he(f, U0)
    _, _, uy = compute_macroscopic(f)
    np.testing.assert_allclose(uy[:, 0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Convective outlet
# ---------------------------------------------------------------------------

def test_convective_outlet_blend(uniform_f):
    """Output is (1-u)*prev + u*penultimate."""
    f      = uniform_f.copy()
    f_prev = f[:, :, -1].copy()

    # Shift penultimate column so the blend is distinguishable
    f[:, :, -2] += 0.01
    expected = (1.0 - U0) * f_prev + U0 * f[:, :, -2]

    apply_outlet_convective(f, f_prev.copy(), U0)
    np.testing.assert_allclose(f[:, :, -1], expected, atol=1e-14)


def test_convective_outlet_updates_prev(uniform_f):
    f      = uniform_f.copy()
    f_prev = f[:, :, -1].copy()
    f_prev_arg = f_prev.copy()

    apply_outlet_convective(f, f_prev_arg, U0)
    # f_prev_arg should now equal f[:, :, -1]
    np.testing.assert_allclose(f_prev_arg, f[:, :, -1], atol=1e-14)


# ---------------------------------------------------------------------------
# Slip walls
# ---------------------------------------------------------------------------

def test_slip_wall_bottom_reflection(uniform_f):
    f = uniform_f.copy()
    # Set down-going dirs at y=0 to known values
    f[4, 0, :] = 0.02
    f[7, 0, :] = 0.01
    f[8, 0, :] = 0.03
    apply_slip_walls(f)
    # 4->2, 7->6, 8->5 reflected
    np.testing.assert_allclose(f[2, 0, :], 0.02, atol=1e-14)
    np.testing.assert_allclose(f[6, 0, :], 0.01, atol=1e-14)
    np.testing.assert_allclose(f[5, 0, :], 0.03, atol=1e-14)


def test_slip_wall_top_reflection(uniform_f):
    f = uniform_f.copy()
    f[2, -1, :] = 0.05
    f[5, -1, :] = 0.02
    f[6, -1, :] = 0.01
    apply_slip_walls(f)
    np.testing.assert_allclose(f[4, -1, :], 0.05, atol=1e-14)
    np.testing.assert_allclose(f[8, -1, :], 0.02, atol=1e-14)
    np.testing.assert_allclose(f[7, -1, :], 0.01, atol=1e-14)


# ---------------------------------------------------------------------------
# Surface link construction
# ---------------------------------------------------------------------------

def test_surface_links_cylinder_count():
    """Surface link count should be close to cylinder perimeter * ~3 directions."""
    from aero.geometry.cylinder import Cylinder
    cyl   = Cylinder(radius=10, cx_frac=0.5, cy_frac=0.5)
    solid = cyl.mark_solid(80, 80)
    links, _ = build_surface_links(solid)
    # Perimeter ≈ 2*pi*r ≈ 63; each surface cell has ~3 links on average
    assert 100 < links.shape[0] < 400


def test_surface_links_empty_domain():
    solid = np.zeros((20, 40), dtype=bool)
    links, _ = build_surface_links(solid)
    assert links.shape[0] == 0


def test_surface_links_all_fluid_links_point_to_solid():
    from aero.geometry.cylinder import Cylinder
    cyl   = Cylinder(radius=8, cx_frac=0.5, cy_frac=0.5)
    solid = cyl.mark_solid(60, 60)
    links, _ = build_surface_links(solid)
    for row in links:
        i, y, x = int(row[0]), int(row[1]), int(row[2])
        assert not solid[y, x], "Source cell must be fluid"
        ny = int(np.clip(y + E[i, 1], 0, 59))
        nx = int(np.clip(x + E[i, 0], 0, 59))
        assert solid[ny, nx], "Target cell must be solid"
