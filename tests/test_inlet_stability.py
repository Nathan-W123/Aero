"""
The regularized velocity inlet keeps BGK stable where plain Zou-He did not.

Both cases are the web UI's defaults.  With Zou-He, each diverged from the
inlet -- the flow at x = 1 ran to four times the free stream while the body's
neighbourhood stayed at 1.5 u0 -- in 800 steps (2D rectangle, omega 1.89) and
200 steps (3D sphere, omega 1.92).  A cylinder of the rectangle's size did the
same, so the body shape was never the cause.
"""

import numpy as np
import pytest

from aero.lbm.boundary import apply_inlet_regularized, apply_inlet_zou_he
from aero.lbm.d2q9 import compute_feq, compute_macroscopic
from aero.web import server as web


def _max_speed_ratio(solver, u0):
    m = solver.macroscopic()
    speed = np.sqrt(sum(c * c for c in m[1:]))
    speed[solver.solid] = 0.0
    return float(np.nanmax(speed)) / u0


def test_default_2d_rectangle_runs_on_bgk():
    pytest.importorskip("numba")
    solver, _ = web._build_solver({"mode": "2d", "shape": "rectangle", "width": 40, "height": 20,
                                   "nx": 400, "ny": 200, "re": 100, "u0": 0.05, "collision": "bgk",
                                   "backend": "numba",
                                   "inlet_perturbation": 0.02})
    solver.run(steps=2000, check_every=10 ** 9, verbose=False)
    assert np.isfinite(solver.Cd_history).all()
    assert _max_speed_ratio(solver, 0.05) < 2.0


@pytest.mark.slow
def test_default_sphere_runs_on_bgk():
    pytest.importorskip("numba")
    solver, _ = web._build_solver({"mode": "3d", "shape": "sphere", "radius": 7, "nx": 96, "ny": 48,
                                   "nz": 48, "re": 100, "u0": 0.05, "collision": "bgk",
                                   "backend": "numba"})
    solver.run(steps=800, check_every=10 ** 9, verbose=False)
    assert np.isfinite(solver.Cd_history).all()
    assert _max_speed_ratio(solver, 0.05) < 2.0


def test_regularized_inlet_imposes_the_velocity_and_keeps_zou_he_density():
    ny, nx, u0 = 12, 6, 0.05
    f = compute_feq(np.ones((ny, nx)), np.full((ny, nx), u0), np.zeros((ny, nx)))
    f[:, :, 0] += np.random.default_rng(0).uniform(-1e-3, 1e-3, f[:, :, 0].shape)
    reg, zh = f.copy(), f.copy()
    apply_inlet_regularized(reg, u0)
    apply_inlet_zou_he(zh, u0)
    rho_r, ux_r, uy_r = compute_macroscopic(reg)
    rho_z, _, _ = compute_macroscopic(zh)
    np.testing.assert_allclose(ux_r[:, 0], u0, atol=1e-14)
    np.testing.assert_allclose(uy_r[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(rho_r[:, 0], rho_z[:, 0], atol=1e-14)
    assert np.array_equal(reg[:, :, 1:], f[:, :, 1:])            # only the inlet column changes


def test_regularized_inlet_leaves_an_equilibrium_alone():
    ny, nx, u0 = 8, 4, 0.05
    f = compute_feq(np.full((ny, nx), 1.02), np.full((ny, nx), u0), np.zeros((ny, nx)))
    g = f.copy()
    apply_inlet_regularized(g, u0)
    np.testing.assert_allclose(g, f, atol=1e-15)


@pytest.mark.parametrize("lattice", ["d3q19", "d3q27"])
def test_3d_regularized_inlet_imposes_the_perturbed_velocity(lattice):
    from aero.lbm import boundary3d as b3
    from aero.lbm.lattice3d import compute_feq as feq3, compute_macroscopic as macro3, get_lattice3d

    lat = get_lattice3d(lattice)
    nz, ny, nx, u0 = 5, 6, 4, 0.05
    shape = (nz, ny, nx)
    f = feq3(np.ones(shape), np.full(shape, u0), np.zeros(shape), np.zeros(shape), lat)
    f[:, :, :, 0] += np.random.default_rng(1).uniform(-1e-3, 1e-3, f[:, :, :, 0].shape)
    b3.apply_inlet_zou_he_3d(f, u0, uz_amp=0.02, step=3, lattice=lat)
    _, ux, uy, uz = macro3(f, lat)
    uz_target = (0.02 * u0 * np.sin(2 * np.pi * np.arange(nz) / nz + 0.17 * 3))[:, None]
    np.testing.assert_allclose(ux[:, :, 0], u0, atol=1e-14)
    np.testing.assert_allclose(uy[:, :, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(uz[:, :, 0], np.broadcast_to(uz_target, (nz, ny)), atol=1e-14)
