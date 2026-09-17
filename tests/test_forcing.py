"""
Validation of the Guo forcing scheme.

Force-driven plane Poiseuille flow is the right acceptance test here: it has
an exact steady solution whose curvature depends only on the applied force and
the viscosity,

    d2u/dy2 = -a / nu

so it pins the force magnitude independently of where the bounce-back wall
effectively sits.  Every way of getting the scheme wrong — dropping the
``1/cs^2`` factor, omitting the half-force velocity correction, applying the
source outside collision — shows up as a wrong curvature.
"""

import numpy as np
import pytest

from aero.lbm.d2q9 import E, W, compute_feq
from aero.lbm.forcing import (
    FORCE_NONE,
    guo_source_term,
    half_force_velocity,
    ibm_acceleration,
    resolve_force_mode,
    uniform_acceleration,
)
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D


def _channel_curvature(u: np.ndarray) -> float:
    """Fit a parabola to the interior of a channel profile; return d2u/dy2."""
    y = np.arange(u.size, dtype=float)
    c2, _, _ = np.polyfit(y[2:-2], u[2:-2], 2)
    return 2.0 * c2


@pytest.mark.parametrize("collision", ["bgk", "trt", "mrt"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("omega", [1.2, 1.7])
def test_body_force_drives_exact_poiseuille_2d(collision, backend, omega):
    """The applied force must reproduce d2u/dy2 = -a/nu exactly."""
    if backend == "numba":
        pytest.importorskip("numba")
    ny, nx = 33, 6
    accel = 1e-6
    nu = (1.0 / omega - 0.5) / 3.0

    solver = Solver(
        Ny=ny, Nx=nx, solid=np.zeros((ny, nx), dtype=bool), omega=omega,
        u0=0.0, D=float(ny), backend=backend, collision=collision,
        wall_bc="noslip", inlet_bc="none", outlet_bc="none",
        body_force_x=accel,
    )
    solver.run(steps=30000, check_every=100000, verbose=False)
    _, ux, _ = solver.macroscopic()

    curvature = _channel_curvature(ux[:, nx // 2])
    assert curvature == pytest.approx(-accel / nu, rel=2e-3)


def test_body_force_drives_exact_poiseuille_3d():
    pytest.importorskip("numba")
    nz, ny, nx = 4, 33, 6
    accel = 1e-6
    omega = 1.5
    nu = (1.0 / omega - 0.5) / 3.0

    solver = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=np.zeros((nz, ny, nx), dtype=bool),
        omega=omega, u0=0.0, D=float(ny), backend="numba", collision="bgk",
        wall_bc="noslip", streamwise_bc="periodic", body_force_x=accel,
    )
    solver.run(steps=30000, check_every=100000, verbose=False)
    _, ux, _, _ = solver.macroscopic()

    curvature = _channel_curvature(ux[nz // 2, :, nx // 2])
    assert curvature == pytest.approx(-accel / nu, rel=2e-3)


def test_poiseuille_profile_is_parabolic():
    """A residual above round-off would mean the source term is y-dependent."""
    pytest.importorskip("numba")
    ny, nx = 33, 6
    solver = Solver(
        Ny=ny, Nx=nx, solid=np.zeros((ny, nx), dtype=bool), omega=1.2,
        u0=0.0, D=float(ny), backend="numba", wall_bc="noslip",
        inlet_bc="none", outlet_bc="none", body_force_x=1e-6,
    )
    solver.run(steps=30000, check_every=100000, verbose=False)
    _, ux, _ = solver.macroscopic()
    u = ux[:, nx // 2]
    y = np.arange(ny, dtype=float)
    fit = np.polyval(np.polyfit(y[2:-2], u[2:-2], 2), y[2:-2])
    assert np.max(np.abs(u[2:-2] - fit)) / u.max() < 1e-6


@pytest.mark.parametrize("omega", [1.0, 1.5, 1.9])
def test_guo_source_injects_the_documented_fraction(omega):
    """
    ``sum_i e_i S_i = (1 - omega/2) F`` — the rest of the force arrives
    through the F/2 in the velocity.  Dropping the 1/cs^2 factor makes this
    ratio cs^2 = 1/3 of what it should be.
    """
    shape = (3, 3)
    rho = np.ones(shape)
    u = (np.zeros(shape), np.zeros(shape))
    acc = (np.full(shape, 1e-3), np.zeros(shape))
    source = guo_source_term(rho, u, acc, omega, E, W)
    injected = np.einsum("i,iyx->yx", E[:, 0].astype(float), source)
    assert injected[1, 1] == pytest.approx((1.0 - 0.5 * omega) * rho[1, 1] * acc[0][1, 1])


def test_half_force_velocity_matches_the_definition():
    rho = np.full((2, 2), 1.2)
    momentum = (np.full((2, 2), 0.06), np.zeros((2, 2)))
    acc = (np.full((2, 2), 0.01), np.full((2, 2), -0.02))
    ux, uy = half_force_velocity(momentum, rho, acc)
    assert ux == pytest.approx(0.06 / 1.2 + 0.005)
    assert uy == pytest.approx(-0.01)


def test_ibm_direct_forcing_reaches_the_wall_velocity():
    """
    Direct forcing must put the *corrected* velocity on the wall velocity.

    a = 2 (u_wall - u_raw), so u_raw + a/2 = u_wall exactly.
    """
    ny, nx = 5, 5
    rho = np.ones((ny, nx))
    f = np.ascontiguousarray(compute_feq(rho, np.zeros((ny, nx)), np.zeros((ny, nx))))
    momentum = (
        np.einsum("i,iyx->yx", E[:, 0].astype(float), f),
        np.einsum("i,iyx->yx", E[:, 1].astype(float), f),
    )
    u_wall = (np.full((ny, nx), 0.05), np.zeros((ny, nx)))
    phi = np.full((ny, nx), 1.0)
    solid = np.zeros((ny, nx), dtype=bool)

    acc = ibm_acceleration(rho, momentum, u_wall, phi, solid)
    ux, _ = half_force_velocity(momentum, rho, (acc[0], acc[1]))
    assert ux[2, 2] == pytest.approx(0.05)


def test_ibm_acceleration_is_confined_to_the_band():
    ny, nx = 4, 4
    rho = np.ones((ny, nx))
    momentum = (np.zeros((ny, nx)), np.zeros((ny, nx)))
    u_wall = (np.full((ny, nx), 0.05), np.zeros((ny, nx)))
    phi = np.array([[-1.0, 0.5, 1.4, 3.0]] * ny)      # solid / band / band / free
    solid = np.zeros((ny, nx), dtype=bool)
    acc = ibm_acceleration(rho, momentum, u_wall, phi, solid)
    assert np.all(acc[0][:, 0] == 0.0)                # phi <= 0
    assert np.all(acc[0][:, 1] > 0.0)
    assert np.all(acc[0][:, 2] > 0.0)
    assert np.all(acc[0][:, 3] == 0.0)                # phi > band


def test_no_force_costs_nothing():
    """With no force the solver must take the FORCE_NONE branch."""
    assert resolve_force_mode(uniform_acceleration(0.0, 0.0), None) == FORCE_NONE


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_unforced_run_is_unaffected_by_the_forcing_machinery(backend):
    """A case with no force must match a solver built without force support."""
    if backend == "numba":
        pytest.importorskip("numba")
    ny, nx = 24, 48
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - 12) ** 2 + (xx - 12) ** 2) <= 16
    common = dict(
        Ny=ny, Nx=nx, solid=solid, omega=1.5, u0=0.05, D=8.0, backend=backend,
    )
    a = Solver(**common).run(steps=40, check_every=100, verbose=False)
    b = Solver(body_force_x=0.0, body_force_y=0.0, **common).run(
        steps=40, check_every=100, verbose=False
    )
    assert a["Cd_mean"] == b["Cd_mean"]


def test_numpy_backend_does_not_use_the_jit_kernels(monkeypatch):
    """
    `backend="numpy"` must stay on the NumPy path even where Numba is present.

    Otherwise the flag is a lie: a user debugging a JIT problem, or needing a
    run that does not depend on the JIT, silently gets the JIT anyway.
    """
    pytest.importorskip("numba")
    from aero.lbm import kernels as k

    assert k.collision_kernel is not k.collision_kernel_numpy
    assert k.stream_kernel is not k.stream_kernel_numpy

    called = []
    monkeypatch.setattr(
        k, "collision_kernel",
        lambda *a, **kw: called.append("jit-collision"),
    )
    monkeypatch.setattr(
        k, "stream_kernel",
        lambda *a, **kw: called.append("jit-stream"),
    )

    ny, nx = 12, 16
    solver = Solver(
        Ny=ny, Nx=nx, solid=np.zeros((ny, nx), dtype=bool), omega=1.5,
        u0=0.05, D=4.0, backend="numpy",
    )
    solver.run(steps=3, check_every=100, verbose=False)
    assert called == []


def test_numpy_and_numba_backends_agree_under_forcing():
    pytest.importorskip("numba")
    ny, nx = 21, 8
    common = dict(
        Ny=ny, Nx=nx, solid=np.zeros((ny, nx), dtype=bool), omega=1.4,
        u0=0.0, D=float(ny), wall_bc="noslip", inlet_bc="none",
        outlet_bc="none", body_force_x=2e-6, body_force_y=1e-6,
    )
    a = Solver(backend="numpy", **common)
    b = Solver(backend="numba", **common)
    a.run(steps=200, check_every=1000, verbose=False)
    b.run(steps=200, check_every=1000, verbose=False)
    np.testing.assert_allclose(a.macroscopic()[1], b.macroscopic()[1], rtol=1e-11)
