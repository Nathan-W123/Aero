"""Tests for the regularized collision operator."""

import numpy as np
import pytest

from aero.lbm.d2q9 import E, W, compute_feq
from aero.lbm.kernels_reg import CS2, HERMITE2
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D


def _cylinder(ny, nx, r):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((yy - ny / 2) ** 2 + (xx - nx / 4) ** 2) <= r * r


# ---------------------------------------------------------------------------
# The Hermite projection
# ---------------------------------------------------------------------------

def test_projection_reproduces_the_second_moment():
    """
    The reconstruction must be a projection: feeding it a Pi and reading the
    second moment back out has to return the same Pi.
    """
    rng = np.random.default_rng(0)
    fneq = 1e-4 * rng.standard_normal(9)
    ex = E[:, 0].astype(float)
    ey = E[:, 1].astype(float)
    pxx = float((ex * ex * fneq).sum())
    pxy = float((ex * ey * fneq).sum())
    pyy = float((ey * ey * fneq).sum())

    f1 = np.array([
        HERMITE2 * W[i] * (
            (ex[i] ** 2 - CS2) * pxx
            + 2.0 * ex[i] * ey[i] * pxy
            + (ey[i] ** 2 - CS2) * pyy
        )
        for i in range(9)
    ])
    assert float((ex * ex * f1).sum()) == pytest.approx(pxx, rel=1e-12)
    assert float((ex * ey * f1).sum()) == pytest.approx(pxy, rel=1e-12)
    assert float((ey * ey * f1).sum()) == pytest.approx(pyy, rel=1e-12)


def test_projection_carries_no_mass_or_momentum():
    """f^(1) lives entirely in the second-order subspace."""
    rng = np.random.default_rng(1)
    fneq = 1e-4 * rng.standard_normal(9)
    ex = E[:, 0].astype(float)
    ey = E[:, 1].astype(float)
    pxx = float((ex * ex * fneq).sum())
    pxy = float((ex * ey * fneq).sum())
    pyy = float((ey * ey * fneq).sum())
    f1 = np.array([
        HERMITE2 * W[i] * (
            (ex[i] ** 2 - CS2) * pxx + 2.0 * ex[i] * ey[i] * pxy
            + (ey[i] ** 2 - CS2) * pyy
        )
        for i in range(9)
    ])
    assert float(f1.sum()) == pytest.approx(0.0, abs=1e-18)
    assert float((ex * f1).sum()) == pytest.approx(0.0, abs=1e-18)
    assert float((ey * f1).sum()) == pytest.approx(0.0, abs=1e-18)


# ---------------------------------------------------------------------------
# Accuracy — must match BGK's viscosity exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega", [1.2, 1.7])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_regularized_reproduces_exact_poiseuille(omega, backend):
    """
    Regularization removes ghost moments only; the hydrodynamics must be
    unchanged, so the force-driven channel curvature stays exact.

    This is also what catches the forcing prefactor: regularization projects
    away the first moment of f_neq that BGK relies on, so the Guo source
    carries 1/2 here instead of (1 - omega/2).  With the BGK prefactor the
    momentum gain is (3/2 - omega/2) F — 10% low at omega=1.2, 35% at 1.7.
    """
    if backend == "numba":
        pytest.importorskip("numba")
    ny, nx = 33, 6
    accel = 1e-6
    nu = (1.0 / omega - 0.5) / 3.0
    solver = Solver(
        Ny=ny, Nx=nx, solid=np.zeros((ny, nx), dtype=bool), omega=omega,
        u0=0.0, D=float(ny), backend=backend, collision="regularized",
        wall_bc="noslip", inlet_bc="none", outlet_bc="none", body_force_x=accel,
    )
    solver.run(steps=25000, check_every=10 ** 9, verbose=False)
    _, ux, _ = solver.macroscopic()
    y = np.arange(ny, dtype=float)
    c2, _, _ = np.polyfit(y[2:-2], ux[2:-2, nx // 2], 2)
    assert 2.0 * c2 == pytest.approx(-accel / nu, rel=2e-3)


def test_regularized_matches_bgk_on_a_resolved_flow():
    """Where BGK is well resolved the two should agree closely."""
    pytest.importorskip("numba")
    ny, nx = 40, 80
    common = dict(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 5.0), omega=1.5, u0=0.05,
        D=10.0, backend="numba",
    )
    a = Solver(collision="bgk", **common).run(steps=500, check_every=10 ** 9, verbose=False)
    b = Solver(collision="regularized", **common).run(steps=500, check_every=10 ** 9, verbose=False)
    assert b["Cd_mean"] == pytest.approx(a["Cd_mean"], rel=0.05)


@pytest.mark.parametrize("collision", ["regularized"])
def test_numpy_and_numba_agree(collision):
    pytest.importorskip("numba")
    ny, nx = 24, 48
    common = dict(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 4.0), omega=1.6, u0=0.05,
        D=8.0, collision=collision,
    )
    a = Solver(backend="numpy", **common)
    b = Solver(backend="numba", **common)
    a.run(steps=60, check_every=10 ** 9, verbose=False)
    b.run(steps=60, check_every=10 ** 9, verbose=False)
    np.testing.assert_allclose(a.macroscopic()[1], b.macroscopic()[1], rtol=1e-10)


# ---------------------------------------------------------------------------
# Stability — the reason the operator exists
# ---------------------------------------------------------------------------

def test_regularized_outlasts_bgk_at_low_viscosity():
    """
    At omega = 1.95 the ghost moments are barely damped and BGK blows up;
    regularization removes them, which is the whole point.
    """
    pytest.importorskip("numba")
    ny, nx = 64, 128
    solid = _cylinder(ny, nx, 6.0)
    common = dict(
        Ny=ny, Nx=nx, solid=solid, omega=1.95, u0=0.09, D=12.0,
        backend="numba", inlet_perturbation=0.05,
    )

    def survives(collision):
        try:
            solver = Solver(collision=collision, **common)
            solver.run(steps=2000, check_every=2000, verbose=False)
            return bool(np.all(np.isfinite(solver.f)))
        except RuntimeError:
            return False

    assert not survives("bgk"), "BGK unexpectedly stable — pick a harder case"
    assert survives("regularized")


def test_solver_3d_accepts_regularized():
    nz, ny, nx = 8, 10, 16
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((zz - 4) ** 2 + (yy - 5) ** 2 + (xx - 4) ** 2) <= 4
    result = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=1.6, u0=0.05, D=4.0,
        backend="numpy", collision="regularized",
    ).run(steps=20, check_every=10 ** 9, verbose=False)
    assert np.isfinite(result["Cd_mean"])


def test_3d_numpy_and_numba_agree():
    pytest.importorskip("numba")
    nz, ny, nx = 8, 10, 16
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((zz - 4) ** 2 + (yy - 5) ** 2 + (xx - 4) ** 2) <= 4
    common = dict(
        Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=1.6, u0=0.05, D=4.0,
        collision="regularized",
    )
    a = Solver3D(backend="numpy", **common)
    b = Solver3D(backend="numba", **common)
    a.run(steps=25, check_every=10 ** 9, verbose=False)
    b.run(steps=25, check_every=10 ** 9, verbose=False)
    np.testing.assert_allclose(a.macroscopic()[1], b.macroscopic()[1], rtol=1e-10)


def test_3d_regularized_poiseuille():
    pytest.importorskip("numba")
    nz, ny, nx = 4, 33, 6
    accel, omega = 1e-6, 1.5
    nu = (1.0 / omega - 0.5) / 3.0
    solver = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=np.zeros((nz, ny, nx), dtype=bool),
        omega=omega, u0=0.0, D=float(ny), backend="numba",
        collision="regularized", wall_bc="noslip",
        streamwise_bc="periodic", body_force_x=accel,
    )
    solver.run(steps=25000, check_every=10 ** 9, verbose=False)
    _, ux, _, _ = solver.macroscopic()
    y = np.arange(ny, dtype=float)
    c2, _, _ = np.polyfit(y[2:-2], ux[nz // 2, 2:-2, nx // 2], 2)
    assert 2.0 * c2 == pytest.approx(-accel / nu, rel=2e-3)


def test_unknown_collision_is_rejected():
    with pytest.raises(ValueError, match="regularized"):
        Solver(
            Ny=8, Nx=8, solid=np.zeros((8, 8), dtype=bool), omega=1.5,
            u0=0.05, D=4.0, collision="nonsense",
        )
