"""Tests for Thermal LBM — Item 13."""

import numpy as np
import pytest

from aero.lbm.d2q9 import E, W
from aero.lbm.thermal import (
    init_g_2d,
    extract_T,
    apply_temperature_bc_2d,
    guo_buoyancy_force_2d,
    collide_g_2d,
    stream_g_2d,
)


@pytest.fixture
def d2q9_arrays():
    ex = E[:, 0].astype(np.int32)
    ey = E[:, 1].astype(np.int32)
    w  = W.copy()
    return ex, ey, w


# ---------------------------------------------------------------------------


def test_thermal_init_uniform(d2q9_arrays):
    """g summed over Q must equal T0 everywhere."""
    ex, ey, w = d2q9_arrays
    T0 = 0.7
    Ny, Nx = 16, 20
    g = init_g_2d(T0, Ny, Nx, w, ex, ey)
    T = extract_T(g)
    np.testing.assert_allclose(T, T0, rtol=1e-12)


def test_thermal_bc_sets_walls(d2q9_arrays):
    """Temperature BC must set wall rows to the prescribed temperatures."""
    ex, ey, w = d2q9_arrays
    Ny, Nx = 12, 16
    g = init_g_2d(0.5, Ny, Nx, w, ex, ey)
    T_hot, T_cold = 1.0, 0.0
    apply_temperature_bc_2d(g, T_hot, T_cold, w, ex, ey)
    T = extract_T(g)
    np.testing.assert_allclose(T[0, :],  T_hot,  rtol=1e-12)
    np.testing.assert_allclose(T[-1, :], T_cold, rtol=1e-12)


def test_buoyancy_sign(d2q9_arrays):
    """Positive (T - T_ref) with positive g_gravity must produce positive net Fy."""
    ex, ey, w = d2q9_arrays
    Ny, Nx = 10, 10
    rho = np.ones((Ny, Nx))
    T   = np.full((Ny, Nx), 1.0)  # hotter than ref
    T_ref = 0.5
    F = guo_buoyancy_force_2d(rho, T, T_ref, g_gravity=0.001, beta=1e-3,
                              ey=ey.astype(float), w=w)
    # F_y = sum_i F[i] * ey[i]. Since ey is non-zero only for directions ±1,
    # net y-momentum deposited should be > 0 when T > T_ref.
    Fy = sum(float(ey[i]) * float(np.sum(F[i])) for i in range(len(w)))
    assert Fy > 0.0, f"Expected positive buoyancy Fy, got {Fy}"


def test_buoyancy_negative_sign(d2q9_arrays):
    """Negative (T - T_ref) must produce negative Fy."""
    ex, ey, w = d2q9_arrays
    Ny, Nx = 10, 10
    rho = np.ones((Ny, Nx))
    T   = np.full((Ny, Nx), 0.0)  # colder than ref
    T_ref = 0.5
    F = guo_buoyancy_force_2d(rho, T, T_ref, g_gravity=0.001, beta=1e-3,
                              ey=ey.astype(float), w=w)
    Fy = sum(float(ey[i]) * float(np.sum(F[i])) for i in range(len(w)))
    assert Fy < 0.0, f"Expected negative buoyancy Fy, got {Fy}"


def test_thermal_1d_diffusion(d2q9_arrays):
    """
    No-flow thermal diffusion: hot bottom (T=1), cold top (T=0).
    After many steps the interior should be approximately linear in y.
    """
    ex, ey, w = d2q9_arrays
    Ny, Nx = 32, 8
    solid = np.zeros((Ny, Nx), dtype=bool)
    g = init_g_2d(0.5, Ny, Nx, w, ex, ey)
    apply_temperature_bc_2d(g, 1.0, 0.0, w, ex, ey)
    omega_T = 1.0 / (3.0 * 1e-1 + 0.5)  # alpha = 0.1 → fast diffusion
    ux = np.zeros((Ny, Nx))
    uy = np.zeros((Ny, Nx))
    for _ in range(3000):
        g = collide_g_2d(g, ux, uy, extract_T(g), omega_T, solid)
        g = stream_g_2d(g, ex, ey)
        apply_temperature_bc_2d(g, 1.0, 0.0, w, ex, ey)

    T = extract_T(g)
    # Interior should be linear: T[j] ≈ 1 - j/(Ny-1)
    j = np.arange(Ny, dtype=float)
    T_linear = 1.0 - j / (Ny - 1)
    # Check interior (exclude boundary rows)
    interior = slice(1, -1)
    np.testing.assert_allclose(T[interior, 0], T_linear[interior], atol=0.03)


def test_thermal_solver2d_integration():
    """Solver2D with thermal=True must run without error for 10 steps."""
    import numpy as np
    from aero.lbm.solver import Solver

    Ny, Nx = 24, 32
    solid = np.zeros((Ny, Nx), dtype=bool)
    sol = Solver(
        Ny=Ny, Nx=Nx,
        solid=solid,
        omega=1.5, u0=0.0, D=10.0,
        backend="numpy",
        thermal=True, T_hot=1.0, T_cold=0.0, alpha_T=0.1,
    )
    for _ in range(10):
        sol._step()
    T = extract_T(sol.g)
    # Temperature must remain bounded between T_cold and T_hot
    assert float(T.min()) >= -0.1
    assert float(T.max()) <=  1.1


def test_thermal_solver2d_buoyancy():
    """With buoyancy enabled, vertical velocity must become nonzero (natural convection)."""
    from aero.lbm.solver import Solver
    from aero.lbm.d2q9 import compute_macroscopic

    Ny, Nx = 24, 24
    solid = np.zeros((Ny, Nx), dtype=bool)
    sol = Solver(
        Ny=Ny, Nx=Nx,
        solid=solid,
        omega=1.5, u0=0.0, D=10.0,
        backend="numpy", inlet_bc="pressure", outlet_bc="pressure",
        thermal=True, T_hot=1.0, T_cold=0.0, alpha_T=0.01,
        buoyancy=True, g_gravity=1e-4, beta=0.1, T_ref=0.5,
    )
    for _ in range(200):
        sol._step()
    _, _, uy = compute_macroscopic(sol.f)
    assert float(np.max(np.abs(uy))) > 1e-10, "Buoyancy should drive nonzero vertical velocity"
