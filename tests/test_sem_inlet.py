"""Tests for the Synthetic Eddy Method (SEM) inlet — Item 16."""

import numpy as np
import pytest

from aero.lbm.sem_inlet import SEMInlet


def _make_sem(Ny=32, Nz=32, u0=0.05, Tu=0.1, L_int=8.0, N=300, seed=42):
    return SEMInlet(Ny=Ny, Nz=Nz, u0=u0, Tu=Tu, L_int=L_int, N_eddies=N, rng_seed=seed)


# ---------------------------------------------------------------------------

def test_sem_zero_Tu():
    """Tu=0 must produce exactly zero fluctuations."""
    sem = SEMInlet(Ny=16, Nz=16, u0=0.05, Tu=0.0, L_int=5.0, N_eddies=50)
    du, dv, dw = sem.fluctuation()
    np.testing.assert_array_equal(du, 0.0)
    np.testing.assert_array_equal(dv, 0.0)
    np.testing.assert_array_equal(dw, 0.0)


def test_sem_output_shape():
    """fluctuation() must return three (Nz, Ny) arrays."""
    sem = _make_sem(Ny=24, Nz=32)
    du, dv, dw = sem.fluctuation()
    assert du.shape == (32, 24)
    assert dv.shape == (32, 24)
    assert dw.shape == (32, 24)


def test_sem_energy_level():
    """RMS of each fluctuation component must be close to Tu*u0 (within 25%)."""
    u0, Tu = 0.05, 0.1
    sem = _make_sem(Ny=48, Nz=48, u0=u0, Tu=Tu, L_int=6.0, N=400, seed=7)
    du, dv, dw = sem.fluctuation()
    target = Tu * u0
    for arr, name in [(du, "u"), (dv, "v"), (dw, "w")]:
        rms = float(np.std(arr))
        assert abs(rms - target) / target < 0.25, (
            f"Component {name}: RMS={rms:.5f}, target={target:.5f}"
        )


def test_sem_divergence_free():
    """
    (v_prime, w_prime) must satisfy ∂v/∂y + ∂w/∂z ≈ 0 at the inlet plane.

    Checked via L2 ratio: ||ky*V + kz*W||_2 / (||V||_2 + ||W||_2)
    which correctly weights by amplitude and should be near machine precision.
    """
    sem = _make_sem(Ny=32, Nz=32, u0=0.05, Tu=0.1, L_int=8.0, N=300, seed=99)
    _, v, w = sem.fluctuation()

    Nz = Ny = 32
    v_hat = np.fft.fft2(v)
    w_hat = np.fft.fft2(w)
    # Skip Nyquist rows (they were zeroed in the projection; exclude from check)
    kz = (np.fft.fftfreq(Nz) * 2.0 * np.pi)[:, np.newaxis]
    ky = (np.fft.fftfreq(Ny) * 2.0 * np.pi)[np.newaxis, :]
    nyq_mask = np.ones((Nz, Ny), dtype=bool)
    nyq_mask[Nz // 2, :] = False
    nyq_mask[:, Ny // 2] = False
    div_sq = np.abs(ky * v_hat + kz * w_hat) ** 2
    vel_sq = np.abs(v_hat) ** 2 + np.abs(w_hat) ** 2
    L2_div = float(np.sqrt(np.sum(div_sq[nyq_mask])))
    L2_vel = float(np.sqrt(np.sum(vel_sq[nyq_mask]))) + 1e-30
    rel_div = L2_div / L2_vel
    assert rel_div < 1e-10, f"L2 divergence ratio = {rel_div:.2e} (expected < 1e-10)"


def test_sem_eddies_advance():
    """After N steps, eddy x-positions must have advanced by N * u0."""
    u0 = 0.05
    sem = _make_sem(Ny=32, Nz=32, u0=u0, Tu=0.05, L_int=5.0, N=10, seed=1)
    x0 = sem.xk.copy()
    steps = 20
    for _ in range(steps):
        sem.step(dt=1.0)
    # Eddies that didn't wrap: x should have advanced by steps * u0
    expected_advance = steps * u0
    # Check at least one eddy hasn't wrapped (for small N and small advance)
    # Use modular arithmetic: at least some should satisfy the advance condition
    # (eddies wrap at x = 3*L_int = 15, total advance = 1.0 < 15)
    moved = sem.xk - x0
    # All eddies that haven't wrapped should have moved exactly steps*u0
    # Those that wrapped have xk reset to -r; detect wraps by large negative jump
    no_wrap = (sem.xk - x0) > -1.0
    if no_wrap.any():
        np.testing.assert_allclose(
            (sem.xk - x0)[no_wrap], expected_advance, atol=1e-12
        )


def test_sem_solver3d_integration():
    """Solver3D with sem_inlet=True must run without error for 5 steps."""
    from aero.geometry3d.sphere import Sphere
    from aero.lbm.solver3d import Solver3D

    Nz, Ny, Nx = 16, 16, 32
    geom = Sphere(radius=3, cx_frac=0.3, cy_frac=0.5, cz_frac=0.5)
    solid = geom.mark_solid(Nz, Ny, Nx)
    sol = Solver3D(
        Nz=Nz, Ny=Ny, Nx=Nx,
        solid=solid,
        omega=1.2, u0=0.05, D=6.0,
        backend="numpy",
        sem_inlet=True, sem_Tu=0.05, sem_L_int=4.0, sem_N=30,
    )
    for _ in range(5):
        sol._step()
