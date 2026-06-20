"""Tests for multi-block static 2:1 z-refinement — Item 14."""

import numpy as np
import pytest

from aero.lbm.multiblock import (
    MultiblockSolver3D,
    _omega_fine,
    _upsample_coarse_to_fine,
    _downsample_fine_to_coarse,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_solid(Nz: int, Ny: int, Nx: int) -> np.ndarray:
    return np.zeros((Nz, Ny, Nx), dtype=bool)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_omega_fine_rescaling():
    """Fine omega should give half the kinematic viscosity of coarse."""
    omega_c = 1.5
    omega_f = _omega_fine(omega_c)
    nu_c = (1.0 / omega_c - 0.5) / 3.0
    nu_f = (1.0 / omega_f - 0.5) / 3.0
    np.testing.assert_allclose(nu_f, nu_c / 2.0, rtol=1e-10)


def test_upsample_shape():
    """Upsampling (Q, Nz_c, Ny, Nx) → (Q, 2*Nz_c, Ny, Nx)."""
    Q, Nz_c, Ny, Nx = 19, 5, 8, 10
    f_c = np.random.default_rng(0).standard_normal((Q, Nz_c, Ny, Nx))
    f_f = _upsample_coarse_to_fine(f_c)
    assert f_f.shape == (Q, 2 * Nz_c, Ny, Nx)


def test_upsample_values():
    """Each coarse z-cell should appear twice in the fine array."""
    Q, Nz_c, Ny, Nx = 2, 4, 3, 3
    f_c = np.arange(Q * Nz_c * Ny * Nx, dtype=float).reshape(Q, Nz_c, Ny, Nx)
    f_f = _upsample_coarse_to_fine(f_c)
    for j in range(Nz_c):
        np.testing.assert_array_equal(f_f[:, 2 * j, :, :],     f_c[:, j, :, :])
        np.testing.assert_array_equal(f_f[:, 2 * j + 1, :, :], f_c[:, j, :, :])


def test_downsample_shape():
    """Downsampling (Q, Nz_f, Ny, Nx) → (Q, Nz_f//2, Ny, Nx)."""
    Q, Nz_f, Ny, Nx = 19, 10, 8, 8
    f_f = np.ones((Q, Nz_f, Ny, Nx))
    f_c = _downsample_fine_to_coarse(f_f)
    assert f_c.shape == (Q, Nz_f // 2, Ny, Nx)


def test_upsample_downsample_roundtrip():
    """Downsample(upsample(f_c)) == f_c (averaging two identical cells)."""
    Q, Nz_c, Ny, Nx = 3, 6, 4, 4
    f_c = np.random.default_rng(7).standard_normal((Q, Nz_c, Ny, Nx))
    f_f = _upsample_coarse_to_fine(f_c)
    f_c2 = _downsample_fine_to_coarse(f_f)
    np.testing.assert_allclose(f_c2, f_c, atol=1e-14)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_multiblock_init():
    """MultiblockSolver3D must construct without error and expose .f."""
    Nz, Ny, Nx = 12, 10, 16
    solid = _empty_solid(Nz, Ny, Nx)
    mb = MultiblockSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx,
        solid=solid,
        refine_z_lo=3, refine_z_hi=9,
        omega=1.5, u0=0.05, D=5.0,
        backend="numpy",
    )
    assert mb.f.shape == (19, Nz, Ny, Nx)
    assert mb._fine.f.shape == (19, 12, Ny, Nx)  # 2*(9-3)=12 fine cells


def test_multiblock_runs_steps():
    """MultiblockSolver3D.step() must run without error."""
    Nz, Ny, Nx = 12, 8, 12
    solid = _empty_solid(Nz, Ny, Nx)
    mb = MultiblockSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx,
        solid=solid,
        refine_z_lo=4, refine_z_hi=8,
        omega=1.5, u0=0.05, D=4.0,
        backend="numpy",
    )
    for _ in range(5):
        mb.step()
    assert mb.step_count == 5


def test_multiblock_mass_conservation():
    """
    Total fluid mass (sum of rho over fluid cells) must be conserved to within 1%
    across 20 multiblock steps. Uses pressure BCs at inlet/outlet.
    """
    from aero.lbm.d3q19 import compute_macroscopic_3d

    Nz, Ny, Nx = 10, 10, 16
    solid = _empty_solid(Nz, Ny, Nx)
    mb = MultiblockSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx,
        solid=solid,
        refine_z_lo=3, refine_z_hi=7,
        omega=1.5, u0=0.05, D=4.0,
        backend="numpy",
        outlet_bc="zerogradient",
        streamwise_bc="periodic",
    )
    rho0_total = compute_macroscopic_3d(mb.f)[0].sum()
    for _ in range(20):
        mb.step()
    rho_final = compute_macroscopic_3d(mb.f)[0].sum()
    rel_err = abs(rho_final - rho0_total) / rho0_total
    assert rel_err < 0.01, f"Mass not conserved: rel_err={rel_err:.2%}"


def test_multiblock_run_interface():
    """MultiblockSolver3D.run() must return dict with expected keys."""
    Nz, Ny, Nx = 8, 8, 12
    solid = _empty_solid(Nz, Ny, Nx)
    mb = MultiblockSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx,
        solid=solid,
        refine_z_lo=2, refine_z_hi=6,
        omega=1.5, u0=0.05, D=4.0,
        backend="numpy",
        streamwise_bc="periodic",
    )
    result = mb.run(steps=10, check_every=5, verbose=False)
    assert "Cd_history" in result
    assert result["steps_completed"] == 10
    assert len(result["Cd_history"]) == 2
