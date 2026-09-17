"""Tests for multi-block static 2:1 refinement."""

import numpy as np
import pytest

from aero.lbm.multiblock import (
    MultiblockSolver3D,
    REFINEMENT,
    _omega_fine,
    _refine_plane,
    _restrict_block,
    _upsample_coarse_to_fine,
    _downsample_fine_to_coarse,
    neq_scale_coarse_to_fine,
    rescale_for_fine_grid,
    rescale_nonequilibrium,
    split_equilibrium,
)
from aero.lbm.d3q19 import compute_feq_3d, compute_macroscopic_3d


def _empty_solid(Nz: int, Ny: int, Nx: int) -> np.ndarray:
    return np.zeros((Nz, Ny, Nx), dtype=bool)


# ---------------------------------------------------------------------------
# Scaling relations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega_c", [1.2, 1.5, 1.8])
def test_fine_grid_keeps_the_reynolds_number(omega_c):
    """
    The refined block must run at the same Reynolds number as the coarse grid.

    Re = u L / nu in lattice units, with u invariant and L doubled, so nu has
    to double.  Halving it instead — which an earlier version did — puts the
    fine block at n**2 = 4x the intended Reynolds number.
    """
    omega_f = _omega_fine(omega_c)
    nu_c = (1.0 / omega_c - 0.5) / 3.0
    nu_f = (1.0 / omega_f - 0.5) / 3.0
    assert nu_f == pytest.approx(REFINEMENT * nu_c)

    u, L = 0.05, 20.0
    re_coarse = u * L / nu_c
    re_fine = u * (REFINEMENT * L) / nu_f
    assert re_fine == pytest.approx(re_coarse)


@pytest.mark.parametrize("omega_c", [1.2, 1.5, 1.8])
def test_filippova_hanel_tau_relation(omega_c):
    """tau_f = 1/2 + n (tau_c - 1/2)."""
    tau_c = 1.0 / omega_c
    tau_f = 1.0 / _omega_fine(omega_c)
    assert tau_f == pytest.approx(0.5 + REFINEMENT * (tau_c - 0.5))
    assert tau_f > 0.5


def test_fine_grid_argument_rescaling():
    """Lengths grow, accelerations shrink, dimensionless settings pass through."""
    kw = dict(
        omega=1.5, u0=0.05, D=10.0, sponge_thickness=8, alpha_T=1e-3,
        body_force_x=2e-6, g_gravity=1e-5, les_cs=0.16, wall_bc="noslip",
    )
    fine = rescale_for_fine_grid(kw)
    assert fine["u0"] == 0.05                      # velocity is invariant
    assert fine["D"] == 20.0                       # cell-counted length doubles
    assert fine["sponge_thickness"] == 16
    assert fine["alpha_T"] == pytest.approx(2e-3)  # diffusivity scales like nu
    assert fine["body_force_x"] == pytest.approx(1e-6)   # acceleration halves
    assert fine["g_gravity"] == pytest.approx(5e-6)
    assert fine["les_cs"] == 0.16                  # dimensionless
    assert fine["wall_bc"] == "noslip"
    assert fine["omega"] == pytest.approx(_omega_fine(1.5))


def test_neq_scale_matches_the_tau_ratio():
    omega_c = 1.5
    omega_f = _omega_fine(omega_c)
    factor = neq_scale_coarse_to_fine(omega_c, omega_f)
    assert factor == pytest.approx((1.0 / omega_f) / (2.0 / omega_c))
    assert 0.0 < factor < 1.0


# ---------------------------------------------------------------------------
# Transfer operators
# ---------------------------------------------------------------------------

def test_refine_plane_shape_and_smoothness():
    rng = np.random.default_rng(0)
    q, ny, nx = 19, 8, 10
    yy, xx = np.mgrid[0:ny, 0:nx]
    plane = np.stack([np.sin(yy / 3.0) * np.cos(xx / 4.0) + 2.0] * q)
    fine = _refine_plane(plane)
    assert fine.shape == (q, 2 * ny, 2 * nx)
    # a bilinear upsample never overshoots the data it came from
    assert fine.min() >= plane.min() - 1e-12
    assert fine.max() <= plane.max() + 1e-12


def test_refine_plane_is_exact_on_a_linear_field():
    """Bilinear interpolation must reproduce a linear field exactly."""
    q, ny, nx = 3, 6, 8
    yy, xx = np.mgrid[0:ny, 0:nx]
    plane = np.stack([2.0 * yy - 3.0 * xx + 1.0] * q).astype(float)
    fine = _refine_plane(plane)
    fy = -0.25 + 0.5 * np.arange(2 * ny)
    fx = -0.25 + 0.5 * np.arange(2 * nx)
    # exact in the interior; edges are clamped rather than extrapolated
    expect = 2.0 * fy[:, None] - 3.0 * fx[None, :] + 1.0
    np.testing.assert_allclose(fine[0][1:-1, 1:-1], expect[1:-1, 1:-1], atol=1e-12)


def test_restrict_block_averages_each_cell_group():
    q = 2
    fine = np.arange(q * 4 * 4 * 4, dtype=float).reshape(q, 4, 4, 4)
    coarse = _restrict_block(fine)
    assert coarse.shape == (q, 2, 2, 2)
    assert coarse[0, 0, 0, 0] == pytest.approx(fine[0, :2, :2, :2].mean())
    assert coarse[1, 1, 1, 1] == pytest.approx(fine[1, 2:, 2:, 2:].mean())


def test_restrict_preserves_a_uniform_field():
    fine = np.full((19, 4, 4, 4), 0.37)
    np.testing.assert_allclose(_restrict_block(fine), 0.37)


def test_nonequilibrium_rescaling_leaves_the_moments_alone():
    """Only the non-equilibrium part may change; rho and u must not."""
    rng = np.random.default_rng(3)
    shape = (4, 5, 6)
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    u = [0.03 * rng.standard_normal(shape) for _ in range(3)]
    f = compute_feq_3d(rho, *u) + 1e-4 * rng.standard_normal((19,) + shape)

    scaled = rescale_nonequilibrium(f, 0.625)
    r0, *u0 = compute_macroscopic_3d(f)
    r1, *u1 = compute_macroscopic_3d(scaled)
    np.testing.assert_allclose(r1, r0, atol=1e-14)
    for a, b in zip(u1, u0):
        np.testing.assert_allclose(b, a, atol=1e-13)

    _, neq0 = split_equilibrium(f)
    _, neq1 = split_equilibrium(scaled)
    np.testing.assert_allclose(neq1, 0.625 * neq0, atol=1e-13)


def test_legacy_z_only_helpers_still_work():
    """The old z-only helpers are kept for callers that used them."""
    q, nz, ny, nx = 19, 5, 8, 10
    f_c = np.random.default_rng(0).standard_normal((q, nz, ny, nx))
    f_f = _upsample_coarse_to_fine(f_c)
    assert f_f.shape == (q, 2 * nz, ny, nx)
    np.testing.assert_allclose(_downsample_fine_to_coarse(f_f), f_c)


# ---------------------------------------------------------------------------
# Solver behaviour
# ---------------------------------------------------------------------------

def _block_solver(**over):
    Nz, Ny, Nx = 12, 6, 8
    kw = dict(omega=1.5, u0=0.02, D=4.0, backend="numpy",
              wall_bc="slip", streamwise_bc="periodic")
    kw.update(over)
    return MultiblockSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx, solid=_empty_solid(Nz, Ny, Nx),
        refine_z_lo=4, refine_z_hi=8, **kw,
    )


def test_fine_block_is_refined_in_every_direction():
    """
    D3Q19 assumes dx = dy = dz, so refining only one axis is not a valid
    lattice refinement.
    """
    sol = _block_solver()
    q, nz, ny, nx = sol._fine.f.shape
    assert (ny, nx) == (2 * sol.Ny, 2 * sol.Nx)
    assert nz == 2 * (sol.refine_z_hi - sol.refine_z_lo) + 2   # + ghost layers


def test_fine_block_keeps_its_own_state():
    """
    The refinement is pointless if the coarse solution is copied over the fine
    interior every step — that is a no-op that costs 8x the work.
    """
    sol = _block_solver()
    sol.step()
    interior = sol._fine.f[:, 1:-1].copy()
    marker = 12345.0
    sol._fine.f[:, 3, 2, 2] = marker
    sol.step()
    # the marked cell must have evolved, not been overwritten by a coarse copy
    assert not np.allclose(sol._fine.f[:, 1:-1], interior)


def test_ghost_layers_are_driven_by_the_coarse_grid():
    sol = _block_solver()
    sol._fine.f[:, 0] = -999.0
    sol._fine.f[:, -1] = -999.0
    sol._inject_ghosts(time_frac=0.0)
    assert np.all(sol._fine.f[:, 0] > -100.0)
    assert np.all(sol._fine.f[:, -1] > -100.0)


def test_rejects_a_block_touching_the_domain_edge():
    """A coarse plane must exist on both sides of each interface."""
    Nz, Ny, Nx = 10, 6, 8
    with pytest.raises(ValueError, match="refine_z_lo"):
        MultiblockSolver3D(
            Nz=Nz, Ny=Ny, Nx=Nx, solid=_empty_solid(Nz, Ny, Nx),
            refine_z_lo=0, refine_z_hi=4, omega=1.5, u0=0.02, D=4.0,
            backend="numpy",
        )


def test_run_reports_the_scaling_it_used():
    sol = _block_solver()
    result = sol.run(steps=3, check_every=100, verbose=False)
    assert result["steps_completed"] == 3
    assert result["omega_fine"] == pytest.approx(_omega_fine(sol.omega_coarse))
    assert 0.0 < result["neq_scale_coarse_to_fine"] < 1.0
    assert len(result["Cd_history"]) == 3


@pytest.mark.slow
def test_refined_block_tracks_a_smooth_wave():
    """
    A decaying shear wave along the refined axis, with no walls anywhere.

    The interface contributes a few percent over a few hundred steps; this
    pins that it stays in that range rather than diverging or damping the wave
    away.
    """
    pytest.importorskip("numba")
    from aero.lbm.solver3d import Solver3D

    Nz, Ny, Nx = 32, 4, 4
    amp, omega, steps = 0.02, 1.5, 300
    nu = (1.0 / omega - 0.5) / 3.0
    k = 2.0 * np.pi / Nz
    solid = _empty_solid(Nz, Ny, Nx)
    kw = dict(omega=omega, u0=0.0, D=8.0, backend="numba",
              wall_bc="slip", streamwise_bc="periodic")

    def seed(solver, centres, ny, nx):
        ux = amp * np.sin(2.0 * np.pi * centres / Nz)
        rho = np.ones((centres.size, ny, nx))
        U = np.broadcast_to(ux[:, None, None], rho.shape).copy()
        Z = np.zeros_like(rho)
        solver.f = np.ascontiguousarray(compute_feq_3d(rho, U, Z, Z))

    ref = Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, **kw)
    seed(ref, np.arange(Nz, dtype=float), Ny, Nx)
    for _ in range(steps):
        ref._step()

    mb = MultiblockSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, refine_z_lo=8, refine_z_hi=24, **kw
    )
    seed(mb._coarse, np.arange(Nz, dtype=float), Ny, Nx)
    nzf = mb._Nz_fine_total
    seed(mb._fine, 8 - 0.75 + 0.5 * np.arange(nzf), 2 * Ny, 2 * Nx)
    for _ in range(steps):
        mb.step()

    analytic = amp * np.exp(-nu * k * k * steps)
    a_ref = np.max(np.abs(ref.macroscopic()[1].mean(axis=(1, 2))))
    a_mb = np.max(np.abs(mb._coarse.macroscopic()[1].mean(axis=(1, 2))))

    assert abs(a_ref - analytic) / analytic < 0.02      # plain solver is accurate
    assert abs(a_mb - analytic) / analytic < 0.06       # interface costs a few %
