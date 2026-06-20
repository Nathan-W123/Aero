"""Tests for moving-wall and synthetic-inflow additions."""

import numpy as np

from aero.lbm.boundary import apply_moving_walls
from aero.lbm.boundary3d import apply_moving_walls_3d
from aero.lbm.d2q9 import compute_feq
from aero.lbm.d3q19 import compute_feq_3d
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D


def test_apply_moving_walls_adds_tangential_momentum_2d():
    rho = np.ones((8, 12))
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    f = compute_feq(rho, ux, uy)
    before = f[8, -1, :].copy()
    apply_moving_walls(f, u_top=0.1)
    assert np.all(f[8, -1, :] > before)


def test_apply_moving_walls_adds_tangential_momentum_3d():
    rho = np.ones((4, 6, 10))
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    uz = np.zeros_like(rho)
    f = compute_feq_3d(rho, ux, uy, uz)
    before = f[9, :, -1, :].copy()
    apply_moving_walls_3d(f, u_top=0.1)
    assert np.all(f[9, :, -1, :] > before)


def test_solver2d_reports_observables_and_synthetic_inflow():
    solid = np.zeros((20, 40), dtype=bool)
    solver = Solver(
        Ny=20,
        Nx=40,
        solid=solid,
        omega=1.0,
        u0=0.05,
        D=10.0,
        backend="numpy",
        synthetic_inflow=True,
        synthetic_inflow_intensity=0.02,
    )
    result = solver.run(steps=40, check_every=20, verbose=False)
    assert "observables" in result
    assert "force_profile_last" in result["observables"]
    assert len(result["Cm_history"]) == 40


def test_solver3d_reports_observables_and_synthetic_inflow():
    solid = np.zeros((6, 10, 20), dtype=bool)
    solver = Solver3D(
        Nz=6,
        Ny=10,
        Nx=20,
        solid=solid,
        omega=1.0,
        u0=0.05,
        D=10.0,
        backend="numpy",
        synthetic_inflow=True,
        synthetic_inflow_intensity=0.02,
    )
    result = solver.run(steps=20, check_every=10, verbose=False)
    assert "observables" in result
    assert "spanwise_force_profile_last" in result["observables"]
    assert len(result["Cmx_history"]) == 20
