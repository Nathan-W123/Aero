"""Tests for streamwise 3D boundary-condition modes."""

import numpy as np

from aero.geometry3d.sphere import Sphere
from aero.lbm.boundary3d import apply_recycling_rescaling_inlet_3d
from aero.lbm.d3q19 import compute_macroscopic_3d
from aero.lbm.solver3d import Solver3D


def test_recycling_rescaling_targets_mean_inlet_speed():
    rho = np.ones((6, 8, 12), dtype=float)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    uz = np.zeros_like(rho)
    ux[:, :, -2] = 0.03
    ux[:, :, -2] += np.linspace(-0.01, 0.01, ux.shape[1])[None, :]
    from aero.lbm.d3q19 import compute_feq_3d

    f = compute_feq_3d(rho, ux, uy, uz)
    apply_recycling_rescaling_inlet_3d(f, 0.05)
    _, ux_new, _, _ = compute_macroscopic_3d(f)
    assert abs(float(np.mean(ux_new[:, :, 0])) - 0.05) < 5e-3


def test_solver3d_periodic_streamwise_mode_runs():
    solid = np.zeros((8, 12, 20), dtype=bool)
    solver = Solver3D(
        Nz=8,
        Ny=12,
        Nx=20,
        solid=solid,
        omega=1.0,
        u0=0.05,
        D=10.0,
        backend="numpy",
        wall_bc="noslip",
        streamwise_bc="periodic",
        body_force_x=1.0e-5,
    )
    result = solver.run(steps=40, check_every=20, verbose=False)
    assert len(result["Cd_history"]) == 40
    assert result["steps_completed"] == 40


def test_solver3d_recycling_mode_runs_with_obstacle():
    geom = Sphere(radius=3, cx_frac=0.35, cy_frac=0.5, cz_frac=0.5)
    solid = geom.mark_solid(10, 12, 24)
    solver = Solver3D(
        Nz=10,
        Ny=12,
        Nx=24,
        solid=solid,
        omega=1.1,
        u0=0.05,
        D=6.0,
        backend="numpy",
        streamwise_bc="recycling",
    )
    result = solver.run(steps=20, check_every=10, verbose=False)
    assert len(result["Cd_history"]) == 20
    assert not np.isnan(result["Cd_mean"])
