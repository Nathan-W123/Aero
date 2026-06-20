"""Tests for pressure/viscous drag decomposition."""

import numpy as np

from aero.geometry.cylinder import Cylinder
from aero.lbm.solver import Solver
from aero.lbm.d2q9 import compute_feq
from aero.forces import compute_force_split_2d
from aero.lbm.boundary import build_surface_links
from aero.stress import compute_stress_2d
from aero.lbm.d2q9 import compute_macroscopic


def test_uniform_flow_stress_near_zero():
    ny, nx = 32, 64
    rho = np.ones((ny, nx))
    ux = np.full((ny, nx), 0.05)
    uy = np.zeros((ny, nx))
    f = compute_feq(rho, ux, uy)
    pi_xx, pi_xy, pi_yy = compute_stress_2d(f, rho, ux, uy)
    assert float(np.max(np.abs(pi_xx))) < 1e-12


def test_drag_split_sums_to_total():
    ny, nx = 60, 120
    cyl = Cylinder(radius=8, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.0, u0=0.05, D=16.0,
        backend="numpy", collision="bgk",
    )
    result = solver.run(steps=200, check_every=200, verbose=False)
    total = result["Cd_mean"]
    split_sum = result["Cd_p_mean"] + result["Cd_v_mean"]
    assert abs(split_sum - total) / max(abs(total), 1e-6) < 0.05


def test_cylinder_both_components_nonzero():
    ny, nx = 60, 120
    cyl = Cylinder(radius=8, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=0.8, u0=0.05, D=16.0,
        backend="numpy", collision="bgk",
    )
    result = solver.run(steps=300, check_every=300, verbose=False)
    assert abs(result["Cd_p_mean"]) > 0.01
    assert abs(result["Cd_v_mean"]) > 0.01
