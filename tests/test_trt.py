"""Tests for TRT collision operator."""

import numpy as np

from aero.geometry.cylinder import Cylinder
from aero.lbm.solver import Solver


def test_trt_runs_without_nan():
    ny, nx = 40, 80
    cyl = Cylinder(radius=6, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.2, u0=0.05, D=12.0,
        backend="numpy", collision="trt",
    )
    result = solver.run(steps=200, check_every=200, verbose=False)
    assert not np.isnan(result["Cd_mean"])
    assert not np.isnan(result["Cd_p_mean"])
    assert abs(result["Cd_p_mean"] + result["Cd_v_mean"] - result["Cd_mean"]) < 1e-10


def test_trt_3d_runs():
    from aero.lbm.solver3d import Solver3D
    from aero.geometry3d.sphere import Sphere

    geom = Sphere(radius=4, cx_frac=0.25, cy_frac=0.5, cz_frac=0.5)
    solid = geom.mark_solid(12, 16, 32)
    solver = Solver3D(
        Nz=12, Ny=16, Nx=32, solid=solid, omega=1.1, u0=0.05, D=8.0,
        backend="numpy", collision="trt",
    )
    result = solver.run(steps=30, check_every=30, verbose=False)
    assert not np.isnan(result["Cd_mean"])
