"""Tests for TRT collision operator."""

import numpy as np
import pytest

from aero.geometry.cylinder import Cylinder
from aero.lbm.solver import Solver
from aero.lbm.trt2d import trt_taus, trt_s_minus


@pytest.mark.parametrize("omega", [1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9, 1.95, 1.99])
def test_trt_tau_plus_carries_the_requested_viscosity(omega):
    """tau_plus must equal the BGK tau, else TRT runs at the wrong Reynolds number."""
    tau_plus, _ = trt_taus(omega, 0.25)
    assert tau_plus == pytest.approx(1.0 / omega)


@pytest.mark.parametrize("omega", [1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9, 1.95, 1.99])
@pytest.mark.parametrize("lam", [3.0 / 16.0, 0.25, 1.0 / 6.0])
def test_trt_rates_stay_in_the_stable_band(omega, lam):
    """Both relaxation rates must satisfy 0 < s < 2 or the collision amplifies."""
    tau_plus, tau_minus = trt_taus(omega, lam)
    assert tau_plus > 0.5
    assert tau_minus > 0.5
    s_minus = trt_s_minus(omega, lam)
    assert 0.0 < s_minus < 2.0
    # magic relation that fixes the effective bounce-back wall position
    assert (tau_plus - 0.5) * (tau_minus - 0.5) == pytest.approx(lam)


@pytest.mark.parametrize("omega", [1.5, 1.7, 1.85])
def test_trt_is_stable_where_bgk_is(omega):
    """
    TRT must at least match BGK stability at ordinary relaxation rates.

    A tau_minus outside (0.5, inf) used to make TRT blow up for every
    omega >= ~1.4, i.e. for essentially every useful viscosity.
    """
    ny, nx = 40, 80
    cyl = Cylinder(radius=5, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=omega, u0=0.05, D=10.0,
        backend="numpy", collision="trt",
    )
    result = solver.run(steps=400, check_every=400, verbose=False)
    assert np.isfinite(result["Cd_mean"])
    assert np.all(np.isfinite(solver.f))
    assert float(np.max(np.abs(solver.f))) < 10.0


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
