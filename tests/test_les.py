"""Tests for Smagorinsky LES helpers."""

import numpy as np

from aero.lbm.les import strain_rate_magnitude_2d, smagorinsky_nu_sgs
from aero.geometry.cylinder import Cylinder
from aero.lbm.solver import Solver


def test_uniform_flow_zero_strain():
    ny, nx = 24, 48
    ux = np.full((ny, nx), 0.05)
    uy = np.zeros((ny, nx))
    fluid = np.ones((ny, nx), dtype=bool)
    s = strain_rate_magnitude_2d(ux, uy, fluid)
    assert float(np.max(s[1:-1, 1:-1])) < 1e-12


def test_les_enabled_short_run():
    ny, nx = 40, 80
    cyl = Cylinder(radius=6, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.0, u0=0.05, D=12.0,
        backend="numpy", les=True, les_cs=0.16,
    )
    result = solver.run(steps=30, check_every=30, verbose=False)
    assert not np.isnan(result["Cd_mean"])


def test_les_omega_field_varies_in_wake():
    from aero.lbm.les import build_omega_field_2d

    ny, nx = 40, 80
    cyl = Cylinder(radius=6, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.0, u0=0.05, D=12.0, backend="numpy",
    )
    for _ in range(40):
        solver._step()
    field = build_omega_field_2d(
        solver.f, solid, ~solid, solver._base_nu, solver.omega, 0.16,
    )
    fluid = field[~solid]
    assert float(fluid.max() - fluid.min()) > 1e-6


def test_smagorinsky_nu_zero_for_uniform():
    s = np.zeros((10, 10))
    assert float(np.max(smagorinsky_nu_sgs(s, 0.16))) == 0.0
