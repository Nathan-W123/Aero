"""Tests for LES helpers and model selection."""

import numpy as np
import pytest

from aero.lbm.les import (
    strain_rate_magnitude_2d,
    smagorinsky_nu_sgs,
    build_omega_field_2d,
    build_omega_field_3d,
    wale_nu_sgs_2d,
    wale_operator_2d,
)
from aero.geometry.cylinder import Cylinder
from aero.lbm.solver import Solver
from cli import build_parser as build_parser_2d


def _uniform_f_2d(ny, nx, u=0.05):
    from aero.lbm.d2q9 import compute_feq
    rho = np.ones((ny, nx))
    return np.ascontiguousarray(
        compute_feq(rho, np.full((ny, nx), u), np.zeros((ny, nx)))
    )


@pytest.mark.parametrize("model", ["smagorinsky", "wale"])
def test_van_driest_damping_accepts_every_les_model(model):
    """
    Van Driest passes a per-cell damped constant; WALE used to reject it.

    The damped constant is a field, so the model kernel must handle an
    array-valued Cs as well as a scalar one.
    """
    ny, nx = 16, 24
    f = _uniform_f_2d(ny, nx)
    solid = np.zeros((ny, nx), dtype=bool)
    phi = np.full((ny, nx), 2.0)
    omega = build_omega_field_2d(
        f, solid, ~solid, 0.1, 1.8, 0.16,
        les_model=model, phi=phi, van_driest=True, van_driest_A=25.0,
    )
    assert omega.shape == (ny, nx)
    assert np.all(np.isfinite(omega))


@pytest.mark.parametrize("model", ["smagorinsky", "wale"])
def test_van_driest_damping_accepts_every_les_model_3d(model):
    from aero.lbm.d3q19 import compute_feq_3d
    nz, ny, nx = 6, 8, 10
    rho = np.ones((nz, ny, nx))
    zero = np.zeros((nz, ny, nx))
    f = np.ascontiguousarray(compute_feq_3d(rho, np.full((nz, ny, nx), 0.05), zero, zero))
    solid = np.zeros((nz, ny, nx), dtype=bool)
    phi = np.full((nz, ny, nx), 2.0)
    omega = build_omega_field_3d(
        f, solid, ~solid, 0.1, 1.8, 0.16,
        les_model=model, phi=phi, van_driest=True, van_driest_A=25.0,
    )
    assert omega.shape == (nz, ny, nx)
    assert np.all(np.isfinite(omega))


def test_wale_nu_sgs_is_linear_in_the_model_constant():
    """A per-cell Cs must scale the operator exactly as a scalar Cs does."""
    rng = np.random.default_rng(0)
    ny, nx = 18, 22
    ux = 0.05 * rng.standard_normal((ny, nx))
    uy = 0.05 * rng.standard_normal((ny, nx))
    fluid = np.ones((ny, nx), dtype=bool)
    op = wale_operator_2d(ux, uy, fluid)
    assert np.allclose(wale_nu_sgs_2d(ux, uy, fluid, 0.16), 0.16 ** 2 * op)
    cs_field = 0.16 * (0.5 + rng.random((ny, nx)))
    assert np.allclose(wale_nu_sgs_2d(ux, uy, fluid, cs_field), cs_field ** 2 * op)


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


def test_cli_accepts_wale_les_model():
    args = build_parser_2d().parse_args(["--les-model", "wale"])
    assert args.les_model == "wale"


def test_solver_accepts_wale_model():
    ny, nx = 24, 48
    solid = np.zeros((ny, nx), dtype=bool)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.0, u0=0.05, D=12.0,
        backend="numpy", les=True, les_model="wale",
    )
    result = solver.run(steps=10, check_every=10, verbose=False)
    assert not np.isnan(result["Cd_mean"])


def test_wale_omega_field_differs_from_smagorinsky():
    ny, nx = 40, 80
    cyl = Cylinder(radius=6, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.0, u0=0.05, D=12.0, backend="numpy",
    )
    for _ in range(40):
        solver._step()
    omega_smag = build_omega_field_2d(
        solver.f, solid, ~solid, solver._base_nu, solver.omega, 0.16, les_model="smagorinsky",
    )
    omega_wale = build_omega_field_2d(
        solver.f, solid, ~solid, solver._base_nu, solver.omega, 0.16, les_model="wale",
    )
    assert float(np.max(np.abs(omega_smag - omega_wale))) > 1e-9


def test_smagorinsky_nu_zero_for_uniform():
    s = np.zeros((10, 10))
    assert float(np.max(smagorinsky_nu_sgs(s, 0.16))) == 0.0


@pytest.mark.parametrize("collision", ["bgk", "trt", "mrt"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_les_reaches_every_collision_operator(collision, backend):
    """
    Enabling LES must change the answer whatever the collision operator is.

    The MRT kernel used to be called without the subgrid omega field, so
    `--collision mrt --les` silently ran a laminar simulation.
    """
    numba = pytest.importorskip("numba") if backend == "numba" else None
    ny, nx = 40, 80
    cyl = Cylinder(radius=6, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    common = dict(
        Ny=ny, Nx=nx, solid=solid, omega=1.6, u0=0.06, D=12.0,
        backend=backend, collision=collision, inlet_perturbation=0.05,
    )
    laminar = Solver(**common).run(steps=120, check_every=120, verbose=False)
    subgrid = Solver(les=True, les_cs=0.6, **common).run(
        steps=120, check_every=120, verbose=False
    )
    assert abs(laminar["Cd_mean"] - subgrid["Cd_mean"]) > 1e-9


@pytest.mark.parametrize("collision", ["bgk", "trt", "mrt"])
def test_numba_and_numpy_backends_agree_under_les(collision):
    pytest.importorskip("numba")
    ny, nx = 32, 64
    cyl = Cylinder(radius=5, cx_frac=1 / 3, cy_frac=0.5)
    solid = cyl.mark_solid(ny, nx)
    common = dict(
        Ny=ny, Nx=nx, solid=solid, omega=1.6, u0=0.06, D=10.0,
        collision=collision, les=True, les_cs=0.3,
    )
    a = Solver(backend="numpy", **common).run(steps=60, check_every=60, verbose=False)
    b = Solver(backend="numba", **common).run(steps=60, check_every=60, verbose=False)
    assert a["Cd_mean"] == pytest.approx(b["Cd_mean"], rel=1e-9, abs=1e-12)
