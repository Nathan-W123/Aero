"""
Benchmark / validation tests against published CFD reference values.

These are deliberately slower than unit tests (5000 steps each).
Run with: pytest tests/test_benchmark.py -v

References:
  - Tritton (1959): Cd ≈ 1.35 at Re=100 (unconfined, experimental)
  - Fornberg (1980): Cd ≈ 1.50 at Re=100 (numerical, steady)
  - With 20% blockage, Cd is elevated roughly 15-25%, so expect ~1.4-1.9
  - Schiller–Naumann sphere drag correlation for Re=20
"""

import numpy as np
import pytest

from aero.benchmarks import assess_literature, schiller_naumann_cd
from aero.geometry.cylinder import Cylinder
from aero.geometry3d.sphere import Sphere
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D


def make_reference_solver(collision: str = "bgk"):
    """Reference case: cylinder Re=100, 400x200, radius=20."""
    Ny, Nx = 200, 400
    radius  = 20
    u0      = 0.05
    Re      = 100.0
    D       = 2.0 * radius
    nu      = u0 * D / Re
    tau     = 3.0 * nu + 0.5
    omega   = 1.0 / tau

    cyl   = Cylinder(radius=radius, cx_frac=1/3, cy_frac=0.5)
    solid = cyl.mark_solid(Ny, Nx)
    return Solver(
        Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D, collision=collision,
    )


def make_sphere_solver3d(re: float = 20.0, collision: str = "bgk"):
    r = 7.0
    Nz, Ny, Nx = 48, 48, 96
    geom = Sphere(radius=r, cx_frac=1.0 / 3.0, cy_frac=0.5, cz_frac=0.5)
    solid = geom.mark_solid(Nz, Ny, Nx)
    u0 = 0.05
    D = 2.0 * r
    nu = u0 * D / re
    omega = 1.0 / (3.0 * nu + 0.5)
    return Solver3D(
        Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D,
        backend="auto", collision=collision,
    )


@pytest.mark.slow
def test_cylinder_cd_reference_range():
    """
    After 5000 steps the time-averaged Cd should be in [1.2, 2.2].
    This accounts for the 20% blockage confinement effect.
    """
    solver = make_reference_solver()
    result = solver.run(steps=5000, check_every=5000, verbose=True)
    Cd = result["Cd_mean"]
    print(f"\n[benchmark] Cd={Cd:.4f}  (expect 1.2 – 2.2 at 5000 steps)")
    status, msg = assess_literature(mode="2d", shape="cylinder", re=100.0, cd=Cd)
    assert status in {"pass", "warn"}, msg
    assert 1.2 <= Cd <= 2.2, f"Cd={Cd:.4f} outside benchmark range"


@pytest.mark.slow
def test_cylinder_vortex_shedding_cl_oscillation():
    """
    After 10 000 steps, Cl should be oscillating: std(Cl[-2000:]) > 0.01.
    A purely steady solution would have std ≈ 0 (Re=100 > Re_crit ≈ 47).
    """
    solver = make_reference_solver()
    solver.inlet_perturbation = 0.02
    result = solver.run(steps=10_000, check_every=5000, verbose=True)
    Cl_std = float(np.std(result["Cl_history"][-2000:]))
    print(f"\n[benchmark] Cl std (last 2000 steps) = {Cl_std:.4f}  (expect > 0.01)")
    assert Cl_std > 0.01, f"No vortex shedding detected: Cl std={Cl_std:.4f}"


@pytest.mark.slow
def test_sphere_re20_cd_literature_band():
    """3D sphere Re=20 — Cd within Schiller–Naumann-informed band."""
    solver = make_sphere_solver3d(re=20.0)
    result = solver.run(steps=3000, check_every=1000, verbose=True)
    cd = result["Cd_mean"]
    ref = schiller_naumann_cd(20.0)
    print(f"\n[benchmark] sphere Re=20 Cd={cd:.4f}  (Schiller–Naumann ≈ {ref:.2f})")
    status, msg = assess_literature(mode="3d", shape="sphere", re=20.0, cd=cd)
    assert not np.isnan(cd)
    assert status in {"pass", "warn"}, msg


def test_bgk_vs_mrt_both_stable_short_run():
    """BGK and MRT should both produce finite Cd over a short 3D sphere run."""
    for collision in ("bgk", "mrt"):
        solver = make_sphere_solver3d(re=20.0, collision=collision)
        result = solver.run(steps=200, check_every=200, verbose=False)
        cd = result["Cd_mean"]
        assert not np.isnan(cd), f"{collision} produced NaN Cd"
        assert 0.1 < cd < 20.0, f"{collision} Cd={cd} out of plausible range"
