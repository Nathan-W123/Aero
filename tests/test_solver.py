"""
Integration and regression tests for the LBM solver.

These tests run short simulations (100–500 steps) to verify that:
  - Mass is conserved over a full timestep loop
  - Uniform inflow develops correctly with no obstacle
  - Cd for a cylinder at Re=100 falls within the physically expected range
    after a warm-up period
"""

import numpy as np
import pytest

from aero.geometry.cylinder import Cylinder
from aero.lbm.d2q9 import compute_macroscopic, compute_feq
from aero.lbm.solver import Solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_solver(Ny=60, Nx=120, radius=None, Re=100.0, u0=0.05):
    if radius is None:
        solid = np.zeros((Ny, Nx), dtype=bool)
        D = 1.0   # placeholder — no obstacle
    else:
        cyl   = Cylinder(radius=radius, cx_frac=1/3, cy_frac=0.5)
        solid = cyl.mark_solid(Ny, Nx)
        D     = 2.0 * radius
    nu    = u0 * D / Re if D > 1.0 else 0.02
    tau   = 3.0 * nu + 0.5
    omega = 1.0 / tau
    return Solver(Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D)


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------

def test_mass_conservation_no_obstacle():
    """
    Total mass (sum of rho over all cells) should be conserved to < 0.1%
    over 200 steps with no obstacle.
    """
    solver = make_solver()
    rho0, _, _ = compute_macroscopic(solver.f)
    mass0 = rho0.sum()

    solver.run(steps=200, check_every=200, verbose=False)

    rho1, _, _ = compute_macroscopic(solver.f)
    mass1 = rho1.sum()

    rel_drift = abs(mass1 - mass0) / mass0
    assert rel_drift < 0.001, f"Mass drifted {rel_drift*100:.4f}%"


# ---------------------------------------------------------------------------
# Uniform flow development
# ---------------------------------------------------------------------------

def test_uniform_flow_no_obstacle():
    """
    With no obstacle, after 300 steps the interior ux should be close to u0.
    Allow ±20% deviation to account for the short run.
    """
    u0     = 0.05
    solver = make_solver(Ny=40, Nx=80, u0=u0)
    solver.run(steps=300, check_every=300, verbose=False)

    _, ux, _ = compute_macroscopic(solver.f)
    interior = ux[1:-1, 5:-5]   # exclude walls and inlet/outlet columns
    assert interior.mean() == pytest.approx(u0, rel=0.20)


# ---------------------------------------------------------------------------
# No-NaN stability
# ---------------------------------------------------------------------------

def test_no_nan_after_steps():
    """100 steps of any reasonable config must not produce NaN."""
    solver = make_solver(Ny=40, Nx=80, radius=8, Re=100)
    solver.run(steps=100, check_every=100, verbose=False)
    assert not np.any(np.isnan(solver.f))


# ---------------------------------------------------------------------------
# Cylinder drag regression
# ---------------------------------------------------------------------------

def test_cylinder_cd_in_physical_range():
    """
    After 3000 warm-up steps, Cd for a cylinder at Re=100 must be in [0.8, 3.0].
    Uses radius=20 / Ny=100 so tau=0.56 (well above the 0.5 stability limit).
    """
    solver = make_solver(Ny=100, Nx=200, radius=20, Re=100, u0=0.05)
    result = solver.run(steps=3000, check_every=3000, verbose=False)
    Cd = result["Cd_mean"]
    assert 0.8 <= Cd <= 3.0, f"Cd={Cd:.4f} outside physical range [0.8, 3.0]"


def test_cylinder_cl_near_zero_mean():
    """
    Time-averaged Cl over the last 20% of a 3000-step run should be near 0.
    Uses the same stable grid as the Cd test.
    """
    solver = make_solver(Ny=100, Nx=200, radius=20, Re=100, u0=0.05)
    result = solver.run(steps=3000, check_every=3000, verbose=False)
    Cl_mean = abs(result["Cl_mean"])
    assert Cl_mean < 0.5, f"|Cl_mean|={Cl_mean:.4f} unexpectedly large"
