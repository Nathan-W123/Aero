"""Tests for 3D inlet perturbation."""

import numpy as np

from aero.geometry3d.sphere import Sphere
from aero.lbm.solver3d import Solver3D


def test_inlet_perturbation_increases_lift_variance():
    geom = Sphere(radius=4, cx_frac=0.25, cy_frac=0.5, cz_frac=0.5)
    solid = geom.mark_solid(12, 16, 32)
    common = dict(Nz=12, Ny=16, Nx=32, solid=solid, omega=1.0, u0=0.05, D=8.0, backend="numpy")

    s0 = Solver3D(**common, inlet_perturbation=0.0)
    s1 = Solver3D(**common, inlet_perturbation=0.02)
    r0 = s0.run(steps=80, check_every=80, verbose=False)
    r1 = s1.run(steps=80, check_every=80, verbose=False)

    var0 = float(np.var(r0["Cly_history"][-40:])) + float(np.var(r0["Clz_history"][-40:]))
    var1 = float(np.var(r1["Cly_history"][-40:])) + float(np.var(r1["Clz_history"][-40:]))
    assert var1 > var0
