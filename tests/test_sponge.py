"""Tests for sponge layer."""

import numpy as np

from aero.lbm.sponge import build_sponge_sigma, apply_sponge_relaxation_2d
from aero.lbm.d2q9 import W, E


def test_sponge_sigma_ramp():
    sigma = build_sponge_sigma(100, 10, 0.2)
    assert sigma[0] == 0.0
    assert sigma[-1] > sigma[-5] > 0.0


def test_sponge_relaxation_reduces_perturbation():
    ny, nx = 20, 40
    f = np.zeros((9, ny, nx))
    f[0] = 1.0
    f[1] = 0.2
    sigma = build_sponge_sigma(nx, 5, 0.5)
    ex = E[:, 0].astype(np.int32)
    ey = E[:, 1].astype(np.int32)
    before = f[1, :, -1].copy()
    apply_sponge_relaxation_2d(f, sigma, 0.05, ex, ey, W)
    assert not np.allclose(f[1, :, -1], before)


def test_sponge_reduces_outlet_disturbance():
    """Sponge layer should damp outlet perturbations more than no sponge."""
    from aero.geometry.cylinder import Cylinder
    from aero.lbm.solver import Solver

    ny, nx = 24, 120
    solid = np.zeros((ny, nx), dtype=bool)
    common = dict(Ny=ny, Nx=nx, solid=solid, omega=1.0, u0=0.05, D=1.0, backend="numpy")

    s_plain = Solver(**common, sponge_thickness=0)
    s_sponge = Solver(**common, sponge_thickness=8, sponge_strength=0.3)
    # Inject a density bump near the outlet
    for s in (s_plain, s_sponge):
        s.f[:, ny // 2, -10] += 0.05
    for _ in range(30):
        s_plain._step()
        s_sponge._step()
    plain_out = float(np.std(s_plain.f[0, :, -5:]))
    sponge_out = float(np.std(s_sponge.f[0, :, -5:]))
    assert sponge_out < plain_out

