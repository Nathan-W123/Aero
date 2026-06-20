"""
Unit tests for D2Q9 lattice constants and equilibrium distribution.

Mathematical invariants verified:
  - Weights sum to 1
  - sum_i feq_i = rho          (mass conservation)
  - sum_i e_ix * feq_i = rho*ux  (x-momentum conservation)
  - sum_i e_iy * feq_i = rho*uy  (y-momentum conservation)
  - feq >= 0 for Ma << 1       (positivity, required for stability)
"""

import numpy as np
import pytest
from aero.lbm.d2q9 import Q, E, W, OPP, CS2, compute_macroscopic, compute_feq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert abs(W.sum() - 1.0) < 1e-14


def test_opposite_directions_are_mutual():
    for i in range(Q):
        assert OPP[OPP[i]] == i, f"OPP[OPP[{i}]] != {i}"


def test_opposite_directions_reverse_velocity():
    for i in range(Q):
        j = OPP[i]
        assert E[i, 0] == -E[j, 0]
        assert E[i, 1] == -E[j, 1]


def test_cs2_value():
    assert abs(CS2 - 1.0 / 3.0) < 1e-15


# ---------------------------------------------------------------------------
# Equilibrium distribution moment constraints
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_flow():
    Ny, Nx = 8, 8
    rho = np.ones((Ny, Nx))
    ux  = np.full((Ny, Nx), 0.05)
    uy  = np.zeros((Ny, Nx))
    return rho, ux, uy


def test_feq_zeroth_moment(uniform_flow):
    """sum_i feq_i = rho"""
    rho, ux, uy = uniform_flow
    feq = compute_feq(rho, ux, uy)
    rho_rec = feq.sum(axis=0)
    np.testing.assert_allclose(rho_rec, rho, atol=1e-13)


def test_feq_first_moment_x(uniform_flow):
    """sum_i e_ix * feq_i = rho * ux"""
    rho, ux, uy = uniform_flow
    feq = compute_feq(rho, ux, uy)
    jx = np.einsum('i,iyx->yx', E[:, 0].astype(float), feq)
    np.testing.assert_allclose(jx, rho * ux, atol=1e-14)


def test_feq_first_moment_y(uniform_flow):
    """sum_i e_iy * feq_i = rho * uy"""
    rho, ux, uy = uniform_flow
    feq = compute_feq(rho, ux, uy)
    jy = np.einsum('i,iyx->yx', E[:, 1].astype(float), feq)
    np.testing.assert_allclose(jy, rho * uy, atol=1e-14)


def test_feq_positivity_low_mach():
    """feq must be >= 0 for Ma << 1 (required for LBM stability)."""
    Ny, Nx = 4, 4
    rho = np.ones((Ny, Nx))
    ux  = np.full((Ny, Nx), 0.05)
    uy  = np.full((Ny, Nx), 0.01)
    feq = compute_feq(rho, ux, uy)
    assert feq.min() >= 0.0, f"Negative feq: {feq.min()}"


def test_feq_rest_state():
    """At ux=uy=0, feq[i] = w[i] * rho."""
    Ny, Nx = 4, 4
    rho = np.full((Ny, Nx), 1.5)
    feq = compute_feq(rho, np.zeros((Ny, Nx)), np.zeros((Ny, Nx)))
    for i in range(Q):
        np.testing.assert_allclose(feq[i], W[i] * rho, atol=1e-14)


# ---------------------------------------------------------------------------
# Macroscopic recovery
# ---------------------------------------------------------------------------

def test_compute_macroscopic_round_trip(uniform_flow):
    """
    If f = feq, recovering macroscopic variables should give back rho, ux, uy.
    """
    rho, ux, uy = uniform_flow
    feq = compute_feq(rho, ux, uy)
    rho_r, ux_r, uy_r = compute_macroscopic(feq)
    np.testing.assert_allclose(rho_r, rho, atol=1e-12)
    np.testing.assert_allclose(ux_r,  ux,  atol=1e-12)
    np.testing.assert_allclose(uy_r,  uy,  atol=1e-12)
