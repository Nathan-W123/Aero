"""Two-relaxation-time (TRT) parameters for D2Q9."""

from __future__ import annotations

import numpy as np


# Smallest (tau_plus - 1/2) used when solving for tau_minus.  tau_plus <= 1/2
# means a non-positive viscosity, which `validate_parameters` already rejects;
# the floor only keeps the relation finite instead of dividing by zero.
_TAU_EPS = 1e-12


def trt_taus(omega: float, magic_lambda: float = 0.25) -> tuple[float, float]:
    """
    Return (tau_plus, tau_minus) from BGK-equivalent omega and magic parameter.

    The symmetric rate carries the viscosity, so tau_plus must equal the BGK
    tau for TRT to reproduce the requested Reynolds number::

        tau_plus  = 1 / omega                      (nu = (tau_plus - 1/2) / 3)

    The antisymmetric rate follows from the TRT "magic" combination, which is
    what fixes the effective bounce-back wall position::

        Lambda    = (tau_plus - 1/2) * (tau_minus - 1/2)
        tau_minus = 1/2 + Lambda / (tau_plus - 1/2)

    ``magic_lambda`` is Lambda.  1/4 puts a bounce-back wall exactly mid-link
    (the usual choice); 1/6 cancels the third-order advection error and 3/16
    is optimal for a Poiseuille profile.

    Both rates stay in the stable band 0 < s < 2 for every tau_plus > 1/2.
    """
    lam = float(magic_lambda)
    tau_plus = 1.0 / omega
    tau_minus = 0.5 + lam / max(tau_plus - 0.5, _TAU_EPS)
    return tau_plus, tau_minus


def trt_s_minus(omega: float, magic_lambda: float = 0.25) -> float:
    """TRT minus-mode relaxation rate s_minus = 1/tau_minus."""
    _, tau_m = trt_taus(omega, magic_lambda)
    return 1.0 / tau_m


def trt_weights_2d(ex: np.ndarray, ey: np.ndarray, tau_plus: float, tau_minus: float):
    """Per-direction weights for TRT collision."""
    cs2 = 1.0 / 3.0
    w_plus = np.empty(len(ex), dtype=np.float64)
    w_minus = np.empty(len(ex), dtype=np.float64)
    for i in range(len(ex)):
        e2 = ex[i] * ex[i] + ey[i] * ey[i]
        w_plus[i] = cs2 - 0.5 * e2
        w_minus[i] = 1.0 - w_plus[i]
    s_plus = 1.0 / tau_plus
    s_minus = 1.0 / tau_minus
    return w_plus, w_minus, s_plus, s_minus
