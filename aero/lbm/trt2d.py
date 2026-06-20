"""Two-relaxation-time (TRT) parameters for D2Q9."""

from __future__ import annotations

import numpy as np


def trt_taus(omega: float, magic_lambda: float = 0.25) -> tuple[float, float]:
    """
    Return (tau_plus, tau_minus) from BGK-equivalent omega and magic parameter.

    omega = 1/tau  =>  tau = 1/omega
    """
    tau = 1.0 / omega
    lam = float(magic_lambda)
    tau_plus = lam * tau + (1.0 - lam)
    tau_minus = (tau_plus * tau - 0.5) / (tau_plus - 0.5)
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
