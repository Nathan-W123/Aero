"""Chapman–Enskog stress tensors from LBM non-equilibrium moments."""

from __future__ import annotations

import numpy as np

from .lbm.d2q9 import E, W, Q, CS2, compute_feq
from .lbm.d3q19 import E3, W3, Q3, CS2_3, compute_feq_3d


def compute_stress_2d(
    f: np.ndarray,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-equilibrium stress tensor components Pi_xx, Pi_xy, Pi_yy.

    Pi_alpha_beta = sum_i c_i_alpha c_i_beta (f_i - feq_i)
    """
    ux_c = ux.copy()
    uy_c = uy.copy()
    feq = compute_feq(rho, ux_c, uy_c)
    f_neq = f - feq

    pi_xx = np.einsum("i,iyx->yx", E[:, 0].astype(np.float64) ** 2, f_neq)
    pi_xy = np.einsum(
        "i,iyx->yx",
        E[:, 0].astype(np.float64) * E[:, 1].astype(np.float64),
        f_neq,
    )
    pi_yy = np.einsum("i,iyx->yx", E[:, 1].astype(np.float64) ** 2, f_neq)
    return pi_xx, pi_xy, pi_yy


def compute_stress_3d(
    f: np.ndarray,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Pi_xx, Pi_xy, Pi_xz, Pi_yy, Pi_yz, Pi_zz."""
    feq = compute_feq_3d(rho, ux, uy, uz)
    f_neq = f - feq
    ex = E3[:, 0].astype(np.float64)
    ey = E3[:, 1].astype(np.float64)
    ez = E3[:, 2].astype(np.float64)

    pi_xx = np.einsum("i,izyx->zyx", ex * ex, f_neq)
    pi_xy = np.einsum("i,izyx->zyx", ex * ey, f_neq)
    pi_xz = np.einsum("i,izyx->zyx", ex * ez, f_neq)
    pi_yy = np.einsum("i,izyx->zyx", ey * ey, f_neq)
    pi_yz = np.einsum("i,izyx->zyx", ey * ez, f_neq)
    pi_zz = np.einsum("i,izyx->zyx", ez * ez, f_neq)
    return pi_xx, pi_xy, pi_xz, pi_yy, pi_yz, pi_zz
