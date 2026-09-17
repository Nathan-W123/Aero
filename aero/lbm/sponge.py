"""Sponge absorption layer near the outlet."""

from __future__ import annotations

import numpy as np


def build_sponge_sigma(nx: int, thickness: int, sigma_max: float) -> np.ndarray:
    """
    Ramp sponge strength over the last ``thickness`` streamwise cells.

    Returns sigma(x) with shape (Nx,) — zero outside the sponge zone.
    """
    if thickness <= 0 or sigma_max <= 0.0:
        return np.zeros(nx, dtype=np.float64)
    sigma = np.zeros(nx, dtype=np.float64)
    start = max(nx - thickness, 0)
    for x in range(start, nx):
        frac = (x - start + 1) / max(thickness, 1)
        sigma[x] = sigma_max * frac * frac
    return sigma


def _sponge_slab(sigma_x: np.ndarray):
    """First column of the sponge zone and its sigma ramp, or None if inactive."""
    active = np.nonzero(sigma_x > 0.0)[0]
    if active.size == 0:
        return None
    start = int(active[0])
    return start, sigma_x[start:]


def _sponge_feq_poly(u_target: float, e_stream: np.ndarray) -> np.ndarray:
    """
    Per-direction equilibrium polynomial for the uniform sponge target state.

    The sponge relaxes toward (ux, uy[, uz]) = (u_target, 0[, 0]), so
    ``e.u = e_x * u_target`` and ``u.u = u_target**2`` are the same in every
    sponge cell.  Only ``rho`` varies, which turns the per-column, per-
    direction equilibrium rebuild into one scalar per direction.
    """
    eu = e_stream.astype(np.float64) * float(u_target)
    usq = float(u_target) * float(u_target)
    return 1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq


def apply_sponge_relaxation_2d(
    f: np.ndarray,
    sigma_x: np.ndarray,
    u_target: float,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
) -> None:
    """Relax f toward feq(rho, u_target, 0) in the sponge zone."""
    slab = _sponge_slab(sigma_x)
    if slab is None:
        return
    start, s = slab
    sub = f[:, :, start:]
    rho = sub.sum(axis=0)                       # before sub is modified
    poly = _sponge_feq_poly(u_target, ex)
    one_minus_s = 1.0 - s
    for i in range(len(w)):
        feq = w[i] * rho * poly[i]
        sub[i] = one_minus_s * sub[i] + s * feq


def apply_sponge_relaxation_3d(
    f: np.ndarray,
    sigma_x: np.ndarray,
    u_target: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
) -> None:
    slab = _sponge_slab(sigma_x)
    if slab is None:
        return
    start, s = slab
    sub = f[:, :, :, start:]
    rho = sub.sum(axis=0)                       # before sub is modified
    poly = _sponge_feq_poly(u_target, ex)
    one_minus_s = 1.0 - s
    for i in range(len(w)):
        feq = w[i] * rho * poly[i]
        sub[i] = one_minus_s * sub[i] + s * feq
