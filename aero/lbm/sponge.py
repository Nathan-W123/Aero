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


def apply_sponge_relaxation_2d(
    f: np.ndarray,
    sigma_x: np.ndarray,
    u_target: float,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
) -> None:
    """Relax f toward feq(rho, u_target, 0) in the sponge zone."""
    _, ny, nx = f.shape
    for x in range(nx):
        s = sigma_x[x]
        if s <= 0.0:
            continue
        rho = f[:, :, x].sum(axis=0)
        ux = np.full(ny, u_target)
        uy = np.zeros(ny)
        usq = ux * ux
        for i in range(len(w)):
            eu = ex[i] * ux + ey[i] * uy
            feq = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
            f[i, :, x] = (1.0 - s) * f[i, :, x] + s * feq


def apply_sponge_relaxation_3d(
    f: np.ndarray,
    sigma_x: np.ndarray,
    u_target: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
) -> None:
    _, nz, ny, nx = f.shape
    for x in range(nx):
        s = sigma_x[x]
        if s <= 0.0:
            continue
        rho = f[:, :, :, x].sum(axis=0)
        ux = np.full((nz, ny), u_target)
        uy = np.zeros((nz, ny))
        uz = np.zeros((nz, ny))
        usq = ux * ux
        for i in range(len(w)):
            eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
            feq = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
            f[i, :, :, x] = (1.0 - s) * f[i, :, :, x] + s * feq
