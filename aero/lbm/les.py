"""Smagorinsky LES subgrid viscosity for LBM."""

from __future__ import annotations

import numpy as np


def strain_rate_magnitude_2d(ux: np.ndarray, uy: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """|S| from central differences; solid cells return 0."""
    ny, nx = ux.shape
    s_mag = np.zeros((ny, nx), dtype=np.float64)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if not fluid[y, x]:
                continue
            dux_dx = 0.5 * (ux[y, x + 1] - ux[y, x - 1])
            dux_dy = 0.5 * (ux[y + 1, x] - ux[y - 1, x])
            duy_dx = 0.5 * (uy[y, x + 1] - uy[y, x - 1])
            duy_dy = 0.5 * (uy[y + 1, x] - uy[y - 1, x])
            s11, s22 = dux_dx, duy_dy
            s12 = 0.5 * (dux_dy + duy_dx)
            s_mag[y, x] = np.sqrt(2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12))
    return s_mag


def strain_rate_magnitude_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    nz, ny, nx = ux.shape
    s_mag = np.zeros((nz, ny, nx), dtype=np.float64)
    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if not fluid[z, y, x]:
                    continue
                dux_dx = 0.5 * (ux[z, y, x + 1] - ux[z, y, x - 1])
                dux_dy = 0.5 * (ux[z, y + 1, x] - ux[z, y - 1, x])
                dux_dz = 0.5 * (ux[z + 1, y, x] - ux[z - 1, y, x])
                duy_dx = 0.5 * (uy[z, y, x + 1] - uy[z, y, x - 1])
                duy_dy = 0.5 * (uy[z, y + 1, x] - uy[z, y - 1, x])
                duy_dz = 0.5 * (uy[z + 1, y, x] - uy[z - 1, y, x])
                duz_dx = 0.5 * (uz[z, y, x + 1] - uz[z, y, x - 1])
                duz_dy = 0.5 * (uz[z, y + 1, x] - uz[z, y - 1, x])
                duz_dz = 0.5 * (uz[z + 1, y, x] - uz[z - 1, y, x])
                s11, s22, s33 = dux_dx, duy_dy, duz_dz
                s12 = 0.5 * (dux_dy + duy_dx)
                s13 = 0.5 * (dux_dz + duz_dx)
                s23 = 0.5 * (duy_dz + duz_dy)
                s_mag[z, y, x] = np.sqrt(
                    2.0 * (s11 * s11 + s22 * s22 + s33 * s33)
                    + 4.0 * (s12 * s12 + s13 * s13 + s23 * s23)
                )
    return s_mag


def smagorinsky_nu_sgs(s_mag: np.ndarray, cs: float, delta: float = 1.0) -> np.ndarray:
    return (cs * delta) ** 2 * s_mag


def omega_from_nu(nu: np.ndarray) -> np.ndarray:
    return 1.0 / (3.0 * nu + 0.5)


def build_omega_field_2d(
    f: np.ndarray,
    solid: np.ndarray,
    fluid: np.ndarray,
    base_nu: float,
    base_omega: float,
    les_cs: float,
) -> np.ndarray:
    """Per-cell relaxation rate from Smagorinsky LES (Ny, Nx)."""
    from .d2q9 import compute_macroscopic

    rho, ux, uy = compute_macroscopic(f)
    s_mag = strain_rate_magnitude_2d(ux, uy, fluid)
    nu_eff = base_nu + smagorinsky_nu_sgs(s_mag, les_cs)
    nu_eff[solid] = base_nu
    return np.ascontiguousarray(omega_from_nu(nu_eff), dtype=np.float64)


def build_omega_field_3d(
    f: np.ndarray,
    solid: np.ndarray,
    fluid: np.ndarray,
    base_nu: float,
    base_omega: float,
    les_cs: float,
) -> np.ndarray:
    """Per-cell relaxation rate from Smagorinsky LES (Nz, Ny, Nx)."""
    from .d3q19 import compute_macroscopic_3d

    rho, ux, uy, uz = compute_macroscopic_3d(f)
    s_mag = strain_rate_magnitude_3d(ux, uy, uz, fluid)
    nu_eff = base_nu + smagorinsky_nu_sgs(s_mag, les_cs)
    nu_eff[solid] = base_nu
    return np.ascontiguousarray(omega_from_nu(nu_eff), dtype=np.float64)
