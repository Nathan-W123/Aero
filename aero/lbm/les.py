"""LES subgrid viscosity helpers for LBM."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Numba JIT strain-rate kernels
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @nb.njit(cache=True, parallel=True)
    def _strain_rate_2d_nb(
        ux: np.ndarray,
        uy: np.ndarray,
        fluid: np.ndarray,
        s_mag: np.ndarray,
    ) -> None:
        ny, nx = ux.shape
        for y in nb.prange(1, ny - 1):
            for x in range(1, nx - 1):
                if not fluid[y, x]:
                    continue
                dux_dx = 0.5 * (ux[y, x + 1] - ux[y, x - 1])
                dux_dy = 0.5 * (ux[y + 1, x] - ux[y - 1, x])
                duy_dx = 0.5 * (uy[y, x + 1] - uy[y, x - 1])
                duy_dy = 0.5 * (uy[y + 1, x] - uy[y - 1, x])
                s11 = dux_dx
                s22 = duy_dy
                s12 = 0.5 * (dux_dy + duy_dx)
                s_mag[y, x] = math.sqrt(
                    2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)
                )

    @nb.njit(cache=True, parallel=True)
    def _strain_rate_3d_nb(
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        fluid: np.ndarray,
        s_mag: np.ndarray,
    ) -> None:
        nz, ny, nx = ux.shape
        for z in nb.prange(1, nz - 1):
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
                    s11 = dux_dx
                    s22 = duy_dy
                    s33 = duz_dz
                    s12 = 0.5 * (dux_dy + duy_dx)
                    s13 = 0.5 * (dux_dz + duz_dx)
                    s23 = 0.5 * (duy_dz + duz_dy)
                    s_mag[z, y, x] = math.sqrt(
                        2.0 * (s11 * s11 + s22 * s22 + s33 * s33)
                        + 4.0 * (s12 * s12 + s13 * s13 + s23 * s23)
                    )


# ---------------------------------------------------------------------------
# NumPy fallbacks
# ---------------------------------------------------------------------------

def _strain_rate_2d_numpy(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
    s_mag: np.ndarray,
) -> None:
    ny, nx = ux.shape
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


def _strain_rate_3d_numpy(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
    s_mag: np.ndarray,
) -> None:
    nz, ny, nx = ux.shape
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


# ---------------------------------------------------------------------------
# Public strain-rate dispatchers
# ---------------------------------------------------------------------------

def strain_rate_magnitude_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    """|S| from central differences; solid cells return 0."""
    s_mag = np.zeros(ux.shape, dtype=np.float64)
    if HAS_NUMBA:
        _strain_rate_2d_nb(ux, uy, fluid, s_mag)
    else:
        _strain_rate_2d_numpy(ux, uy, fluid, s_mag)
    return s_mag


def strain_rate_magnitude_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    s_mag = np.zeros(ux.shape, dtype=np.float64)
    if HAS_NUMBA:
        _strain_rate_3d_nb(ux, uy, uz, fluid, s_mag)
    else:
        _strain_rate_3d_numpy(ux, uy, uz, fluid, s_mag)
    return s_mag


# ---------------------------------------------------------------------------
# SGS viscosity helpers
# ---------------------------------------------------------------------------

def smagorinsky_nu_sgs(s_mag: np.ndarray, cs: float, delta: float = 1.0) -> np.ndarray:
    return (cs * delta) ** 2 * s_mag


def wale_nu_sgs_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
    cs: float,
    delta: float = 1.0,
) -> np.ndarray:
    """WALE SGS viscosity for 2D using local velocity-gradient tensors."""
    ny, nx = ux.shape
    nu = np.zeros_like(ux, dtype=np.float64)
    cdelta2 = (cs * delta) ** 2
    eps = 1e-30
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if not fluid[y, x]:
                continue
            g11 = 0.5 * (ux[y, x + 1] - ux[y, x - 1])
            g12 = 0.5 * (ux[y + 1, x] - ux[y - 1, x])
            g21 = 0.5 * (uy[y, x + 1] - uy[y, x - 1])
            g22 = 0.5 * (uy[y + 1, x] - uy[y - 1, x])

            s11 = g11
            s22 = g22
            s12 = 0.5 * (g12 + g21)
            s_sq = s11 * s11 + s22 * s22 + 2.0 * s12 * s12

            g_sq11 = g11 * g11 + g12 * g21
            g_sq12 = g11 * g12 + g12 * g22
            g_sq21 = g21 * g11 + g22 * g21
            g_sq22 = g21 * g12 + g22 * g22
            trace = 0.5 * (g_sq11 + g_sq22)
            sd11 = g_sq11 - trace
            sd22 = g_sq22 - trace
            sd12 = 0.5 * (g_sq12 + g_sq21)
            sd_sq = sd11 * sd11 + sd22 * sd22 + 2.0 * sd12 * sd12

            num = sd_sq ** 1.5
            den = (s_sq ** 2.5) + (sd_sq ** 1.25) + eps
            nu[y, x] = cdelta2 * num / den
    return nu


def wale_nu_sgs_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
    cs: float,
    delta: float = 1.0,
) -> np.ndarray:
    """WALE SGS viscosity for 3D."""
    nz, ny, nx = ux.shape
    nu = np.zeros_like(ux, dtype=np.float64)
    cdelta2 = (cs * delta) ** 2
    eps = 1e-30
    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if not fluid[z, y, x]:
                    continue
                g = np.array([
                    [
                        0.5 * (ux[z, y, x + 1] - ux[z, y, x - 1]),
                        0.5 * (ux[z, y + 1, x] - ux[z, y - 1, x]),
                        0.5 * (ux[z + 1, y, x] - ux[z - 1, y, x]),
                    ],
                    [
                        0.5 * (uy[z, y, x + 1] - uy[z, y, x - 1]),
                        0.5 * (uy[z, y + 1, x] - uy[z, y - 1, x]),
                        0.5 * (uy[z + 1, y, x] - uy[z - 1, y, x]),
                    ],
                    [
                        0.5 * (uz[z, y, x + 1] - uz[z, y, x - 1]),
                        0.5 * (uz[z, y + 1, x] - uz[z, y - 1, x]),
                        0.5 * (uz[z + 1, y, x] - uz[z - 1, y, x]),
                    ],
                ], dtype=np.float64)
                s = 0.5 * (g + g.T)
                s_sq = np.sum(s * s)
                g_sq = g @ g
                sd = 0.5 * (g_sq + g_sq.T) - np.eye(3) * np.trace(g_sq) / 3.0
                sd_sq = np.sum(sd * sd)
                num = sd_sq ** 1.5
                den = (s_sq ** 2.5) + (sd_sq ** 1.25) + eps
                nu[z, y, x] = cdelta2 * num / den
    return nu


def omega_from_nu(nu: np.ndarray) -> np.ndarray:
    return 1.0 / (3.0 * nu + 0.5)


# ---------------------------------------------------------------------------
# Omega field builders (called every timestep from solver)
# ---------------------------------------------------------------------------

def build_omega_field_2d(
    f: np.ndarray,
    solid: np.ndarray,
    fluid: np.ndarray,
    base_nu: float,
    base_omega: float,
    les_cs: float,
    les_model: str = "smagorinsky",
    phi: Optional[np.ndarray] = None,
    van_driest: bool = False,
    van_driest_A: float = 25.0,
) -> np.ndarray:
    """Per-cell relaxation rate from selectable LES model (Ny, Nx)."""
    from .d2q9 import compute_macroscopic

    rho, ux, uy = compute_macroscopic(f)
    if les_model == "wale":
        if van_driest and phi is not None:
            y_wall = np.abs(phi)
            damping = (1.0 - np.exp(-y_wall / van_driest_A)) ** 2
            nu_sgs = wale_nu_sgs_2d(ux, uy, fluid, les_cs * damping)
        else:
            nu_sgs = wale_nu_sgs_2d(ux, uy, fluid, les_cs)
    else:
        s_mag = strain_rate_magnitude_2d(ux, uy, fluid)
        if van_driest and phi is not None:
            y_wall = np.abs(phi)
            damping = (1.0 - np.exp(-y_wall / van_driest_A)) ** 2
            nu_sgs = (les_cs * damping) ** 2 * s_mag
        else:
            nu_sgs = les_cs ** 2 * s_mag

    nu_eff = base_nu + nu_sgs
    nu_eff[solid] = base_nu
    return np.ascontiguousarray(omega_from_nu(nu_eff), dtype=np.float64)


def build_omega_field_3d(
    f: np.ndarray,
    solid: np.ndarray,
    fluid: np.ndarray,
    base_nu: float,
    base_omega: float,
    les_cs: float,
    les_model: str = "smagorinsky",
    phi: Optional[np.ndarray] = None,
    van_driest: bool = False,
    van_driest_A: float = 25.0,
) -> np.ndarray:
    """Per-cell relaxation rate from selectable LES model (Nz, Ny, Nx)."""
    from .d3q19 import compute_macroscopic_3d

    rho, ux, uy, uz = compute_macroscopic_3d(f)
    if les_model == "wale":
        if van_driest and phi is not None:
            y_wall = np.abs(phi)
            damping = (1.0 - np.exp(-y_wall / van_driest_A)) ** 2
            nu_sgs = wale_nu_sgs_3d(ux, uy, uz, fluid, les_cs * damping)
        else:
            nu_sgs = wale_nu_sgs_3d(ux, uy, uz, fluid, les_cs)
    else:
        s_mag = strain_rate_magnitude_3d(ux, uy, uz, fluid)
        if van_driest and phi is not None:
            y_wall = np.abs(phi)
            damping = (1.0 - np.exp(-y_wall / van_driest_A)) ** 2
            nu_sgs = (les_cs * damping) ** 2 * s_mag
        else:
            nu_sgs = les_cs ** 2 * s_mag

    nu_eff = base_nu + nu_sgs
    nu_eff[solid] = base_nu
    return np.ascontiguousarray(omega_from_nu(nu_eff), dtype=np.float64)
