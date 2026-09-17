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
    def _wale_operator_2d_nb(
        ux: np.ndarray,
        uy: np.ndarray,
        fluid: np.ndarray,
        op: np.ndarray,
    ) -> None:
        """WALE operator sd^(3/2) / (s^(5/2) + sd^(5/4)); scale by (Cw*delta)^2."""
        ny, nx = ux.shape
        eps = 1e-30
        for y in nb.prange(1, ny - 1):
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
                op[y, x] = num / den

    @nb.njit(cache=True, parallel=True)
    def _wale_operator_3d_nb(
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        fluid: np.ndarray,
        op: np.ndarray,
    ) -> None:
        """3D WALE operator; scale by (Cw*delta)^2 to get nu_sgs."""
        nz, ny, nx = ux.shape
        eps = 1e-30
        for z in nb.prange(1, nz - 1):
            for y in range(1, ny - 1):
                for x in range(1, nx - 1):
                    if not fluid[z, y, x]:
                        continue
                    g11 = 0.5 * (ux[z, y, x + 1] - ux[z, y, x - 1])
                    g12 = 0.5 * (ux[z, y + 1, x] - ux[z, y - 1, x])
                    g13 = 0.5 * (ux[z + 1, y, x] - ux[z - 1, y, x])
                    g21 = 0.5 * (uy[z, y, x + 1] - uy[z, y, x - 1])
                    g22 = 0.5 * (uy[z, y + 1, x] - uy[z, y - 1, x])
                    g23 = 0.5 * (uy[z + 1, y, x] - uy[z - 1, y, x])
                    g31 = 0.5 * (uz[z, y, x + 1] - uz[z, y, x - 1])
                    g32 = 0.5 * (uz[z, y + 1, x] - uz[z, y - 1, x])
                    g33 = 0.5 * (uz[z + 1, y, x] - uz[z - 1, y, x])

                    # s = 0.5 (g + g^T);  s_sq = sum(s*s)
                    s12 = 0.5 * (g12 + g21)
                    s13 = 0.5 * (g13 + g31)
                    s23 = 0.5 * (g23 + g32)
                    s_sq = (
                        g11 * g11 + g22 * g22 + g33 * g33
                        + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)
                    )

                    # g2 = g @ g
                    a11 = g11 * g11 + g12 * g21 + g13 * g31
                    a12 = g11 * g12 + g12 * g22 + g13 * g32
                    a13 = g11 * g13 + g12 * g23 + g13 * g33
                    a21 = g21 * g11 + g22 * g21 + g23 * g31
                    a22 = g21 * g12 + g22 * g22 + g23 * g32
                    a23 = g21 * g13 + g22 * g23 + g23 * g33
                    a31 = g31 * g11 + g32 * g21 + g33 * g31
                    a32 = g31 * g12 + g32 * g22 + g33 * g32
                    a33 = g31 * g13 + g32 * g23 + g33 * g33

                    # sd = 0.5 (g2 + g2^T) - I tr(g2)/3
                    trace = (a11 + a22 + a33) / 3.0
                    sd11 = a11 - trace
                    sd22 = a22 - trace
                    sd33 = a33 - trace
                    sd12 = 0.5 * (a12 + a21)
                    sd13 = 0.5 * (a13 + a31)
                    sd23 = 0.5 * (a23 + a32)
                    sd_sq = (
                        sd11 * sd11 + sd22 * sd22 + sd33 * sd33
                        + 2.0 * (sd12 * sd12 + sd13 * sd13 + sd23 * sd23)
                    )

                    num = sd_sq ** 1.5
                    den = (s_sq ** 2.5) + (sd_sq ** 1.25) + eps
                    op[z, y, x] = num / den

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

_INNER_2D = (slice(1, -1), slice(1, -1))
_INNER_3D = (slice(1, -1), slice(1, -1), slice(1, -1))


def _grad_2d(u: np.ndarray) -> tuple:
    """Central differences on the interior; returns (du/dx, du/dy)."""
    du_dx = 0.5 * (u[1:-1, 2:] - u[1:-1, :-2])
    du_dy = 0.5 * (u[2:, 1:-1] - u[:-2, 1:-1])
    return du_dx, du_dy


def _grad_3d(u: np.ndarray) -> tuple:
    """Central differences on the interior; returns (du/dx, du/dy, du/dz)."""
    du_dx = 0.5 * (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2])
    du_dy = 0.5 * (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1])
    du_dz = 0.5 * (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1])
    return du_dx, du_dy, du_dz


def _strain_rate_2d_numpy(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
    s_mag: np.ndarray,
) -> None:
    dux_dx, dux_dy = _grad_2d(ux)
    duy_dx, duy_dy = _grad_2d(uy)
    s12 = 0.5 * (dux_dy + duy_dx)
    mag = np.sqrt(2.0 * (dux_dx ** 2 + duy_dy ** 2 + 2.0 * s12 ** 2))
    s_mag[_INNER_2D] = np.where(fluid[_INNER_2D], mag, 0.0)


def _strain_rate_3d_numpy(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
    s_mag: np.ndarray,
) -> None:
    dux_dx, dux_dy, dux_dz = _grad_3d(ux)
    duy_dx, duy_dy, duy_dz = _grad_3d(uy)
    duz_dx, duz_dy, duz_dz = _grad_3d(uz)
    s12 = 0.5 * (dux_dy + duy_dx)
    s13 = 0.5 * (dux_dz + duz_dx)
    s23 = 0.5 * (duy_dz + duz_dy)
    mag = np.sqrt(
        2.0 * (dux_dx ** 2 + duy_dy ** 2 + duz_dz ** 2)
        + 4.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )
    s_mag[_INNER_3D] = np.where(fluid[_INNER_3D], mag, 0.0)


def _wale_operator_2d_numpy(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
    op: np.ndarray,
) -> None:
    eps = 1e-30
    g11, g12 = _grad_2d(ux)
    g21, g22 = _grad_2d(uy)

    s12 = 0.5 * (g12 + g21)
    s_sq = g11 * g11 + g22 * g22 + 2.0 * s12 * s12

    a11 = g11 * g11 + g12 * g21
    a12 = g11 * g12 + g12 * g22
    a21 = g21 * g11 + g22 * g21
    a22 = g21 * g12 + g22 * g22
    trace = 0.5 * (a11 + a22)
    sd11 = a11 - trace
    sd22 = a22 - trace
    sd12 = 0.5 * (a12 + a21)
    sd_sq = sd11 * sd11 + sd22 * sd22 + 2.0 * sd12 * sd12

    val = sd_sq ** 1.5 / ((s_sq ** 2.5) + (sd_sq ** 1.25) + eps)
    op[_INNER_2D] = np.where(fluid[_INNER_2D], val, 0.0)


def _wale_operator_3d_numpy(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
    op: np.ndarray,
) -> None:
    eps = 1e-30
    g11, g12, g13 = _grad_3d(ux)
    g21, g22, g23 = _grad_3d(uy)
    g31, g32, g33 = _grad_3d(uz)

    s12 = 0.5 * (g12 + g21)
    s13 = 0.5 * (g13 + g31)
    s23 = 0.5 * (g23 + g32)
    s_sq = (
        g11 * g11 + g22 * g22 + g33 * g33
        + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)
    )

    a11 = g11 * g11 + g12 * g21 + g13 * g31
    a12 = g11 * g12 + g12 * g22 + g13 * g32
    a13 = g11 * g13 + g12 * g23 + g13 * g33
    a21 = g21 * g11 + g22 * g21 + g23 * g31
    a22 = g21 * g12 + g22 * g22 + g23 * g32
    a23 = g21 * g13 + g22 * g23 + g23 * g33
    a31 = g31 * g11 + g32 * g21 + g33 * g31
    a32 = g31 * g12 + g32 * g22 + g33 * g32
    a33 = g31 * g13 + g32 * g23 + g33 * g33

    trace = (a11 + a22 + a33) / 3.0
    sd11 = a11 - trace
    sd22 = a22 - trace
    sd33 = a33 - trace
    sd12 = 0.5 * (a12 + a21)
    sd13 = 0.5 * (a13 + a31)
    sd23 = 0.5 * (a23 + a32)
    sd_sq = (
        sd11 * sd11 + sd22 * sd22 + sd33 * sd33
        + 2.0 * (sd12 * sd12 + sd13 * sd13 + sd23 * sd23)
    )

    val = sd_sq ** 1.5 / ((s_sq ** 2.5) + (sd_sq ** 1.25) + eps)
    op[_INNER_3D] = np.where(fluid[_INNER_3D], val, 0.0)


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


def wale_operator_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    """
    Dimensionless 2D WALE operator sd^(3/2) / (s^(5/2) + sd^(5/4)).

    ``nu_sgs = (Cw * delta)**2 * operator``.  Keeping the model constant out
    of the kernel lets it be a scalar *or* a per-cell field (e.g. Van Driest
    damped) without changing the inner loop.
    """
    op = np.zeros(ux.shape, dtype=np.float64)
    if HAS_NUMBA:
        _wale_operator_2d_nb(ux, uy, fluid, op)
    else:
        _wale_operator_2d_numpy(ux, uy, fluid, op)
    return op


def wale_operator_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    """Dimensionless 3D WALE operator; see :func:`wale_operator_2d`."""
    op = np.zeros(ux.shape, dtype=np.float64)
    if HAS_NUMBA:
        _wale_operator_3d_nb(ux, uy, uz, fluid, op)
    else:
        _wale_operator_3d_numpy(ux, uy, uz, fluid, op)
    return op


def wale_nu_sgs_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    fluid: np.ndarray,
    cs: float,
    delta: float = 1.0,
) -> np.ndarray:
    """
    WALE SGS viscosity for 2D using local velocity-gradient tensors.

    ``cs`` may be a scalar or a per-cell array (Van Driest damping).
    """
    return np.asarray((cs * delta) ** 2) * wale_operator_2d(ux, uy, fluid)


def wale_nu_sgs_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fluid: np.ndarray,
    cs: float,
    delta: float = 1.0,
) -> np.ndarray:
    """WALE SGS viscosity for 3D.  ``cs`` may be a scalar or a per-cell array."""
    return np.asarray((cs * delta) ** 2) * wale_operator_3d(ux, uy, uz, fluid)


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
