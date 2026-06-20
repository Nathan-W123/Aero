"""Guo et al. (2002) immersed boundary forcing for LBM."""

from __future__ import annotations

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Numba JIT kernels
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @nb.njit(cache=True, parallel=True)
    def _guo_forcing_2d_nb(
        f: np.ndarray,
        phi: np.ndarray,
        tau: float,
        u_wall_x: np.ndarray,
        u_wall_y: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
        w: np.ndarray,
        solid: np.ndarray,
    ) -> None:
        Q = f.shape[0]
        ny, nx = phi.shape
        delta = 1.5
        coeff = 1.0 - 0.5 / tau
        for y in nb.prange(ny):
            for x in range(nx):
                if solid[y, x] or phi[y, x] <= 0.0 or phi[y, x] > delta:
                    continue
                rho = 0.0
                for i in range(Q):
                    rho += f[i, y, x]
                if rho <= 0.0:
                    continue
                inv_rho = 1.0 / rho
                ux = 0.0
                uy = 0.0
                for i in range(Q):
                    ux += ex[i] * f[i, y, x]
                    uy += ey[i] * f[i, y, x]
                ux *= inv_rho
                uy *= inv_rho
                fx = coeff * (u_wall_x[y, x] - ux)
                fy = coeff * (u_wall_y[y, x] - uy)
                for i in range(Q):
                    eu = ex[i] * fx + ey[i] * fy
                    f[i, y, x] += w[i] * eu

    @nb.njit(cache=True, parallel=True)
    def _guo_forcing_3d_nb(
        f: np.ndarray,
        phi: np.ndarray,
        tau: float,
        u_wall_x: np.ndarray,
        u_wall_y: np.ndarray,
        u_wall_z: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
        w: np.ndarray,
        solid: np.ndarray,
    ) -> None:
        Q = f.shape[0]
        nz, ny, nx = phi.shape
        delta = 1.5
        coeff = 1.0 - 0.5 / tau
        for z in nb.prange(nz):
            for y in range(ny):
                for x in range(nx):
                    if solid[z, y, x] or phi[z, y, x] <= 0.0 or phi[z, y, x] > delta:
                        continue
                    rho = 0.0
                    for i in range(Q):
                        rho += f[i, z, y, x]
                    if rho <= 0.0:
                        continue
                    inv_rho = 1.0 / rho
                    ux = 0.0
                    uy = 0.0
                    uz = 0.0
                    for i in range(Q):
                        ux += ex[i] * f[i, z, y, x]
                        uy += ey[i] * f[i, z, y, x]
                        uz += ez[i] * f[i, z, y, x]
                    ux *= inv_rho
                    uy *= inv_rho
                    uz *= inv_rho
                    fx = coeff * (u_wall_x[z, y, x] - ux)
                    fy = coeff * (u_wall_y[z, y, x] - uy)
                    fz = coeff * (u_wall_z[z, y, x] - uz)
                    for i in range(Q):
                        eu = ex[i] * fx + ey[i] * fy + ez[i] * fz
                        f[i, z, y, x] += w[i] * eu


# ---------------------------------------------------------------------------
# NumPy fallbacks (renamed from original)
# ---------------------------------------------------------------------------

def _guo_forcing_2d_numpy(
    f: np.ndarray,
    phi: np.ndarray,
    tau: float,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
    solid: np.ndarray,
) -> None:
    delta = 1.5
    q = f.shape[0]
    coeff = (1.0 - 0.5 / tau)
    exf = ex.astype(np.float64)
    eyf = ey.astype(np.float64)
    for y in range(phi.shape[0]):
        for x in range(phi.shape[1]):
            if solid[y, x] or phi[y, x] <= 0.0 or phi[y, x] > delta:
                continue
            rho = f[:, y, x].sum()
            if rho <= 0.0:
                continue
            ux = np.dot(exf, f[:, y, x]) / rho
            uy = np.dot(eyf, f[:, y, x]) / rho
            fx = coeff * (u_wall_x[y, x] - ux)
            fy = coeff * (u_wall_y[y, x] - uy)
            for i in range(q):
                eu = ex[i] * fx + ey[i] * fy
                f[i, y, x] += w[i] * eu


def _guo_forcing_3d_numpy(
    f: np.ndarray,
    phi: np.ndarray,
    tau: float,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    u_wall_z: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
    solid: np.ndarray,
) -> None:
    delta = 1.5
    q = f.shape[0]
    coeff = (1.0 - 0.5 / tau)
    exf = ex.astype(np.float64)
    eyf = ey.astype(np.float64)
    ezf = ez.astype(np.float64)
    nz, ny, nx = phi.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if solid[z, y, x] or phi[z, y, x] <= 0.0 or phi[z, y, x] > delta:
                    continue
                rho = f[:, z, y, x].sum()
                if rho <= 0.0:
                    continue
                ux = np.dot(exf, f[:, z, y, x]) / rho
                uy = np.dot(eyf, f[:, z, y, x]) / rho
                uz = np.dot(ezf, f[:, z, y, x]) / rho
                fx = coeff * (u_wall_x[z, y, x] - ux)
                fy = coeff * (u_wall_y[z, y, x] - uy)
                fz = coeff * (u_wall_z[z, y, x] - uz)
                for i in range(q):
                    eu = ex[i] * fx + ey[i] * fy + ez[i] * fz
                    f[i, z, y, x] += w[i] * eu


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------

def apply_guo_forcing_2d(
    f: np.ndarray,
    phi: np.ndarray,
    tau: float,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
    solid: np.ndarray,
) -> None:
    """
    Add Guo forcing in IBM band (0 < phi <= delta).

    phi < 0 => solid; phi > delta => free stream; 0 < phi <= 1.5 => IBM band.
    """
    if HAS_NUMBA:
        _guo_forcing_2d_nb(f, phi, tau, u_wall_x, u_wall_y,
                           ex.astype(np.float64), ey.astype(np.float64), w, solid)
    else:
        _guo_forcing_2d_numpy(f, phi, tau, u_wall_x, u_wall_y, ex, ey, w, solid)


def apply_guo_forcing_3d(
    f: np.ndarray,
    phi: np.ndarray,
    tau: float,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    u_wall_z: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
    solid: np.ndarray,
) -> None:
    if HAS_NUMBA:
        _guo_forcing_3d_nb(f, phi, tau, u_wall_x, u_wall_y, u_wall_z,
                           ex.astype(np.float64), ey.astype(np.float64),
                           ez.astype(np.float64), w, solid)
    else:
        _guo_forcing_3d_numpy(f, phi, tau, u_wall_x, u_wall_y, u_wall_z,
                              ex, ey, ez, w, solid)


def apply_uniform_body_force_2d(
    f: np.ndarray,
    tau: float,
    fx: float,
    fy: float,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
    solid: np.ndarray,
) -> None:
    """Apply a uniform Guo-style body force to all fluid nodes."""
    if abs(fx) < 1e-16 and abs(fy) < 1e-16:
        return
    cs2 = 1.0 / 3.0
    coeff = (1.0 - 0.5 / tau)
    q = f.shape[0]
    for y in range(f.shape[1]):
        for x in range(f.shape[2]):
            if solid[y, x]:
                continue
            rho = f[:, y, x].sum()
            if rho <= 0.0:
                continue
            ux = np.dot(ex.astype(np.float64), f[:, y, x]) / rho
            uy = np.dot(ey.astype(np.float64), f[:, y, x]) / rho
            for i in range(q):
                eu = ex[i] * ux + ey[i] * uy
                ef = ex[i] * fx + ey[i] * fy
                uf = ux * fx + uy * fy
                term = (ef - uf) / cs2 + (eu * ef) / (cs2 * cs2)
                f[i, y, x] += coeff * w[i] * rho * term


def apply_uniform_body_force_3d(
    f: np.ndarray,
    tau: float,
    fx: float,
    fy: float,
    fz: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
    solid: np.ndarray,
) -> None:
    """Apply a uniform Guo-style body force to all fluid nodes."""
    if abs(fx) < 1e-16 and abs(fy) < 1e-16 and abs(fz) < 1e-16:
        return
    cs2 = 1.0 / 3.0
    coeff = (1.0 - 0.5 / tau)
    q = f.shape[0]
    nz, ny, nx = f.shape[1:]
    exf = ex.astype(np.float64)
    eyf = ey.astype(np.float64)
    ezf = ez.astype(np.float64)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if solid[z, y, x]:
                    continue
                rho = f[:, z, y, x].sum()
                if rho <= 0.0:
                    continue
                ux = np.dot(exf, f[:, z, y, x]) / rho
                uy = np.dot(eyf, f[:, z, y, x]) / rho
                uz = np.dot(ezf, f[:, z, y, x]) / rho
                uf = ux * fx + uy * fy + uz * fz
                for i in range(q):
                    eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
                    ef = ex[i] * fx + ey[i] * fy + ez[i] * fz
                    term = (ef - uf) / cs2 + (eu * ef) / (cs2 * cs2)
                    f[i, z, y, x] += coeff * w[i] * rho * term
