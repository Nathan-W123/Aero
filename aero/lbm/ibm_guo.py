"""Guo et al. (2002) immersed boundary forcing for LBM."""

from __future__ import annotations

import numpy as np


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
    delta = 1.5
    q = f.shape[0]
    coeff = (1.0 - 0.5 / tau)
    for y in range(phi.shape[0]):
        for x in range(phi.shape[1]):
            if solid[y, x] or phi[y, x] <= 0.0 or phi[y, x] > delta:
                continue
            rho = f[:, y, x].sum()
            if rho <= 0.0:
                continue
            ux = np.dot(ex.astype(np.float64), f[:, y, x]) / rho
            uy = np.dot(ey.astype(np.float64), f[:, y, x]) / rho
            fx = coeff * (u_wall_x[y, x] - ux)
            fy = coeff * (u_wall_y[y, x] - uy)
            for i in range(q):
                eu = ex[i] * fx + ey[i] * fy
                f[i, y, x] += w[i] * eu


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
    delta = 1.5
    q = f.shape[0]
    coeff = (1.0 - 0.5 / tau)
    nz, ny, nx = phi.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if solid[z, y, x] or phi[z, y, x] <= 0.0 or phi[z, y, x] > delta:
                    continue
                rho = f[:, z, y, x].sum()
                if rho <= 0.0:
                    continue
                ux = np.dot(ex.astype(np.float64), f[:, z, y, x]) / rho
                uy = np.dot(ey.astype(np.float64), f[:, z, y, x]) / rho
                uz = np.dot(ez.astype(np.float64), f[:, z, y, x]) / rho
                fx = coeff * (u_wall_x[z, y, x] - ux)
                fy = coeff * (u_wall_y[z, y, x] - uy)
                fz = coeff * (u_wall_z[z, y, x] - uz)
                for i in range(q):
                    eu = ex[i] * fx + ey[i] * fy + ez[i] * fz
                    f[i, z, y, x] += w[i] * eu
