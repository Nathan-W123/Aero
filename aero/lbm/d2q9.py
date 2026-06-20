"""
D2Q9 Lattice Boltzmann constants and core operations.

Lattice velocity indexing:
  6  2  5
  3  0  1
  7  4  8
"""

import numpy as np
from typing import Tuple

# --- Lattice constants ---

Q = 9

# Lattice velocity vectors: e[i] = (ex, ey)
E = np.array([
    [ 0,  0],  # 0 — rest
    [ 1,  0],  # 1 — East
    [ 0,  1],  # 2 — North
    [-1,  0],  # 3 — West
    [ 0, -1],  # 4 — South
    [ 1,  1],  # 5 — NE
    [-1,  1],  # 6 — NW
    [-1, -1],  # 7 — SW
    [ 1, -1],  # 8 — SE
], dtype=np.int8)

# Weights
W = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36,
], dtype=np.float64)

# Opposite direction index (used for bounce-back)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)

# Lattice speed of sound squared
CS2 = 1.0 / 3.0


def compute_macroscopic(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rho, ux, uy from distribution function f.

    Parameters
    ----------
    f : ndarray, shape (9, Ny, Nx)

    Returns
    -------
    rho : ndarray (Ny, Nx)
    ux  : ndarray (Ny, Nx)
    uy  : ndarray (Ny, Nx)
    """
    rho = f.sum(axis=0)
    # Avoid division by zero in degenerate cells
    inv_rho = np.where(rho > 0, 1.0 / rho, 0.0)
    ux = inv_rho * np.einsum('i,iyx->yx', E[:, 0].astype(np.float64), f)
    uy = inv_rho * np.einsum('i,iyx->yx', E[:, 1].astype(np.float64), f)
    return rho, ux, uy


def compute_feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """
    Compute equilibrium distribution for all 9 directions.

    feq[i] = w[i] * rho * (1 + 3*(e.u) + 4.5*(e.u)^2 - 1.5*(u.u))

    Parameters
    ----------
    rho : ndarray (Ny, Nx)
    ux  : ndarray (Ny, Nx)
    uy  : ndarray (Ny, Nx)

    Returns
    -------
    feq : ndarray (9, Ny, Nx)
    """
    usq = ux * ux + uy * uy                      # (Ny, Nx)
    feq = np.empty((Q, *rho.shape), dtype=np.float64)

    for i in range(Q):
        eu = E[i, 0] * ux + E[i, 1] * uy         # (Ny, Nx)
        feq[i] = W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)

    return feq
