"""
D3Q19 Lattice Boltzmann constants and core operations.

Lattice velocity indexing (ex, ey, ez):
  i=0:  ( 0, 0, 0)  rest         w=1/3
  i=1:  (+1, 0, 0)  +x           w=1/18
  i=2:  (-1, 0, 0)  -x           w=1/18
  i=3:  ( 0,+1, 0)  +y           w=1/18
  i=4:  ( 0,-1, 0)  -y           w=1/18
  i=5:  ( 0, 0,+1)  +z           w=1/18
  i=6:  ( 0, 0,-1)  -z           w=1/18
  i=7:  (+1,+1, 0)  +x+y         w=1/36
  i=8:  (-1,+1, 0)  -x+y         w=1/36
  i=9:  (+1,-1, 0)  +x-y         w=1/36
  i=10: (-1,-1, 0)  -x-y         w=1/36
  i=11: (+1, 0,+1)  +x+z         w=1/36
  i=12: (-1, 0,+1)  -x+z         w=1/36
  i=13: (+1, 0,-1)  +x-z         w=1/36
  i=14: (-1, 0,-1)  -x-z         w=1/36
  i=15: ( 0,+1,+1)  +y+z         w=1/36
  i=16: ( 0,-1,+1)  -y+z         w=1/36
  i=17: ( 0,+1,-1)  +y-z         w=1/36
  i=18: ( 0,-1,-1)  -y-z         w=1/36

Domain layout: f shape = (Q, Nz, Ny, Nx)
  x : streamwise (inlet left, outlet right)
  y : vertical   (walls at y=0 bottom and y=Ny-1 top)
  z : spanwise   (periodic)
"""

import numpy as np
from typing import Tuple

Q3 = 19

# Lattice velocity vectors: E3[i] = (ex, ey, ez)
E3 = np.array([
    [ 0,  0,  0],  # 0
    [ 1,  0,  0],  # 1
    [-1,  0,  0],  # 2
    [ 0,  1,  0],  # 3
    [ 0, -1,  0],  # 4
    [ 0,  0,  1],  # 5
    [ 0,  0, -1],  # 6
    [ 1,  1,  0],  # 7
    [-1,  1,  0],  # 8
    [ 1, -1,  0],  # 9
    [-1, -1,  0],  # 10
    [ 1,  0,  1],  # 11
    [-1,  0,  1],  # 12
    [ 1,  0, -1],  # 13
    [-1,  0, -1],  # 14
    [ 0,  1,  1],  # 15
    [ 0, -1,  1],  # 16
    [ 0,  1, -1],  # 17
    [ 0, -1, -1],  # 18
], dtype=np.int8)

# Weights
W3 = np.array([
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=np.float64)

# Opposite direction (bounce-back partner)
OPP3 = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15],
                dtype=np.int8)

# y-mirror index: same ex, ez but ey flipped (for specular slip walls)
Y_MIR3 = np.array([0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 16, 15, 18, 17],
                  dtype=np.int8)

# Lattice speed of sound squared
CS2_3 = 1.0 / 3.0


def compute_macroscopic_3d(
    f: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rho, ux, uy, uz from distribution function f.

    Parameters
    ----------
    f : ndarray (19, Nz, Ny, Nx)

    Returns
    -------
    rho, ux, uy, uz : ndarray (Nz, Ny, Nx) each
    """
    rho = f.sum(axis=0)
    inv_rho = np.where(rho > 0.0, 1.0 / rho, 0.0)
    ex = E3[:, 0].astype(np.float64)
    ey = E3[:, 1].astype(np.float64)
    ez = E3[:, 2].astype(np.float64)
    ux = inv_rho * np.einsum('i,izyx->zyx', ex, f)
    uy = inv_rho * np.einsum('i,izyx->zyx', ey, f)
    uz = inv_rho * np.einsum('i,izyx->zyx', ez, f)
    return rho, ux, uy, uz


def compute_feq_3d(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> np.ndarray:
    """
    Compute D3Q19 equilibrium distribution.

    feq[i] = w[i] * rho * (1 + 3*(e·u) + 4.5*(e·u)^2 - 1.5*(u·u))

    Parameters
    ----------
    rho, ux, uy, uz : ndarray (Nz, Ny, Nx)

    Returns
    -------
    feq : ndarray (19, Nz, Ny, Nx)
    """
    usq = ux*ux + uy*uy + uz*uz
    feq = np.empty((Q3, *rho.shape), dtype=np.float64)
    for i in range(Q3):
        eu = E3[i, 0] * ux + E3[i, 1] * uy + E3[i, 2] * uz
        feq[i] = W3[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
    return feq
