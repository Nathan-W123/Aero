"""
Immersed-boundary acceleration field for the Guo forcing scheme.

This module used to add a source term directly to ``f`` before collision.
That is the wrong place for it: the Guo scheme couples the source term to the
equilibrium through the half-force velocity, so forcing applied outside
collision cannot be consistent.  Forcing now lives inside the collision
kernels (see :mod:`aero.lbm.forcing`); what remains here is building the
acceleration field those kernels consume.

The old source term was also missing its ``1/cs^2`` factor, which made the
immersed boundary push with exactly one third of the requested force.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .forcing import ibm_acceleration
from .physics import inv_positive

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


def _is_numpy(*arrays: np.ndarray) -> bool:
    """True when every array is a host NumPy array (i.e. Numba can take it)."""
    return all(isinstance(a, np.ndarray) for a in arrays)


def _moments(f: np.ndarray, e_cols: Tuple[np.ndarray, ...], xp=np):
    """Density and momentum components from ``f``, backend-agnostic."""
    rho = f.sum(axis=0)
    flat = f.reshape(f.shape[0], -1)
    momentum = tuple(
        (e.astype(np.float64) @ flat).reshape(rho.shape) for e in e_cols
    )
    return rho, momentum


def ibm_acceleration_2d(
    f: np.ndarray,
    phi: np.ndarray,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    solid: np.ndarray,
    band: float = 1.5,
) -> np.ndarray:
    """
    Direct-forcing IBM acceleration field, shape ``(2, Ny, Nx)``.

    Non-zero only in the band ``0 < phi <= band``; see
    :func:`aero.lbm.forcing.ibm_acceleration` for the scheme.
    """
    rho, momentum = _moments(f, (ex, ey))
    return ibm_acceleration(
        rho, momentum, (u_wall_x, u_wall_y), phi, solid, band=band
    )


def ibm_acceleration_3d(
    f: np.ndarray,
    phi: np.ndarray,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    u_wall_z: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    solid: np.ndarray,
    band: float = 1.5,
) -> np.ndarray:
    """Direct-forcing IBM acceleration field, shape ``(3, Nz, Ny, Nx)``."""
    rho, momentum = _moments(f, (ex, ey, ez))
    return ibm_acceleration(
        rho, momentum, (u_wall_x, u_wall_y, u_wall_z), phi, solid, band=band
    )
