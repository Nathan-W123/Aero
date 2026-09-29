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

With the force at full strength a second problem showed.  The forcing held a
1.5-cell band of fluid *outside* the surface at rest -- a body 3 cells wider
than the geometry -- while drag was still read by momentum exchange at the
voxel surface inside that band, which saw almost none of it.  Now the body is
the forcing alone: weight 1 inside, 0 outside, blended across one cell at
``phi = 0``; the solver collides every cell as fluid and takes the drag as the
reaction to this force.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .forcing import ibm_weight
from .physics import inv_positive


def ibm_cells(phi: np.ndarray, width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flat indices of the cells the immersed body acts on, and their weights.

    Only the body and its one-cell skin carry any weight (see
    :func:`aero.lbm.forcing.ibm_weight`), so everything below touches those
    cells and nothing else.  ``phi`` does not change during a run: callers can
    compute this once.
    """
    weight = ibm_weight(phi, width).ravel()
    cells = np.flatnonzero(weight)
    return cells, weight[cells]


def ibm_forcing(
    f: np.ndarray,
    cells: np.ndarray,
    weights: np.ndarray,
    u_wall: Tuple[np.ndarray, ...],
    e_cols: Tuple[np.ndarray, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct-forcing acceleration field, and the force it puts into the fluid.

    Returns ``(acc, force)``: ``acc`` has shape ``(D, *grid)`` and is zero
    away from the body; ``force`` has shape ``(D, len(cells))`` and is
    ``rho * a`` on the forced cells -- per step, the momentum the Guo source
    adds there.  Its negative, summed, is the force on the body.
    """
    grid = f.shape[1:]
    ndim = len(e_cols)
    fc = f.reshape(f.shape[0], -1)[:, cells]
    rho = fc.sum(axis=0)
    inv_rho = inv_positive(rho)
    w = weights * (rho > 0.0)
    acc = np.zeros((ndim,) + grid, dtype=np.float64)
    flat = acc.reshape(ndim, -1)
    force = np.empty((ndim, cells.size), dtype=np.float64)
    for d, (e, uw) in enumerate(zip(e_cols, u_wall)):
        m = np.asarray(e, dtype=np.float64) @ fc
        uw_c = np.broadcast_to(np.asarray(uw, dtype=np.float64), grid).reshape(-1)[cells]
        a = 2.0 * w * (uw_c - m * inv_rho)
        flat[d, cells] = a
        force[d] = rho * a
    return acc, force


def ibm_acceleration_2d(
    f: np.ndarray,
    phi: np.ndarray,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    width: float = 1.0,
) -> np.ndarray:
    """Direct-forcing IBM acceleration field, shape ``(2, Ny, Nx)``."""
    cells, weights = ibm_cells(phi, width)
    return ibm_forcing(f, cells, weights, (u_wall_x, u_wall_y), (ex, ey))[0]


def ibm_acceleration_3d(
    f: np.ndarray,
    phi: np.ndarray,
    u_wall_x: np.ndarray,
    u_wall_y: np.ndarray,
    u_wall_z: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    width: float = 1.0,
) -> np.ndarray:
    """Direct-forcing IBM acceleration field, shape ``(3, Nz, Ny, Nx)``."""
    cells, weights = ibm_cells(phi, width)
    return ibm_forcing(f, cells, weights, (u_wall_x, u_wall_y, u_wall_z), (ex, ey, ez))[0]
