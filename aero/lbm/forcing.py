"""
Guo et al. (2002) forcing scheme — shared definitions for 2D and 3D.

A body force enters LBM in two coupled places, and both are required: getting
one without the other leaves a first-order error in the momentum.

1.  **The source term** added to the post-collision populations::

        S_i = (1 - omega/2) * w_i * [ (e_i - u)·F / cs^2
                                      + (e_i·u)(e_i·F) / cs^4 ]

    With cs^2 = 1/3 this is ``(1 - omega/2) * w_i * (3*(e_i·F - u·F)
    + 9*(e_i·u)(e_i·F))``.  Dropping the ``1/cs^2`` factor injects exactly
    ``cs^2 = 1/3`` of the intended force, because
    ``sum_i e_i w_i (e_i·F) = cs^2 F``.

2.  **The half-force correction** to the macroscopic velocity::

        rho u = sum_i e_i f_i + F/2

    used for the equilibrium, for the source term above, and for anything
    reported to the caller.

Together they give ``sum_i e_i f_post + F/2 = rho u + F/2``, i.e. exactly one
unit of acceleration per timestep, independent of omega.

Throughout this module the force is carried as an **acceleration** ``a``
(force per unit mass), so ``F = rho * a``.  That is the natural form for
gravity and for a pressure-gradient-equivalent channel forcing, and it makes
the Boussinesq buoyancy term fall out directly.

Multi-rate operators
--------------------
TRT and MRT relax different moments at different rates, so the single
``(1 - omega/2)`` prefactor becomes one factor per rate:

* TRT — split ``S_i`` into symmetric and antisymmetric parts and apply
  ``(1 - s_plus/2)`` and ``(1 - s_minus/2)``.  The parts are cheap in closed
  form: ``S_plus = w_i rho (9 (e_i·u)(e_i·a) - 3 u·a)`` and
  ``S_minus = 3 w_i rho (e_i·a)``.
* MRT — transform ``S_i`` to moment space and apply ``(I - S/2)`` there.

Both reduce to the BGK form when every rate equals omega.

Force modes
-----------
The kernels take the force in one of three modes, so the common cases cost
nothing:

``FORCE_NONE``     no forcing; the kernels skip every force branch.
``FORCE_UNIFORM``  one acceleration vector for the whole domain, passed as a
                   small ``(D,)`` array — no per-cell memory traffic.
``FORCE_FIELD``    a per-cell ``(D, *grid)`` acceleration field, for IBM,
                   buoyancy, or any combination of sources.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

FORCE_NONE = 0
FORCE_UNIFORM = 1
FORCE_FIELD = 2

CS2 = 1.0 / 3.0
INV_CS2 = 3.0
INV_CS4 = 9.0


def guo_source_term(
    rho: np.ndarray,
    u: Tuple[np.ndarray, ...],
    acc: Tuple[np.ndarray, ...],
    omega,
    e: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Reference (vectorised) Guo source term for an arbitrary lattice.

    Parameters
    ----------
    rho   : density field
    u     : velocity components, already half-force corrected
    acc   : acceleration components (force per unit mass)
    omega : scalar or per-cell relaxation rate
    e     : (Q, D) int lattice velocities
    w     : (Q,) weights

    Returns
    -------
    S : (Q, *grid) source term to add to the post-collision populations.
    """
    q = e.shape[0]
    prefactor = 1.0 - 0.5 * np.asarray(omega)
    u_dot_a = sum(uc * ac for uc, ac in zip(u, acc))
    out = np.empty((q,) + np.shape(rho), dtype=np.float64)
    for i in range(q):
        e_dot_a = sum(float(e[i, d]) * ac for d, ac in enumerate(acc))
        e_dot_u = sum(float(e[i, d]) * uc for d, uc in enumerate(u))
        out[i] = (
            prefactor * w[i] * rho
            * (INV_CS2 * (e_dot_a - u_dot_a) + INV_CS4 * e_dot_u * e_dot_a)
        )
    return out


def half_force_velocity(
    momentum: Tuple[np.ndarray, ...],
    rho: np.ndarray,
    acc: Optional[Tuple[np.ndarray, ...]],
) -> Tuple[np.ndarray, ...]:
    """
    ``u = (sum_i e_i f_i) / rho + a/2`` — the velocity consistent with Guo forcing.

    With ``acc=None`` this is the plain macroscopic velocity.
    """
    from .physics import inv_positive

    inv_rho = inv_positive(rho)
    if acc is None:
        return tuple(m * inv_rho for m in momentum)
    return tuple(m * inv_rho + 0.5 * ac for m, ac in zip(momentum, acc))


def uniform_acceleration(*components: float) -> np.ndarray:
    """Pack a uniform acceleration into the ``(D,)`` array the kernels expect."""
    return np.asarray(components, dtype=np.float64)


def resolve_force_mode(
    uniform: np.ndarray,
    field: Optional[np.ndarray],
) -> int:
    """Pick the cheapest mode that covers the requested forcing."""
    if field is not None:
        return FORCE_FIELD
    if np.any(np.abs(uniform) > 0.0):
        return FORCE_UNIFORM
    return FORCE_NONE


def dummy_force_field(ndim: int) -> np.ndarray:
    """
    A minimal correctly-typed array to hand a kernel when no field is in use.

    Numba needs a concrete array type even on a branch it never takes.
    """
    return np.zeros((ndim,) + (1,) * ndim, dtype=np.float64)


def ibm_acceleration(
    rho: np.ndarray,
    momentum: Tuple[np.ndarray, ...],
    u_wall: Tuple[np.ndarray, ...],
    phi: np.ndarray,
    solid: np.ndarray,
    band: float = 1.5,
) -> np.ndarray:
    """
    Direct-forcing IBM acceleration inside the band ``0 < phi <= band``.

    Direct forcing asks the *corrected* velocity to reach the wall velocity::

        u_raw + a/2 = u_wall   =>   a = 2 (u_wall - u_raw)

    The factor of two is what makes the immersed boundary actually reach the
    target in one step under the half-force convention; without it the wall
    condition is under-relaxed.

    Returns a ``(D, *grid)`` acceleration field, zero outside the band.
    """
    from .physics import inv_positive

    inv_rho = inv_positive(rho)
    active = (~solid) & (phi > 0.0) & (phi <= band) & (rho > 0.0)
    acc = np.zeros((len(momentum),) + rho.shape, dtype=np.float64)
    for d, (m, uw) in enumerate(zip(momentum, u_wall)):
        acc[d] = np.where(active, 2.0 * (uw - m * inv_rho), 0.0)
    return acc


def boussinesq_acceleration(
    temperature: np.ndarray,
    T_ref: float,
    g_gravity: float,
    beta: float,
    ndim: int,
    axis: int = 1,
) -> np.ndarray:
    """
    Boussinesq buoyancy as an acceleration field, ``a = g beta (T - T_ref)``.

    Expressed as an acceleration the Boussinesq term needs no separate
    treatment: it joins whatever else is forcing the flow and goes through the
    same Guo source term, which also gives it the half-force correction it
    previously lacked.  ``axis`` selects the gravity direction (y by default).
    """
    acc = np.zeros((ndim,) + temperature.shape, dtype=np.float64)
    acc[axis] = g_gravity * beta * (temperature - T_ref)
    return acc
