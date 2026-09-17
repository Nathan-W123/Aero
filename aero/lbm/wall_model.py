"""
Equilibrium (algebraic) wall model for LES.

Without one, an LES has to resolve the viscous sublayer: the first cell needs
``y+ ~ 1``, and since ``y+ ~ Re^0.9`` that is unaffordable at aerodynamic
Reynolds numbers.  Van Driest damping does not substitute for this — it damps
the subgrid viscosity near a wall the simulation has *already* resolved, and
says nothing about what to do when the first node lands at ``y+ = 200``.

A wall model instead assumes the near-wall layer is in equilibrium and reads
the wall shear stress off a law of the wall, then feeds that stress back as an
effective viscosity at the wall-adjacent cells.  The simulation no longer has
to resolve the sublayer; it only has to put its first node somewhere the law
of the wall is valid (the log layer, ``y+ >~ 30``).

Law of the wall
---------------
Reichardt's single formula is used rather than a branched viscous/log fit,
because it is smooth through the buffer layer and therefore does not make the
Newton solve below jump around when a cell sits at ``y+ ~ 10``::

    u+ = (1/kappa) ln(1 + kappa y+)
         + 7.8 [ 1 - exp(-y+/11) - (y+/11) exp(-0.33 y+) ]

with ``kappa = 0.41``.  It reduces to ``u+ = y+`` as ``y+ -> 0`` and to the
standard log law at large ``y+``, so a run that *is* wall-resolved is not
disturbed by having the model switched on: across ``y+ < 2`` it asks for
within ``2e-4`` of the molecular viscosity, and over most of that range for
exactly the molecular viscosity, because Reichardt crosses ``u+ = y+`` from
below at ``y+ ~ 0.2`` and the clamp below turns the sub-molecular request on
the far side into an exact no-op.  It is therefore safe to leave enabled and
only starts doing real work past ``y+ ~ 5``.

Solving for the stress
----------------------
Given the wall-parallel velocity ``u_p`` at distance ``y`` from the wall, find
``u_tau`` from ``u_p / u_tau = f(y u_tau / nu)``.  That is one scalar equation
per wall cell; Newton's method on ``g(u_tau) = u_tau f(y u_tau/nu) - u_p``
converges in a handful of iterations from the laminar guess
``u_tau = sqrt(nu u_p / y)``.

Applying it
-----------
The modelled stress is imposed as an effective viscosity at the wall-adjacent
cell, ``nu_w = tau_w y / u_p``, which is the value that reproduces ``tau_w``
from the resolved velocity gradient the solver actually has.  It is written
into the same ``omega`` field the LES models use, so it composes with them and
needs no change to the collision kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

KAPPA = 0.41
B_REICHARDT = 7.8


def reichardt_u_plus(y_plus: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """
    Reichardt's law of the wall, valid from the sublayer through the log layer.

    Smooth everywhere, unlike a branched viscous/log fit, so the Newton solve
    in :func:`friction_velocity` has no kink to get stuck on.
    """
    y_plus = np.maximum(np.asarray(y_plus, dtype=np.float64), 0.0)
    return (
        np.log1p(kappa * y_plus) / kappa
        + B_REICHARDT * (
            1.0 - np.exp(-y_plus / 11.0)
            - (y_plus / 11.0) * np.exp(-0.33 * y_plus)
        )
    )


def _reichardt_derivative(y_plus: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """d(u+)/d(y+), for the Newton step."""
    y_plus = np.maximum(np.asarray(y_plus, dtype=np.float64), 0.0)
    return (
        1.0 / (1.0 + kappa * y_plus)
        + B_REICHARDT * (
            np.exp(-y_plus / 11.0) / 11.0
            - np.exp(-0.33 * y_plus) / 11.0
            + 0.33 * (y_plus / 11.0) * np.exp(-0.33 * y_plus)
        )
    )


def friction_velocity(
    u_parallel: np.ndarray,
    wall_distance: np.ndarray,
    nu: float,
    *,
    iterations: int = 12,
    kappa: float = KAPPA,
) -> np.ndarray:
    """
    Solve ``u_p = u_tau * f(y u_tau / nu)`` for ``u_tau``, per cell.

    Newton from the laminar estimate ``sqrt(nu u_p / y)``, which is the exact
    answer in the sublayer and a good starting point above it.  Zero velocity
    gives zero stress, which is handled without dividing by anything.
    """
    u_p = np.abs(np.asarray(u_parallel, dtype=np.float64))
    y = np.maximum(np.asarray(wall_distance, dtype=np.float64), 1e-30)
    nu = float(nu)

    active = u_p > 1e-300
    u_tau = np.where(active, np.sqrt(nu * u_p / y), 0.0)

    for _ in range(iterations):
        safe = np.where(u_tau > 1e-300, u_tau, 1.0)
        y_plus = y * safe / nu
        f = reichardt_u_plus(y_plus, kappa)
        df = _reichardt_derivative(y_plus, kappa)
        # g(u_tau) = u_tau f(y u_tau/nu) - u_p
        g = safe * f - u_p
        dg = f + safe * df * (y / nu)
        step = np.where(np.abs(dg) > 1e-300, g / dg, 0.0)
        u_tau = np.where(active, np.maximum(safe - step, 1e-30), 0.0)

    return np.where(active, u_tau, 0.0)


def wall_shear_stress(
    u_parallel: np.ndarray,
    wall_distance: np.ndarray,
    nu: float,
    rho: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """``tau_w = rho u_tau^2`` from the law of the wall."""
    u_tau = friction_velocity(u_parallel, wall_distance, nu, **kwargs)
    rho_arr = 1.0 if rho is None else np.asarray(rho, dtype=np.float64)
    return rho_arr * u_tau * u_tau


def effective_wall_viscosity(
    u_parallel: np.ndarray,
    wall_distance: np.ndarray,
    nu: float,
    **kwargs,
) -> np.ndarray:
    """
    Viscosity that reproduces the modelled stress from the resolved gradient.

    ``nu_w = tau_w y / (rho u_p)``: the solver has ``du/dy ~ u_p / y`` at the
    wall-adjacent cell, so this is the value that makes ``rho nu_w du/dy``
    equal the stress the law of the wall predicts.  It is never allowed below
    the molecular viscosity — the model adds turbulent transport, it does not
    remove the molecular part.
    """
    u_p = np.abs(np.asarray(u_parallel, dtype=np.float64))
    y = np.maximum(np.asarray(wall_distance, dtype=np.float64), 1e-30)
    u_tau = friction_velocity(u_p, y, nu, **kwargs)
    nu_w = np.where(u_p > 1e-300, u_tau * u_tau * y / np.where(u_p > 1e-300, u_p, 1.0), nu)
    return np.maximum(nu_w, nu)


def wall_adjacent_mask(solid: np.ndarray) -> np.ndarray:
    """
    Fluid cells with at least one face-neighbouring solid cell.

    Face neighbours only: those are the cells whose wall distance is a clean
    half spacing, which is what the model assumes.
    """
    solid = np.asarray(solid, dtype=bool)
    neighbour = np.zeros_like(solid)
    for axis in range(solid.ndim):
        neighbour |= np.roll(solid, 1, axis=axis)
        neighbour |= np.roll(solid, -1, axis=axis)
    return neighbour & (~solid)


def apply_wall_model(
    omega_field: np.ndarray,
    velocity: Tuple[np.ndarray, ...],
    solid: np.ndarray,
    nu: float,
    *,
    wall_distance: float = 0.5,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Overwrite ``omega`` at wall-adjacent cells with the modelled value.

    Returns a new field; the input is not modified.  Because the result is an
    ordinary omega field, this composes with Smagorinsky or WALE — run the
    subgrid model first, then this, and the wall cells take the modelled value
    while everything else keeps the subgrid one.
    """
    if mask is None:
        mask = wall_adjacent_mask(solid)
    if not mask.any():
        return omega_field

    u_mag = np.sqrt(sum(np.asarray(u, dtype=np.float64) ** 2 for u in velocity))
    nu_w = effective_wall_viscosity(
        u_mag[mask], np.full(int(mask.sum()), float(wall_distance)), nu, **kwargs
    )
    out = np.array(omega_field, dtype=np.float64, copy=True)
    out[mask] = 1.0 / (3.0 * nu_w + 0.5)
    return out
