"""
Aerodynamic force computation via the momentum exchange method.

For each obstacle surface link (i, y, x) — a fluid cell whose direction-i
neighbour is solid — the force contribution is:

  delta_F += e[i] * (f_pre[i, y, x] + f_post[opp[i], y, x])

Summed over all surface links, this gives the total lattice-unit force on
the obstacle.  Normalised by the dynamic pressure head gives Cd and Cl.
"""

import numpy as np
from typing import Tuple
from .lbm.d2q9 import E, OPP


def compute_forces(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    rho0: float,
    u0: float,
) -> Tuple[float, float]:
    """
    Compute instantaneous drag and lift coefficients.

    Parameters
    ----------
    f_pre   : ndarray (9, Ny, Nx) — post-collision / pre-streaming distributions
    f_post  : ndarray (9, Ny, Nx) — post-streaming / post-BC distributions
    links   : ndarray int32 (N, 3) — surface links [i, y, x]
    rho0    : float — reference density (lattice units)
    u0      : float — inlet velocity (lattice units)

    Returns
    -------
    Cd, Cl : float — drag and lift coefficients
             Cd > 0 means force in +x (downstream) direction
             Cl > 0 means force in +y direction
    """
    if links.shape[0] == 0:
        return 0.0, 0.0

    i_arr   = links[:, 0]
    y_arr   = links[:, 1]
    x_arr   = links[:, 2]
    opp_arr = OPP[i_arr]

    # Momentum exchange
    f_out = f_pre[i_arr,   y_arr, x_arr]   # leaving fluid toward solid
    f_in  = f_post[opp_arr, y_arr, x_arr]  # returning after bounce-back

    ex = E[i_arr, 0].astype(np.float64)
    ey = E[i_arr, 1].astype(np.float64)

    Fx_lbm = float(np.sum(ex * (f_out + f_in)))
    Fy_lbm = float(np.sum(ey * (f_out + f_in)))

    # Dynamic pressure reference (per unit depth, D is baked into link count
    # but we need the user to supply D separately for normalisation).
    # Here we return raw lattice forces; cli.py normalises using D.
    # To keep this function self-contained we return a sentinel tuple that
    # cli/solver normalises.  However, for solver.py's internal history we
    # also need D.  We therefore accept D as an optional parameter via the
    # module-level helper below and keep this function returning raw forces.
    return Fx_lbm, Fy_lbm


def forces_to_coefficients(
    Fx_lbm: float,
    Fy_lbm: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float]:
    """
    Convert raw lattice forces to dimensionless drag/lift coefficients.

    Cd = Fx / (0.5 * rho0 * u0^2 * D)
    Cl = Fy / (0.5 * rho0 * u0^2 * D)
    """
    F_dyn = 0.5 * rho0 * u0 * u0 * D
    if F_dyn == 0.0:
        return 0.0, 0.0
    return Fx_lbm / F_dyn, Fy_lbm / F_dyn
