"""
3D aerodynamic force computation via the momentum exchange method.

For each surface link (i, z, y, x) — a fluid cell whose direction-i
neighbour is solid — the force contribution is:

  delta_F += e[i] * (f_pre[i,z,y,x] + f_post[opp[i],z,y,x])

Summed over all surface links gives total lattice-unit force on the obstacle.
Cd = Fx / (0.5*rho0*u0^2*D),  Cl_y = Fy/...,  Cl_z = Fz/...
"""

import numpy as np
from typing import Tuple
from .lbm.d3q19 import E3, OPP3


def compute_forces_3d(
    f_pre:  np.ndarray,
    f_post: np.ndarray,
    links:  np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute raw lattice-unit drag and lift forces.

    Parameters
    ----------
    f_pre   : (19, Nz, Ny, Nx) — post-collision / pre-streaming distributions
    f_post  : (19, Nz, Ny, Nx) — post-streaming / post-BC distributions
    links   : (N, 4) int32     — surface links [i, z, y, x]

    Returns
    -------
    Fx, Fy, Fz : float — raw lattice forces (dimensionless coefficients require normalisation)
    """
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0

    i_arr   = links[:, 0]
    z_arr   = links[:, 1]
    y_arr   = links[:, 2]
    x_arr   = links[:, 3]
    opp_arr = OPP3[i_arr]

    f_out = f_pre [i_arr,   z_arr, y_arr, x_arr]
    f_in  = f_post[opp_arr, z_arr, y_arr, x_arr]
    mom   = f_out + f_in

    ex = E3[i_arr, 0].astype(np.float64)
    ey = E3[i_arr, 1].astype(np.float64)
    ez = E3[i_arr, 2].astype(np.float64)

    return (
        float(np.sum(ex * mom)),
        float(np.sum(ey * mom)),
        float(np.sum(ez * mom)),
    )


def forces_to_coefficients_3d(
    Fx: float,
    Fy: float,
    Fz: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float, float]:
    """
    Convert raw lattice forces to dimensionless Cd, Cl_y, Cl_z.

    Reference area = D^2 (frontal area of a sphere/cube with diameter D).
    Cd = Fx / (0.5 * rho0 * u0^2 * D^2)
    """
    F_dyn = 0.5 * rho0 * u0 * u0 * D * D
    if F_dyn == 0.0:
        return 0.0, 0.0, 0.0
    return Fx / F_dyn, Fy / F_dyn, Fz / F_dyn
