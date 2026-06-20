"""
Aerodynamic force computation via the momentum exchange method.

Pressure/viscous decomposition splits each link's momentum exchange into
equilibrium (pressure) and non-equilibrium (viscous) parts using local
Chapman–Enskog moments — equivalent to integrating stress at the surface
and guaranteed to sum to the total momentum-exchange force.
"""

import numpy as np
from typing import Dict, Tuple
from .lbm.d2q9 import E, OPP, compute_feq, compute_macroscopic


def compute_forces(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    rho0: float,
    u0: float,
) -> Tuple[float, float]:
    """Compute raw lattice-unit drag and lift via momentum exchange."""
    if links.shape[0] == 0:
        return 0.0, 0.0

    i_arr   = links[:, 0]
    y_arr   = links[:, 1]
    x_arr   = links[:, 2]
    opp_arr = OPP[i_arr]

    f_out = f_pre[i_arr,   y_arr, x_arr]
    f_in  = f_post[opp_arr, y_arr, x_arr]

    ex = E[i_arr, 0].astype(np.float64)
    ey = E[i_arr, 1].astype(np.float64)

    Fx_lbm = float(np.sum(ex * (f_out + f_in)))
    Fy_lbm = float(np.sum(ey * (f_out + f_in)))
    return Fx_lbm, Fy_lbm


def compute_force_split_2d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Pressure and viscous force components on the obstacle (lattice units).

    At each surface link the momentum exchange is split into equilibrium
    (pressure) and non-equilibrium (viscous) contributions.
    """
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    fx_p = fy_p = fx_v = fy_v = 0.0
    for i, y, x in links:
        opp = int(OPP[i])
        ex = float(E[i, 0])
        ey = float(E[i, 1])
        f_out = float(f_pre[i, y, x])
        f_in = float(f_post[opp, y, x])

        rho, ux, uy = compute_macroscopic(f_pre[:, y : y + 1, x : x + 1])
        feq = compute_feq(rho, ux, uy)
        feq_out = float(feq[i, 0, 0])
        feq_in = float(feq[opp, 0, 0])

        mom_p = feq_out + feq_in
        mom_v = (f_out - feq_out) + (f_in - feq_in)
        fx_p += ex * mom_p
        fy_p += ey * mom_p
        fx_v += ex * mom_v
        fy_v += ey * mom_v
    return fx_p, fy_p, fx_v, fy_v


def forces_to_coefficients(
    Fx_lbm: float,
    Fy_lbm: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float]:
    F_dyn = 0.5 * rho0 * u0 * u0 * D
    if F_dyn == 0.0:
        return 0.0, 0.0
    return Fx_lbm / F_dyn, Fy_lbm / F_dyn


def split_to_coefficients(
    fx_p: float,
    fy_p: float,
    fx_v: float,
    fy_v: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float, float, float]:
    F_dyn = 0.5 * rho0 * u0 * u0 * D
    if F_dyn == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return fx_p / F_dyn, fy_p / F_dyn, fx_v / F_dyn, fy_v / F_dyn


def compute_force_moment_2d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    center_x: float,
    center_y: float,
) -> Tuple[float, float, float]:
    """Return raw Fx, Fy, and z-moment from momentum exchange."""
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0
    i_arr = links[:, 0]
    y_arr = links[:, 1]
    x_arr = links[:, 2]
    opp_arr = OPP[i_arr]
    mom = f_pre[i_arr, y_arr, x_arr] + f_post[opp_arr, y_arr, x_arr]
    dfx = E[i_arr, 0].astype(np.float64) * mom
    dfy = E[i_arr, 1].astype(np.float64) * mom
    rx = x_arr.astype(np.float64) - float(center_x)
    ry = y_arr.astype(np.float64) - float(center_y)
    mz = float(np.sum(rx * dfy - ry * dfx))
    return float(np.sum(dfx)), float(np.sum(dfy)), mz


def moment_to_coefficient_2d(
    mz: float,
    rho0: float,
    u0: float,
    D: float,
) -> float:
    denom = 0.5 * rho0 * u0 * u0 * D * D
    if denom == 0.0:
        return 0.0
    return mz / denom


def force_profile_2d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    ny: int,
) -> Dict[str, list[float]]:
    """Return cross-stream integrated force profiles by y-index."""
    fx = np.zeros(int(ny), dtype=np.float64)
    fy = np.zeros(int(ny), dtype=np.float64)
    if links.shape[0] == 0:
        return {"y": list(range(int(ny))), "fx": fx.tolist(), "fy": fy.tolist()}
    i_arr = links[:, 0]
    y_arr = links[:, 1]
    opp_arr = OPP[i_arr]
    mom = f_pre[i_arr, links[:, 1], links[:, 2]] + f_post[opp_arr, links[:, 1], links[:, 2]]
    dfx = E[i_arr, 0].astype(np.float64) * mom
    dfy = E[i_arr, 1].astype(np.float64) * mom
    np.add.at(fx, y_arr, dfx)
    np.add.at(fy, y_arr, dfy)
    return {"y": list(range(int(ny))), "fx": fx.tolist(), "fy": fy.tolist()}
