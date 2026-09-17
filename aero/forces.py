"""
Aerodynamic force computation via the momentum exchange method.

Pressure/viscous decomposition splits each link's momentum exchange into
equilibrium (pressure) and non-equilibrium (viscous) parts using local
Chapman–Enskog moments — equivalent to integrating stress at the surface
and guaranteed to sum to the total momentum-exchange force.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .lbm.d2q9 import E, W, OPP, compute_feq, compute_macroscopic
from .lbm.physics import inv_positive


def _link_arrays(links: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Split a (N,3) link table into i/y/x/opp index arrays (all intp)."""
    i_arr = links[:, 0].astype(np.intp)
    y_arr = links[:, 1].astype(np.intp)
    x_arr = links[:, 2].astype(np.intp)
    return i_arr, y_arr, x_arr, OPP[i_arr].astype(np.intp)


def _link_feq_pair(
    f_pre: np.ndarray,
    i_arr: np.ndarray,
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    opp_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equilibrium values feq_i and feq_opp[i] at every surface-link node.

    The node moments are taken from ``f_pre`` (post-collision, pre-streaming),
    matching the per-link Chapman-Enskog split used for the pressure/viscous
    decomposition.  One gather over the link nodes replaces a Python-level
    loop that rebuilt the full 9-direction equilibrium per link.
    """
    f_node = f_pre[:, y_arr, x_arr]                      # (9, N_links)
    rho = f_node.sum(axis=0)
    inv_rho = inv_positive(rho)
    ex_all = E[:, 0].astype(np.float64)
    ey_all = E[:, 1].astype(np.float64)
    ux = inv_rho * (ex_all @ f_node)
    uy = inv_rho * (ey_all @ f_node)
    usq = ux * ux + uy * uy

    eu_out = ex_all[i_arr] * ux + ey_all[i_arr] * uy
    eu_in = ex_all[opp_arr] * ux + ey_all[opp_arr] * uy
    feq_out = W[i_arr] * rho * (1.0 + 3.0 * eu_out + 4.5 * eu_out * eu_out - 1.5 * usq)
    feq_in = W[opp_arr] * rho * (1.0 + 3.0 * eu_in + 4.5 * eu_in * eu_in - 1.5 * usq)
    return feq_out, feq_in


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

    i_arr, y_arr, x_arr, opp_arr = _link_arrays(links)
    feq_out, feq_in = _link_feq_pair(f_pre, i_arr, y_arr, x_arr, opp_arr)

    f_out = f_pre[i_arr, y_arr, x_arr]
    f_in = f_post[opp_arr, y_arr, x_arr]

    mom_p = feq_out + feq_in
    mom_v = (f_out - feq_out) + (f_in - feq_in)

    ex = E[i_arr, 0].astype(np.float64)
    ey = E[i_arr, 1].astype(np.float64)
    return (
        float(np.sum(ex * mom_p)),
        float(np.sum(ey * mom_p)),
        float(np.sum(ex * mom_v)),
        float(np.sum(ey * mom_v)),
    )


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
    ny = int(ny)
    if links.shape[0] == 0:
        zeros = [0.0] * ny
        return {"y": list(range(ny)), "fx": list(zeros), "fy": list(zeros)}
    i_arr, y_arr, x_arr, opp_arr = _link_arrays(links)
    mom = f_pre[i_arr, y_arr, x_arr] + f_post[opp_arr, y_arr, x_arr]
    # bincount is an order of magnitude faster than np.add.at for this pattern
    fx = np.bincount(y_arr, weights=E[i_arr, 0].astype(np.float64) * mom, minlength=ny)
    fy = np.bincount(y_arr, weights=E[i_arr, 1].astype(np.float64) * mom, minlength=ny)
    return {"y": list(range(ny)), "fx": fx.tolist(), "fy": fy.tolist()}


def surface_diagnostics_2d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    ny: int,
    want_profile: bool = True,
) -> Dict[str, object]:
    """
    Every momentum-exchange surface observable from a single gather.

    The solver needs the total force, the pressure/viscous split, the moment
    and (optionally) the sectional profile every timestep.  Calling the four
    public helpers separately re-gathers the same link values four times;
    this does the gather once and derives all of them from it.

    Returns keys: ``fx``, ``fy``, ``mz``, ``fx_p``, ``fy_p``, ``fx_v``,
    ``fy_v``, ``profile`` (None when ``want_profile`` is False).
    """
    ny = int(ny)
    if links.shape[0] == 0:
        empty: Optional[Dict[str, list]] = None
        if want_profile:
            zeros = [0.0] * ny
            empty = {"y": list(range(ny)), "fx": list(zeros), "fy": list(zeros)}
        return {
            "fx": 0.0, "fy": 0.0, "mz": 0.0,
            "fx_p": 0.0, "fy_p": 0.0, "fx_v": 0.0, "fy_v": 0.0,
            "profile": empty,
        }

    i_arr, y_arr, x_arr, opp_arr = _link_arrays(links)
    f_out = f_pre[i_arr, y_arr, x_arr]
    f_in = f_post[opp_arr, y_arr, x_arr]
    mom = f_out + f_in

    ex = E[i_arr, 0].astype(np.float64)
    ey = E[i_arr, 1].astype(np.float64)
    dfx = ex * mom
    dfy = ey * mom

    rx = x_arr.astype(np.float64) - float(center_x)
    ry = y_arr.astype(np.float64) - float(center_y)

    feq_out, feq_in = _link_feq_pair(f_pre, i_arr, y_arr, x_arr, opp_arr)
    mom_p = feq_out + feq_in
    mom_v = (f_out - feq_out) + (f_in - feq_in)

    profile = None
    if want_profile:
        fx_prof = np.bincount(y_arr, weights=dfx, minlength=ny)
        fy_prof = np.bincount(y_arr, weights=dfy, minlength=ny)
        profile = {
            "y": list(range(ny)),
            "fx": fx_prof.tolist(),
            "fy": fy_prof.tolist(),
        }

    return {
        "fx": float(np.sum(dfx)),
        "fy": float(np.sum(dfy)),
        "mz": float(np.sum(rx * dfy - ry * dfx)),
        "fx_p": float(np.sum(ex * mom_p)),
        "fy_p": float(np.sum(ey * mom_p)),
        "fx_v": float(np.sum(ex * mom_v)),
        "fy_v": float(np.sum(ey * mom_v)),
        "profile": profile,
    }
