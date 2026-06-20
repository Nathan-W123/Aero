"""3D aerodynamic force computation and pressure/viscous split."""

import numpy as np
from typing import Dict, Tuple
from .lbm.d3q19 import E3, OPP3, compute_feq_3d, compute_macroscopic_3d


def compute_forces_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
) -> Tuple[float, float, float]:
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0

    i_arr   = links[:, 0]
    z_arr   = links[:, 1]
    y_arr   = links[:, 2]
    x_arr   = links[:, 3]
    opp_arr = OPP3[i_arr]

    f_out = f_pre[i_arr,   z_arr, y_arr, x_arr]
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


def compute_force_split_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    if links.shape[0] == 0:
        return (0.0,) * 6

    fx_p = fy_p = fz_p = fx_v = fy_v = fz_v = 0.0
    for i, z, y, x in links:
        opp = int(OPP3[i])
        ex = float(E3[i, 0])
        ey = float(E3[i, 1])
        ez = float(E3[i, 2])
        f_out = float(f_pre[i, z, y, x])
        f_in = float(f_post[opp, z, y, x])

        rho, ux, uy, uz = compute_macroscopic_3d(f_pre[:, z : z + 1, y : y + 1, x : x + 1])
        feq = compute_feq_3d(rho, ux, uy, uz)
        feq_out = float(feq[i, 0, 0, 0])
        feq_in = float(feq[opp, 0, 0, 0])

        mom_p = feq_out + feq_in
        mom_v = (f_out - feq_out) + (f_in - feq_in)
        fx_p += ex * mom_p
        fy_p += ey * mom_p
        fz_p += ez * mom_p
        fx_v += ex * mom_v
        fy_v += ey * mom_v
        fz_v += ez * mom_v
    return fx_p, fy_p, fz_p, fx_v, fy_v, fz_v


def forces_to_coefficients_3d(
    Fx: float,
    Fy: float,
    Fz: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float, float]:
    F_dyn = 0.5 * rho0 * u0 * u0 * D * D
    if F_dyn == 0.0:
        return 0.0, 0.0, 0.0
    return Fx / F_dyn, Fy / F_dyn, Fz / F_dyn


def split_to_coefficients_3d(
    fx_p: float,
    fy_p: float,
    fz_p: float,
    fx_v: float,
    fy_v: float,
    fz_v: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float, float, float, float, float]:
    F_dyn = 0.5 * rho0 * u0 * u0 * D * D
    if F_dyn == 0.0:
        return (0.0,) * 6
    return (
        fx_p / F_dyn, fy_p / F_dyn, fz_p / F_dyn,
        fx_v / F_dyn, fy_v / F_dyn, fz_v / F_dyn,
    )


def compute_force_moment_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    center_z: float,
) -> Tuple[float, float, float, float, float, float]:
    """Return raw Fx, Fy, Fz and moments Mx, My, Mz."""
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    i_arr = links[:, 0]
    z_arr = links[:, 1]
    y_arr = links[:, 2]
    x_arr = links[:, 3]
    opp_arr = OPP3[i_arr]
    mom = f_pre[i_arr, z_arr, y_arr, x_arr] + f_post[opp_arr, z_arr, y_arr, x_arr]
    dfx = E3[i_arr, 0].astype(np.float64) * mom
    dfy = E3[i_arr, 1].astype(np.float64) * mom
    dfz = E3[i_arr, 2].astype(np.float64) * mom
    rx = x_arr.astype(np.float64) - float(center_x)
    ry = y_arr.astype(np.float64) - float(center_y)
    rz = z_arr.astype(np.float64) - float(center_z)
    mx = float(np.sum(ry * dfz - rz * dfy))
    my = float(np.sum(rz * dfx - rx * dfz))
    mz = float(np.sum(rx * dfy - ry * dfx))
    return float(np.sum(dfx)), float(np.sum(dfy)), float(np.sum(dfz)), mx, my, mz


def moments_to_coefficients_3d(
    mx: float,
    my: float,
    mz: float,
    rho0: float,
    u0: float,
    D: float,
) -> Tuple[float, float, float]:
    denom = 0.5 * rho0 * u0 * u0 * D * D * D
    if denom == 0.0:
        return 0.0, 0.0, 0.0
    return mx / denom, my / denom, mz / denom


def spanwise_force_profile_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    nz: int,
) -> Dict[str, list[float]]:
    """Return sectional force sums per spanwise z-index."""
    fx = np.zeros(int(nz), dtype=np.float64)
    fy = np.zeros(int(nz), dtype=np.float64)
    fz = np.zeros(int(nz), dtype=np.float64)
    if links.shape[0] == 0:
        idx = list(range(int(nz)))
        return {"z": idx, "fx": fx.tolist(), "fy": fy.tolist(), "fz": fz.tolist()}
    i_arr = links[:, 0]
    z_arr = links[:, 1]
    y_arr = links[:, 2]
    x_arr = links[:, 3]
    opp_arr = OPP3[i_arr]
    mom = f_pre[i_arr, z_arr, y_arr, x_arr] + f_post[opp_arr, z_arr, y_arr, x_arr]
    np.add.at(fx, z_arr, E3[i_arr, 0].astype(np.float64) * mom)
    np.add.at(fy, z_arr, E3[i_arr, 1].astype(np.float64) * mom)
    np.add.at(fz, z_arr, E3[i_arr, 2].astype(np.float64) * mom)
    return {"z": list(range(int(nz))), "fx": fx.tolist(), "fy": fy.tolist(), "fz": fz.tolist()}
