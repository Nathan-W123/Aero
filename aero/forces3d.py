"""3D aerodynamic force computation and pressure/viscous split."""

import numpy as np
from typing import Dict, Optional, Tuple
from .lbm.d3q19 import E3, W3, OPP3, compute_feq_3d, compute_macroscopic_3d
from .lbm.lattice3d import D3Q19, Lattice3D
from .lbm.physics import inv_positive


def _link_arrays_3d(
    links: np.ndarray, lat: Lattice3D = D3Q19
) -> Tuple[np.ndarray, ...]:
    """Split a (N,4) link table into i/z/y/x/opp index arrays (all intp)."""
    i_arr = links[:, 0].astype(np.intp)
    z_arr = links[:, 1].astype(np.intp)
    y_arr = links[:, 2].astype(np.intp)
    x_arr = links[:, 3].astype(np.intp)
    return i_arr, z_arr, y_arr, x_arr, lat.OPP[i_arr].astype(np.intp)


def _link_feq_pair_3d(
    f_pre: np.ndarray,
    i_arr: np.ndarray,
    z_arr: np.ndarray,
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    opp_arr: np.ndarray,
    lat: Lattice3D = D3Q19,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equilibrium values feq_i and feq_opp[i] at every surface-link node.

    Moments come from ``f_pre`` (post-collision, pre-streaming).  One gather
    over the link nodes replaces a Python-level loop that rebuilt the full
    19-direction equilibrium per link.
    """
    f_node = f_pre[:, z_arr, y_arr, x_arr]               # (Q, N_links)
    rho = f_node.sum(axis=0)
    inv_rho = inv_positive(rho)
    ex_all, ey_all, ez_all = lat.ex, lat.ey, lat.ez
    ux = inv_rho * (ex_all @ f_node)
    uy = inv_rho * (ey_all @ f_node)
    uz = inv_rho * (ez_all @ f_node)
    usq = ux * ux + uy * uy + uz * uz

    eu_out = ex_all[i_arr] * ux + ey_all[i_arr] * uy + ez_all[i_arr] * uz
    eu_in = ex_all[opp_arr] * ux + ey_all[opp_arr] * uy + ez_all[opp_arr] * uz
    poly_out = 1.0 + 3.0 * eu_out + 4.5 * eu_out * eu_out - 1.5 * usq
    poly_in = 1.0 + 3.0 * eu_in + 4.5 * eu_in * eu_in - 1.5 * usq
    if lat.h3:
        # must match the collision's equilibrium, or the pressure/viscous
        # split leaks the H3 term into the "viscous" residual
        poly_out = poly_out + 4.5 * (eu_out ** 3 - eu_out * usq)
        poly_in = poly_in + 4.5 * (eu_in ** 3 - eu_in * usq)
    feq_out = lat.W[i_arr] * rho * poly_out
    feq_in = lat.W[opp_arr] * rho * poly_in
    return feq_out, feq_in


def compute_forces_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    lattice: Lattice3D = D3Q19,
) -> Tuple[float, float, float]:
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0

    i_arr   = links[:, 0]
    z_arr   = links[:, 1]
    y_arr   = links[:, 2]
    x_arr   = links[:, 3]
    opp_arr = lattice.OPP[i_arr]

    f_out = f_pre[i_arr,   z_arr, y_arr, x_arr]
    f_in  = f_post[opp_arr, z_arr, y_arr, x_arr]
    mom   = f_out + f_in

    ex = lattice.ex[i_arr]
    ey = lattice.ey[i_arr]
    ez = lattice.ez[i_arr]

    return (
        float(np.sum(ex * mom)),
        float(np.sum(ey * mom)),
        float(np.sum(ez * mom)),
    )


def compute_force_split_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    lattice: Lattice3D = D3Q19,
) -> Tuple[float, float, float, float, float, float]:
    if links.shape[0] == 0:
        return (0.0,) * 6

    i_arr, z_arr, y_arr, x_arr, opp_arr = _link_arrays_3d(links, lattice)
    feq_out, feq_in = _link_feq_pair_3d(
        f_pre, i_arr, z_arr, y_arr, x_arr, opp_arr, lattice
    )

    f_out = f_pre[i_arr, z_arr, y_arr, x_arr]
    f_in = f_post[opp_arr, z_arr, y_arr, x_arr]

    mom_p = feq_out + feq_in
    mom_v = (f_out - feq_out) + (f_in - feq_in)

    ex = lattice.ex[i_arr]
    ey = lattice.ey[i_arr]
    ez = lattice.ez[i_arr]
    return (
        float(np.sum(ex * mom_p)),
        float(np.sum(ey * mom_p)),
        float(np.sum(ez * mom_p)),
        float(np.sum(ex * mom_v)),
        float(np.sum(ey * mom_v)),
        float(np.sum(ez * mom_v)),
    )


def projected_frontal_area(solid: np.ndarray) -> float:
    """
    Frontal area of a voxelised body: the number of (z, y) columns that hit
    solid anywhere along the streamwise x axis.

    This is the area the momentum-exchange force actually acts over, and it
    works for any shape including an STL, so it is the solver's default
    reference area.  For analytic bodies the geometry classes also expose the
    exact value (pi r^2 for a sphere, h x d for a box) for comparison with
    published coefficients; the two differ only by voxelisation, ~1-3%.
    """
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 3:
        raise ValueError("projected_frontal_area expects a (Nz, Ny, Nx) mask")
    return float(solid.any(axis=2).sum())


def forces_to_coefficients_3d(
    Fx: float,
    Fy: float,
    Fz: float,
    rho0: float,
    u0: float,
    area: float,
) -> Tuple[float, float, float]:
    """
    ``C = F / (1/2 rho0 u0^2 A)`` with ``A`` the frontal reference area.

    ``area`` is an *area*, not a length.  An earlier version took the
    reference length ``D`` and used ``D^2``, which is the frontal area of
    nothing but a square box: for a sphere it under-reports every
    coefficient by pi/4, a 21.5% error that looked like a physics problem.
    """
    F_dyn = 0.5 * rho0 * u0 * u0 * area
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
    area: float,
) -> Tuple[float, float, float, float, float, float]:
    F_dyn = 0.5 * rho0 * u0 * u0 * area
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
    lattice: Lattice3D = D3Q19,
) -> Tuple[float, float, float, float, float, float]:
    """Return raw Fx, Fy, Fz and moments Mx, My, Mz."""
    if links.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    i_arr = links[:, 0]
    z_arr = links[:, 1]
    y_arr = links[:, 2]
    x_arr = links[:, 3]
    opp_arr = lattice.OPP[i_arr]
    mom = f_pre[i_arr, z_arr, y_arr, x_arr] + f_post[opp_arr, z_arr, y_arr, x_arr]
    dfx = lattice.ex[i_arr] * mom
    dfy = lattice.ey[i_arr] * mom
    dfz = lattice.ez[i_arr] * mom
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
    area: float,
    D: float,
) -> Tuple[float, float, float]:
    """Moment coefficients, ``M / (1/2 rho0 u0^2 A D)``."""
    denom = 0.5 * rho0 * u0 * u0 * area * D
    if denom == 0.0:
        return 0.0, 0.0, 0.0
    return mx / denom, my / denom, mz / denom


def spanwise_force_profile_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    nz: int,
    lattice: Lattice3D = D3Q19,
) -> Dict[str, list[float]]:
    """Return sectional force sums per spanwise z-index."""
    nz = int(nz)
    if links.shape[0] == 0:
        zeros = [0.0] * nz
        return {
            "z": list(range(nz)),
            "fx": list(zeros), "fy": list(zeros), "fz": list(zeros),
        }
    i_arr, z_arr, y_arr, x_arr, opp_arr = _link_arrays_3d(links, lattice)
    mom = f_pre[i_arr, z_arr, y_arr, x_arr] + f_post[opp_arr, z_arr, y_arr, x_arr]
    # bincount is an order of magnitude faster than np.add.at for this pattern
    fx = np.bincount(z_arr, weights=lattice.ex[i_arr] * mom, minlength=nz)
    fy = np.bincount(z_arr, weights=lattice.ey[i_arr] * mom, minlength=nz)
    fz = np.bincount(z_arr, weights=lattice.ez[i_arr] * mom, minlength=nz)
    return {"z": list(range(nz)), "fx": fx.tolist(), "fy": fy.tolist(), "fz": fz.tolist()}


def surface_diagnostics_3d(
    f_pre: np.ndarray,
    f_post: np.ndarray,
    links: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    center_z: float,
    nz: int,
    want_profile: bool = True,
    lattice: Lattice3D = D3Q19,
) -> Dict[str, object]:
    """
    Every momentum-exchange surface observable from a single gather.

    3D counterpart of :func:`aero.forces.surface_diagnostics_2d`.  Returns
    keys ``fx``, ``fy``, ``fz``, ``mx``, ``my``, ``mz``, the six split
    components ``f{x,y,z}_{p,v}``, and ``profile``.
    """
    nz = int(nz)
    if links.shape[0] == 0:
        empty: Optional[Dict[str, list]] = None
        if want_profile:
            zeros = [0.0] * nz
            empty = {
                "z": list(range(nz)),
                "fx": list(zeros), "fy": list(zeros), "fz": list(zeros),
            }
        out: Dict[str, object] = {k: 0.0 for k in (
            "fx", "fy", "fz", "mx", "my", "mz",
            "fx_p", "fy_p", "fz_p", "fx_v", "fy_v", "fz_v",
        )}
        out["profile"] = empty
        return out

    i_arr, z_arr, y_arr, x_arr, opp_arr = _link_arrays_3d(links, lattice)
    f_out = f_pre[i_arr, z_arr, y_arr, x_arr]
    f_in = f_post[opp_arr, z_arr, y_arr, x_arr]
    mom = f_out + f_in

    ex = lattice.ex[i_arr]
    ey = lattice.ey[i_arr]
    ez = lattice.ez[i_arr]
    dfx = ex * mom
    dfy = ey * mom
    dfz = ez * mom

    rx = x_arr.astype(np.float64) - float(center_x)
    ry = y_arr.astype(np.float64) - float(center_y)
    rz = z_arr.astype(np.float64) - float(center_z)

    feq_out, feq_in = _link_feq_pair_3d(
        f_pre, i_arr, z_arr, y_arr, x_arr, opp_arr, lattice
    )
    mom_p = feq_out + feq_in
    mom_v = (f_out - feq_out) + (f_in - feq_in)

    profile = None
    if want_profile:
        profile = {
            "z": list(range(nz)),
            "fx": np.bincount(z_arr, weights=dfx, minlength=nz).tolist(),
            "fy": np.bincount(z_arr, weights=dfy, minlength=nz).tolist(),
            "fz": np.bincount(z_arr, weights=dfz, minlength=nz).tolist(),
        }

    return {
        "fx": float(np.sum(dfx)),
        "fy": float(np.sum(dfy)),
        "fz": float(np.sum(dfz)),
        "mx": float(np.sum(ry * dfz - rz * dfy)),
        "my": float(np.sum(rz * dfx - rx * dfz)),
        "mz": float(np.sum(rx * dfy - ry * dfx)),
        "fx_p": float(np.sum(ex * mom_p)),
        "fy_p": float(np.sum(ey * mom_p)),
        "fz_p": float(np.sum(ez * mom_p)),
        "fx_v": float(np.sum(ex * mom_v)),
        "fy_v": float(np.sum(ey * mom_v)),
        "fz_v": float(np.sum(ez * mom_v)),
        "profile": profile,
    }
