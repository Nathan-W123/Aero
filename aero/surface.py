"""
Per-surface-node observables: pressure coefficient, wall shear stress, y+.

The link table already carries everything needed — each entry is a fluid cell
and the lattice direction pointing at the body — so these come out of one
gather, the same pass the force computation uses.  Links are grouped by node
first, because a single link direction is not the surface normal; see
:func:`surface_nodes`.

Definitions (lattice units throughout; the coefficients are dimensionless)
-------------------------------------------------------------------------
Pressure coefficient
    ``Cp = (p - p_ref) / (1/2 rho_ref u_ref^2)``  with ``p = cs^2 rho``.
    Sampling ``rho`` at the fluid cell adjacent to the wall is the standard
    LBM surface-pressure estimate; it sits half a lattice spacing off the
    wall, which is the same place bounce-back puts the wall.

Wall shear stress
    From the non-equilibrium momentum flux, which is what carries the viscous
    stress in LBM and needs no finite differencing of the velocity field:

        Pi_ab^neq = sum_i c_ia c_ib (f_i - f_i^eq)
        tau_ab    = -(1 - omega/2) * Pi_ab^neq

    The ``(1 - omega/2)`` factor is the same second-order correction that
    appears in the forcing scheme.  The wall shear is the tangential part of
    ``tau . n`` for the wall normal ``n``.

    This is evaluated at the first fluid node, half a lattice spacing off the
    wall, so on a steep near-wall gradient it reads slightly below the value
    *at* the wall — for plane Poiseuille the offset is the exact
    ``du/dy(0.5)`` rather than ``du/dy(0)``, about 2-3% on a 40-cell channel.
    That is inherent to a node-based estimate, not an error in the stress.

Friction velocity and y+
    ``u_tau = sqrt(|tau_w| / rho)`` and ``y+ = y u_tau / nu``.  The first
    fluid node sits half a cell from a bounce-back wall, so ``y = 0.5``.
    This is the number that says whether an LES actually resolved its
    boundary layer: y+ ~ 1 is resolved, y+ >> 30 needs a wall model.

Skin friction
    ``Cf = |tau_w| / (1/2 rho_ref u_ref^2)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CS2 = 1.0 / 3.0


def _link_columns(links: np.ndarray, ndim: int):
    """Split a link table into direction and coordinate index arrays."""
    i_arr = links[:, 0].astype(np.intp)
    coords = tuple(links[:, k + 1].astype(np.intp) for k in range(ndim))
    return i_arr, coords


def surface_nodes(links: np.ndarray, e: np.ndarray, w: np.ndarray):
    """
    Group a link table by node and give each node a surface normal.

    A single link direction is **not** the surface normal.  A flat wall below
    a fluid row produces three links per node in D2Q9 — the axial one and two
    diagonals — and only the axial one points along the normal; using each
    link's own direction would compute the shear on three different planes and
    average them.

    The normal comes from the weighted sum of the links at a node,
    ``n ~ -sum_i w_i e_i``, which recovers the exact normal on a flat face and
    a sensibly smoothed one on a staircase.

    Returns ``(coords, normals, counts)`` where ``coords`` is ``(M, ndim)``
    unique node indices in grid order.
    """
    ndim = e.shape[1]
    if links.shape[0] == 0:
        return (np.empty((0, ndim), dtype=np.intp),
                np.empty((0, ndim), dtype=np.float64),
                np.empty(0, dtype=np.intp))

    i_arr, coords = _link_columns(links, ndim)
    node_index = np.stack(coords, axis=1)
    unique, inverse, counts = np.unique(
        node_index, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()

    weighted = -(w[i_arr][:, None] * e[i_arr].astype(np.float64))     # (N, ndim)
    normals = np.zeros((unique.shape[0], ndim), dtype=np.float64)
    for d in range(ndim):
        normals[:, d] = np.bincount(
            inverse, weights=weighted[:, d], minlength=unique.shape[0]
        )
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals, norm, out=np.zeros_like(normals), where=norm > 1e-30
    )
    return unique, normals, counts


def surface_fields(
    f: np.ndarray,
    links: np.ndarray,
    *,
    e: np.ndarray,
    w: np.ndarray,
    omega: float,
    rho_ref: float = 1.0,
    u_ref: float = 1.0,
    nu: Optional[float] = None,
    omega_field: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Per-link surface quantities from the current distribution.

    Parameters
    ----------
    f           : (Q, *grid) distribution, post-streaming
    links       : (N, 1+ndim) link table, columns ``[i, (z,) y, x]``
    e, w        : lattice velocities (Q, ndim) and weights (Q,)
    omega       : relaxation rate; ``omega_field`` overrides it per cell (LES)
    rho_ref, u_ref : normalisation for the coefficients
    nu          : kinematic viscosity; derived from ``omega`` when omitted

    Returns a dict of ``(M,)`` arrays, one entry per surface *node* (links are
    grouped by node — see :func:`surface_nodes`), plus the node coordinates so
    callers can map values back onto the geometry.

    Note the two index orders in play: ``coords`` follows the grid array
    layout (``(y, x)`` in 2D, ``(z, y, x)`` in 3D) while ``normal`` and
    ``shear_vector`` are lattice vectors, ordered ``(x, y[, z])`` like ``e``.
    Everything derived from them is a magnitude, so the two never mix.
    """
    ndim = e.shape[1]
    if links.shape[0] == 0:
        empty = np.empty(0, dtype=np.float64)
        out = {k: empty.copy() for k in
               ("cp", "pressure", "rho", "tau_wall", "cf", "u_tau", "y_plus")}
        out["normal"] = np.empty((0, ndim), dtype=np.float64)
        out["shear_vector"] = np.empty((0, ndim), dtype=np.float64)
        out["coords"] = np.empty((0, ndim), dtype=np.intp)
        out["links_per_node"] = np.empty(0, dtype=np.intp)
        return out

    node_coords, normal, link_counts = surface_nodes(links, e, w)
    coords = tuple(node_coords[:, d] for d in range(ndim))
    node = f[(slice(None),) + coords]                    # (Q, M)

    rho = node.sum(axis=0)
    inv_rho = np.where(rho > 0.0, 1.0 / np.where(rho > 0.0, rho, 1.0), 0.0)
    e_f = e.astype(np.float64)
    u = np.stack([inv_rho * (e_f[:, d] @ node) for d in range(ndim)])   # (ndim, N)

    # equilibrium at the link node
    usq = np.einsum("dn,dn->n", u, u)
    eu = e_f @ u                                          # (Q, N)
    feq = w[:, None] * rho[None, :] * (
        1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq[None, :]
    )
    fneq = node - feq

    # non-equilibrium momentum flux Pi_ab = sum_i c_ia c_ib fneq_i
    pi = np.empty((ndim, ndim, rho.size), dtype=np.float64)
    for a in range(ndim):
        for b in range(a, ndim):
            pi[a, b] = (e_f[:, a] * e_f[:, b]) @ fneq
            pi[b, a] = pi[a, b]

    om = omega if omega_field is None else omega_field[coords]
    prefactor = -(1.0 - 0.5 * np.asarray(om))
    stress = prefactor * pi                               # tau_ab

    traction = np.einsum("abn,nb->na", stress, normal)    # tau . n
    normal_part = np.einsum("na,na->n", traction, normal)
    shear_vec = traction - normal_part[:, None] * normal
    tau_wall = np.linalg.norm(shear_vec, axis=1)

    dyn = 0.5 * rho_ref * u_ref * u_ref
    pressure = CS2 * rho
    p_ref = CS2 * rho_ref

    if nu is None:
        nu = (1.0 / np.asarray(om) - 0.5) / 3.0
    u_tau = np.sqrt(tau_wall * inv_rho)
    # first fluid node sits half a cell off a mid-link bounce-back wall
    y_plus = 0.5 * u_tau / np.maximum(np.asarray(nu), 1e-30)

    return {
        "coords": node_coords,
        "normal": normal,
        "links_per_node": link_counts,
        "rho": rho,
        "pressure": pressure,
        "cp": (pressure - p_ref) / dyn if dyn > 0 else np.zeros_like(pressure),
        "tau_wall": tau_wall,
        "shear_vector": shear_vec,
        "cf": tau_wall / dyn if dyn > 0 else np.zeros_like(tau_wall),
        "u_tau": u_tau,
        "y_plus": np.broadcast_to(y_plus, tau_wall.shape).copy(),
    }


def surface_summary(fields: Dict[str, np.ndarray]) -> Dict[str, object]:
    """JSON-safe reduction of :func:`surface_fields`, for results.json."""
    if fields["rho"].size == 0:
        return {"links": 0}

    yp = fields["y_plus"]
    return {
        "links": int(fields["rho"].size),
        "cp_min": float(np.min(fields["cp"])),
        "cp_max": float(np.max(fields["cp"])),
        "cp_mean": float(np.mean(fields["cp"])),
        "cf_mean": float(np.mean(fields["cf"])),
        "cf_max": float(np.max(fields["cf"])),
        "u_tau_mean": float(np.mean(fields["u_tau"])),
        "y_plus_min": float(np.min(yp)),
        "y_plus_mean": float(np.mean(yp)),
        "y_plus_max": float(np.max(yp)),
        "wall_resolution": _wall_resolution_verdict(float(np.mean(yp))),
    }


def _wall_resolution_verdict(y_plus_mean: float) -> str:
    """
    Plain-language reading of the mean y+.

    The usual LES thresholds: y+ ~ 1 resolves the viscous sublayer, up to ~5
    is still inside it, 5-30 is the buffer layer where neither a resolved nor
    a modelled treatment is clean, and above 30 the first node is in the log
    layer and a wall model is required.
    """
    if y_plus_mean <= 1.0:
        return "wall-resolved (y+ <= 1)"
    if y_plus_mean <= 5.0:
        return "viscous sublayer (y+ <= 5)"
    if y_plus_mean <= 30.0:
        return "buffer layer (5 < y+ <= 30) — marginal for both resolved and modelled walls"
    return "log layer (y+ > 30) — needs a wall model"


def surface_profile(
    fields: Dict[str, np.ndarray],
    axis: int,
    length: int,
    quantity: str = "cp",
) -> Dict[str, list]:
    """
    Average a surface quantity into bins along one grid axis.

    Useful for a Cp distribution down the span or along the chord, where the
    raw per-link scatter is harder to read than a sectional mean.
    """
    coords = fields["coords"]
    if coords.shape[0] == 0:
        return {"index": list(range(length)), quantity: [0.0] * length, "count": [0] * length}
    idx = coords[:, axis]
    counts = np.bincount(idx, minlength=length)
    totals = np.bincount(idx, weights=fields[quantity], minlength=length)
    mean = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    return {
        "index": list(range(length)),
        quantity: mean.tolist(),
        "count": counts.astype(int).tolist(),
    }
