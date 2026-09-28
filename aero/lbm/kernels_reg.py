"""
Regularized collision (Latt & Chopard 2006) for D2Q9.

BGK relaxes every moment of ``f`` at the same rate, including the
non-hydrodynamic "ghost" moments that carry no physics.  At low viscosity
(omega -> 2) those ghost moments are barely damped, and they are what makes
BGK go unstable before the flow itself does.

Regularization throws them away each step.  The non-equilibrium part is
projected onto the second-order Hermite basis and rebuilt from that
projection alone::

    Pi_ab^neq = sum_i c_ia c_ib (f_i - f_i^eq)
    f_i^(1)   = (w_i / (2 cs^4)) * (c_ia c_ib - cs^2 delta_ab) * Pi_ab^neq
    f_i^post  = f_i^eq + (1 - omega) f_i^(1)

which is BGK for the hydrodynamic moments and complete annihilation for
everything above them.  With cs^2 = 1/3 the prefactor ``1/(2 cs^4)`` is 4.5.

This keeps BGK's viscosity and its second-order accuracy — the Chapman-Enskog
expansion only ever used the second-order part — while removing the ghost
modes that limit the stable range.  It costs one extra pass over the
directions, cheaper than MRT's two matrix products.

Forcing needs one change from BGK, and it is easy to miss.  BGK keeps the
first moment of ``f_neq``, which under Guo forcing is ``-F/2``; regularization
projects that away, because ``f^(1)`` is built from the second moment alone
and has zero first moment.  Matching the momentum balance then gives

    sum_i c_i f_post = rho u + kappa F  ==  m + F   only for  kappa = 1/2

so the source carries ``1/2`` here rather than ``(1 - omega/2)``.  Using the
BGK prefactor instead leaves a momentum gain of ``(3/2 - omega/2) F``, i.e.
10% low at omega=1.2 and 35% low at omega=1.7 — a force-driven channel shows
it immediately.

Matching the *second* moment as well needs the force's own contribution
folded into the regularized moment,

    Pi_reg = Pi^neq + (u_a F_b + u_b F_a) / 2

which makes the regularized operator reproduce BGK's stress exactly.
"""

from __future__ import annotations

import numpy as np

from .d2q9 import E, W, compute_feq, compute_macroscopic
from .physics import inv_positive

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    nb = None  # type: ignore[assignment]

CS2 = 1.0 / 3.0
#: 1 / (2 cs^4) for cs^2 = 1/3
HERMITE2 = 4.5


def regularized_collision_numpy(
    f: np.ndarray,
    f_post: np.ndarray,
    solid: np.ndarray,
    omega: float,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
    omega_field: np.ndarray | None = None,
    acc: np.ndarray | None = None,
) -> None:
    """Pure-NumPy regularized collision with optional Guo forcing."""
    rho, ux, uy = compute_macroscopic(f)
    fluid = ~solid
    if acc is not None:
        ax = np.where(fluid, acc[0], 0.0)
        ay = np.where(fluid, acc[1], 0.0)
        ux = ux + 0.5 * ax           # half-force correction
        uy = uy + 0.5 * ay
    else:
        ax = ay = None
    ux = np.where(solid, 0.0, ux)
    uy = np.where(solid, 0.0, uy)

    feq = compute_feq(rho, ux, uy)
    fneq = f - feq

    exf = ex.astype(np.float64)
    eyf = ey.astype(np.float64)
    pi_xx = np.einsum("i,iyx->yx", exf * exf, fneq)
    pi_xy = np.einsum("i,iyx->yx", exf * eyf, fneq)
    pi_yy = np.einsum("i,iyx->yx", eyf * eyf, fneq)

    om = omega if omega_field is None else omega_field
    ua = None if ax is None else ux * ax + uy * ay
    if ax is not None:
        # fold the force's own second moment in, so the stress matches BGK
        fx = rho * ax
        fy = rho * ay
        pi_xx = pi_xx + ux * fx
        pi_xy = pi_xy + 0.5 * (ux * fy + uy * fx)
        pi_yy = pi_yy + uy * fy

    for i in range(f.shape[0]):
        hxx = exf[i] * exf[i] - CS2
        hxy = exf[i] * eyf[i]
        hyy = eyf[i] * eyf[i] - CS2
        f1 = HERMITE2 * w[i] * (hxx * pi_xx + 2.0 * hxy * pi_xy + hyy * pi_yy)
        f_post[i] = feq[i] + (1.0 - om) * f1
        if ax is not None:
            eu = exf[i] * ux + eyf[i] * uy
            ea = exf[i] * ax + eyf[i] * ay
            # prefactor 1/2, not (1 - omega/2): see the module docstring
            f_post[i] += 0.5 * w[i] * rho * (3.0 * (ea - ua) + 9.0 * eu * ea)


if _HAS_NUMBA:
    # x-tile length, as in kernels.collision_kernel.
    _TILE = 64

    @nb.njit(cache=True, parallel=True)
    def regularized_collision_kernel(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        w: np.ndarray,
        omega_field: np.ndarray,
        use_omega_field: bool,
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
    ) -> None:
        """
        Regularized collision with Guo forcing, parallel over rows.

        Tiled like the BGK kernel so every inner loop runs along x at unit
        stride and vectorises; per cell the operations and their order are
        those of the one-cell-at-a-time form, so results are bit-identical.
        """
        q, ny, nx = f.shape
        for y in nb.prange(ny):
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            ua_t = np.empty(_TILE)
            om_t = np.empty(_TILE)
            ax_t = np.empty(_TILE)
            ay_t = np.empty(_TILE)
            pxx_t = np.empty(_TILE)
            pxy_t = np.empty(_TILE)
            pyy_t = np.empty(_TILE)
            feq_t = np.empty((q, _TILE))

            for x0 in range(0, nx, _TILE):
                n = nx - x0
                if n > _TILE:
                    n = _TILE

                for k in range(n):
                    rho_t[k] = 0.0
                    ux_t[k] = 0.0
                    uy_t[k] = 0.0
                for i in range(q):
                    exi = ex[i]
                    eyi = ey[i]
                    for k in range(n):
                        fi = f[i, y, x0 + k]
                        rho_t[k] += fi
                        ux_t[k] += exi * fi
                        uy_t[k] += eyi * fi

                for k in range(n):
                    rho = rho_t[k]
                    inv_r = 1.0 / rho if rho > 0.0 else 0.0
                    ax = 0.0
                    ay = 0.0
                    if force_mode == 1:
                        ax = acc_uniform[0]
                        ay = acc_uniform[1]
                    elif force_mode == 2:
                        ax = acc_field[0, y, x0 + k]
                        ay = acc_field[1, y, x0 + k]
                    if solid[y, x0 + k]:
                        ux = 0.0
                        uy = 0.0
                        ax = 0.0
                        ay = 0.0
                    else:
                        ux = ux_t[k] * inv_r + 0.5 * ax
                        uy = uy_t[k] * inv_r + 0.5 * ay
                    ux_t[k] = ux
                    uy_t[k] = uy
                    ax_t[k] = ax
                    ay_t[k] = ay
                    usq_t[k] = ux * ux + uy * uy
                    ua_t[k] = ux * ax + uy * ay
                    om_t[k] = omega_field[y, x0 + k] if use_omega_field else omega
                    pxx_t[k] = 0.0
                    pxy_t[k] = 0.0
                    pyy_t[k] = 0.0

                # equilibrium and the second moment of the non-equilibrium part
                for i in range(q):
                    exi = ex[i]
                    eyi = ey[i]
                    wi = w[i]
                    cxx = exi * exi
                    cxy = exi * eyi
                    cyy = eyi * eyi
                    for k in range(n):
                        eu = exi * ux_t[k] + eyi * uy_t[k]
                        feq = wi * rho_t[k] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq_t[k])
                        feq_t[i, k] = feq
                        d = f[i, y, x0 + k] - feq
                        pxx_t[k] += cxx * d
                        pxy_t[k] += cxy * d
                        pyy_t[k] += cyy * d

                if force_mode != 0:
                    # fold the force's own second moment in
                    for k in range(n):
                        rho = rho_t[k]
                        ux = ux_t[k]
                        uy = uy_t[k]
                        fx = rho * ax_t[k]
                        fy = rho * ay_t[k]
                        pxx_t[k] += ux * fx
                        pxy_t[k] += 0.5 * (ux * fy + uy * fx)
                        pyy_t[k] += uy * fy

                for i in range(q):
                    exi = ex[i]
                    eyi = ey[i]
                    wi = w[i]
                    hxx = exi * exi - CS2
                    hxy = exi * eyi
                    hyy = eyi * eyi - CS2
                    hw = HERMITE2 * wi
                    for k in range(n):
                        f1 = hw * (hxx * pxx_t[k] + 2.0 * hxy * pxy_t[k] + hyy * pyy_t[k])
                        out = feq_t[i, k] + (1.0 - om_t[k]) * f1
                        if force_mode != 0:
                            eu = exi * ux_t[k] + eyi * uy_t[k]
                            ea = exi * ax_t[k] + eyi * ay_t[k]
                            # prefactor 1/2, not (1 - omega/2)
                            out += 0.5 * wi * rho_t[k] * (3.0 * (ea - ua_t[k]) + 9.0 * eu * ea)
                        f_post[i, y, x0 + k] = out
else:
    def regularized_collision_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for the JIT regularized kernel")
