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
        q, ny, nx = f.shape
        for y in nb.prange(ny):
            feq = np.empty(q)
            for x in range(nx):
                rho = 0.0
                mx = 0.0
                my = 0.0
                for i in range(q):
                    fi = f[i, y, x]
                    rho += fi
                    mx += ex[i] * fi
                    my += ey[i] * fi
                inv_r = 1.0 / rho if rho > 0.0 else 0.0

                ax = 0.0
                ay = 0.0
                if force_mode == 1:
                    ax = acc_uniform[0]
                    ay = acc_uniform[1]
                elif force_mode == 2:
                    ax = acc_field[0, y, x]
                    ay = acc_field[1, y, x]

                if solid[y, x]:
                    ux = 0.0
                    uy = 0.0
                    ax = 0.0
                    ay = 0.0
                else:
                    ux = mx * inv_r + 0.5 * ax
                    uy = my * inv_r + 0.5 * ay
                usq = ux * ux + uy * uy
                ua = ux * ax + uy * ay

                for i in range(q):
                    eu = ex[i] * ux + ey[i] * uy
                    feq[i] = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)

                # second-order moment of the non-equilibrium part
                pxx = 0.0
                pxy = 0.0
                pyy = 0.0
                for i in range(q):
                    d = f[i, y, x] - feq[i]
                    pxx += ex[i] * ex[i] * d
                    pxy += ex[i] * ey[i] * d
                    pyy += ey[i] * ey[i] * d

                if force_mode != 0:
                    # fold the force's own second moment in
                    fx = rho * ax
                    fy = rho * ay
                    pxx += ux * fx
                    pxy += 0.5 * (ux * fy + uy * fx)
                    pyy += uy * fy

                om = omega_field[y, x] if use_omega_field else omega
                for i in range(q):
                    hxx = ex[i] * ex[i] - CS2
                    hxy = ex[i] * ey[i]
                    hyy = ey[i] * ey[i] - CS2
                    f1 = HERMITE2 * w[i] * (hxx * pxx + 2.0 * hxy * pxy + hyy * pyy)
                    out = feq[i] + (1.0 - om) * f1
                    if force_mode != 0:
                        eu = ex[i] * ux + ey[i] * uy
                        ea = ex[i] * ax + ey[i] * ay
                        # prefactor 1/2, not (1 - omega/2)
                        out += 0.5 * w[i] * rho * (3.0 * (ea - ua) + 9.0 * eu * ea)
                    f_post[i, y, x] = out
else:
    def regularized_collision_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for the JIT regularized kernel")
