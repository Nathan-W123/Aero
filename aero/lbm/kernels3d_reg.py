"""
Regularized collision (Latt & Chopard 2006) for D3Q19.

Same scheme as :mod:`aero.lbm.kernels_reg`, with the six independent
components of the second-order non-equilibrium moment.
"""

from __future__ import annotations

import numpy as np

from .lattice3d import D3Q19, compute_feq, compute_macroscopic
from .kernels_reg import CS2, HERMITE2

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


def regularized_collision_numpy_3d(
    f: np.ndarray,
    f_post: np.ndarray,
    solid: np.ndarray,
    omega: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
    omega_field: np.ndarray | None = None,
    acc: np.ndarray | None = None,
    lattice=D3Q19,
) -> None:
    """Pure-NumPy regularized collision with optional Guo forcing."""
    rho, ux, uy, uz = compute_macroscopic(f, lattice)
    fluid = ~solid
    if acc is not None:
        ax = np.where(fluid, acc[0], 0.0)
        ay = np.where(fluid, acc[1], 0.0)
        az = np.where(fluid, acc[2], 0.0)
        ux = ux + 0.5 * ax           # half-force correction
        uy = uy + 0.5 * ay
        uz = uz + 0.5 * az
    else:
        ax = ay = az = None
    ux = np.where(solid, 0.0, ux)
    uy = np.where(solid, 0.0, uy)
    uz = np.where(solid, 0.0, uz)

    feq = compute_feq(rho, ux, uy, uz, lattice)
    fneq = f - feq

    e = [ex.astype(np.float64), ey.astype(np.float64), ez.astype(np.float64)]
    pi = {}
    for a in range(3):
        for b in range(a, 3):
            pi[(a, b)] = np.einsum("i,izyx->zyx", e[a] * e[b], fneq)

    om = omega if omega_field is None else omega_field
    ua = None if ax is None else ux * ax + uy * ay + uz * az
    if ax is not None:
        # fold the force's own second moment in, so the stress matches BGK
        uvec = (ux, uy, uz)
        fvec = (rho * ax, rho * ay, rho * az)
        for a in range(3):
            for b in range(a, 3):
                pi[(a, b)] = pi[(a, b)] + 0.5 * (
                    uvec[a] * fvec[b] + uvec[b] * fvec[a]
                )

    for i in range(f.shape[0]):
        f1 = np.zeros_like(rho)
        for a in range(3):
            for b in range(3):
                lo, hi = (a, b) if a <= b else (b, a)
                h = e[a][i] * e[b][i] - (CS2 if a == b else 0.0)
                f1 = f1 + h * pi[(lo, hi)]
        f_post[i] = feq[i] + (1.0 - om) * (HERMITE2 * w[i] * f1)
        if ax is not None:
            eu = e[0][i] * ux + e[1][i] * uy + e[2][i] * uz
            ea = e[0][i] * ax + e[1][i] * ay + e[2][i] * az
            # prefactor 1/2, not (1 - omega/2): see kernels_reg
            f_post[i] += 0.5 * w[i] * rho * (3.0 * (ea - ua) + 9.0 * eu * ea)


if _HAS_NUMBA:
    from .kernels3d import _store_row

    # x-tile length, as in kernels3d.collision_kernel_3d.
    _TILE = 64

    @nb.njit(cache=True, parallel=True)
    def regularized_collision_kernel_3d(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
        w: np.ndarray,
        omega_field: np.ndarray,
        use_omega_field: bool,
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
        h3: float = 0.0,
        stream: bool = False,
    ) -> None:
        """
        Regularized collision with Guo forcing, parallel over z slices.

        Tiled like the BGK kernel: a run of x cells goes through the moment,
        equilibrium/stress and reconstruction passes together, and every inner
        loop walks one direction plane at unit stride, so it vectorises.
        Gathering the Q values of one cell at a time instead made this kernel
        4x slower than BGK.  Each cell still sees the same operations in the
        same order as the per-cell form, so the results are bit-identical.
        ``stream`` pushes each value straight to its streamed position, as in
        :func:`aero.lbm.kernels3d.collision_kernel_3d`.
        """
        q, nz, ny, nx = f.shape
        for z in nb.prange(nz):
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            uz_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            ua_t = np.empty(_TILE)
            om_t = np.empty(_TILE)
            ax_t = np.empty(_TILE)
            ay_t = np.empty(_TILE)
            az_t = np.empty(_TILE)
            pxx_t = np.empty(_TILE)
            pxy_t = np.empty(_TILE)
            pxz_t = np.empty(_TILE)
            pyy_t = np.empty(_TILE)
            pyz_t = np.empty(_TILE)
            pzz_t = np.empty(_TILE)
            feq_t = np.empty((q, _TILE))
            out_t = np.empty(_TILE)

            for y in range(ny):
                for x0 in range(0, nx, _TILE):
                    n = nx - x0
                    if n > _TILE:
                        n = _TILE

                    # density and momentum
                    for k in range(n):
                        rho_t[k] = 0.0
                        ux_t[k] = 0.0
                        uy_t[k] = 0.0
                        uz_t[k] = 0.0
                    for i in range(q):
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        for k in range(n):
                            fi = f[i, z, y, x0 + k]
                            rho_t[k] += fi
                            ux_t[k] += exi * fi
                            uy_t[k] += eyi * fi
                            uz_t[k] += ezi * fi

                    # velocity (half-force corrected), solids at rest
                    for k in range(n):
                        rho = rho_t[k]
                        inv_r = 1.0 / rho if rho > 0.0 else 0.0
                        ax = 0.0
                        ay = 0.0
                        az = 0.0
                        if force_mode == 1:
                            ax = acc_uniform[0]
                            ay = acc_uniform[1]
                            az = acc_uniform[2]
                        elif force_mode == 2:
                            ax = acc_field[0, z, y, x0 + k]
                            ay = acc_field[1, z, y, x0 + k]
                            az = acc_field[2, z, y, x0 + k]
                        if solid[z, y, x0 + k]:
                            ux = 0.0
                            uy = 0.0
                            uz = 0.0
                            ax = 0.0
                            ay = 0.0
                            az = 0.0
                        else:
                            ux = ux_t[k] * inv_r + 0.5 * ax
                            uy = uy_t[k] * inv_r + 0.5 * ay
                            uz = uz_t[k] * inv_r + 0.5 * az
                        ux_t[k] = ux
                        uy_t[k] = uy
                        uz_t[k] = uz
                        ax_t[k] = ax
                        ay_t[k] = ay
                        az_t[k] = az
                        usq_t[k] = ux * ux + uy * uy + uz * uz
                        ua_t[k] = ux * ax + uy * ay + uz * az
                        om_t[k] = omega_field[z, y, x0 + k] if use_omega_field else omega
                        pxx_t[k] = 0.0
                        pxy_t[k] = 0.0
                        pxz_t[k] = 0.0
                        pyy_t[k] = 0.0
                        pyz_t[k] = 0.0
                        pzz_t[k] = 0.0

                    # equilibrium and the non-equilibrium second moment
                    for i in range(q):
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        wi = w[i]
                        cxx = exi * exi
                        cxy = exi * eyi
                        cxz = exi * ezi
                        cyy = eyi * eyi
                        cyz = eyi * ezi
                        czz = ezi * ezi
                        for k in range(n):
                            eu = exi * ux_t[k] + eyi * uy_t[k] + ezi * uz_t[k]
                            usq = usq_t[k]
                            feq = wi * rho_t[k] * (
                                1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
                                + h3 * 4.5 * (eu * eu * eu - eu * usq)
                            )
                            feq_t[i, k] = feq
                            d = f[i, z, y, x0 + k] - feq
                            pxx_t[k] += cxx * d
                            pxy_t[k] += cxy * d
                            pxz_t[k] += cxz * d
                            pyy_t[k] += cyy * d
                            pyz_t[k] += cyz * d
                            pzz_t[k] += czz * d

                    if force_mode != 0:
                        # fold the force's own second moment in
                        for k in range(n):
                            rho = rho_t[k]
                            ux = ux_t[k]
                            uy = uy_t[k]
                            uz = uz_t[k]
                            fxx = rho * ax_t[k]
                            fyy = rho * ay_t[k]
                            fzz = rho * az_t[k]
                            pxx_t[k] += ux * fxx
                            pyy_t[k] += uy * fyy
                            pzz_t[k] += uz * fzz
                            pxy_t[k] += 0.5 * (ux * fyy + uy * fxx)
                            pxz_t[k] += 0.5 * (ux * fzz + uz * fxx)
                            pyz_t[k] += 0.5 * (uy * fzz + uz * fyy)

                    # rebuild f from feq and the regularized stress
                    for i in range(q):
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        wi = w[i]
                        hxx = exi * exi - CS2
                        hyy = eyi * eyi - CS2
                        hzz = ezi * ezi - CS2
                        cxy = exi * eyi
                        cxz = exi * ezi
                        cyz = eyi * ezi
                        hw = HERMITE2 * wi
                        for k in range(n):
                            f1 = hw * (
                                hxx * pxx_t[k] + hyy * pyy_t[k] + hzz * pzz_t[k]
                                + 2.0 * (cxy * pxy_t[k] + cxz * pxz_t[k] + cyz * pyz_t[k])
                            )
                            out = feq_t[i, k] + (1.0 - om_t[k]) * f1
                            if force_mode != 0:
                                eu = exi * ux_t[k] + eyi * uy_t[k] + ezi * uz_t[k]
                                ea = exi * ax_t[k] + eyi * ay_t[k] + ezi * az_t[k]
                                # prefactor 1/2, not (1 - omega/2)
                                out += 0.5 * wi * rho_t[k] * (
                                    3.0 * (ea - ua_t[k]) + 9.0 * eu * ea
                                )
                            out_t[k] = out
                        _store_row(f_post, i, z, y, x0, n, out_t, stream, exi, eyi, ezi)
else:
    def regularized_collision_kernel_3d(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for the JIT regularized 3D kernel")
