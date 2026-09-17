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
    ) -> None:
        q, nz, ny, nx = f.shape
        for z in nb.prange(nz):
            feq = np.empty(q)
            for y in range(ny):
                for x in range(nx):
                    rho = 0.0
                    mx = 0.0
                    my = 0.0
                    mz = 0.0
                    for i in range(q):
                        fi = f[i, z, y, x]
                        rho += fi
                        mx += ex[i] * fi
                        my += ey[i] * fi
                        mz += ez[i] * fi
                    inv_r = 1.0 / rho if rho > 0.0 else 0.0

                    ax = 0.0
                    ay = 0.0
                    az = 0.0
                    if force_mode == 1:
                        ax = acc_uniform[0]
                        ay = acc_uniform[1]
                        az = acc_uniform[2]
                    elif force_mode == 2:
                        ax = acc_field[0, z, y, x]
                        ay = acc_field[1, z, y, x]
                        az = acc_field[2, z, y, x]

                    if solid[z, y, x]:
                        ux = 0.0
                        uy = 0.0
                        uz = 0.0
                        ax = 0.0
                        ay = 0.0
                        az = 0.0
                    else:
                        ux = mx * inv_r + 0.5 * ax
                        uy = my * inv_r + 0.5 * ay
                        uz = mz * inv_r + 0.5 * az
                    usq = ux * ux + uy * uy + uz * uz
                    ua = ux * ax + uy * ay + uz * az

                    for i in range(q):
                        eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
                        feq[i] = w[i] * rho * (
                            1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
                            + h3 * 4.5 * (eu * eu * eu - eu * usq)
                        )

                    pxx = 0.0; pxy = 0.0; pxz = 0.0
                    pyy = 0.0; pyz = 0.0; pzz = 0.0
                    for i in range(q):
                        d = f[i, z, y, x] - feq[i]
                        pxx += ex[i] * ex[i] * d
                        pxy += ex[i] * ey[i] * d
                        pxz += ex[i] * ez[i] * d
                        pyy += ey[i] * ey[i] * d
                        pyz += ey[i] * ez[i] * d
                        pzz += ez[i] * ez[i] * d

                    if force_mode != 0:
                        # fold the force's own second moment in
                        fxx = rho * ax
                        fyy = rho * ay
                        fzz = rho * az
                        pxx += ux * fxx
                        pyy += uy * fyy
                        pzz += uz * fzz
                        pxy += 0.5 * (ux * fyy + uy * fxx)
                        pxz += 0.5 * (ux * fzz + uz * fxx)
                        pyz += 0.5 * (uy * fzz + uz * fyy)

                    om = omega_field[z, y, x] if use_omega_field else omega
                    for i in range(q):
                        hxx = ex[i] * ex[i] - CS2
                        hyy = ey[i] * ey[i] - CS2
                        hzz = ez[i] * ez[i] - CS2
                        f1 = HERMITE2 * w[i] * (
                            hxx * pxx + hyy * pyy + hzz * pzz
                            + 2.0 * (ex[i] * ey[i] * pxy
                                     + ex[i] * ez[i] * pxz
                                     + ey[i] * ez[i] * pyz)
                        )
                        out = feq[i] + (1.0 - om) * f1
                        if force_mode != 0:
                            eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
                            ea = ex[i] * ax + ey[i] * ay + ez[i] * az
                            # prefactor 1/2, not (1 - omega/2)
                            out += 0.5 * w[i] * rho * (
                                3.0 * (ea - ua) + 9.0 * eu * ea
                            )
                        f_post[i, z, y, x] = out
else:
    def regularized_collision_kernel_3d(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for the JIT regularized 3D kernel")
