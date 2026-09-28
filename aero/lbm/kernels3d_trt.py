"""TRT collision kernel for D3Q19 / D3Q27."""

from __future__ import annotations

import numpy as np

from .trt2d import trt_s_minus
from .kernels_trt import _trt_s_minus_field, _trt_s_minus_numba
from .lattice3d import D3Q19, compute_feq, compute_macroscopic

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


def trt_collision_numpy_3d(
    f: np.ndarray,
    f_post: np.ndarray,
    solid: np.ndarray,
    omega: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    w: np.ndarray,
    magic_lambda: float = 0.25,
    omega_field: np.ndarray | None = None,
    acc: np.ndarray | None = None,
    lattice=D3Q19,
) -> None:
    rho, ux, uy, uz = compute_macroscopic(f, lattice)
    ux = ux.copy()
    uy = uy.copy()
    uz = uz.copy()
    fluid = ~solid
    if acc is not None:
        ax = np.where(fluid, acc[0], 0.0)
        ay = np.where(fluid, acc[1], 0.0)
        az = np.where(fluid, acc[2], 0.0)
        ux = ux + 0.5 * ax          # half-force correction
        uy = uy + 0.5 * ay
        uz = uz + 0.5 * az
    ux[solid] = 0.0
    uy[solid] = 0.0
    uz[solid] = 0.0
    feq = compute_feq(rho, ux, uy, uz, lattice)
    if omega_field is None:
        om = omega
        s_minus = trt_s_minus(omega, magic_lambda)
    else:
        om = omega_field
        s_minus = _trt_s_minus_field(omega_field, magic_lambda)
    ua = None if acc is None else ux * ax + uy * ay + uz * az

    for i in range(f.shape[0]):
        j = int(lattice.OPP[i])
        f_plus = 0.5 * (f[i] + f[j])
        f_minus = 0.5 * (f[i] - f[j])
        feq_plus = 0.5 * (feq[i] + feq[j])
        feq_minus = 0.5 * (feq[i] - feq[j])
        f_post[i] = (
            feq[i]
            + (1.0 - om) * (f_plus - feq_plus)
            + (1.0 - s_minus) * (f_minus - feq_minus)
        )
        if acc is not None:
            eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
            ea = ex[i] * ax + ey[i] * ay + ez[i] * az
            s_p = w[i] * rho * (9.0 * eu * ea - 3.0 * ua)
            s_m = w[i] * rho * 3.0 * ea
            f_post[i] += (1.0 - 0.5 * om) * s_p + (1.0 - 0.5 * s_minus) * s_m


if _HAS_NUMBA:
    # x-tile length, as in kernels3d.collision_kernel_3d.
    _TILE = 64

    @nb.njit(cache=True, parallel=True)
    def trt_collision_kernel_3d(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
        w: np.ndarray,
        opp: np.ndarray,
        magic_lambda: float,
        omega_field: np.ndarray,
        use_omega_field: bool,
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
        h3: float = 0.0,
    ) -> None:
        """
        TRT collision with Guo forcing, parallel over z slices.

        Tiled like the BGK kernel: a run of x cells goes through each pass
        together so the inner loops walk direction planes at unit stride and
        vectorise.  Per cell the operations and their order are those of the
        one-cell-at-a-time form, so results are bit-identical.
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
            sm_t = np.empty(_TILE)
            ax_t = np.empty(_TILE)
            ay_t = np.empty(_TILE)
            az_t = np.empty(_TILE)
            feq_t = np.empty((q, _TILE))
            for y in range(ny):
                for x0 in range(0, nx, _TILE):
                    n = nx - x0
                    if n > _TILE:
                        n = _TILE

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
                            ux = uy = uz = 0.0
                            ax = ay = az = 0.0
                        else:
                            # half-force correction: rho u = sum e_i f_i + F/2
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
                        om = omega_field[z, y, x0 + k] if use_omega_field else omega
                        om_t[k] = om
                        sm_t[k] = _trt_s_minus_numba(om, magic_lambda)

                    for i in range(q):
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        wi = w[i]
                        for k in range(n):
                            eu = exi * ux_t[k] + eyi * uy_t[k] + ezi * uz_t[k]
                            usq = usq_t[k]
                            feq_t[i, k] = wi * rho_t[k] * (
                                1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
                                + h3 * 4.5 * (eu * eu * eu - eu * usq)
                            )

                    for i in range(q):
                        j = opp[i]
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        wi = w[i]
                        for k in range(n):
                            fi = f[i, z, y, x0 + k]
                            fj = f[j, z, y, x0 + k]
                            f_plus = 0.5 * (fi + fj)
                            f_minus = 0.5 * (fi - fj)
                            feq_plus = 0.5 * (feq_t[i, k] + feq_t[j, k])
                            feq_minus = 0.5 * (feq_t[i, k] - feq_t[j, k])
                            om = om_t[k]
                            sm = sm_t[k]
                            out = (
                                feq_t[i, k]
                                + (1.0 - om) * (f_plus - feq_plus)
                                + (1.0 - sm) * (f_minus - feq_minus)
                            )
                            if force_mode != 0:
                                # The Guo source splits in closed form; each part
                                # carries the rate that relaxes it.
                                eu = exi * ux_t[k] + eyi * uy_t[k] + ezi * uz_t[k]
                                ea = exi * ax_t[k] + eyi * ay_t[k] + ezi * az_t[k]
                                s_p = wi * rho_t[k] * (9.0 * eu * ea - 3.0 * ua_t[k])
                                s_m = wi * rho_t[k] * 3.0 * ea
                                out += (1.0 - 0.5 * om) * s_p + (1.0 - 0.5 * sm) * s_m
                            f_post[i, z, y, x0 + k] = out
else:
    def trt_collision_kernel_3d(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for TRT 3D kernel")
