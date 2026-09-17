"""TRT collision kernel for D3Q19."""

from __future__ import annotations

import numpy as np

from .trt2d import trt_s_minus
from .kernels_trt import _trt_s_minus_field, _trt_s_minus_numba
from .d3q19 import OPP3, compute_feq_3d, compute_macroscopic_3d

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
) -> None:
    rho, ux, uy, uz = compute_macroscopic_3d(f)
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
    feq = compute_feq_3d(rho, ux, uy, uz)
    if omega_field is None:
        om = omega
        s_minus = trt_s_minus(omega, magic_lambda)
    else:
        om = omega_field
        s_minus = _trt_s_minus_field(omega_field, magic_lambda)
    ua = None if acc is None else ux * ax + uy * ay + uz * az

    for i in range(f.shape[0]):
        j = int(OPP3[i])
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
    ) -> None:
        q, nz, ny, nx = f.shape
        for z in nb.prange(nz):
            # Hoisted out of the cell loop: one heap allocation per cell per
            # timestep otherwise.
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
                        ux = uy = uz = 0.0
                        ax = ay = az = 0.0
                    else:
                        # half-force correction: rho u = sum e_i f_i + F/2
                        ux = mx * inv_r + 0.5 * ax
                        uy = my * inv_r + 0.5 * ay
                        uz = mz * inv_r + 0.5 * az
                    usq = ux * ux + uy * uy + uz * uz
                    ua = ux * ax + uy * ay + uz * az
                    for i in range(q):
                        eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
                        feq[i] = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
                    om = omega_field[z, y, x] if use_omega_field else omega
                    sm = _trt_s_minus_numba(om, magic_lambda)
                    for i in range(q):
                        j = opp[i]
                        f_plus = 0.5 * (f[i, z, y, x] + f[j, z, y, x])
                        f_minus = 0.5 * (f[i, z, y, x] - f[j, z, y, x])
                        feq_plus = 0.5 * (feq[i] + feq[j])
                        feq_minus = 0.5 * (feq[i] - feq[j])
                        out = (
                            feq[i]
                            + (1.0 - om) * (f_plus - feq_plus)
                            + (1.0 - sm) * (f_minus - feq_minus)
                        )
                        if force_mode != 0:
                            # The Guo source splits in closed form; each part
                            # carries the rate that relaxes it.
                            eu = ex[i] * ux + ey[i] * uy + ez[i] * uz
                            ea = ex[i] * ax + ey[i] * ay + ez[i] * az
                            s_p = w[i] * rho * (9.0 * eu * ea - 3.0 * ua)
                            s_m = w[i] * rho * 3.0 * ea
                            out += (1.0 - 0.5 * om) * s_p + (1.0 - 0.5 * sm) * s_m
                        f_post[i, z, y, x] = out
else:
    def trt_collision_kernel_3d(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for TRT 3D kernel")
