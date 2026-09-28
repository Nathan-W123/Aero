"""TRT collision kernel for D2Q9 (symmetric/antisymmetric split)."""

from __future__ import annotations

import numpy as np

from .trt2d import trt_taus, trt_s_minus
from .d2q9 import OPP, compute_feq, compute_macroscopic

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


_TAU_EPS = 1e-12


def _trt_s_minus_field(omega_field: np.ndarray, magic_lambda: float) -> np.ndarray:
    """Per-cell s_minus for a spatially varying omega (LES).  See `trt_taus`."""
    lam = float(magic_lambda)
    tau_plus = 1.0 / omega_field
    tau_minus = 0.5 + lam / np.maximum(tau_plus - 0.5, _TAU_EPS)
    return 1.0 / tau_minus


def trt_collision_numpy(
    f: np.ndarray,
    f_post: np.ndarray,
    solid: np.ndarray,
    omega: float,
    ex: np.ndarray,
    ey: np.ndarray,
    w: np.ndarray,
    magic_lambda: float = 0.25,
    omega_field: np.ndarray | None = None,
    acc: np.ndarray | None = None,
) -> None:
    rho, ux, uy = compute_macroscopic(f)
    ux_c = ux.copy()
    uy_c = uy.copy()
    fluid = ~solid
    if acc is not None:
        ax = np.where(fluid, acc[0], 0.0)
        ay = np.where(fluid, acc[1], 0.0)
        ux_c = ux_c + 0.5 * ax          # half-force correction
        uy_c = uy_c + 0.5 * ay
    ux_c[solid] = 0.0
    uy_c[solid] = 0.0
    feq = compute_feq(rho, ux_c, uy_c)
    if omega_field is None:
        s_minus = trt_s_minus(omega, magic_lambda)
        om = omega
    else:
        om = omega_field
        s_minus = _trt_s_minus_field(omega_field, magic_lambda)
    ua = None if acc is None else ux_c * ax + uy_c * ay

    for i in range(f.shape[0]):
        j = int(OPP[i])
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
            eu = ex[i] * ux_c + ey[i] * uy_c
            ea = ex[i] * ax + ey[i] * ay
            s_p = w[i] * rho * (9.0 * eu * ea - 3.0 * ua)
            s_m = w[i] * rho * 3.0 * ea
            f_post[i] += (1.0 - 0.5 * om) * s_p + (1.0 - 0.5 * s_minus) * s_m


if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _trt_s_minus_numba(omega: float, lam: float) -> float:
        """Scalar s_minus; must stay in step with `trt_taus`."""
        tau_plus = 1.0 / omega
        denom = tau_plus - 0.5
        if denom < 1e-12:
            denom = 1e-12
        tau_minus = 0.5 + lam / denom
        return 1.0 / tau_minus

    # x-tile length, as in kernels.collision_kernel.
    _TILE = 64

    @nb.njit(cache=True, parallel=True)
    def trt_collision_kernel(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        w: np.ndarray,
        opp: np.ndarray,
        magic_lambda: float,
        omega_field: np.ndarray,
        use_omega_field: bool,
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
    ) -> None:
        """
        TRT collision with Guo forcing, parallel over rows, tiled along x so
        the inner loops vectorise.  Per cell the operations and their order
        are those of the one-cell-at-a-time form, so results are bit-identical.
        """
        q, ny, nx = f.shape
        for y in nb.prange(ny):
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            ua_t = np.empty(_TILE)
            om_t = np.empty(_TILE)
            sm_t = np.empty(_TILE)
            ax_t = np.empty(_TILE)
            ay_t = np.empty(_TILE)
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
                        # half-force correction: rho u = sum e_i f_i + F/2
                        ux = ux_t[k] * inv_r + 0.5 * ax
                        uy = uy_t[k] * inv_r + 0.5 * ay
                    ux_t[k] = ux
                    uy_t[k] = uy
                    ax_t[k] = ax
                    ay_t[k] = ay
                    usq_t[k] = ux * ux + uy * uy
                    ua_t[k] = ux * ax + uy * ay
                    om = omega_field[y, x0 + k] if use_omega_field else omega
                    om_t[k] = om
                    sm_t[k] = _trt_s_minus_numba(om, magic_lambda)

                for i in range(q):
                    exi = ex[i]
                    eyi = ey[i]
                    wi = w[i]
                    for k in range(n):
                        eu = exi * ux_t[k] + eyi * uy_t[k]
                        feq_t[i, k] = wi * rho_t[k] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq_t[k])

                for i in range(q):
                    j = opp[i]
                    exi = ex[i]
                    eyi = ey[i]
                    wi = w[i]
                    for k in range(n):
                        fi = f[i, y, x0 + k]
                        fj = f[j, y, x0 + k]
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
                            eu = exi * ux_t[k] + eyi * uy_t[k]
                            ea = exi * ax_t[k] + eyi * ay_t[k]
                            s_plus = wi * rho_t[k] * (9.0 * eu * ea - 3.0 * ua_t[k])
                            s_minus = wi * rho_t[k] * 3.0 * ea
                            out += (1.0 - 0.5 * om) * s_plus + (1.0 - 0.5 * sm) * s_minus
                        f_post[i, y, x0 + k] = out
else:
    def trt_collision_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for TRT kernel")
