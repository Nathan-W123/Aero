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
        q, ny, nx = f.shape
        for y in nb.prange(ny):
            # Hoisted out of the cell loop: one heap allocation per cell per
            # timestep otherwise.
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
                    # half-force correction: rho u = sum e_i f_i + F/2
                    ux = mx * inv_r + 0.5 * ax
                    uy = my * inv_r + 0.5 * ay
                usq = ux * ux + uy * uy
                ua = ux * ax + uy * ay
                for i in range(q):
                    eu = ex[i] * ux + ey[i] * uy
                    feq[i] = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
                om = omega_field[y, x] if use_omega_field else omega
                sm = _trt_s_minus_numba(om, magic_lambda)
                for i in range(q):
                    j = opp[i]
                    f_plus = 0.5 * (f[i, y, x] + f[j, y, x])
                    f_minus = 0.5 * (f[i, y, x] - f[j, y, x])
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
                        eu = ex[i] * ux + ey[i] * uy
                        ea = ex[i] * ax + ey[i] * ay
                        s_plus = w[i] * rho * (9.0 * eu * ea - 3.0 * ua)
                        s_minus = w[i] * rho * 3.0 * ea
                        out += (1.0 - 0.5 * om) * s_plus + (1.0 - 0.5 * sm) * s_minus
                    f_post[i, y, x] = out
else:
    def trt_collision_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for TRT kernel")
