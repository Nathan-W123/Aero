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
) -> None:
    rho, ux, uy = compute_macroscopic(f)
    ux_c = ux.copy()
    uy_c = uy.copy()
    ux_c[solid] = 0.0
    uy_c[solid] = 0.0
    feq = compute_feq(rho, ux_c, uy_c)
    if omega_field is None:
        s_minus = trt_s_minus(omega, magic_lambda)
        om = omega
    else:
        om = omega_field
        s_minus = _trt_s_minus_field(omega_field, magic_lambda)

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
                if solid[y, x]:
                    ux = 0.0
                    uy = 0.0
                else:
                    ux = mx * inv_r
                    uy = my * inv_r
                usq = ux * ux + uy * uy
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
                    f_post[i, y, x] = (
                        feq[i]
                        + (1.0 - om) * (f_plus - feq_plus)
                        + (1.0 - sm) * (f_minus - feq_minus)
                    )
else:
    def trt_collision_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Numba required for TRT kernel")
