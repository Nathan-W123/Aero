"""
MRT collision kernel for D2Q9.

Strategy: transform f → moment space m = M9 @ f, relax each moment
independently via s vector, then transform back f_post = M9_inv @ m_post.

The equilibrium is computed in distribution space (feq) and then
transformed: m_eq = M9 @ feq.  This avoids hardcoding analytical moment
equilibria and is exact.

Numba path: explicit 9×9 matrix-vector loops per cell, prange over rows.
NumPy fallback: vectorised einsum over all cells at once.
"""

import numpy as np
from typing import Optional

from .mrt2d import (
    M9, M9_inv, build_s_vec, magic_sq, VISCOUS_MODES, ENERGY_FLUX_MODES,
)
from .physics import inv_positive
from .forcing import FORCE_NONE, FORCE_UNIFORM, FORCE_FIELD, dummy_force_field

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    nb = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Numba path
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    # x-tile length: five Q x tile scratch arrays have to stay in cache.
    _TILE = 32

    @nb.njit(cache=True, parallel=True)
    def _mrt_collision_2d(
        f:     np.ndarray,   # (9, Ny, Nx)
        f_post: np.ndarray,  # (9, Ny, Nx) — output
        solid: np.ndarray,   # (Ny, Nx) bool
        s:     np.ndarray,   # (9,) relaxation rates
        M:     np.ndarray,   # (9, 9)
        Minv:  np.ndarray,   # (9, 9)
        ex:    np.ndarray,   # (9,) int32
        ey:    np.ndarray,   # (9,) int32
        w:     np.ndarray,   # (9,) float64
        omega_field: np.ndarray,  # (Ny, Nx) per-cell relaxation rate (LES)
        use_omega_field: bool,
        visc_modes: np.ndarray,   # rows of s carrying the viscosity
        flux_modes: np.ndarray,   # rows of s tied to it via magic_sq
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
    ) -> None:
        """
        MRT collision with Guo forcing, parallel over rows, tiled along x so
        the moment transforms vectorise.  Per cell every sum runs in the same
        order as the one-cell-at-a-time form, so results are bit-identical.
        """
        Q, Ny, Nx = f.shape
        for y in nb.prange(Ny):
            # which rates the subgrid viscosity replaces: 1 viscous, 2 the flux
            # rates tied to it (assigned second, so they win a tie)
            kind = np.zeros(Q, dtype=np.int64)
            for kk in range(visc_modes.shape[0]):
                kind[visc_modes[kk]] = 1
            for kk in range(flux_modes.shape[0]):
                kind[flux_modes[kk]] = 2
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            ua_t = np.empty(_TILE)
            ax_t = np.empty(_TILE)
            ay_t = np.empty(_TILE)
            sv_t = np.empty(_TILE)
            sq_t = np.empty(_TILE)
            acc_t = np.empty(_TILE)
            feq_t = np.empty((Q, _TILE))
            src_t = np.empty((Q, _TILE))
            m_t = np.empty((Q, _TILE))
            meq_t = np.empty((Q, _TILE))
            mpost_t = np.empty((Q, _TILE))
            for x0 in range(0, Nx, _TILE):
                n = Nx - x0
                if n > _TILE:
                    n = _TILE

                # --- macroscopic ---
                for k in range(n):
                    rho_t[k] = 0.0
                    ux_t[k] = 0.0
                    uy_t[k] = 0.0
                for i in range(Q):
                    exi = ex[i]
                    eyi = ey[i]
                    for k in range(n):
                        fi = f[i, y, x0 + k]
                        rho_t[k] += fi
                        ux_t[k] += exi * fi
                        uy_t[k] += eyi * fi
                for k in range(n):
                    rho = rho_t[k]
                    inv_rho = 1.0 / rho if rho > 0.0 else 0.0
                    ax = 0.0
                    ay = 0.0
                    if force_mode == 1:
                        ax = acc_uniform[0]
                        ay = acc_uniform[1]
                    elif force_mode == 2:
                        ax = acc_field[0, y, x0 + k]
                        ay = acc_field[1, y, x0 + k]
                    if solid[y, x0 + k]:
                        ux = 0.0; uy = 0.0
                        ax = 0.0; ay = 0.0
                    else:
                        # half-force correction: rho u = sum e_i f_i + F/2
                        ux = ux_t[k] * inv_rho + 0.5 * ax
                        uy = uy_t[k] * inv_rho + 0.5 * ay
                    ux_t[k] = ux
                    uy_t[k] = uy
                    ax_t[k] = ax
                    ay_t[k] = ay
                    usq_t[k] = ux * ux + uy * uy
                    ua_t[k] = ux * ax + uy * ay
                    if use_omega_field:
                        sv = omega_field[y, x0 + k]
                        sv_t[k] = sv
                        sq_t[k] = 8.0 * (2.0 - sv) / (8.0 - sv)

                # --- feq and the Guo source in distribution space ---
                for i in range(Q):
                    exi = ex[i]
                    eyi = ey[i]
                    wi = w[i]
                    for k in range(n):
                        eu = exi * ux_t[k] + eyi * uy_t[k]
                        feq_t[i, k] = wi * rho_t[k] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq_t[k])
                    if force_mode != 0:
                        for k in range(n):
                            eu = exi * ux_t[k] + eyi * uy_t[k]
                            ea = exi * ax_t[k] + eyi * ay_t[k]
                            src_t[i, k] = wi * rho_t[k] * (3.0 * (ea - ua_t[k]) + 9.0 * eu * ea)

                # --- m = M @ f, m_eq = M @ feq ---
                for a in range(Q):
                    for k in range(n):
                        m_t[a, k] = 0.0
                        meq_t[a, k] = 0.0
                    for b in range(Q):
                        mab = M[a, b]
                        for k in range(n):
                            m_t[a, k] += mab * f[b, y, x0 + k]
                            meq_t[a, k] += mab * feq_t[b, k]

                # --- relax: m_post = m - s*(m - m_eq) + (I - S/2) M src ---
                for a in range(Q):
                    sa = s[a]
                    ka = kind[a] if use_omega_field else 0
                    if force_mode != 0:
                        for k in range(n):
                            acc_t[k] = 0.0
                        for b in range(Q):
                            mab = M[a, b]
                            for k in range(n):
                                acc_t[k] += mab * src_t[b, k]
                    for k in range(n):
                        if ka == 1:
                            sk = sv_t[k]
                        elif ka == 2:
                            sk = sq_t[k]
                        else:
                            sk = sa
                        if force_mode == 0:
                            mpost_t[a, k] = m_t[a, k] - sk * (m_t[a, k] - meq_t[a, k])
                        else:
                            mpost_t[a, k] = (
                                m_t[a, k] - sk * (m_t[a, k] - meq_t[a, k])
                                + (1.0 - 0.5 * sk) * acc_t[k]
                            )

                # --- f_post = Minv @ m_post ---
                for i in range(Q):
                    for k in range(n):
                        acc_t[k] = 0.0
                    for a in range(Q):
                        mia = Minv[i, a]
                        for k in range(n):
                            acc_t[k] += mia * mpost_t[a, k]
                    for k in range(n):
                        f_post[i, y, x0 + k] = acc_t[k]

else:
    def _mrt_collision_2d(f, f_post, solid, s, M, Minv, ex, ey, w,
                          omega_field, use_omega_field, visc_modes, flux_modes,
                          acc_uniform, acc_field, force_mode):
        pass  # replaced by numpy fallback below


# ---------------------------------------------------------------------------
# NumPy fallback (always available)
# ---------------------------------------------------------------------------

def _mrt_collision_2d_numpy(
    f:      np.ndarray,
    f_post: np.ndarray,
    solid:  np.ndarray,
    s:      np.ndarray,
    M:      np.ndarray,
    Minv:   np.ndarray,
    ex:     np.ndarray,
    ey:     np.ndarray,
    w:      np.ndarray,
    omega_field: Optional[np.ndarray] = None,
    acc: Optional[np.ndarray] = None,
) -> None:
    """Vectorised NumPy MRT collision for D2Q9."""
    Q, Ny, Nx = f.shape
    N = Ny * Nx
    f_flat = f.reshape(Q, N)           # (9, N)

    # macroscopic
    rho = f_flat.sum(axis=0)           # (N,)
    inv_rho = inv_positive(rho)
    ux = inv_rho * (ex.astype(np.float64) @ f_flat)
    uy = inv_rho * (ey.astype(np.float64) @ f_flat)
    solid_flat = solid.ravel()
    if acc is None:
        ax = ay = None
    else:
        fluid_flat = ~solid_flat
        ax = np.where(fluid_flat, np.asarray(acc[0]).reshape(N), 0.0)
        ay = np.where(fluid_flat, np.asarray(acc[1]).reshape(N), 0.0)
        ux = ux + 0.5 * ax          # half-force correction
        uy = uy + 0.5 * ay
    ux[solid_flat] = 0.0
    uy[solid_flat] = 0.0

    # feq  (Q, N)
    usq = ux * ux + uy * uy
    feq = np.empty((Q, N))
    for i in range(Q):
        eu = ex[i] * ux + ey[i] * uy
        feq[i] = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)

    # transform to moment space
    m     = M @ f_flat    # (9, N)
    m_eq  = M @ feq       # (9, N)

    # relax — a subgrid model makes the viscous rates a per-cell field
    if omega_field is None:
        s_eff = s[:, None]
    else:
        s_eff = np.repeat(s[:, None], N, axis=1)
        sv = np.asarray(omega_field, dtype=np.float64).reshape(N)
        s_eff[VISCOUS_MODES, :] = sv
        s_eff[ENERGY_FLUX_MODES, :] = magic_sq(sv)
    m_post = m - s_eff * (m - m_eq)
    if acc is not None:
        # Guo source, transformed to moment space and scaled by (I - S/2)
        ua = ux * ax + uy * ay
        src = np.empty((Q, N))
        for i in range(Q):
            eu = ex[i] * ux + ey[i] * uy
            ea = ex[i] * ax + ey[i] * ay
            src[i] = w[i] * rho * (3.0 * (ea - ua) + 9.0 * eu * ea)
        m_post = m_post + (1.0 - 0.5 * s_eff) * (M @ src)

    # back to distribution space
    f_out = Minv @ m_post  # (9, N)
    f_post[:] = f_out.reshape(Q, Ny, Nx)


# ---------------------------------------------------------------------------
# Public entry point — picks Numba or NumPy automatically
# ---------------------------------------------------------------------------

class MRTKernel2D:
    """Holds pre-built matrices and dispatches to the right backend."""

    def __init__(self, omega: float, use_numba: bool = True):
        self.M    = np.ascontiguousarray(M9,     dtype=np.float64)
        self.Minv = np.ascontiguousarray(M9_inv, dtype=np.float64)
        self.s    = build_s_vec(omega)
        self._use_numba = use_numba and _HAS_NUMBA
        # Numba needs a concretely-typed array even when the flag is False
        self._omega_dummy = np.zeros((1, 1), dtype=np.float64)
        self._acc_dummy_u = np.zeros(2, dtype=np.float64)
        self._acc_dummy_f = dummy_force_field(2)

    def collide(
        self,
        f:      np.ndarray,
        f_post: np.ndarray,
        solid:  np.ndarray,
        ex:     np.ndarray,
        ey:     np.ndarray,
        w:      np.ndarray,
        omega_field: Optional[np.ndarray] = None,
        acc_uniform: Optional[np.ndarray] = None,
        acc_field: Optional[np.ndarray] = None,
        force_mode: int = FORCE_NONE,
    ) -> None:
        """
        MRT collision.  ``omega_field`` (LES) overrides the viscous rates
        per cell; the tuned energy/ghost rates are left alone.  Guo forcing
        enters in moment space scaled by ``(I - S/2)``.
        """
        if self._use_numba:
            use_field = omega_field is not None
            _mrt_collision_2d(
                f, f_post, solid, self.s, self.M, self.Minv, ex, ey, w,
                omega_field if use_field else self._omega_dummy, use_field,
                VISCOUS_MODES, ENERGY_FLUX_MODES,
                acc_uniform if acc_uniform is not None else self._acc_dummy_u,
                acc_field if acc_field is not None else self._acc_dummy_f,
                force_mode,
            )
        else:
            acc = None
            if force_mode == FORCE_UNIFORM:
                acc = np.broadcast_to(
                    np.asarray(acc_uniform).reshape(2, 1, 1), (2,) + f.shape[1:]
                )
            elif force_mode == FORCE_FIELD:
                acc = acc_field
            _mrt_collision_2d_numpy(
                f, f_post, solid, self.s, self.M, self.Minv, ex, ey, w,
                omega_field, acc,
            )
