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
        Q, Ny, Nx = f.shape
        for y in nb.prange(Ny):
            # Scratch hoisted out of the cell loop: allocating these per cell
            # costs one heap allocation per cell per timestep.
            feq = np.empty(Q)
            m = np.empty(Q)
            m_eq = np.empty(Q)
            m_post = np.empty(Q)
            src = np.empty(Q)
            s_cell = s.copy()
            for x in range(Nx):
                if use_omega_field:
                    # Subgrid viscosity only moves the viscous rates; the
                    # conserved and ghost modes keep their tuned values.
                    sv = omega_field[y, x]
                    for k in range(visc_modes.shape[0]):
                        s_cell[visc_modes[k]] = sv
                    sq_local = 8.0 * (2.0 - sv) / (8.0 - sv)
                    for k in range(flux_modes.shape[0]):
                        s_cell[flux_modes[k]] = sq_local
                # --- macroscopic ---
                rho = 0.0; mx = 0.0; my = 0.0
                for i in range(Q):
                    fi = f[i, y, x]; rho += fi
                    mx += ex[i] * fi; my += ey[i] * fi
                inv_rho = 1.0 / rho if rho > 0.0 else 0.0

                ax = 0.0
                ay = 0.0
                if force_mode == 1:
                    ax = acc_uniform[0]
                    ay = acc_uniform[1]
                elif force_mode == 2:
                    ax = acc_field[0, y, x]
                    ay = acc_field[1, y, x]

                if solid[y, x]:
                    ux = 0.0; uy = 0.0
                    ax = 0.0; ay = 0.0
                else:
                    # half-force correction: rho u = sum e_i f_i + F/2
                    ux = mx * inv_rho + 0.5 * ax
                    uy = my * inv_rho + 0.5 * ay

                usq = ux * ux + uy * uy
                ua = ux * ax + uy * ay

                # --- feq ---
                for i in range(Q):
                    eu = ex[i] * ux + ey[i] * uy
                    feq[i] = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)

                # --- Guo source in distribution space ---
                if force_mode != 0:
                    for i in range(Q):
                        eu = ex[i] * ux + ey[i] * uy
                        ea = ex[i] * ax + ey[i] * ay
                        src[i] = w[i] * rho * (3.0 * (ea - ua) + 9.0 * eu * ea)

                # --- m = M @ f, m_eq = M @ feq ---
                for a in range(Q):
                    acc_m = 0.0
                    acc_eq = 0.0
                    for b in range(Q):
                        acc_m += M[a, b] * f[b, y, x]
                        acc_eq += M[a, b] * feq[b]
                    m[a] = acc_m
                    m_eq[a] = acc_eq

                # --- relax: m_post = m - s*(m - m_eq) + (I - S/2) M src ---
                if force_mode == 0:
                    for a in range(Q):
                        m_post[a] = m[a] - s_cell[a] * (m[a] - m_eq[a])
                else:
                    for a in range(Q):
                        acc_s = 0.0
                        for b in range(Q):
                            acc_s += M[a, b] * src[b]
                        m_post[a] = (
                            m[a] - s_cell[a] * (m[a] - m_eq[a])
                            + (1.0 - 0.5 * s_cell[a]) * acc_s
                        )

                # --- f_post = Minv @ m_post ---
                for i in range(Q):
                    acc = 0.0
                    for a in range(Q):
                        acc += Minv[i, a] * m_post[a]
                    f_post[i, y, x] = acc

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
