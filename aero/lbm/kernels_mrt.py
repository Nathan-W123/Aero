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
from .mrt2d import M9, M9_inv, build_s_vec

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
    ) -> None:
        Q, Ny, Nx = f.shape
        for y in nb.prange(Ny):
            for x in range(Nx):
                # --- macroscopic ---
                rho = 0.0; mx = 0.0; my = 0.0
                for i in range(Q):
                    fi = f[i, y, x]; rho += fi
                    mx += ex[i] * fi; my += ey[i] * fi
                inv_rho = 1.0 / rho if rho > 0.0 else 0.0
                if solid[y, x]:
                    ux = 0.0; uy = 0.0
                else:
                    ux = mx * inv_rho; uy = my * inv_rho

                usq = ux * ux + uy * uy

                # --- feq ---
                feq = np.empty(Q)
                for i in range(Q):
                    eu = ex[i] * ux + ey[i] * uy
                    feq[i] = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)

                # --- m = M @ f, m_eq = M @ feq ---
                m     = np.zeros(Q)
                m_eq  = np.zeros(Q)
                for a in range(Q):
                    for b in range(Q):
                        m[a]    += M[a, b] * f[b, y, x]
                        m_eq[a] += M[a, b] * feq[b]

                # --- relax: m_post = m - s*(m - m_eq) ---
                m_post = np.zeros(Q)
                for a in range(Q):
                    m_post[a] = m[a] - s[a] * (m[a] - m_eq[a])

                # --- f_post = Minv @ m_post ---
                for i in range(Q):
                    acc = 0.0
                    for a in range(Q):
                        acc += Minv[i, a] * m_post[a]
                    f_post[i, y, x] = acc

else:
    def _mrt_collision_2d(f, f_post, solid, s, M, Minv, ex, ey, w):
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
) -> None:
    """Vectorised NumPy MRT collision for D2Q9."""
    Q, Ny, Nx = f.shape
    N = Ny * Nx
    f_flat = f.reshape(Q, N)           # (9, N)

    # macroscopic
    rho = f_flat.sum(axis=0)           # (N,)
    inv_rho = np.where(rho > 0.0, 1.0 / rho, 0.0)
    ux = inv_rho * (ex.astype(np.float64) @ f_flat)
    uy = inv_rho * (ey.astype(np.float64) @ f_flat)
    solid_flat = solid.ravel()
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

    # relax
    m_post = m - s[:, None] * (m - m_eq)

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

    def collide(
        self,
        f:      np.ndarray,
        f_post: np.ndarray,
        solid:  np.ndarray,
        ex:     np.ndarray,
        ey:     np.ndarray,
        w:      np.ndarray,
    ) -> None:
        if self._use_numba:
            _mrt_collision_2d(f, f_post, solid, self.s, self.M, self.Minv, ex, ey, w)
        else:
            _mrt_collision_2d_numpy(f, f_post, solid, self.s, self.M, self.Minv, ex, ey, w)
