"""
MRT collision kernel for D3Q19.

Same strategy as kernels_mrt.py but with 19×19 matrices.
prange over z slices.
"""

import numpy as np
from .mrt3d import M19, M19_inv, build_s3_vec

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
    def _mrt_collision_3d(
        f:      np.ndarray,   # (19, Nz, Ny, Nx)
        f_post: np.ndarray,   # (19, Nz, Ny, Nx)
        solid:  np.ndarray,   # (Nz, Ny, Nx) bool
        s:      np.ndarray,   # (19,)
        M:      np.ndarray,   # (19, 19)
        Minv:   np.ndarray,   # (19, 19)
        ex:     np.ndarray,   # (19,) int32
        ey:     np.ndarray,   # (19,) int32
        ez:     np.ndarray,   # (19,) int32
        w:      np.ndarray,   # (19,) float64
    ) -> None:
        Q, Nz, Ny, Nx = f.shape
        for z in nb.prange(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    # macroscopic
                    rho = 0.0; mx = 0.0; my = 0.0; mz = 0.0
                    for i in range(Q):
                        fi = f[i, z, y, x]
                        rho += fi
                        mx  += ex[i] * fi
                        my  += ey[i] * fi
                        mz  += ez[i] * fi
                    inv_rho = 1.0 / rho if rho > 0.0 else 0.0
                    if solid[z, y, x]:
                        ux = 0.0; uy = 0.0; uz = 0.0
                    else:
                        ux = mx * inv_rho
                        uy = my * inv_rho
                        uz = mz * inv_rho

                    usq = ux*ux + uy*uy + uz*uz

                    # feq
                    feq = np.empty(Q)
                    for i in range(Q):
                        eu = ex[i]*ux + ey[i]*uy + ez[i]*uz
                        feq[i] = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)

                    # m = M @ f_cell,  m_eq = M @ feq
                    m    = np.zeros(Q)
                    m_eq = np.zeros(Q)
                    for a in range(Q):
                        for b in range(Q):
                            m[a]    += M[a, b] * f[b, z, y, x]
                            m_eq[a] += M[a, b] * feq[b]

                    # relax
                    m_post = np.zeros(Q)
                    for a in range(Q):
                        m_post[a] = m[a] - s[a] * (m[a] - m_eq[a])

                    # f_post = Minv @ m_post
                    for i in range(Q):
                        acc = 0.0
                        for a in range(Q):
                            acc += Minv[i, a] * m_post[a]
                        f_post[i, z, y, x] = acc

else:
    def _mrt_collision_3d(f, f_post, solid, s, M, Minv, ex, ey, ez, w):
        pass  # replaced by numpy fallback below


# ---------------------------------------------------------------------------
# NumPy fallback
# ---------------------------------------------------------------------------

def _mrt_collision_3d_numpy(
    f:      np.ndarray,
    f_post: np.ndarray,
    solid:  np.ndarray,
    s:      np.ndarray,
    M:      np.ndarray,
    Minv:   np.ndarray,
    ex:     np.ndarray,
    ey:     np.ndarray,
    ez:     np.ndarray,
    w:      np.ndarray,
) -> None:
    Q, Nz, Ny, Nx = f.shape
    N = Nz * Ny * Nx
    f_flat = f.reshape(Q, N)

    rho = f_flat.sum(axis=0)
    inv_rho = np.where(rho > 0.0, 1.0 / rho, 0.0)
    ux = inv_rho * (ex.astype(np.float64) @ f_flat)
    uy = inv_rho * (ey.astype(np.float64) @ f_flat)
    uz = inv_rho * (ez.astype(np.float64) @ f_flat)
    solid_flat = solid.ravel()
    ux[solid_flat] = 0.0
    uy[solid_flat] = 0.0
    uz[solid_flat] = 0.0

    usq = ux*ux + uy*uy + uz*uz
    feq = np.empty((Q, N))
    for i in range(Q):
        eu = ex[i]*ux + ey[i]*uy + ez[i]*uz
        feq[i] = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)

    m      = M @ f_flat
    m_eq   = M @ feq
    m_post = m - s[:, None] * (m - m_eq)
    f_post[:] = (Minv @ m_post).reshape(Q, Nz, Ny, Nx)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class MRTKernel3D:
    def __init__(self, omega: float, use_numba: bool = True):
        self.M    = np.ascontiguousarray(M19,     dtype=np.float64)
        self.Minv = np.ascontiguousarray(M19_inv, dtype=np.float64)
        self.s    = build_s3_vec(omega)
        self._use_numba = use_numba and _HAS_NUMBA

    def collide(
        self,
        f:      np.ndarray,
        f_post: np.ndarray,
        solid:  np.ndarray,
        ex:     np.ndarray,
        ey:     np.ndarray,
        ez:     np.ndarray,
        w:      np.ndarray,
    ) -> None:
        if self._use_numba:
            _mrt_collision_3d(
                f, f_post, solid, self.s, self.M, self.Minv, ex, ey, ez, w
            )
        else:
            _mrt_collision_3d_numpy(
                f, f_post, solid, self.s, self.M, self.Minv, ex, ey, ez, w
            )
