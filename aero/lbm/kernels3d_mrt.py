"""
MRT collision kernel for D3Q19.

Same strategy as kernels_mrt.py but with 19×19 matrices.
prange over z slices.
"""

import numpy as np
from typing import Optional

from .mrt3d import M19, M19_inv, build_s3_vec, VISCOUS_MODES
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
        omega_field: np.ndarray,  # (Nz, Ny, Nx) per-cell relaxation rate (LES)
        use_omega_field: bool,
        visc_modes: np.ndarray,   # rows of s carrying the viscosity
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
    ) -> None:
        """
        MRT collision with Guo forcing, parallel over z slices.

        Tiled along x: the moment transforms run as Q x Q scalar-times-row
        updates over a tile, which vectorise, instead of Q x Q scalar products
        per cell, which do not.  Per cell every sum runs in the same order as
        the one-cell-at-a-time form, so results are bit-identical.
        """
        Q, Nz, Ny, Nx = f.shape
        for z in nb.prange(Nz):
            # which rates the subgrid viscosity replaces
            visc = np.zeros(Q, dtype=np.bool_)
            for kk in range(visc_modes.shape[0]):
                visc[visc_modes[kk]] = True
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            uz_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            ua_t = np.empty(_TILE)
            ax_t = np.empty(_TILE)
            ay_t = np.empty(_TILE)
            az_t = np.empty(_TILE)
            sv_t = np.empty(_TILE)
            acc_t = np.empty(_TILE)
            feq_t = np.empty((Q, _TILE))
            src_t = np.empty((Q, _TILE))
            m_t = np.empty((Q, _TILE))
            meq_t = np.empty((Q, _TILE))
            mpost_t = np.empty((Q, _TILE))
            for y in range(Ny):
                for x0 in range(0, Nx, _TILE):
                    n = Nx - x0
                    if n > _TILE:
                        n = _TILE

                    # macroscopic
                    for k in range(n):
                        rho_t[k] = 0.0
                        ux_t[k] = 0.0
                        uy_t[k] = 0.0
                        uz_t[k] = 0.0
                    for i in range(Q):
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
                        inv_rho = 1.0 / rho if rho > 0.0 else 0.0
                        ax = 0.0; ay = 0.0; az = 0.0
                        if force_mode == 1:
                            ax = acc_uniform[0]
                            ay = acc_uniform[1]
                            az = acc_uniform[2]
                        elif force_mode == 2:
                            ax = acc_field[0, z, y, x0 + k]
                            ay = acc_field[1, z, y, x0 + k]
                            az = acc_field[2, z, y, x0 + k]
                        if solid[z, y, x0 + k]:
                            ux = 0.0; uy = 0.0; uz = 0.0
                            ax = 0.0; ay = 0.0; az = 0.0
                        else:
                            # half-force correction: rho u = sum e_i f_i + F/2
                            ux = ux_t[k] * inv_rho + 0.5 * ax
                            uy = uy_t[k] * inv_rho + 0.5 * ay
                            uz = uz_t[k] * inv_rho + 0.5 * az
                        ux_t[k] = ux
                        uy_t[k] = uy
                        uz_t[k] = uz
                        ax_t[k] = ax
                        ay_t[k] = ay
                        az_t[k] = az
                        usq_t[k] = ux*ux + uy*uy + uz*uz
                        ua_t[k] = ux*ax + uy*ay + uz*az
                        if use_omega_field:
                            sv_t[k] = omega_field[z, y, x0 + k]

                    # feq, and the Guo source in distribution space
                    for i in range(Q):
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        wi = w[i]
                        for k in range(n):
                            eu = exi*ux_t[k] + eyi*uy_t[k] + ezi*uz_t[k]
                            feq_t[i, k] = wi * rho_t[k] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq_t[k])
                        if force_mode != 0:
                            for k in range(n):
                                eu = exi*ux_t[k] + eyi*uy_t[k] + ezi*uz_t[k]
                                ea = exi*ax_t[k] + eyi*ay_t[k] + ezi*az_t[k]
                                src_t[i, k] = wi * rho_t[k] * (3.0*(ea - ua_t[k]) + 9.0*eu*ea)

                    # m = M @ f,  m_eq = M @ feq
                    for a in range(Q):
                        for k in range(n):
                            m_t[a, k] = 0.0
                            meq_t[a, k] = 0.0
                        for b in range(Q):
                            mab = M[a, b]
                            for k in range(n):
                                m_t[a, k] += mab * f[b, z, y, x0 + k]
                                meq_t[a, k] += mab * feq_t[b, k]

                    # relax: m_post = m - s (m - m_eq) + (I - S/2) M src.
                    # Subgrid viscosity only moves the viscous rates; the
                    # conserved and ghost modes keep their tuned values.
                    for a in range(Q):
                        sa = s[a]
                        per_cell = use_omega_field and visc[a]
                        if force_mode == 0:
                            for k in range(n):
                                sk = sv_t[k] if per_cell else sa
                                mpost_t[a, k] = m_t[a, k] - sk * (m_t[a, k] - meq_t[a, k])
                        else:
                            for k in range(n):
                                acc_t[k] = 0.0
                            for b in range(Q):
                                mab = M[a, b]
                                for k in range(n):
                                    acc_t[k] += mab * src_t[b, k]
                            for k in range(n):
                                sk = sv_t[k] if per_cell else sa
                                mpost_t[a, k] = (
                                    m_t[a, k] - sk * (m_t[a, k] - meq_t[a, k])
                                    + (1.0 - 0.5 * sk) * acc_t[k]
                                )

                    # f_post = Minv @ m_post
                    for i in range(Q):
                        for k in range(n):
                            acc_t[k] = 0.0
                        for a in range(Q):
                            mia = Minv[i, a]
                            for k in range(n):
                                acc_t[k] += mia * mpost_t[a, k]
                        for k in range(n):
                            f_post[i, z, y, x0 + k] = acc_t[k]

else:
    def _mrt_collision_3d(f, f_post, solid, s, M, Minv, ex, ey, ez, w,
                          omega_field, use_omega_field, visc_modes,
                          acc_uniform, acc_field, force_mode):
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
    omega_field: Optional[np.ndarray] = None,
    acc: Optional[np.ndarray] = None,
) -> None:
    Q, Nz, Ny, Nx = f.shape
    N = Nz * Ny * Nx
    f_flat = f.reshape(Q, N)

    rho = f_flat.sum(axis=0)
    inv_rho = inv_positive(rho)
    ux = inv_rho * (ex.astype(np.float64) @ f_flat)
    uy = inv_rho * (ey.astype(np.float64) @ f_flat)
    uz = inv_rho * (ez.astype(np.float64) @ f_flat)
    solid_flat = solid.ravel()
    if acc is None:
        ax = ay = az = None
    else:
        fluid_flat = ~solid_flat
        ax = np.where(fluid_flat, np.asarray(acc[0]).reshape(N), 0.0)
        ay = np.where(fluid_flat, np.asarray(acc[1]).reshape(N), 0.0)
        az = np.where(fluid_flat, np.asarray(acc[2]).reshape(N), 0.0)
        ux = ux + 0.5 * ax          # half-force correction
        uy = uy + 0.5 * ay
        uz = uz + 0.5 * az
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
    # a subgrid model makes the viscous rates a per-cell field
    if omega_field is None:
        s_eff = s[:, None]
    else:
        s_eff = np.repeat(s[:, None], N, axis=1)
        s_eff[VISCOUS_MODES, :] = np.asarray(omega_field, dtype=np.float64).reshape(N)
    m_post = m - s_eff * (m - m_eq)
    if acc is not None:
        # Guo source, transformed to moment space and scaled by (I - S/2)
        ua = ux*ax + uy*ay + uz*az
        src = np.empty((Q, N))
        for i in range(Q):
            eu = ex[i]*ux + ey[i]*uy + ez[i]*uz
            ea = ex[i]*ax + ey[i]*ay + ez[i]*az
            src[i] = w[i] * rho * (3.0*(ea - ua) + 9.0*eu*ea)
        m_post = m_post + (1.0 - 0.5 * s_eff) * (M @ src)
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
        # Numba needs a concretely-typed array even when the flag is False
        self._omega_dummy = np.zeros((1, 1, 1), dtype=np.float64)
        self._acc_dummy_u = np.zeros(3, dtype=np.float64)
        self._acc_dummy_f = dummy_force_field(3)

    def collide(
        self,
        f:      np.ndarray,
        f_post: np.ndarray,
        solid:  np.ndarray,
        ex:     np.ndarray,
        ey:     np.ndarray,
        ez:     np.ndarray,
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
            _mrt_collision_3d(
                f, f_post, solid, self.s, self.M, self.Minv, ex, ey, ez, w,
                omega_field if use_field else self._omega_dummy, use_field,
                VISCOUS_MODES,
                acc_uniform if acc_uniform is not None else self._acc_dummy_u,
                acc_field if acc_field is not None else self._acc_dummy_f,
                force_mode,
            )
        else:
            acc = None
            if force_mode == FORCE_UNIFORM:
                acc = np.broadcast_to(
                    np.asarray(acc_uniform).reshape(3, 1, 1, 1), (3,) + f.shape[1:]
                )
            elif force_mode == FORCE_FIELD:
                acc = acc_field
            _mrt_collision_3d_numpy(
                f, f_post, solid, self.s, self.M, self.Minv, ex, ey, ez, w,
                omega_field, acc,
            )
