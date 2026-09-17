"""
Numba-JIT accelerated D3Q19 kernels.

collision_kernel_3d  — fused macroscopic + BGK collision
stream_kernel_3d     — 3D push streaming

f layout: (Q, Nz, Ny, Nx), C-contiguous float64.
ex/ey/ez: (Q,) int32.  w: (Q,) float64.

Pure-NumPy fallbacks (same signatures) are provided when Numba is absent.
"""

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None  # type: ignore[assignment]

# x-tile length used by the collision kernel (see kernels.py).
_TILE = 64


if HAS_NUMBA:
    @nb.njit(cache=True, parallel=True)
    def collision_kernel_3d(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
        w: np.ndarray,
        omega_field: np.ndarray,
        use_omega_field: bool,
    ) -> None:
        """
        Fused macroscopic + BGK collision for D3Q19.  Parallel over z slices.

        A tile of x is carried through three passes so every inner loop walks
        one direction plane at unit stride, instead of gathering Q strided
        values per cell.
        """
        Q, Nz, Ny, Nx = f.shape
        for z in nb.prange(Nz):
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            uz_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            om_t = np.empty(_TILE)

            for y in range(Ny):
                for x0 in range(0, Nx, _TILE):
                    n = Nx - x0
                    if n > _TILE:
                        n = _TILE

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
                        if solid[z, y, x0 + k]:
                            ux_t[k] = 0.0
                            uy_t[k] = 0.0
                            uz_t[k] = 0.0
                        else:
                            ux_t[k] = ux_t[k] * inv_rho
                            uy_t[k] = uy_t[k] * inv_rho
                            uz_t[k] = uz_t[k] * inv_rho
                        usq_t[k] = (
                            ux_t[k] * ux_t[k] + uy_t[k] * uy_t[k] + uz_t[k] * uz_t[k]
                        )
                        om_t[k] = (
                            omega_field[z, y, x0 + k] if use_omega_field else omega
                        )

                    for i in range(Q):
                        exi = ex[i]
                        eyi = ey[i]
                        ezi = ez[i]
                        wi = w[i]
                        for k in range(n):
                            eu = exi * ux_t[k] + eyi * uy_t[k] + ezi * uz_t[k]
                            feqi = wi * rho_t[k] * (
                                1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq_t[k]
                            )
                            om = om_t[k]
                            f_post[i, z, y, x0 + k] = (
                                (1.0 - om) * f[i, z, y, x0 + k] + om * feqi
                            )

    @nb.njit(cache=True, parallel=True)
    def stream_kernel_3d(
        f_src: np.ndarray,
        f_dst: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
    ) -> None:
        """
        Push streaming for D3Q19. f_dst[i,(z+ez)%Nz,(y+ey)%Ny,(x+ex)%Nx]=f_src[i,z,y,x].

        The x loop is innermost so source and destination both run at unit
        stride: the x shift is a contiguous block move plus one wrapped
        element, instead of a per-cell modulo and a scattered write.  For a
        fixed i the (z,y) shift is a bijection, so prange over z is race-free.
        """
        Q, Nz, Ny, Nx = f_src.shape
        for z in nb.prange(Nz):
            for i in range(Q):
                zn = (z + ez[i]) % Nz
                exi = ex[i]
                eyi = ey[i]
                for y in range(Ny):
                    yn = (y + eyi) % Ny
                    if exi == 0:
                        for x in range(Nx):
                            f_dst[i, zn, yn, x] = f_src[i, z, y, x]
                    elif exi == 1:
                        f_dst[i, zn, yn, 0] = f_src[i, z, y, Nx - 1]
                        for x in range(1, Nx):
                            f_dst[i, zn, yn, x] = f_src[i, z, y, x - 1]
                    elif exi == -1:
                        f_dst[i, zn, yn, Nx - 1] = f_src[i, z, y, 0]
                        for x in range(Nx - 1):
                            f_dst[i, zn, yn, x] = f_src[i, z, y, x + 1]
                    else:
                        # General |ex| > 1 fallback (unused by D3Q19)
                        for x in range(Nx):
                            f_dst[i, zn, yn, (x + exi) % Nx] = f_src[i, z, y, x]

else:
    def collision_kernel_3d(f, f_post, solid, omega, ex, ey, ez, w, omega_field, use_omega_field):
        """NumPy fallback for collision_kernel_3d."""
        collision_kernel_3d_xp(f, f_post, solid, omega, ex, ey, ez, w, omega_field, use_omega_field, xp=np)

    def stream_kernel_3d(f_src, f_dst, ex, ey, ez):
        """NumPy fallback for stream_kernel_3d."""
        stream_kernel_3d_xp(f_src, f_dst, ex, ey, ez, xp=np)


def collision_kernel_3d_xp(f, f_post, solid, omega, ex, ey, ez, w, omega_field, use_omega_field, xp=np):
    """Array-backend-agnostic BGK collision (works with numpy or cupy)."""
    rho = f.sum(axis=0)
    # Guard the denominator before dividing: 1.0 / rho over the raw array
    # divides by zero in empty cells (RuntimeWarning) before discarding them.
    positive = rho > 0.0
    inv_rho = xp.where(positive, 1.0 / xp.where(positive, rho, 1.0), 0.0)
    ux = inv_rho * xp.einsum('i,izyx->zyx', ex.astype(xp.float64), f)
    uy = inv_rho * xp.einsum('i,izyx->zyx', ey.astype(xp.float64), f)
    uz = inv_rho * xp.einsum('i,izyx->zyx', ez.astype(xp.float64), f)
    ux[solid] = 0.0; uy[solid] = 0.0; uz[solid] = 0.0
    usq = ux*ux + uy*uy + uz*uz
    om = omega_field if use_omega_field else xp.full(ux.shape, omega)
    for i in range(f.shape[0]):
        eu = ex[i]*ux + ey[i]*uy + ez[i]*uz
        feqi = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
        f_post[i] = (1.0 - om) * f[i] + om * feqi


def stream_kernel_3d_xp(f_src, f_dst, ex, ey, ez, xp=np):
    """Array-backend-agnostic streaming (works with numpy or cupy)."""
    for i in range(f_src.shape[0]):
        tmp = xp.roll(f_src[i], int(ez[i]), axis=0)
        tmp = xp.roll(tmp,      int(ey[i]), axis=1)
        f_dst[i] = xp.roll(tmp, int(ex[i]), axis=2)
