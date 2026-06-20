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
        """Fused macroscopic + BGK collision for D3Q19.  Parallel over z slices."""
        Q, Nz, Ny, Nx = f.shape
        for z in nb.prange(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    rho = 0.0; mx = 0.0; my = 0.0; mz = 0.0
                    for i in range(Q):
                        fi  = f[i, z, y, x]
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
                    om = omega_field[z, y, x] if use_omega_field else omega
                    for i in range(Q):
                        eu   = ex[i]*ux + ey[i]*uy + ez[i]*uz
                        feqi = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
                        f_post[i, z, y, x] = (1.0 - om) * f[i, z, y, x] + om * feqi

    @nb.njit(cache=True, parallel=True)
    def stream_kernel_3d(
        f_src: np.ndarray,
        f_dst: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
        ez: np.ndarray,
    ) -> None:
        """Push streaming for D3Q19. f_dst[i,(z+ez)%Nz,(y+ey)%Ny,(x+ex)%Nx]=f_src[i,z,y,x]."""
        Q, Nz, Ny, Nx = f_src.shape
        for z in nb.prange(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    for i in range(Q):
                        zn = (z + ez[i]) % Nz
                        yn = (y + ey[i]) % Ny
                        xn = (x + ex[i]) % Nx
                        f_dst[i, zn, yn, xn] = f_src[i, z, y, x]

else:
    def collision_kernel_3d(f, f_post, solid, omega, ex, ey, ez, w, omega_field, use_omega_field):
        """NumPy fallback for collision_kernel_3d."""
        rho = f.sum(axis=0)
        inv_rho = np.where(rho > 0.0, 1.0 / rho, 0.0)
        ux = inv_rho * np.einsum('i,izyx->zyx', ex.astype(np.float64), f)
        uy = inv_rho * np.einsum('i,izyx->zyx', ey.astype(np.float64), f)
        uz = inv_rho * np.einsum('i,izyx->zyx', ez.astype(np.float64), f)
        ux[solid] = 0.0; uy[solid] = 0.0; uz[solid] = 0.0
        usq = ux*ux + uy*uy + uz*uz
        om = omega_field if use_omega_field else np.full(ux.shape, omega)
        for i in range(19):
            eu = ex[i]*ux + ey[i]*uy + ez[i]*uz
            feqi = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
            f_post[i] = (1.0 - om) * f[i] + om * feqi

    def stream_kernel_3d(f_src, f_dst, ex, ey, ez):
        """NumPy fallback for stream_kernel_3d."""
        for i in range(19):
            tmp = np.roll(f_src[i], int(ez[i]), axis=0)
            tmp = np.roll(tmp,      int(ey[i]), axis=1)
            f_dst[i] = np.roll(tmp, int(ex[i]), axis=2)
