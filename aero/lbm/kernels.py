"""
Numba-JIT accelerated LBM kernels for D2Q9.

Two kernels cover the entire hot path per timestep:

  collision_kernel  — fused macroscopic + BGK collision, operates in-place
  stream_kernel     — push streaming into a pre-allocated output array

When Numba is not installed (HAS_NUMBA=False) the module exposes
pure-NumPy fallbacks with the same signatures so the rest of the code
never needs to branch on import.

Math notes
----------
BGK collision:
    feq[i] = w[i] * rho * (1 + 3*(e·u) + 4.5*(e·u)^2 - 1.5*(u·u))
    f_post[i] = (1 - omega) * f[i] + omega * feq[i]

Push streaming (periodic, bijective):
    f_dst[i, (y+ey)%Ny, (x+ex)%Nx] = f_src[i, y, x]
    Each (i, yn, xn) is written exactly once — no data race with prange.

Solid cells: velocity clamped to zero before collision (bounce-back handled
separately by apply_bounce_back in boundary.py).
"""

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Numba JIT kernels (defined only when Numba is available)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @nb.njit(cache=True, parallel=True)
    def collision_kernel(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        w: np.ndarray,
    ) -> None:
        """
        Fused macroscopic + BGK collision.

        Reads f (Q, Ny, Nx), writes post-collision state into f_post (same
        shape).  Both arrays must be C-contiguous float64.

        Solid cells use ux=uy=0 so that collision returns feq(rho, 0, 0);
        the mid-link bounce-back in apply_bounce_back() corrects the
        streamed values afterwards.
        """
        Q, Ny, Nx = f.shape
        for y in nb.prange(Ny):
            for x in range(Nx):
                # --- macroscopic ---
                rho = 0.0
                mx  = 0.0
                my  = 0.0
                for i in range(Q):
                    fi   = f[i, y, x]
                    rho += fi
                    mx  += ex[i] * fi
                    my  += ey[i] * fi

                inv_rho = 1.0 / rho if rho > 0.0 else 0.0
                if solid[y, x]:
                    ux = 0.0
                    uy = 0.0
                else:
                    ux = mx * inv_rho
                    uy = my * inv_rho

                usq = ux * ux + uy * uy

                # --- BGK ---
                for i in range(Q):
                    eu    = ex[i] * ux + ey[i] * uy
                    feqi  = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
                    f_post[i, y, x] = (1.0 - omega) * f[i, y, x] + omega * feqi

    @nb.njit(cache=True, parallel=True)
    def stream_kernel(
        f_src: np.ndarray,
        f_dst: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
    ) -> None:
        """
        Push streaming: f_dst[i,(y+ey)%Ny,(x+ex)%Nx] = f_src[i,y,x].

        For a given direction i, the shift (ey[i], ex[i]) is a bijection on
        the periodic grid, so each output cell is written exactly once.
        prange over y rows is therefore race-free.
        """
        Q, Ny, Nx = f_src.shape
        for y in nb.prange(Ny):
            for x in range(Nx):
                for i in range(Q):
                    yn = (y + ey[i]) % Ny
                    xn = (x + ex[i]) % Nx
                    f_dst[i, yn, xn] = f_src[i, y, x]

# ---------------------------------------------------------------------------
# Pure-NumPy fallbacks — same signatures, used when Numba is absent
# ---------------------------------------------------------------------------

else:
    def collision_kernel(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        w: np.ndarray,
    ) -> None:
        """NumPy fallback for collision_kernel (used when Numba is not installed)."""
        rho = f.sum(axis=0)
        inv_rho = np.where(rho > 0.0, 1.0 / rho, 0.0)
        ux = inv_rho * np.einsum('i,iyx->yx', ex.astype(np.float64), f)
        uy = inv_rho * np.einsum('i,iyx->yx', ey.astype(np.float64), f)
        ux[solid] = 0.0
        uy[solid] = 0.0
        usq = ux * ux + uy * uy
        for i in range(9):
            eu = ex[i] * ux + ey[i] * uy
            feqi = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
            f_post[i] = (1.0 - omega) * f[i] + omega * feqi

    def stream_kernel(
        f_src: np.ndarray,
        f_dst: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
    ) -> None:
        """NumPy fallback for stream_kernel (used when Numba is not installed)."""
        for i in range(9):
            f_dst[i] = np.roll(
                np.roll(f_src[i], int(ey[i]), axis=0),
                int(ex[i]), axis=1,
            )
