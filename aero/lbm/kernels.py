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

from .physics import inv_positive

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None  # type: ignore[assignment]

# x-tile length used by the collision kernel.  Sized so one tile's worth of
# scratch plus the Q source/destination streams stay resident in L1.
_TILE = 64


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
        omega_field: np.ndarray,
        use_omega_field: bool,
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
    ) -> None:
        """
        Fused macroscopic + BGK collision, with Guo forcing.

        Reads f (Q, Ny, Nx), writes post-collision state into f_post (same
        shape).  Both arrays must be C-contiguous float64.

        Solid cells use ux=uy=0 so that collision returns feq(rho, 0, 0);
        the mid-link bounce-back in apply_bounce_back() corrects the
        streamed values afterwards.  They take no force.

        ``force_mode`` is FORCE_NONE / FORCE_UNIFORM / FORCE_FIELD from
        forcing.py.  When forcing is active the velocity carries the
        half-force correction ``u = m/rho + a/2`` and the Guo source term is
        added to the post-collision populations.
        """
        Q, Ny, Nx = f.shape
        for y in nb.prange(Ny):
            # Per-row scratch: a tile of x is carried through the three passes
            # so every inner loop walks one direction plane at unit stride
            # instead of gathering Q strided values per cell.
            rho_t = np.empty(_TILE)
            ux_t = np.empty(_TILE)
            uy_t = np.empty(_TILE)
            usq_t = np.empty(_TILE)
            om_t = np.empty(_TILE)
            ax_t = np.zeros(_TILE)
            ay_t = np.zeros(_TILE)
            ua_t = np.zeros(_TILE)

            for x0 in range(0, Nx, _TILE):
                n = Nx - x0
                if n > _TILE:
                    n = _TILE

                # --- macroscopic moments ---
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

                # --- local acceleration ---
                if force_mode == 1:
                    for k in range(n):
                        ax_t[k] = acc_uniform[0]
                        ay_t[k] = acc_uniform[1]
                elif force_mode == 2:
                    for k in range(n):
                        ax_t[k] = acc_field[0, y, x0 + k]
                        ay_t[k] = acc_field[1, y, x0 + k]

                for k in range(n):
                    rho = rho_t[k]
                    inv_rho = 1.0 / rho if rho > 0.0 else 0.0
                    if solid[y, x0 + k]:
                        ux_t[k] = 0.0
                        uy_t[k] = 0.0
                        ax_t[k] = 0.0
                        ay_t[k] = 0.0
                    else:
                        # half-force correction: rho u = sum e_i f_i + F/2
                        ux_t[k] = ux_t[k] * inv_rho + 0.5 * ax_t[k]
                        uy_t[k] = uy_t[k] * inv_rho + 0.5 * ay_t[k]
                    usq_t[k] = ux_t[k] * ux_t[k] + uy_t[k] * uy_t[k]
                    ua_t[k] = ux_t[k] * ax_t[k] + uy_t[k] * ay_t[k]
                    om_t[k] = omega_field[y, x0 + k] if use_omega_field else omega

                # --- BGK (+ Guo source) ---
                if force_mode == 0:
                    for i in range(Q):
                        exi = ex[i]
                        eyi = ey[i]
                        wi = w[i]
                        for k in range(n):
                            eu = exi * ux_t[k] + eyi * uy_t[k]
                            feqi = wi * rho_t[k] * (
                                1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq_t[k]
                            )
                            om = om_t[k]
                            f_post[i, y, x0 + k] = (
                                (1.0 - om) * f[i, y, x0 + k] + om * feqi
                            )
                else:
                    for i in range(Q):
                        exi = ex[i]
                        eyi = ey[i]
                        wi = w[i]
                        for k in range(n):
                            eu = exi * ux_t[k] + eyi * uy_t[k]
                            feqi = wi * rho_t[k] * (
                                1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq_t[k]
                            )
                            om = om_t[k]
                            ea = exi * ax_t[k] + eyi * ay_t[k]
                            src = (1.0 - 0.5 * om) * wi * rho_t[k] * (
                                3.0 * (ea - ua_t[k]) + 9.0 * eu * ea
                            )
                            f_post[i, y, x0 + k] = (
                                (1.0 - om) * f[i, y, x0 + k] + om * feqi + src
                            )

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

        The row loop is the innermost one so that both source and destination
        run at unit stride: the x shift becomes a contiguous block move plus a
        single wrapped element, instead of a per-cell modulo and a scattered
        write across Q planes.
        """
        Q, Ny, Nx = f_src.shape
        for y in nb.prange(Ny):
            for i in range(Q):
                yn = (y + ey[i]) % Ny
                exi = ex[i]
                if exi == 0:
                    for x in range(Nx):
                        f_dst[i, yn, x] = f_src[i, y, x]
                elif exi == 1:
                    f_dst[i, yn, 0] = f_src[i, y, Nx - 1]
                    for x in range(1, Nx):
                        f_dst[i, yn, x] = f_src[i, y, x - 1]
                elif exi == -1:
                    f_dst[i, yn, Nx - 1] = f_src[i, y, 0]
                    for x in range(Nx - 1):
                        f_dst[i, yn, x] = f_src[i, y, x + 1]
                else:
                    # General |ex| > 1 fallback (unused by D2Q9)
                    for x in range(Nx):
                        f_dst[i, yn, (x + exi) % Nx] = f_src[i, y, x]

# ---------------------------------------------------------------------------
# Pure-NumPy implementations
#
# These are always defined, not only when Numba is missing: `backend="numpy"`
# is a contract the solver has to be able to honour even on a machine that has
# Numba installed (for debugging a JIT problem, or for a run that must not
# depend on the JIT at all).
# ---------------------------------------------------------------------------

if True:
    def collision_kernel_numpy(
        f: np.ndarray,
        f_post: np.ndarray,
        solid: np.ndarray,
        omega: float,
        ex: np.ndarray,
        ey: np.ndarray,
        w: np.ndarray,
        omega_field: np.ndarray,
        use_omega_field: bool,
        acc_uniform: np.ndarray,
        acc_field: np.ndarray,
        force_mode: int,
    ) -> None:
        """Pure-NumPy BGK collision with Guo forcing."""
        Q = f.shape[0]
        rho = f.sum(axis=0)
        inv_rho = inv_positive(rho)
        ux = inv_rho * np.einsum('i,iyx->yx', ex.astype(np.float64), f)
        uy = inv_rho * np.einsum('i,iyx->yx', ey.astype(np.float64), f)
        fluid = ~solid
        if force_mode == 1:
            ax = np.where(fluid, acc_uniform[0], 0.0)
            ay = np.where(fluid, acc_uniform[1], 0.0)
        elif force_mode == 2:
            ax = np.where(fluid, acc_field[0], 0.0)
            ay = np.where(fluid, acc_field[1], 0.0)
        else:
            ax = ay = None
        if ax is not None:
            ux = ux + 0.5 * ax          # half-force correction
            uy = uy + 0.5 * ay
        ux[solid] = 0.0
        uy[solid] = 0.0
        usq = ux * ux + uy * uy
        om = omega_field if use_omega_field else np.full(ux.shape, omega)
        ua = None if ax is None else ux * ax + uy * ay
        for i in range(Q):
            eu = ex[i] * ux + ey[i] * uy
            feqi = w[i] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq)
            f_post[i] = (1.0 - om) * f[i] + om * feqi
            if ax is not None:
                ea = ex[i] * ax + ey[i] * ay
                f_post[i] += (1.0 - 0.5 * om) * w[i] * rho * (
                    3.0 * (ea - ua) + 9.0 * eu * ea
                )

    def stream_kernel_numpy(
        f_src: np.ndarray,
        f_dst: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
    ) -> None:
        """Pure-NumPy push streaming."""
        for i in range(f_src.shape[0]):
            f_dst[i] = np.roll(
                np.roll(f_src[i], int(ey[i]), axis=0),
                int(ex[i]), axis=1,
            )


if not HAS_NUMBA:
    # Without Numba the public names are the NumPy ones.
    collision_kernel = collision_kernel_numpy
    stream_kernel = stream_kernel_numpy
