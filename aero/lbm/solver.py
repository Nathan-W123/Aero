"""
LBM D2Q9 wind tunnel solver.

Timestep loop:
  1. BGK collision (fused with macroscopic) → f_post
  2. Save f_pre = f_post  (needed for mid-link bounce-back)
  3. Stream: push each direction to its neighbour
  4. Apply BCs: bounce-back → inlet → outlet → walls
  5. Record forces via momentum exchange

BC options
----------
inlet_bc  : "velocity"  — Zou-He velocity (impose ux=u0, uy=0)
            "pressure"  — Zou-He pressure (impose rho=rho_in, uy=0)
outlet_bc : "convective" — advective non-reflecting outflow (default)
            "pressure"   — Zou-He pressure (impose rho=rho_out, uy=0)
            "zerogradient" — f[:,-1] = f[:,-2]
wall_bc   : "slip"    — specular reflection (default)
            "noslip"  — full bounce-back (use with Poiseuille benchmarks)

Backend options
---------------
backend : "auto"   — use Numba if installed, else NumPy (default)
          "numba"  — require Numba (ImportError if missing)
          "numpy"  — always use pure NumPy
"""

import numpy as np
import pathlib
import time
from typing import Callable, Optional

from .d2q9 import Q, E, W, compute_macroscopic, compute_feq
from . import kernels as _kernels
from .kernels_mrt import MRTKernel2D
from .boundary import (
    apply_inlet_zou_he,
    apply_inlet_zou_he_pressure,
    apply_outlet_convective,
    apply_outlet_zero_gradient,
    apply_outlet_zou_he_pressure,
    apply_slip_walls,
    apply_noslip_walls,
    apply_bounce_back,
    build_surface_links,
)
from ..forces import compute_forces, forces_to_coefficients
from ..diagnostics import check_stability, check_convergence


class Solver:
    def __init__(
        self,
        Ny: int,
        Nx: int,
        solid: np.ndarray,
        omega: float,
        u0: float,
        D: float,
        rho0: float = 1.0,
        wall_bc: str = "slip",
        inlet_bc: str = "velocity",
        outlet_bc: str = "convective",
        rho_in: float = 1.0,
        rho_out: float = 1.0,
        backend: str = "auto",
        collision: str = "bgk",
        inlet_perturbation: float = 0.0,
    ):
        """
        Parameters
        ----------
        Ny, Nx    : int     — grid dimensions
        solid     : ndarray bool (Ny, Nx) — obstacle mask
        omega     : float   — BGK relaxation rate (1/tau)
        u0        : float   — inlet velocity in lattice units (used for velocity BC
                              and convective outlet; ignored for pressure-driven flow)
        D         : float   — characteristic length in lattice cells
        rho0      : float   — reference density
        wall_bc   : "slip" | "noslip"
        inlet_bc  : "velocity" | "pressure"
        outlet_bc : "convective" | "pressure" | "zerogradient"
        rho_in    : float   — inlet density (only for inlet_bc="pressure")
        rho_out   : float   — outlet density (only for outlet_bc="pressure")
        backend   : "auto" | "numpy" | "numba" — compute backend
        collision : "bgk" | "mrt" — collision operator
        inlet_perturbation : float — sinusoidal uy fraction of u0 at inlet (2D shedding)
        """
        self.Ny = Ny
        self.Nx = Nx
        self.solid = solid
        self.fluid = ~solid
        self.omega = omega
        self.u0 = u0
        self.D = D
        self.rho0 = rho0
        self.wall_bc = wall_bc
        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        self.rho_in = rho_in
        self.rho_out = rho_out
        self.inlet_perturbation = max(float(inlet_perturbation), 0.0)
        if collision not in ("bgk", "mrt"):
            raise ValueError(f"Unknown collision '{collision}'. Choose 'bgk' or 'mrt'.")
        self.collision = collision

        # --- Backend selection ---
        if backend == "auto":
            self._use_numba = _kernels.HAS_NUMBA
        elif backend == "numba":
            if not _kernels.HAS_NUMBA:
                raise ImportError("backend='numba' requested but Numba is not installed.")
            self._use_numba = True
        elif backend == "numpy":
            self._use_numba = False
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'auto', 'numpy', or 'numba'.")
        self.backend: str = "numba" if self._use_numba else "numpy"

        # Pre-cast lattice arrays for kernels (int32 ex/ey, float64 w)
        self._ex = E[:, 0].astype(np.int32)
        self._ey = E[:, 1].astype(np.int32)
        self._w  = W.copy()
        self._mrt_kernel = None

        # Ensure solid is bool and C-contiguous (required by Numba kernels)
        self.solid = np.ascontiguousarray(solid, dtype=np.bool_)

        # Precompute surface links for bounce-back
        self.surface_links = build_surface_links(self.solid)

        # Initialize f to equilibrium everywhere (including solid cells).
        # Must be C-contiguous float64 for Numba kernels.
        rho_init = np.full((Ny, Nx), rho0)
        ux_init  = np.full((Ny, Nx), u0)
        uy_init  = np.zeros((Ny, Nx))
        self.f   = np.ascontiguousarray(compute_feq(rho_init, ux_init, uy_init), dtype=np.float64)

        # Convective outlet BC: stores the outlet column (shape 9, Ny)
        # from the previous timestep.  Initialised to equilibrium.
        self._f_outlet_prev = compute_feq(rho_init, ux_init, uy_init)[:, :, -1].copy()
        if self.collision == "mrt":
            self._mrt_kernel = MRTKernel2D(omega, use_numba=self._use_numba)

        # Step counter — incremented by run(); preserved across checkpoint loads
        self.step_count: int = 0

        # State available after run()
        self.rho: Optional[np.ndarray] = None
        self.ux:  Optional[np.ndarray] = None
        self.uy:  Optional[np.ndarray] = None
        self.Cd_history: list = []
        self.Cl_history: list = []

    # ------------------------------------------------------------------
    # Private: single timestep
    # ------------------------------------------------------------------

    def _step(self) -> tuple:
        """Execute one LBM timestep. Returns instantaneous (Cd, Cl)."""
        f = self.f

        if self.collision == "mrt":
            f_post = np.empty_like(f)
            self._mrt_kernel.collide(f, f_post, self.solid, self._ex, self._ey, self._w)
            f_pre = f_post.copy()
            if self._use_numba:
                f_new = np.empty_like(f_post)
                _kernels.stream_kernel(f_post, f_new, self._ex, self._ey)
                f_post = f_new
            else:
                for i in range(Q):
                    f_post[i] = np.roll(f_post[i], shift=int(E[i, 1]), axis=0)
                    f_post[i] = np.roll(f_post[i], shift=int(E[i, 0]), axis=1)

        elif self._use_numba:
            # --- Numba fast path ---

            # 1+2. Fused macroscopic + BGK collision
            f_post = np.empty_like(f)
            _kernels.collision_kernel(
                f, f_post, self.solid, self.omega,
                self._ex, self._ey, self._w,
            )

            # 3. Pre-streaming snapshot for mid-link bounce-back
            f_pre = f_post.copy()

            # 4. Push streaming into a fresh array
            f_new = np.empty_like(f_post)
            _kernels.stream_kernel(f_post, f_new, self._ex, self._ey)
            f_post = f_new

        else:
            # --- NumPy fallback path ---

            # 1. Macroscopic variables
            rho, ux, uy = compute_macroscopic(f)

            # 2. BGK collision; solid cells relax to zero-velocity equilibrium
            ux_c = ux.copy()
            uy_c = uy.copy()
            ux_c[self.solid] = 0.0
            uy_c[self.solid] = 0.0
            feq    = compute_feq(rho, ux_c, uy_c)
            f_post = (1.0 - self.omega) * f + self.omega * feq

            # 3. Pre-streaming snapshot for mid-link bounce-back
            f_pre = f_post.copy()

            # 4. Stream
            for i in range(Q):
                f_post[i] = np.roll(f_post[i], shift=int(E[i, 1]), axis=0)
                f_post[i] = np.roll(f_post[i], shift=int(E[i, 0]), axis=1)

        # 5. Boundary conditions — order matters (same for both backends)
        apply_bounce_back(f_post, f_pre, self.surface_links)

        if self.inlet_bc == "velocity":
            apply_inlet_zou_he(
                f_post,
                self.u0,
                uy_amp=self.inlet_perturbation,
                step=self.step_count + 1,
            )
        elif self.inlet_bc == "pressure":
            apply_inlet_zou_he_pressure(f_post, self.rho_in)

        if self.outlet_bc == "convective":
            apply_outlet_convective(f_post, self._f_outlet_prev, self.u0)
        elif self.outlet_bc == "pressure":
            apply_outlet_zou_he_pressure(f_post, self.rho_out)
        elif self.outlet_bc == "zerogradient":
            apply_outlet_zero_gradient(f_post)

        if self.wall_bc == "slip":
            apply_slip_walls(f_post)
        elif self.wall_bc == "noslip":
            apply_noslip_walls(f_post)

        self.f = f_post
        self.step_count += 1

        # 6. Forces
        Fx_lbm, Fy_lbm = compute_forces(f_pre, f_post, self.surface_links, self.rho0, self.u0)
        Cd, Cl = forces_to_coefficients(Fx_lbm, Fy_lbm, self.rho0, self.u0, self.D)
        return Cd, Cl

    # ------------------------------------------------------------------
    # Public: run simulation
    # ------------------------------------------------------------------

    def run(
        self,
        steps: int,
        check_every: int = 500,
        verbose: bool = True,
        callback: Optional[Callable] = None,
        checkpoint_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the simulation for `steps` timesteps.

        Parameters
        ----------
        steps            : int  — total LBM steps
        check_every      : int  — frequency of stability checks and stdout progress
        verbose          : bool — print per-check progress line
        callback         : optional callable(step, Cd, Cl, rho, ux, uy)
                           called every check_every steps
        checkpoint_every : optional int — save checkpoint every N steps
        checkpoint_dir   : optional str — directory for checkpoint files

        Returns
        -------
        dict  keys: Cd_mean, Cl_mean, Cd_std, Cl_std,
                    Cd_history, Cl_history, rho, ux, uy
        """
        if verbose:
            print(f"  Backend   : {self.backend}")
            print(f"  Collision : {self.collision}")

        t_start    = time.perf_counter()
        avg_window = max(1, steps // 5)   # average over last 20 % of run

        for step in range(1, steps + 1):
            Cd, Cl = self._step()
            self.Cd_history.append(Cd)
            self.Cl_history.append(Cl)

            if checkpoint_every and checkpoint_dir and step % checkpoint_every == 0:
                ckpt_path = pathlib.Path(checkpoint_dir) / f"checkpoint_{self.step_count:08d}.npz"
                self.save_checkpoint(str(ckpt_path))

            if step % check_every == 0:
                rho, ux, uy = compute_macroscopic(self.f)
                check_stability(self.f, rho, ux, uy, step)

                elapsed        = time.perf_counter() - t_start
                steps_per_sec  = step / elapsed
                window         = min(check_every, len(self.Cd_history))
                Cd_avg         = float(np.mean(self.Cd_history[-window:]))
                Cl_avg         = float(np.mean(self.Cl_history[-window:]))

                if verbose:
                    print(
                        f"  step {step:6d}/{steps}  "
                        f"Cd={Cd_avg:+.4f}  Cl={Cl_avg:+.4f}  "
                        f"[{steps_per_sec:.0f} steps/s]"
                    )

                if callback is not None:
                    callback(step, Cd, Cl, rho, ux, uy)

        # Final macroscopic state
        rho, ux, uy        = compute_macroscopic(self.f)
        self.rho, self.ux, self.uy = rho, ux, uy

        Cd_arr = np.array(self.Cd_history[-avg_window:])
        Cl_arr = np.array(self.Cl_history[-avg_window:])

        return {
            "Cd_mean":    float(np.mean(Cd_arr)),
            "Cl_mean":    float(np.mean(Cl_arr)),
            "Cd_std":     float(np.std(Cd_arr)),
            "Cl_std":     float(np.std(Cl_arr)),
            "Cd_history": self.Cd_history,
            "Cl_history": self.Cl_history,
            "rho": rho,
            "ux":  ux,
            "uy":  uy,
        }

    # ------------------------------------------------------------------
    # Checkpoint: save / load solver state
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save full solver state to a .npz file.

        Stores: f array, _f_outlet_prev, Cd/Cl histories, step count.
        Does NOT store geometry or solver parameters — those come from
        the SimulationCase config and must be used to reconstruct the
        Solver before calling load_checkpoint().
        """
        np.savez_compressed(
            path,
            f=self.f,
            f_outlet_prev=self._f_outlet_prev,
            Cd_history=np.array(self.Cd_history, dtype=np.float64),
            Cl_history=np.array(self.Cl_history, dtype=np.float64),
            step_count=np.array(self.step_count, dtype=np.int64),
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Restore solver state from a .npz checkpoint.

        The Solver must already be constructed with the same geometry and
        parameters as when the checkpoint was saved.  Call this before run().
        """
        data = np.load(path)
        self.f                = data["f"]
        self._f_outlet_prev   = data["f_outlet_prev"]
        self.Cd_history       = list(data["Cd_history"])
        self.Cl_history       = list(data["Cl_history"])
        self.step_count       = int(data["step_count"])
