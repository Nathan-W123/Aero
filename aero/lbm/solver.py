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

from .d2q9 import Q, E, W, OPP, compute_macroscopic, compute_feq
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
    apply_bounce_back_bouzidi,
    build_surface_links,
)
from ..forces import (
    compute_forces,
    compute_force_split_2d,
    forces_to_coefficients,
    split_to_coefficients,
)
from ..diagnostics import check_stability, detect_statistical_stationarity
from .sponge import build_sponge_sigma, apply_sponge_relaxation_2d
from .les import (
    strain_rate_magnitude_2d,
    smagorinsky_nu_sgs,
    omega_from_nu,
    build_omega_field_2d,
)
from .physics import base_nu_from_omega
from .trt2d import trt_taus
from . import kernels_trt as _ktrt
from .ibm_guo import apply_guo_forcing_2d


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
        trt_lambda: float = 0.25,
        sponge_thickness: int = 0,
        sponge_strength: float = 0.1,
        les: bool = False,
        les_cs: float = 0.16,
        ibm_enabled: bool = False,
        phi: Optional[np.ndarray] = None,
        bouzidi: bool = False,
        van_driest: bool = False,
        van_driest_A: float = 25.0,
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
        collision : "bgk" | "mrt" | "trt" — collision operator
        inlet_perturbation : float — sinusoidal uy fraction of u0 at inlet (2D shedding)
        trt_lambda : float — TRT magic parameter (default 0.25)
        sponge_thickness : int — outlet sponge cells (0=off)
        sponge_strength : float — max sponge relaxation rate
        les : bool — enable Smagorinsky LES
        les_cs : float — Smagorinsky constant
        ibm_enabled : bool — Guo IBM forcing
        phi : optional signed-distance field for IBM
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
        self.trt_lambda = float(trt_lambda)
        self.sponge_thickness = int(sponge_thickness)
        self.sponge_strength = float(sponge_strength)
        self.les = bool(les)
        self.les_cs = float(les_cs)
        self.ibm_enabled = bool(ibm_enabled)
        self.phi = phi
        self.bouzidi = bool(bouzidi)
        self.van_driest = bool(van_driest)
        self.van_driest_A = float(van_driest_A)
        self.tau = 1.0 / omega
        self._base_nu = base_nu_from_omega(omega)
        if collision not in ("bgk", "mrt", "trt"):
            raise ValueError(f"Unknown collision '{collision}'. Choose 'bgk', 'mrt', or 'trt'.")
        self.collision = collision
        _, self._s_minus_trt = trt_taus(omega, self.trt_lambda)
        self._s_minus_trt = 1.0 / self._s_minus_trt

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
        _phi_for_q = phi if (bouzidi and phi is not None) else None
        self.surface_links, self.q_vals = build_surface_links(self.solid, _phi_for_q)

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

        self._sponge_sigma = build_sponge_sigma(self.Nx, self.sponge_thickness, self.sponge_strength)
        self._u_wall_x = np.zeros((Ny, Nx))
        self._u_wall_y = np.zeros((Ny, Nx))
        self._omega_dummy = np.zeros((Ny, Nx), dtype=np.float64)

        # Step counter — incremented by run(); preserved across checkpoint loads
        self.step_count: int = 0

        # State available after run()
        self.rho: Optional[np.ndarray] = None
        self.ux:  Optional[np.ndarray] = None
        self.uy:  Optional[np.ndarray] = None
        self.Cd_history: list = []
        self.Cl_history: list = []
        self.Cd_p_history: list = []
        self.Cd_v_history: list = []

    # ------------------------------------------------------------------
    # Private: single timestep
    # ------------------------------------------------------------------

    def _step(self) -> tuple:
        """Execute one LBM timestep. Returns instantaneous (Cd, Cl)."""
        f = self.f

        if self.ibm_enabled and self.phi is not None:
            apply_guo_forcing_2d(
                f, self.phi, self.tau,
                self._u_wall_x, self._u_wall_y,
                self._ex, self._ey, self._w, self.solid,
            )

        omega_field = self._omega_dummy
        use_omega_field = False
        if self.les:
            omega_field = build_omega_field_2d(
                f, self.solid, self.fluid, self._base_nu, self.omega, self.les_cs,
                phi=self.phi, van_driest=self.van_driest, van_driest_A=self.van_driest_A,
            )
            use_omega_field = True
        omega_use = self.omega

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

        elif self.collision == "trt":
            f_post = np.empty_like(f)
            if self._use_numba and _ktrt._HAS_NUMBA:
                _ktrt.trt_collision_kernel(
                    f, f_post, self.solid, omega_use,
                    self._ex, self._ey, self._w,
                    OPP.astype(np.int32), self.trt_lambda,
                    omega_field, use_omega_field,
                )
            else:
                _ktrt.trt_collision_numpy(
                    f, f_post, self.solid, omega_use,
                    self._ex, self._ey, self._w, self.trt_lambda,
                    omega_field if use_omega_field else None,
                )
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
                f, f_post, self.solid, omega_use,
                self._ex, self._ey, self._w,
                omega_field, use_omega_field,
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
            om = omega_field if use_omega_field else omega_use
            f_post = (1.0 - om) * f + om * feq

            # 3. Pre-streaming snapshot for mid-link bounce-back
            f_pre = f_post.copy()

            # 4. Stream
            for i in range(Q):
                f_post[i] = np.roll(f_post[i], shift=int(E[i, 1]), axis=0)
                f_post[i] = np.roll(f_post[i], shift=int(E[i, 0]), axis=1)

        # 5. Boundary conditions — order matters (same for both backends)
        if not self.ibm_enabled:
            if self.bouzidi:
                apply_bounce_back_bouzidi(f_post, f_pre, self.surface_links, self.q_vals)
            else:
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

        if self.sponge_thickness > 0:
            apply_sponge_relaxation_2d(
                f_post, self._sponge_sigma, self.u0,
                self._ex, self._ey, self._w,
            )

        if self.wall_bc == "slip":
            apply_slip_walls(f_post)
        elif self.wall_bc == "noslip":
            apply_noslip_walls(f_post)

        self.f = f_post
        self.step_count += 1

        # 6. Forces
        Fx_lbm, Fy_lbm = compute_forces(f_pre, f_post, self.surface_links, self.rho0, self.u0)
        Cd, Cl = forces_to_coefficients(Fx_lbm, Fy_lbm, self.rho0, self.u0, self.D)
        fx_p, fy_p, fx_v, fy_v = compute_force_split_2d(f_pre, f_post, self.surface_links)
        cdp, _, cdv, _ = split_to_coefficients(fx_p, fy_p, fx_v, fy_v, self.rho0, self.u0, self.D)
        self._last_cd_p = cdp
        self._last_cd_v = cdv
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
        auto_stop: bool = False,
        convergence_window: int = 5000,
        convergence_tol: float = 0.01,
        strouhal_window: int = 4096,
        strouhal_tol: float = 0.05,
        hdf5_path: Optional[str] = None,
        hdf5_every: Optional[int] = None,
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

        from ..hdf5_writer import HDF5Writer
        hdf5_writer = HDF5Writer(hdf5_path, Ny=self.Ny, Nx=self.Nx, dims=2) if hdf5_path else None

        t_start    = time.perf_counter()
        avg_window = max(1, steps // 5)   # average over last 20 % of run
        stop_reason = "max_steps"
        last_convergence = None
        last_strouhal = None

        for step in range(1, steps + 1):
            Cd, Cl = self._step()
            self.Cd_history.append(Cd)
            self.Cl_history.append(Cl)
            self.Cd_p_history.append(getattr(self, "_last_cd_p", 0.0))
            self.Cd_v_history.append(getattr(self, "_last_cd_v", 0.0))

            if checkpoint_every and checkpoint_dir and step % checkpoint_every == 0:
                ckpt_path = pathlib.Path(checkpoint_dir) / f"checkpoint_{self.step_count:08d}.npz"
                self.save_checkpoint(str(ckpt_path))

            if hdf5_writer and hdf5_every and step % hdf5_every == 0:
                from ..lbm.d2q9 import compute_macroscopic as _cm2
                _rho, _ux, _uy = _cm2(self.f)
                hdf5_writer.write_step(self.step_count, ux=_ux, uy=_uy, rho=_rho)

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

                if auto_stop:
                    last_convergence, last_strouhal, should_stop = detect_statistical_stationarity(
                        self.Cd_history,
                        self.Cl_history,
                        self.D,
                        self.u0,
                        convergence_window=convergence_window,
                        convergence_tol=convergence_tol,
                        strouhal_window=strouhal_window,
                        strouhal_tol=strouhal_tol,
                    )
                    if should_stop:
                        stop_reason = "auto_converged"
                        break

        # Final macroscopic state
        rho, ux, uy        = compute_macroscopic(self.f)
        self.rho, self.ux, self.uy = rho, ux, uy

        Cd_arr = np.array(self.Cd_history[-avg_window:])
        Cl_arr = np.array(self.Cl_history[-avg_window:])
        Cdp_arr = np.array(self.Cd_p_history[-avg_window:])
        Cdv_arr = np.array(self.Cd_v_history[-avg_window:])

        if hdf5_writer:
            hdf5_writer.close()
        return {
            "Cd_mean":    float(np.mean(Cd_arr)),
            "Cl_mean":    float(np.mean(Cl_arr)),
            "Cd_std":     float(np.std(Cd_arr)),
            "Cl_std":     float(np.std(Cl_arr)),
            "Cd_p_mean":  float(np.mean(Cdp_arr)),
            "Cd_v_mean":  float(np.mean(Cdv_arr)),
            "Cd_history": self.Cd_history,
            "Cl_history": self.Cl_history,
            "Cd_p_history": self.Cd_p_history,
            "Cd_v_history": self.Cd_v_history,
            "steps_completed": self.step_count,
            "stop_reason": stop_reason,
            "convergence_report": last_convergence,
            "strouhal_report": last_strouhal,
            "rho": rho,
            "ux":  ux,
            "uy":  uy,
        }

    # ------------------------------------------------------------------
    # Checkpoint: save / load solver state
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save full solver state to a .npz file, including all init parameters.

        The companion `from_checkpoint(path)` classmethod can reconstruct the
        solver from scratch without requiring the caller to know any original
        parameters.
        """
        import json
        params = dict(
            Ny=self.Ny, Nx=self.Nx, omega=self.omega, u0=self.u0, D=self.D,
            rho0=self.rho0, wall_bc=self.wall_bc, inlet_bc=self.inlet_bc,
            outlet_bc=self.outlet_bc, rho_in=self.rho_in, rho_out=self.rho_out,
            inlet_perturbation=self.inlet_perturbation, trt_lambda=self.trt_lambda,
            sponge_thickness=self.sponge_thickness, sponge_strength=self.sponge_strength,
            les=self.les, les_cs=self.les_cs, ibm_enabled=self.ibm_enabled,
            bouzidi=self.bouzidi, van_driest=self.van_driest, van_driest_A=self.van_driest_A,
            collision=self.collision, backend=self.backend,
        )
        kwargs: dict = dict(
            f=self.f,
            f_outlet_prev=self._f_outlet_prev,
            solid=self.solid,
            Cd_history=np.array(self.Cd_history, dtype=np.float64),
            Cl_history=np.array(self.Cl_history, dtype=np.float64),
            step_count=np.array(self.step_count, dtype=np.int64),
            surface_links=self.surface_links,
            q_vals=self.q_vals,
            params_json=np.array(json.dumps(params)),
        )
        if self.phi is not None:
            kwargs["phi"] = self.phi
        np.savez_compressed(path, **kwargs)

    def load_checkpoint(self, path: str) -> None:
        """
        Restore dynamic solver state (f, histories, step) from a .npz checkpoint.

        Call this on an already-constructed Solver when you have the geometry
        parameters at hand.  For a full stateless restart use `from_checkpoint`.
        """
        data = np.load(path, allow_pickle=False)
        self.f                = data["f"]
        self._f_outlet_prev   = data["f_outlet_prev"]
        self.Cd_history       = list(data["Cd_history"])
        self.Cl_history       = list(data["Cl_history"])
        self.step_count       = int(data["step_count"])
        # Restore pre-computed link tables so geometry doesn't need rebuilding
        if "surface_links" in data:
            self.surface_links = data["surface_links"]
        if "q_vals" in data:
            self.q_vals = data["q_vals"]

    @classmethod
    def from_checkpoint(cls, path: str) -> "Solver":
        """
        Reconstruct a Solver from a checkpoint without any external parameters.

        The checkpoint must have been saved with the current `save_checkpoint`
        (which embeds all init params as JSON).  Returns a fully-initialised
        Solver ready to continue running.
        """
        import json
        data = np.load(path, allow_pickle=False)
        params = json.loads(str(data["params_json"]))
        solid = data["solid"]
        phi   = data["phi"] if "phi" in data else None
        solver = cls(solid=solid, phi=phi, **params)
        solver.load_checkpoint(path)
        return solver
