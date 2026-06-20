"""
D3Q19 3D LBM wind tunnel solver.

Timestep loop:
  1. Fused macroscopic + BGK collision → f_post
  2. f_pre = f_post.copy()   (pre-streaming snapshot for bounce-back)
  3. Push streaming into f_new
  4. BCs: bounce-back → inlet → outlet → walls
  5. Forces: momentum exchange → Cd, Cl_y, Cl_z

Domain: (Q, Nz, Ny, Nx)
  x = streamwise, y = vertical (walls), z = spanwise (periodic default)

BC options
----------
inlet_bc  : "velocity"      — Zou-He 3D velocity (ux=u0, uy=uz=0)
outlet_bc : "convective"    — advective non-reflecting outflow (default)
            "zerogradient"  — copy from x=Nx-2
wall_bc   : "slip"          — specular reflection in y (default)
            "noslip"        — full bounce-back in y

Backend
-------
backend : "auto" | "numpy" | "numba"
"""

import numpy as np
import pathlib
import time
from typing import Callable, Optional, List

from .d3q19 import Q3, E3, W3, compute_macroscopic_3d, compute_feq_3d
from . import kernels3d as _k3
from .kernels3d import collision_kernel_3d_xp, stream_kernel_3d_xp
from .kernels3d_mrt import MRTKernel3D
from .boundary3d import (
    apply_inlet_zou_he_3d,
    apply_inlet_sem_3d,
    apply_inlet_velocity_field_3d,
    apply_outlet_zero_gradient_3d,
    apply_outlet_convective_3d,
    apply_streamwise_periodic_3d,
    apply_recycling_rescaling_inlet_3d,
    apply_slip_walls_3d,
    apply_noslip_walls_3d,
    apply_moving_walls_3d,
    build_surface_links_3d,
    apply_bounce_back_3d,
    apply_bounce_back_bouzidi_3d,
)
from ..forces3d import (
    compute_forces_3d,
    compute_force_split_3d,
    compute_force_moment_3d,
    forces_to_coefficients_3d,
    moments_to_coefficients_3d,
    spanwise_force_profile_3d,
    split_to_coefficients_3d,
)
from ..diagnostics import check_stability, detect_statistical_stationarity
from ..observables import coefficient_spectrum
from .sponge import build_sponge_sigma, apply_sponge_relaxation_3d
from .les import strain_rate_magnitude_3d, smagorinsky_nu_sgs, build_omega_field_3d
from .physics import base_nu_from_omega
from .trt2d import trt_taus
from . import kernels3d_trt as _ktrt3
from .ibm_guo import apply_guo_forcing_3d, apply_uniform_body_force_3d


class Solver3D:
    def __init__(
        self,
        Nz: int,
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
        streamwise_bc: str = "open",
        backend: str = "auto",
        collision: str = "bgk",
        inlet_perturbation: float = 0.0,
        trt_lambda: float = 0.25,
        sponge_thickness: int = 0,
        sponge_strength: float = 0.1,
        les: bool = False,
        les_cs: float = 0.16,
        les_model: str = "smagorinsky",
        ibm_enabled: bool = False,
        phi: Optional[np.ndarray] = None,
        bouzidi: bool = False,
        van_driest: bool = False,
        van_driest_A: float = 25.0,
        body_force_x: float = 0.0,
        body_force_y: float = 0.0,
        body_force_z: float = 0.0,
        wall_velocity_top: float = 0.0,
        wall_velocity_bottom: float = 0.0,
        synthetic_inflow: bool = False,
        synthetic_inflow_intensity: float = 0.03,
        synthetic_inflow_seed: int = 12345,
        sem_inlet: bool = False,
        sem_Tu: float = 0.05,
        sem_L_int: float = 10.0,
        sem_N: int = 200,
        thermal: bool = False,
        T_hot: float = 1.0,
        T_cold: float = 0.0,
        alpha_T: float = 1e-3,
        buoyancy: bool = False,
        g_gravity: float = 0.0,
        beta: float = 1e-3,
        T_ref: float = 0.5,
    ):
        """
        Parameters
        ----------
        Nz, Ny, Nx : int         — grid dimensions (span, height, streamwise)
        solid       : bool ndarray (Nz, Ny, Nx)
        omega       : float       — BGK relaxation rate (1/tau)
        u0          : float       — inlet velocity (lattice units)
        D           : float       — reference length (lattice cells)
        rho0        : float       — reference density
        wall_bc     : "slip" | "noslip"
        inlet_bc    : "velocity"  (only option in Phase 4)
        outlet_bc   : "convective" | "zerogradient"
        backend     : "auto" | "numpy" | "numba"
        collision   : "bgk" | "mrt" | "trt"
        inlet_perturbation : spanwise uz fraction of u0 at inlet
        """
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx
        self.solid = np.ascontiguousarray(solid, dtype=np.bool_)
        self.fluid = ~self.solid
        self.omega = omega
        self.u0 = u0
        self.D = D
        self.rho0 = rho0
        self.wall_bc = wall_bc
        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        if streamwise_bc not in ("open", "periodic", "recycling"):
            raise ValueError(
                f"Unknown streamwise_bc '{streamwise_bc}'. Choose 'open', 'periodic', or 'recycling'."
            )
        self.streamwise_bc = streamwise_bc
        self.inlet_perturbation = max(float(inlet_perturbation), 0.0)
        self.trt_lambda = float(trt_lambda)
        self.sponge_thickness = int(sponge_thickness)
        self.sponge_strength = float(sponge_strength)
        self.les = bool(les)
        self.les_cs = float(les_cs)
        if les_model not in ("smagorinsky", "wale"):
            raise ValueError("les_model must be 'smagorinsky' or 'wale'.")
        self.les_model = str(les_model)
        self.ibm_enabled = bool(ibm_enabled)
        self.phi = phi
        self.bouzidi = bool(bouzidi)
        self.van_driest = bool(van_driest)
        self.van_driest_A = float(van_driest_A)
        self.body_force_x = float(body_force_x)
        self.body_force_y = float(body_force_y)
        self.body_force_z = float(body_force_z)
        self.wall_velocity_top = float(wall_velocity_top)
        self.wall_velocity_bottom = float(wall_velocity_bottom)
        self.synthetic_inflow = bool(synthetic_inflow)
        self.synthetic_inflow_intensity = float(synthetic_inflow_intensity)
        self._rng = np.random.default_rng(int(synthetic_inflow_seed))
        self.sem_inlet = bool(sem_inlet)
        self._sem = None
        if self.sem_inlet:
            from .sem_inlet import SEMInlet
            self._sem = SEMInlet(Ny, Nz, u0, float(sem_Tu), float(sem_L_int), int(sem_N))
        self.thermal = bool(thermal)
        self.T_hot = float(T_hot)
        self.T_cold = float(T_cold)
        self.alpha_T = float(alpha_T)
        self.omega_T = float(1.0 / (3.0 * alpha_T + 0.5))
        self.buoyancy = bool(buoyancy)
        self.g_gravity = float(g_gravity)
        self.beta = float(beta)
        self.T_ref = float(T_ref)
        self.tau = 1.0 / omega
        self._base_nu = base_nu_from_omega(omega)
        if collision not in ("bgk", "mrt", "trt"):
            raise ValueError(f"Unknown collision '{collision}'. Choose 'bgk', 'mrt', or 'trt'.")
        self.collision = collision
        _, self._s_minus_trt = trt_taus(omega, self.trt_lambda)
        self._s_minus_trt = 1.0 / self._s_minus_trt

        # Backend
        self._xp = np
        self._use_cupy = False
        if backend == "auto":
            self._use_numba = _k3.HAS_NUMBA
        elif backend == "numba":
            if not _k3.HAS_NUMBA:
                raise ImportError("backend='numba' requested but Numba is not installed.")
            self._use_numba = True
        elif backend == "numpy":
            self._use_numba = False
        elif backend == "cupy":
            try:
                import cupy as cp  # type: ignore
                self._xp = cp
                self._use_cupy = True
                self._use_numba = False
            except ImportError:
                raise ImportError(
                    "backend='cupy' requested but cupy is not installed. "
                    "Install it with: pip install cupy-cudaXX"
                )
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'auto', 'numpy', 'numba', or 'cupy'.")
        self.backend: str = "cupy" if self._use_cupy else ("numba" if self._use_numba else "numpy")

        # Pre-cast lattice arrays
        self._ex = E3[:, 0].astype(np.int32)
        self._ey = E3[:, 1].astype(np.int32)
        self._ez = E3[:, 2].astype(np.int32)
        self._w  = W3.copy()
        # Keep numpy copies for thermal (needed even when CuPy overwrites above)
        self._ex_np = E3[:, 0].astype(np.int32)
        self._ey_np = E3[:, 1].astype(np.int32)
        self._ez_np = E3[:, 2].astype(np.int32)
        self._w_np  = W3.copy()
        self._mrt_kernel = None

        self.solid = np.ascontiguousarray(solid, dtype=np.bool_)
        _phi_for_q = phi if (bouzidi and phi is not None) else None
        self.surface_links, self.q_vals = build_surface_links_3d(self.solid, _phi_for_q)

        # Initialize f to equilibrium
        rho_init = np.full((Nz, Ny, Nx), rho0)
        ux_init  = np.full((Nz, Ny, Nx), u0)
        uy_init  = np.zeros((Nz, Ny, Nx))
        uz_init  = np.zeros((Nz, Ny, Nx))
        self.f   = np.ascontiguousarray(
            compute_feq_3d(rho_init, ux_init, uy_init, uz_init), dtype=np.float64
        )
        self._f_outlet_prev = self.f[:, :, :, -1].copy()
        if self.collision == "mrt":
            self._mrt_kernel = MRTKernel3D(omega, use_numba=self._use_numba)

        self._sponge_sigma = build_sponge_sigma(self.Nx, self.sponge_thickness, self.sponge_strength)
        self._u_wall_x = np.zeros((Nz, Ny, Nx))
        self._u_wall_y = np.zeros((Nz, Ny, Nx))
        self._u_wall_z = np.zeros((Nz, Ny, Nx))
        self._omega_dummy = np.zeros((Nz, Ny, Nx), dtype=np.float64)
        self._inlet_uy_field = np.zeros((Nz, Ny), dtype=np.float64)
        self._inlet_uz_field = np.zeros((Nz, Ny), dtype=np.float64)

        # Transfer domain arrays to GPU when using CuPy
        if self._use_cupy:
            xp = self._xp
            self.f = xp.asarray(self.f)
            self._f_outlet_prev = xp.asarray(self._f_outlet_prev)
            self.solid = xp.asarray(self.solid)
            self._ex = xp.asarray(self._ex)
            self._ey = xp.asarray(self._ey)
            self._ez = xp.asarray(self._ez)
            self._w  = xp.asarray(self._w)
            self._omega_dummy = xp.asarray(self._omega_dummy)
            self._u_wall_x = xp.asarray(self._u_wall_x)
            self._u_wall_y = xp.asarray(self._u_wall_y)
            self._u_wall_z = xp.asarray(self._u_wall_z)
            self.surface_links = xp.asarray(self.surface_links)
            self.q_vals = xp.asarray(self.q_vals)

        self.step_count: int = 0

        # Thermal distribution
        self.g: Optional[np.ndarray] = None
        if self.thermal:
            from .thermal import init_g_3d
            self.g = init_g_3d(self.T_ref, Nz, Ny, Nx, self._w_np, self._ex_np, self._ey_np, self._ez_np)

        self.Cd_history: List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []
        self.Cmx_history: List[float] = []
        self.Cmy_history: List[float] = []
        self.Cmz_history: List[float] = []
        self.Cd_p_history: List[float] = []
        self.Cd_v_history: List[float] = []
        solid_z, solid_y, solid_x = np.where(self.solid)
        self._ref_center_z = float(np.mean(solid_z)) if solid_z.size else 0.5 * (Nz - 1)
        self._ref_center_y = float(np.mean(solid_y)) if solid_y.size else 0.5 * (Ny - 1)
        self._ref_center_x = float(np.mean(solid_x)) if solid_x.size else 0.5 * (Nx - 1)

    def _update_synthetic_inflow(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.synthetic_inflow:
            shape = (self.Nz, self.Ny)
            return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)
        noise_y = self._rng.standard_normal((self.Nz, self.Ny))
        noise_z = self._rng.standard_normal((self.Nz, self.Ny))
        filt_y = (
            noise_y + np.roll(noise_y, 1, axis=0) + np.roll(noise_y, -1, axis=0)
            + np.roll(noise_y, 1, axis=1) + np.roll(noise_y, -1, axis=1)
        ) / 5.0
        filt_z = (
            noise_z + np.roll(noise_z, 1, axis=0) + np.roll(noise_z, -1, axis=0)
            + np.roll(noise_z, 1, axis=1) + np.roll(noise_z, -1, axis=1)
        ) / 5.0
        self._inlet_uy_field = 0.8 * self._inlet_uy_field + 0.2 * filt_y
        self._inlet_uz_field = 0.8 * self._inlet_uz_field + 0.2 * filt_z
        self._inlet_uy_field -= float(np.mean(self._inlet_uy_field))
        self._inlet_uz_field -= float(np.mean(self._inlet_uz_field))
        target = self.synthetic_inflow_intensity * self.u0
        rms = float(np.sqrt(np.mean(self._inlet_uy_field ** 2 + self._inlet_uz_field ** 2)))
        if rms > 1e-12 and target > 0.0:
            scale = target / rms
            self._inlet_uy_field *= scale
            self._inlet_uz_field *= scale
        return self._inlet_uy_field.copy(), self._inlet_uz_field.copy()

    # ------------------------------------------------------------------

    def _step(self) -> tuple:
        f = self.f

        if self.ibm_enabled and self.phi is not None:
            apply_guo_forcing_3d(
                f, self.phi, self.tau,
                self._u_wall_x, self._u_wall_y, self._u_wall_z,
                self._ex, self._ey, self._ez, self._w, self.solid,
            )

        omega_field = self._omega_dummy
        use_omega_field = False
        if self.les:
            omega_field = build_omega_field_3d(
                f, self.solid, self.fluid, self._base_nu, self.omega, self.les_cs,
                les_model=self.les_model,
                phi=self.phi, van_driest=self.van_driest, van_driest_A=self.van_driest_A,
            )
            use_omega_field = True
        omega_use = self.omega

        if self.collision == "mrt":
            f_post = np.empty_like(f)
            self._mrt_kernel.collide(
                f, f_post, self.solid, self._ex, self._ey, self._ez, self._w
            )
            f_pre = f_post.copy()
            f_new = np.empty_like(f_post)
            _k3.stream_kernel_3d(f_post, f_new, self._ex, self._ey, self._ez)
            f_post = f_new
        elif self.collision == "trt":
            f_post = np.empty_like(f)
            if self._use_numba and _ktrt3._HAS_NUMBA:
                from .d3q19 import OPP3
                _ktrt3.trt_collision_kernel_3d(
                    f, f_post, self.solid, omega_use,
                    self._ex, self._ey, self._ez, self._w,
                    OPP3.astype(np.int32), self.trt_lambda,
                    omega_field, use_omega_field,
                )
            else:
                _ktrt3.trt_collision_numpy_3d(
                    f, f_post, self.solid, omega_use,
                    self._ex, self._ey, self._ez, self._w, self.trt_lambda,
                    omega_field if use_omega_field else None,
                )
            f_pre = f_post.copy()
            f_new = np.empty_like(f_post)
            _k3.stream_kernel_3d(f_post, f_new, self._ex, self._ey, self._ez)
            f_post = f_new
        elif self._use_cupy:
            xp = self._xp
            f_post = xp.empty_like(f)
            collision_kernel_3d_xp(
                f, f_post, self.solid, omega_use,
                self._ex, self._ey, self._ez, self._w,
                omega_field, use_omega_field, xp=xp,
            )
            f_pre = f_post.copy()
            f_new = xp.empty_like(f_post)
            stream_kernel_3d_xp(f_post, f_new, self._ex, self._ey, self._ez, xp=xp)
            f_post = f_new
        elif self._use_numba:
            f_post = np.empty_like(f)
            _k3.collision_kernel_3d(
                f, f_post, self.solid, omega_use,
                self._ex, self._ey, self._ez, self._w,
                omega_field, use_omega_field,
            )
            f_pre = f_post.copy()
            f_new = np.empty_like(f_post)
            _k3.stream_kernel_3d(f_post, f_new, self._ex, self._ey, self._ez)
            f_post = f_new
        else:
            rho, ux, uy, uz = compute_macroscopic_3d(f)
            ux[self.solid] = 0.0
            uy[self.solid] = 0.0
            uz[self.solid] = 0.0
            feq    = compute_feq_3d(rho, ux, uy, uz)
            om = omega_field if use_omega_field else omega_use
            f_post = (1.0 - om) * f + om * feq
            f_pre  = f_post.copy()
            f_new  = np.empty_like(f_post)
            _k3.stream_kernel_3d(f_post, f_new, self._ex, self._ey, self._ez)
            f_post = f_new

        if any(abs(val) > 0.0 for val in (self.body_force_x, self.body_force_y, self.body_force_z)):
            apply_uniform_body_force_3d(
                f_post,
                self.tau,
                self.body_force_x,
                self.body_force_y,
                self.body_force_z,
                self._ex,
                self._ey,
                self._ez,
                self._w,
                self.solid,
            )

        # BCs
        if not self.ibm_enabled:
            if self.bouzidi:
                apply_bounce_back_bouzidi_3d(f_post, f_pre, self.surface_links, self.q_vals)
            else:
                apply_bounce_back_3d(f_post, f_pre, self.surface_links)

        if self.streamwise_bc == "open":
            if self.inlet_bc == "velocity":
                if self._sem is not None:
                    self._sem.step()
                    du, dv, dw = self._sem.fluctuation()
                    apply_inlet_sem_3d(f_post, self.u0, du, dv, dw)
                elif self.synthetic_inflow:
                    uy_in, uz_in = self._update_synthetic_inflow()
                    apply_inlet_velocity_field_3d(
                        f_post,
                        np.full((self.Nz, self.Ny), self.u0, dtype=np.float64),
                        uy_in,
                        uz_in,
                    )
                else:
                    apply_inlet_zou_he_3d(
                        f_post,
                        self.u0,
                        uz_amp=self.inlet_perturbation,
                        step=self.step_count + 1,
                    )

            if self.outlet_bc == "convective":
                apply_outlet_convective_3d(f_post, self._f_outlet_prev, self.u0)
            else:
                apply_outlet_zero_gradient_3d(f_post)
        elif self.streamwise_bc == "periodic":
            apply_streamwise_periodic_3d(f_post)
        elif self.streamwise_bc == "recycling":
            apply_recycling_rescaling_inlet_3d(f_post, self.u0)

        if self.sponge_thickness > 0:
            apply_sponge_relaxation_3d(
                f_post, self._sponge_sigma, self.u0,
                self._ex, self._ey, self._ez, self._w,
            )

        if self.wall_bc == "slip":
            apply_slip_walls_3d(f_post)
        elif self.wall_bc == "noslip":
            apply_noslip_walls_3d(f_post)
        elif self.wall_bc == "moving":
            apply_moving_walls_3d(
                f_post,
                u_top=self.wall_velocity_top,
                u_bottom=self.wall_velocity_bottom,
            )

        # Thermal step (CPU only; g stays on numpy even with CuPy f)
        if self.thermal and self.g is not None:
            from .thermal import (
                collide_g_3d, stream_g_3d, extract_T,
                apply_temperature_bc_3d, guo_buoyancy_force_3d,
            )
            f_cpu = f_post.get() if self._use_cupy else f_post
            rho_now, ux_now, uy_now, uz_now = compute_macroscopic_3d(f_cpu)
            T_now = extract_T(self.g)
            if self.buoyancy:
                from .d3q19 import E3
                F_b = guo_buoyancy_force_3d(
                    rho_now, T_now, self.T_ref,
                    self.g_gravity, self.beta, E3[:, 1], self._w_np,
                )
                if self._use_cupy:
                    f_post = f_post + self._xp.asarray(F_b)
                else:
                    f_post = f_post + F_b
            solid_cpu = self.solid.get() if self._use_cupy else self.solid
            g_post = collide_g_3d(self.g, ux_now, uy_now, uz_now, T_now, self.omega_T, solid_cpu)
            g_post = stream_g_3d(g_post, self._ex_np, self._ey_np, self._ez_np)
            apply_temperature_bc_3d(g_post, self.T_hot, self.T_cold, self._w_np, self._ex_np, self._ey_np, self._ez_np)
            self.g = g_post

        self.f = f_post
        self.step_count += 1

        # For force computation, bring GPU arrays to CPU if needed
        if self._use_cupy:
            f_pre_cpu  = f_pre.get()
            f_post_cpu = f_post.get()
            links_cpu  = self.surface_links.get()
        else:
            f_pre_cpu  = f_pre
            f_post_cpu = f_post
            links_cpu  = self.surface_links

        Fx, Fy, Fz = compute_forces_3d(f_pre_cpu, f_post_cpu, links_cpu)
        Cd, Cly, Clz = forces_to_coefficients_3d(Fx, Fy, Fz, self.rho0, self.u0, self.D)
        fx_p, fy_p, fz_p, fx_v, fy_v, fz_v = compute_force_split_3d(f_pre_cpu, f_post_cpu, links_cpu)
        cdp, _, _, cdv, _, _ = split_to_coefficients_3d(
            fx_p, fy_p, fz_p, fx_v, fy_v, fz_v, self.rho0, self.u0, self.D,
        )
        self._last_cd_p = cdp
        self._last_cd_v = cdv
        _, _, _, mx, my, mz = compute_force_moment_3d(
            f_pre_cpu,
            f_post_cpu,
            links_cpu,
            center_x=self._ref_center_x,
            center_y=self._ref_center_y,
            center_z=self._ref_center_z,
        )
        cmx, cmy, cmz = moments_to_coefficients_3d(mx, my, mz, self.rho0, self.u0, self.D)
        self._last_cmx = cmx
        self._last_cmy = cmy
        self._last_cmz = cmz
        self._last_spanwise_profile = spanwise_force_profile_3d(
            f_pre_cpu, f_post_cpu, links_cpu, nz=self.Nz
        )
        return Cd, Cly, Clz

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
        Run the 3D simulation for `steps` timesteps.

        Returns
        -------
        dict  keys: Cd_mean, Cly_mean, Clz_mean, Cd_std, Cly_std, Clz_std,
                    Cd_history, Cly_history, Clz_history, rho, ux, uy, uz
        """
        if verbose:
            print(f"  Backend   : {self.backend}  (3D D3Q19)")
            print(f"  Collision : {self.collision}")

        from ..hdf5_writer import HDF5Writer
        hdf5_writer = (
            HDF5Writer(hdf5_path, Nz=self.Nz, Ny=self.Ny, Nx=self.Nx, dims=3)
            if hdf5_path else None
        )

        t_start    = time.perf_counter()
        avg_window = max(1, steps // 5)
        stop_reason = "max_steps"
        last_convergence = None
        last_strouhal = None

        for step in range(1, steps + 1):
            Cd, Cly, Clz = self._step()
            self.Cd_history.append(Cd)
            self.Cly_history.append(Cly)
            self.Clz_history.append(Clz)
            self.Cmx_history.append(getattr(self, "_last_cmx", 0.0))
            self.Cmy_history.append(getattr(self, "_last_cmy", 0.0))
            self.Cmz_history.append(getattr(self, "_last_cmz", 0.0))
            self.Cd_p_history.append(getattr(self, "_last_cd_p", 0.0))
            self.Cd_v_history.append(getattr(self, "_last_cd_v", 0.0))

            if checkpoint_every and checkpoint_dir and step % checkpoint_every == 0:
                ckpt = pathlib.Path(checkpoint_dir) / f"checkpoint3d_{self.step_count:08d}.npz"
                self.save_checkpoint(str(ckpt))

            if hdf5_writer and hdf5_every and step % hdf5_every == 0:
                from ..lbm.d3q19 import compute_macroscopic_3d as _cm3
                from .thermal import extract_T as _extract_T
                _f_cpu = self.f.get() if self._use_cupy else self.f
                _rho, _ux, _uy, _uz = _cm3(_f_cpu)
                _scalar = _extract_T(self.g) if (self.thermal and self.g is not None) else None
                hdf5_writer.write_step(self.step_count, ux=_ux, uy=_uy, uz=_uz, rho=_rho, scalar=_scalar)

            if step % check_every == 0:
                # Use scalar stability check on the midplane slice
                f_mid = self.f[:, self.Nz // 2, :, :]
                rho_s = f_mid.sum(axis=0)
                inv_r = np.where(rho_s > 0, 1.0 / rho_s, 0.0)
                ux_s  = inv_r * np.einsum('i,iyx->yx', self._ex.astype(np.float64), f_mid)
                uy_s  = inv_r * np.einsum('i,iyx->yx', self._ey.astype(np.float64), f_mid)
                check_stability(f_mid, rho_s, ux_s, uy_s, step)

                elapsed       = time.perf_counter() - t_start
                sps           = step / elapsed
                w             = min(check_every, len(self.Cd_history))
                Cd_avg        = float(np.mean(self.Cd_history[-w:]))
                Cly_avg       = float(np.mean(self.Cly_history[-w:]))

                if verbose:
                    print(
                        f"  step {step:6d}/{steps}  "
                        f"Cd={Cd_avg:+.4f}  Cl_y={Cly_avg:+.4f}  "
                        f"[{sps:.0f} steps/s]",
                        flush=True,
                    )

                if callback is not None:
                    callback(step, Cd, Cly, Clz)

                if auto_stop:
                    lift_history = self.Cly_history
                    if np.std(self.Clz_history[-min(len(self.Clz_history), strouhal_window):]) > np.std(
                        self.Cly_history[-min(len(self.Cly_history), strouhal_window):]
                    ):
                        lift_history = self.Clz_history
                    last_convergence, last_strouhal, should_stop = detect_statistical_stationarity(
                        self.Cd_history,
                        lift_history,
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

        rho, ux, uy, uz = compute_macroscopic_3d(self.f)

        Cd_arr  = np.array(self.Cd_history[-avg_window:])
        Cly_arr = np.array(self.Cly_history[-avg_window:])
        Clz_arr = np.array(self.Clz_history[-avg_window:])
        Cmx_arr = np.array(self.Cmx_history[-avg_window:]) if self.Cmx_history else np.zeros(1)
        Cmy_arr = np.array(self.Cmy_history[-avg_window:]) if self.Cmy_history else np.zeros(1)
        Cmz_arr = np.array(self.Cmz_history[-avg_window:]) if self.Cmz_history else np.zeros(1)
        Cdp_arr = np.array(self.Cd_p_history[-avg_window:])
        Cdv_arr = np.array(self.Cd_v_history[-avg_window:])
        scalar = None
        scalar_stats = None
        if self.thermal and self.g is not None:
            from .thermal import extract_T
            scalar = extract_T(self.g)
            scalar_stats = {
                "enabled": True,
                "mean": float(np.mean(scalar)),
                "std": float(np.std(scalar)),
                "min": float(np.min(scalar)),
                "max": float(np.max(scalar)),
                "wall_hot": float(self.T_hot),
                "wall_cold": float(self.T_cold),
                "diffusivity": float(self.alpha_T),
                "buoyancy": bool(self.buoyancy),
                "gravity": float(self.g_gravity),
                "beta": float(self.beta),
                "reference": float(self.T_ref),
            }

        if hdf5_writer:
            hdf5_writer.close()
        return {
            "Cd_mean":    float(np.mean(Cd_arr)),
            "Cly_mean":   float(np.mean(Cly_arr)),
            "Clz_mean":   float(np.mean(Clz_arr)),
            "Cd_std":     float(np.std(Cd_arr)),
            "Cly_std":    float(np.std(Cly_arr)),
            "Clz_std":    float(np.std(Clz_arr)),
            "Cmx_mean":   float(np.mean(Cmx_arr)),
            "Cmy_mean":   float(np.mean(Cmy_arr)),
            "Cmz_mean":   float(np.mean(Cmz_arr)),
            "Cmx_std":    float(np.std(Cmx_arr)),
            "Cmy_std":    float(np.std(Cmy_arr)),
            "Cmz_std":    float(np.std(Cmz_arr)),
            "Cd_p_mean":  float(np.mean(Cdp_arr)),
            "Cd_v_mean":  float(np.mean(Cdv_arr)),
            "Cd_history":  self.Cd_history,
            "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history,
            "Cmx_history": self.Cmx_history,
            "Cmy_history": self.Cmy_history,
            "Cmz_history": self.Cmz_history,
            "Cd_p_history": self.Cd_p_history,
            "Cd_v_history": self.Cd_v_history,
            "observables": {
                "cd_spectrum": coefficient_spectrum(self.Cd_history),
                "cly_spectrum": coefficient_spectrum(self.Cly_history),
                "clz_spectrum": coefficient_spectrum(self.Clz_history),
                "cmz_spectrum": coefficient_spectrum(self.Cmz_history),
                "spanwise_force_profile_last": getattr(self, "_last_spanwise_profile", None),
            },
            "steps_completed": self.step_count,
            "stop_reason": stop_reason,
            "convergence_report": last_convergence,
            "strouhal_report": last_strouhal,
            "rho": rho, "ux": ux, "uy": uy, "uz": uz,
            "scalar": scalar,
            "scalar_stats": scalar_stats,
        }

    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save full solver state including all init parameters for stateless restart.

        Use `Solver3D.from_checkpoint(path)` to rebuild without any external
        geometry or parameter knowledge.
        """
        import json
        f_cpu  = self.f.get() if self._use_cupy else self.f
        fp_cpu = self._f_outlet_prev.get() if self._use_cupy else self._f_outlet_prev
        sl_cpu = self.surface_links.get() if self._use_cupy else self.surface_links
        qv_cpu = self.q_vals.get() if self._use_cupy else self.q_vals
        solid_cpu = self.solid.get() if self._use_cupy else self.solid
        params = dict(
            Nz=self.Nz, Ny=self.Ny, Nx=self.Nx, omega=self.omega, u0=self.u0, D=self.D,
            rho0=self.rho0, wall_bc=self.wall_bc, inlet_bc=self.inlet_bc,
            outlet_bc=self.outlet_bc, streamwise_bc=self.streamwise_bc,
            backend="numpy",  # always restart on CPU; user can switch after
            collision=self.collision, inlet_perturbation=self.inlet_perturbation,
            trt_lambda=self.trt_lambda, sponge_thickness=self.sponge_thickness,
            sponge_strength=self.sponge_strength, les=self.les, les_cs=self.les_cs,
            les_model=self.les_model,
            ibm_enabled=self.ibm_enabled, bouzidi=self.bouzidi,
            van_driest=self.van_driest, van_driest_A=self.van_driest_A,
            body_force_x=self.body_force_x, body_force_y=self.body_force_y,
            body_force_z=self.body_force_z,
            wall_velocity_top=self.wall_velocity_top,
            wall_velocity_bottom=self.wall_velocity_bottom,
            synthetic_inflow=self.synthetic_inflow,
            synthetic_inflow_intensity=self.synthetic_inflow_intensity,
            thermal=self.thermal,
            T_hot=self.T_hot,
            T_cold=self.T_cold,
            alpha_T=self.alpha_T,
            buoyancy=self.buoyancy,
            g_gravity=self.g_gravity,
            beta=self.beta,
            T_ref=self.T_ref,
        )
        kwargs: dict = dict(
            f=f_cpu, f_outlet_prev=fp_cpu,
            solid=solid_cpu, surface_links=sl_cpu, q_vals=qv_cpu,
            Cd_history=np.array(self.Cd_history,  dtype=np.float64),
            Cly_history=np.array(self.Cly_history, dtype=np.float64),
            Clz_history=np.array(self.Clz_history, dtype=np.float64),
            Cmx_history=np.array(self.Cmx_history, dtype=np.float64),
            Cmy_history=np.array(self.Cmy_history, dtype=np.float64),
            Cmz_history=np.array(self.Cmz_history, dtype=np.float64),
            step_count=np.array(self.step_count, dtype=np.int64),
            params_json=np.array(json.dumps(params)),
        )
        if self.phi is not None:
            phi_cpu = self.phi.get() if self._use_cupy else self.phi
            kwargs["phi"] = phi_cpu
        if self.g is not None:
            kwargs["g"] = self.g
        np.savez_compressed(path, **kwargs)

    def load_checkpoint(self, path: str) -> None:
        """
        Restore dynamic state (f, histories, step, link tables) from a checkpoint.

        Restores pre-computed surface links and q_vals so geometry doesn't need
        rebuilding.  For stateless restart use `from_checkpoint` instead.
        """
        data = np.load(path, allow_pickle=False)
        f_np  = data["f"]
        fp_np = data["f_outlet_prev"]
        self.f              = self._xp.asarray(f_np)  if self._use_cupy else f_np
        self._f_outlet_prev = self._xp.asarray(fp_np) if self._use_cupy else fp_np
        self.Cd_history  = list(data["Cd_history"])
        self.Cly_history = list(data["Cly_history"])
        self.Clz_history = list(data["Clz_history"])
        if "Cmx_history" in data:
            self.Cmx_history = list(data["Cmx_history"])
        if "Cmy_history" in data:
            self.Cmy_history = list(data["Cmy_history"])
        if "Cmz_history" in data:
            self.Cmz_history = list(data["Cmz_history"])
        self.step_count  = int(data["step_count"])
        if "g" in data:
            self.g = data["g"]
        if "surface_links" in data:
            sl = data["surface_links"]
            self.surface_links = self._xp.asarray(sl) if self._use_cupy else sl
        if "q_vals" in data:
            qv = data["q_vals"]
            self.q_vals = self._xp.asarray(qv) if self._use_cupy else qv

    @classmethod
    def from_checkpoint(cls, path: str, backend: Optional[str] = None) -> "Solver3D":
        """
        Reconstruct a Solver3D entirely from a checkpoint file.

        No geometry objects or parameter knowledge needed.  The checkpoint must
        have been written by the current `save_checkpoint` (embeds params as JSON).

        Parameters
        ----------
        path    : path to the .npz checkpoint
        backend : override the saved backend (e.g. "cupy" to restart on GPU)
        """
        import json
        data = np.load(path, allow_pickle=False)
        params = json.loads(str(data["params_json"]))
        if backend is not None:
            params["backend"] = backend
        solid = data["solid"]
        phi   = data["phi"] if "phi" in data else None
        solver = cls(solid=solid, phi=phi, **params)
        solver.load_checkpoint(path)
        return solver
