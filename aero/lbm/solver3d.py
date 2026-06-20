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
from .kernels3d_mrt import MRTKernel3D
from .boundary3d import (
    apply_inlet_zou_he_3d,
    apply_outlet_zero_gradient_3d,
    apply_outlet_convective_3d,
    apply_slip_walls_3d,
    apply_noslip_walls_3d,
    build_surface_links_3d,
    apply_bounce_back_3d,
)
from ..forces3d import (
    compute_forces_3d,
    compute_force_split_3d,
    forces_to_coefficients_3d,
    split_to_coefficients_3d,
)
from ..diagnostics import check_stability  # reuse scalar stability check
from .sponge import build_sponge_sigma, apply_sponge_relaxation_3d
from .les import strain_rate_magnitude_3d, smagorinsky_nu_sgs, build_omega_field_3d
from .physics import base_nu_from_omega
from .trt2d import trt_taus
from . import kernels3d_trt as _ktrt3
from .ibm_guo import apply_guo_forcing_3d


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
        self.inlet_perturbation = max(float(inlet_perturbation), 0.0)
        self.trt_lambda = float(trt_lambda)
        self.sponge_thickness = int(sponge_thickness)
        self.sponge_strength = float(sponge_strength)
        self.les = bool(les)
        self.les_cs = float(les_cs)
        self.ibm_enabled = bool(ibm_enabled)
        self.phi = phi
        self.tau = 1.0 / omega
        self._base_nu = base_nu_from_omega(omega)
        if collision not in ("bgk", "mrt", "trt"):
            raise ValueError(f"Unknown collision '{collision}'. Choose 'bgk', 'mrt', or 'trt'.")
        self.collision = collision
        _, self._s_minus_trt = trt_taus(omega, self.trt_lambda)
        self._s_minus_trt = 1.0 / self._s_minus_trt

        # Backend
        if backend == "auto":
            self._use_numba = _k3.HAS_NUMBA
        elif backend == "numba":
            if not _k3.HAS_NUMBA:
                raise ImportError("backend='numba' requested but Numba is not installed.")
            self._use_numba = True
        elif backend == "numpy":
            self._use_numba = False
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'auto', 'numpy', or 'numba'.")
        self.backend: str = "numba" if self._use_numba else "numpy"

        # Pre-cast lattice arrays
        self._ex = E3[:, 0].astype(np.int32)
        self._ey = E3[:, 1].astype(np.int32)
        self._ez = E3[:, 2].astype(np.int32)
        self._w  = W3.copy()
        self._mrt_kernel = None

        self.solid = np.ascontiguousarray(solid, dtype=np.bool_)
        self.surface_links = build_surface_links_3d(self.solid)

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

        self.step_count: int = 0
        self.Cd_history: List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []
        self.Cd_p_history: List[float] = []
        self.Cd_v_history: List[float] = []

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

        # BCs
        if not self.ibm_enabled:
            apply_bounce_back_3d(f_post, f_pre, self.surface_links)

        if self.inlet_bc == "velocity":
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

        if self.sponge_thickness > 0:
            apply_sponge_relaxation_3d(
                f_post, self._sponge_sigma, self.u0,
                self._ex, self._ey, self._ez, self._w,
            )

        if self.wall_bc == "slip":
            apply_slip_walls_3d(f_post)
        elif self.wall_bc == "noslip":
            apply_noslip_walls_3d(f_post)

        self.f = f_post
        self.step_count += 1

        Fx, Fy, Fz = compute_forces_3d(f_pre, f_post, self.surface_links)
        Cd, Cly, Clz = forces_to_coefficients_3d(Fx, Fy, Fz, self.rho0, self.u0, self.D)
        fx_p, fy_p, fz_p, fx_v, fy_v, fz_v = compute_force_split_3d(f_pre, f_post, self.surface_links)
        cdp, _, _, cdv, _, _ = split_to_coefficients_3d(
            fx_p, fy_p, fz_p, fx_v, fy_v, fz_v, self.rho0, self.u0, self.D,
        )
        self._last_cd_p = cdp
        self._last_cd_v = cdv
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

        t_start    = time.perf_counter()
        avg_window = max(1, steps // 5)

        for step in range(1, steps + 1):
            Cd, Cly, Clz = self._step()
            self.Cd_history.append(Cd)
            self.Cly_history.append(Cly)
            self.Clz_history.append(Clz)
            self.Cd_p_history.append(getattr(self, "_last_cd_p", 0.0))
            self.Cd_v_history.append(getattr(self, "_last_cd_v", 0.0))

            if checkpoint_every and checkpoint_dir and step % checkpoint_every == 0:
                ckpt = pathlib.Path(checkpoint_dir) / f"checkpoint3d_{self.step_count:08d}.npz"
                self.save_checkpoint(str(ckpt))

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
                        f"[{sps:.0f} steps/s]"
                    )

                if callback is not None:
                    callback(step, Cd, Cly, Clz)

        rho, ux, uy, uz = compute_macroscopic_3d(self.f)

        Cd_arr  = np.array(self.Cd_history[-avg_window:])
        Cly_arr = np.array(self.Cly_history[-avg_window:])
        Clz_arr = np.array(self.Clz_history[-avg_window:])
        Cdp_arr = np.array(self.Cd_p_history[-avg_window:])
        Cdv_arr = np.array(self.Cd_v_history[-avg_window:])

        return {
            "Cd_mean":    float(np.mean(Cd_arr)),
            "Cly_mean":   float(np.mean(Cly_arr)),
            "Clz_mean":   float(np.mean(Clz_arr)),
            "Cd_std":     float(np.std(Cd_arr)),
            "Cly_std":    float(np.std(Cly_arr)),
            "Clz_std":    float(np.std(Clz_arr)),
            "Cd_p_mean":  float(np.mean(Cdp_arr)),
            "Cd_v_mean":  float(np.mean(Cdv_arr)),
            "Cd_history":  self.Cd_history,
            "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history,
            "Cd_p_history": self.Cd_p_history,
            "Cd_v_history": self.Cd_v_history,
            "rho": rho, "ux": ux, "uy": uy, "uz": uz,
        }

    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        np.savez_compressed(
            path,
            f=self.f,
            f_outlet_prev=self._f_outlet_prev,
            Cd_history=np.array(self.Cd_history,  dtype=np.float64),
            Cly_history=np.array(self.Cly_history, dtype=np.float64),
            Clz_history=np.array(self.Clz_history, dtype=np.float64),
            step_count=np.array(self.step_count, dtype=np.int64),
        )

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path)
        self.f               = data["f"]
        self._f_outlet_prev  = data["f_outlet_prev"]
        self.Cd_history      = list(data["Cd_history"])
        self.Cly_history     = list(data["Cly_history"])
        self.Clz_history     = list(data["Clz_history"])
        self.step_count      = int(data["step_count"])
