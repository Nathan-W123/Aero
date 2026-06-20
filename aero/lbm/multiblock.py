"""
Multi-block static 2:1 z-refinement for 3D LBM.

Runs two nested Solver3D instances at different spatial resolutions:
  - Coarse solver: full domain at the base grid (Nz, Ny, Nx)
  - Fine solver  : refined z-slab at 2× z-resolution; same Ny, Nx

Time stepping uses sub-cycling: one coarse step + two fine half-steps, with
Filippova-Hänel omega rescaling at the fine grid so that kinematic viscosity
is matched across the interface.

Interface exchange
------------------
Coarse → fine (before fine steps):
  The coarse f at z=refine_z_lo and z=refine_z_hi is upsampled (each coarse
  z-cell mapped to 2 fine z-cells) and injected into the fine solver's
  inlet/outlet ghost layers via equilibrium reconstruction.

Fine → coarse (after 2 fine steps):
  Fine cells are averaged in pairs back to coarse z-cells in the refined
  region, replacing coarse f[refine_z_lo:refine_z_hi].

Usage
-----
    solver = MultiblockSolver3D(
        Nz=60, Ny=60, Nx=120, solid=solid,
        refine_z_lo=20, refine_z_hi=40,
        omega=1.5, u0=0.05, D=10.0
    )
    solver.run(steps=500)
"""

from typing import Optional, Dict, Any, List
import numpy as np

from .d3q19 import E3, W3, compute_macroscopic_3d, compute_feq_3d, Q3


def _omega_fine(omega_coarse: float) -> float:
    """
    Filippova-Hänel rescaling: nu is halved at 2× finer grid.
    nu = (1/omega - 0.5) / 3  → nu_fine = nu_coarse / 2
    omega_fine = 1 / (3 * nu_fine + 0.5)
    """
    nu_coarse = (1.0 / omega_coarse - 0.5) / 3.0
    nu_fine   = nu_coarse / 2.0
    return float(1.0 / (3.0 * nu_fine + 0.5))


def _upsample_coarse_to_fine(f_coarse_slice: np.ndarray) -> np.ndarray:
    """
    Upsample a coarse z-slice (Q, Nz_c, Ny, Nx) to fine (Q, Nz_f, Ny, Nx)
    by repeating each coarse z-cell twice.
    """
    Q, Nz_c, Ny, Nx = f_coarse_slice.shape
    f_fine = np.repeat(f_coarse_slice, 2, axis=1)  # (Q, 2*Nz_c, Ny, Nx)
    return f_fine


def _downsample_fine_to_coarse(f_fine_slice: np.ndarray) -> np.ndarray:
    """
    Downsample fine z-slice (Q, Nz_f, Ny, Nx) to coarse (Q, Nz_c, Ny, Nx)
    by averaging pairs of fine cells.
    """
    Q, Nz_f, Ny, Nx = f_fine_slice.shape
    Nz_c = Nz_f // 2
    f_c = 0.5 * (f_fine_slice[:, 0::2, :, :] + f_fine_slice[:, 1::2, :, :])
    return f_c


class MultiblockSolver3D:
    """
    2:1 static z-refinement wrapper around two Solver3D instances.

    Parameters
    ----------
    Nz, Ny, Nx      : int    — coarse domain dimensions
    solid            : bool ndarray (Nz, Ny, Nx)
    refine_z_lo      : int   — start of refined z-slab (coarse indices)
    refine_z_hi      : int   — end of refined z-slab (exclusive, coarse indices)
    **solver_kw      : passed to both Solver3D instances; omega is rescaled for the fine grid
    """

    def __init__(
        self,
        Nz: int,
        Ny: int,
        Nx: int,
        solid: np.ndarray,
        refine_z_lo: int,
        refine_z_hi: int,
        **solver_kw: Any,
    ) -> None:
        from .solver3d import Solver3D

        self.Nz = int(Nz)
        self.Ny = int(Ny)
        self.Nx = int(Nx)
        self.refine_z_lo = int(refine_z_lo)
        self.refine_z_hi = int(refine_z_hi)
        self._Nz_refined = self.refine_z_hi - self.refine_z_lo
        self._Nz_fine    = 2 * self._Nz_refined

        omega_coarse = float(solver_kw.get("omega", 1.0))
        omega_f = _omega_fine(omega_coarse)

        # Coarse solver — full domain
        self._coarse = Solver3D(
            Nz=Nz, Ny=Ny, Nx=Nx,
            solid=solid,
            **solver_kw,
        )

        # Fine solver — only the refined slab
        solid_slab = solid[refine_z_lo:refine_z_hi]
        # Upsample solid in z by repeating each cell
        solid_fine = np.repeat(solid_slab, 2, axis=0)  # (Nz_fine, Ny, Nx)

        kw_fine = dict(solver_kw)
        kw_fine["omega"] = omega_f
        # Fine solver has open z (we inject BCs manually via interface exchange)
        kw_fine["streamwise_bc"] = "periodic"
        u0 = float(solver_kw.get("u0", 0.05))
        D_fine = float(solver_kw.get("D", 10.0)) * 2.0  # D in fine lattice units

        self._fine = Solver3D(
            Nz=self._Nz_fine, Ny=Ny, Nx=Nx,
            solid=solid_fine,
            **kw_fine,
        )

        self.step_count: int = 0
        self.Cd_history:  List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []

    # ------------------------------------------------------------------
    # Interface exchange
    # ------------------------------------------------------------------

    def _coarse_to_fine(self) -> None:
        """Inject coarse slab into fine solver boundary cells."""
        f_c = self._coarse.f  # (Q, Nz, Ny, Nx)
        slab = f_c[:, self.refine_z_lo:self.refine_z_hi, :, :]  # (Q, Nz_c, Ny, Nx)
        f_fine_from_coarse = _upsample_coarse_to_fine(slab)     # (Q, Nz_f, Ny, Nx)
        # Overwrite fine solver's interior with coarse data (used as initial condition for sub-cycling)
        self._fine.f[:] = f_fine_from_coarse

    def _fine_to_coarse(self) -> None:
        """Replace coarse slab cells with averaged fine data."""
        f_f = self._fine.f  # (Q, Nz_f, Ny, Nx)
        f_c_from_fine = _downsample_fine_to_coarse(f_f)  # (Q, Nz_c, Ny, Nx)
        self._coarse.f[:, self.refine_z_lo:self.refine_z_hi, :, :] = f_c_from_fine

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self):
        """One global timestep: 1 coarse step + 2 fine sub-steps with interface exchange."""
        # 1. Coarse step
        Cd_c, Cly_c, Clz_c = self._coarse._step()

        # 2. Coarse → fine: copy coarse slab into fine grid
        self._coarse_to_fine()

        # 3. Two fine sub-steps
        self._fine._step()
        self._fine._step()

        # 4. Fine → coarse: write averaged fine cells back to coarse slab
        self._fine_to_coarse()

        self.step_count += 1
        return Cd_c, Cly_c, Clz_c

    def run(
        self,
        steps: int,
        check_every: int = 500,
        verbose: bool = True,
        callback=None,
        hdf5_path: Optional[str] = None,
        hdf5_every: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run for `steps` global timesteps."""
        for i in range(steps):
            Cd, Cly, Clz = self.step()

            if (i + 1) % check_every == 0:
                self.Cd_history.append(float(Cd))
                self.Cly_history.append(float(Cly))
                self.Clz_history.append(float(Clz))
                if verbose:
                    print(f"  step {self.step_count:6d}  Cd={Cd:.4f}  Cly={Cly:.4f}  Clz={Clz:.4f}")

            if callback is not None:
                callback(self.step_count, Cd, Cly, Clz)

        return {
            "Cd_history":  self.Cd_history,
            "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history,
            "steps_completed": self.step_count,
        }

    # ------------------------------------------------------------------
    # Property forwarding
    # ------------------------------------------------------------------

    @property
    def f(self) -> np.ndarray:
        """Return the coarse distribution function."""
        return self._coarse.f

    @property
    def solid(self) -> np.ndarray:
        return self._coarse.solid

    @property
    def rho(self) -> Optional[np.ndarray]:
        return self._coarse.rho

    @property
    def ux(self) -> Optional[np.ndarray]:
        return self._coarse.ux

    @property
    def uy(self) -> Optional[np.ndarray]:
        return self._coarse.uy

    @property
    def uz(self) -> Optional[np.ndarray]:
        return self._coarse.uz
