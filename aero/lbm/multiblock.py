"""
Multi-block static 2:1 refinement for 3D LBM.

A refined block covers a slab of the domain at twice the resolution **in every
direction**.  The refinement has to be isotropic: D3Q19 assumes dx = dy = dz,
so halving only one axis is not a valid lattice refinement, it is a different
(and wrong) discretisation.

Scaling
-------
Acoustic (convective) scaling, the usual choice for a 2:1 LBM block:
``dx_f = dx_c/2``, ``dt_f = dt_c/2``, so the fine block runs two sub-steps per
coarse step and the lattice velocity is unchanged.  The lattice viscosity then
**doubles** (Filippova-Hanel: ``tau_f = 1/2 + n (tau_c - 1/2)``) — see
:func:`_omega_fine` for why that direction is right, and
:func:`rescale_for_fine_grid` for how the other arguments follow.

Interface transfer
------------------
Density and velocity are grid-independent under this scaling, so the
equilibrium part of ``f`` transfers unchanged.  The non-equilibrium part does
not: it is proportional to ``tau`` times the lattice velocity gradient, and the
lattice gradient halves when dx halves.  So

    f_neq_fine = f_neq_coarse * tau_f / (2 tau_c)

with the inverse on the way back.  Copying ``f`` across the interface without
this rescaling gets the viscous stress wrong by that factor, which is the
whole reason a refinement interface needs more than interpolation.

Coarse -> fine (before each fine sub-step):
  Fill the fine block's two z ghost layers by interpolating the coarse
  solution — bilinear in (y, x), linear in z onto the staggered fine cell
  centres, and linear in time, since each sub-step needs the coarse state at
  its own start rather than at the ends of the coarse step.

Fine -> coarse (after both sub-steps):
  Average each 2x2x2 fine cell group back onto its coarse cell and rescale the
  non-equilibrium part.

The fine block keeps its own state between steps; only the ghost layers come
from the coarse grid.  Overwriting the fine interior every step — as an
earlier version did — makes the refinement a no-op that costs 8x the work and
returns the coarse answer.

Accuracy and limits
-------------------
Measured on a decaying shear wave resolved with 32 coarse cells, with the
variation along the refined axis and no walls anywhere: the plain solver is
0.5% from the analytic decay and the refined run is 3.2%, i.e. the interface
contributes about 3% over 300 coarse steps.  That is a working refinement, not
a publication-grade one; treat the block interface as a known error source and
validate any case that depends on it.

A block that spans the full cross-section **up to a bounce-back wall is not
geometrically consistent**.  ``apply_noslip_walls_3d`` reflects on the node, so
the coarse channel spans ``Ny - 1`` coarse cells while the fine one spans
``2 Ny - 1`` fine = ``Ny - 0.5`` coarse: the two grids describe channels of
different width and cannot agree near the wall.  Keep refined blocks away from
bounce-back walls, or use a cross-section that is periodic or slip.

Usage
-----
    solver = MultiblockSolver3D(
        Nz=60, Ny=60, Nx=120, solid=solid,
        refine_z_lo=20, refine_z_hi=40,
        omega=1.5, u0=0.05, D=10.0,
    )
    solver.run(steps=500)
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .d3q19 import E3, W3, Q3, compute_macroscopic_3d, compute_feq_3d


REFINEMENT = 2  # dx_coarse / dx_fine


def _omega_fine(omega_coarse: float, n: int = REFINEMENT) -> float:
    """
    Filippova-Hanel relaxation rescaling: ``tau_f = 1/2 + n (tau_c - 1/2)``.

    Equivalently ``nu_lattice`` is multiplied by ``n`` on the finer grid.  That
    direction is not a typo and is worth spelling out, because the intuitive
    guess is the opposite.  Under acoustic scaling ``dt`` and ``dx`` shrink
    together, and

        nu_lattice = nu_physical * dt / dx**2

    so with ``dt/dx`` held fixed, halving ``dx`` doubles ``nu_lattice``.  The
    check that settles it is the Reynolds number, which must be the same on
    both grids: ``Re = u_lat L_lat / nu_lat`` with ``u_lat`` invariant and
    ``L_lat`` doubled needs ``nu_lat`` doubled too.  Halving it instead runs
    the refined block at ``n**2`` times the intended Reynolds number.
    """
    tau_coarse = 1.0 / omega_coarse
    tau_fine = 0.5 + n * (tau_coarse - 0.5)
    return float(1.0 / tau_fine)


def rescale_for_fine_grid(solver_kw: dict, n: int = REFINEMENT) -> dict:
    """
    Convert coarse-grid solver arguments to their fine-grid values.

    Under acoustic scaling (``dt ~ dx``, ``n`` sub-steps per coarse step):

    ===========================  ====================  ==================
    quantity                     lattice scaling       factor for n=2
    ===========================  ====================  ==================
    velocity  u                  invariant             1
    length in cells  D           ``* n``               2
    viscosity  nu                ``* n``               2  (via omega)
    diffusivity  alpha           ``* n``               2
    acceleration  a              ``/ n``               1/2
    ===========================  ====================  ==================

    Dimensionless settings (model constants, intensities, BC names) pass
    through untouched.
    """
    kw = dict(solver_kw)
    kw["omega"] = _omega_fine(float(solver_kw.get("omega", 1.0)), n)

    for key in ("D", "sem_L_int"):
        if kw.get(key):
            kw[key] = float(kw[key]) * n
    for key in ("sponge_thickness",):
        if kw.get(key):
            kw[key] = int(round(float(kw[key]) * n))
    for key in ("alpha_T",):
        if kw.get(key):
            kw[key] = float(kw[key]) * n
    # accelerations: a_lattice = a_physical * dt^2 / dx  ->  divides by n
    for key in ("body_force_x", "body_force_y", "body_force_z", "g_gravity"):
        if kw.get(key):
            kw[key] = float(kw[key]) / n
    return kw


def neq_scale_coarse_to_fine(omega_coarse: float, omega_fine: float) -> float:
    """
    Factor applied to ``f_neq`` when moving from the coarse to the fine grid.

    ``f_neq ~ tau * grad_lattice(u)``, and the lattice gradient halves when dx
    halves, so the factor is ``tau_f / (2 tau_c)``.
    """
    tau_c = 1.0 / omega_coarse
    tau_f = 1.0 / omega_fine
    return tau_f / (2.0 * tau_c)


def split_equilibrium(f: np.ndarray):
    """Return ``(feq, fneq)`` for a distribution array."""
    rho, ux, uy, uz = compute_macroscopic_3d(f)
    feq = compute_feq_3d(rho, ux, uy, uz)
    return feq, f - feq


def rescale_nonequilibrium(f: np.ndarray, factor: float) -> np.ndarray:
    """Rebuild ``f`` with its non-equilibrium part scaled by ``factor``."""
    feq, fneq = split_equilibrium(f)
    return feq + factor * fneq


def _refine_plane(plane: np.ndarray) -> np.ndarray:
    """
    Bilinear 2x upsample of a ``(Q, Ny, Nx)`` interface plane.

    Fine cell ``i`` sits at coarse coordinate ``-0.25 + 0.5 i``, i.e. the two
    fine cells inside a coarse cell are offset by +-1/4 of a coarse spacing.
    Sampling is clamped at the edges.
    """
    q, ny, nx = plane.shape
    fine_y = -0.25 + 0.5 * np.arange(2 * ny)
    fine_x = -0.25 + 0.5 * np.arange(2 * nx)

    def _weights(coord, n):
        lo = np.clip(np.floor(coord).astype(int), 0, n - 1)
        hi = np.clip(lo + 1, 0, n - 1)
        frac = np.clip(coord - lo, 0.0, 1.0)
        return lo, hi, frac

    y_lo, y_hi, wy = _weights(fine_y, ny)
    x_lo, x_hi, wx = _weights(fine_x, nx)

    a = plane[:, y_lo, :][:, :, x_lo]
    b = plane[:, y_lo, :][:, :, x_hi]
    c = plane[:, y_hi, :][:, :, x_lo]
    d = plane[:, y_hi, :][:, :, x_hi]
    wy_ = wy[None, :, None]
    wx_ = wx[None, None, :]
    return (
        a * (1 - wy_) * (1 - wx_)
        + b * (1 - wy_) * wx_
        + c * wy_ * (1 - wx_)
        + d * wy_ * wx_
    )


def _restrict_block(f_fine: np.ndarray) -> np.ndarray:
    """Average each 2x2x2 fine cell group onto its coarse cell."""
    q, nz, ny, nx = f_fine.shape
    return f_fine.reshape(q, nz // 2, 2, ny // 2, 2, nx // 2, 2).mean(axis=(2, 4, 6))


# Kept for backwards compatibility with the previous z-only API ---------------

def _upsample_coarse_to_fine(f_coarse_slice: np.ndarray) -> np.ndarray:
    """Repeat each coarse z-cell twice (0th order, z only)."""
    return np.repeat(f_coarse_slice, 2, axis=1)


def _downsample_fine_to_coarse(f_fine_slice: np.ndarray) -> np.ndarray:
    """Average pairs of fine z-cells (z only)."""
    return 0.5 * (f_fine_slice[:, 0::2, :, :] + f_fine_slice[:, 1::2, :, :])


class MultiblockSolver3D:
    """
    Static 2:1 refined block embedded in a coarse domain.

    Parameters
    ----------
    Nz, Ny, Nx      : coarse domain dimensions
    solid           : bool ndarray (Nz, Ny, Nx)
    refine_z_lo     : first coarse z-plane inside the refined block
    refine_z_hi     : one past the last coarse z-plane (exclusive)
    solid_fine      : optional (2*Nz_r, 2*Ny, 2*Nx) mask re-voxelised at fine
                      resolution.  Defaults to repeating the coarse mask, which
                      keeps the staircase of the coarse geometry.
    **solver_kw     : passed to both Solver3D instances; omega, D and any
                      cell-counted length are rescaled for the fine block.
    """

    def __init__(
        self,
        Nz: int,
        Ny: int,
        Nx: int,
        solid: np.ndarray,
        refine_z_lo: int,
        refine_z_hi: int,
        solid_fine: Optional[np.ndarray] = None,
        restrict_band: int = 2,
        **solver_kw: Any,
    ) -> None:
        from .solver3d import Solver3D

        self.Nz = int(Nz)
        self.Ny = int(Ny)
        self.Nx = int(Nx)
        self.refine_z_lo = int(refine_z_lo)
        self.refine_z_hi = int(refine_z_hi)
        if not 0 < self.refine_z_lo < self.refine_z_hi <= Nz:
            raise ValueError(
                "refined block must satisfy 0 < refine_z_lo < refine_z_hi <= Nz "
                "so that a coarse plane exists on both sides of the interface"
            )
        self.restrict_band = int(restrict_band)
        self._Nz_refined = self.refine_z_hi - self.refine_z_lo
        # two fine planes per coarse plane, plus one ghost layer either side
        self._Nz_fine = 2 * self._Nz_refined
        self._Nz_fine_total = self._Nz_fine + 2

        self.omega_coarse = float(solver_kw.get("omega", 1.0))
        self.omega_fine = _omega_fine(self.omega_coarse)
        self._neq_c2f = neq_scale_coarse_to_fine(self.omega_coarse, self.omega_fine)
        self._neq_f2c = 1.0 / self._neq_c2f

        self._coarse = Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, **solver_kw)

        if solid_fine is None:
            slab = solid[self.refine_z_lo:self.refine_z_hi]
            solid_fine = np.repeat(np.repeat(np.repeat(slab, 2, axis=0), 2, axis=1), 2, axis=2)
        solid_fine = np.asarray(solid_fine, dtype=bool)
        expected = (self._Nz_fine, 2 * Ny, 2 * Nx)
        if solid_fine.shape != expected:
            raise ValueError(f"solid_fine must have shape {expected}, got {solid_fine.shape}")
        # pad with the neighbouring plane so the ghost layers carry geometry too
        solid_padded = np.concatenate(
            [solid_fine[:1], solid_fine, solid_fine[-1:]], axis=0
        )

        kw_fine = rescale_for_fine_grid(solver_kw, REFINEMENT)
        # The block's z faces are driven by the interface, not by a BC.
        kw_fine["streamwise_bc"] = solver_kw.get("streamwise_bc", "open")

        self._fine = Solver3D(
            Nz=self._Nz_fine_total, Ny=2 * Ny, Nx=2 * Nx,
            solid=solid_padded,
            **kw_fine,
        )

        # Coarse interface planes from the *previous* coarse step, needed to
        # interpolate the half-time level the first fine sub-step sits at.
        self._prev_lo: Optional[np.ndarray] = None
        self._prev_hi: Optional[np.ndarray] = None

        self.step_count: int = 0
        self.Cd_history: List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []

    # ------------------------------------------------------------------
    # Interface transfer
    # ------------------------------------------------------------------

    def _coarse_interface_planes(self):
        """
        The two coarse plane pairs bracketing the block's z faces.

        Returns ``(lo_pair, hi_pair)`` where each pair is the coarse planes on
        either side of the corresponding ghost layer.
        """
        f_c = self._coarse.f
        lo = (f_c[:, self.refine_z_lo - 1], f_c[:, self.refine_z_lo])
        hi_index = min(self.refine_z_hi, self.Nz - 1)
        hi = (f_c[:, self.refine_z_hi - 1], f_c[:, hi_index])
        return lo, hi

    @staticmethod
    def _interp_pair(pair, weight_first: float) -> np.ndarray:
        return weight_first * pair[0] + (1.0 - weight_first) * pair[1]

    def _inject_ghosts(self, time_frac: float) -> None:
        """
        Set the fine block's z ghost layers from the coarse solution.

        ``time_frac`` is the time level the ghost must represent, as a fraction
        of the coarse step: the populations a sub-step streams in are the ones
        at its *start*, so sub-step 1 wants 0.0 and sub-step 2 wants 0.5.  The
        coarse state is known only at the ends of its own step, so anything in
        between is interpolated linearly in time from the planes saved before
        the coarse step advanced.
        """
        lo_now, hi_now = self._coarse_interface_planes()

        if self._prev_lo is None or time_frac >= 1.0:
            lo_pair, hi_pair = lo_now, hi_now
        else:
            lo_pair = tuple(
                (1.0 - time_frac) * prev + time_frac * now
                for prev, now in zip(self._prev_lo, lo_now)
            )
            hi_pair = tuple(
                (1.0 - time_frac) * prev + time_frac * now
                for prev, now in zip(self._prev_hi, hi_now)
            )

        # Ghost centres sit 3/4 of a coarse cell outside the block face.
        plane_lo = self._interp_pair(lo_pair, 0.75)
        plane_hi = self._interp_pair(hi_pair, 0.25)

        for plane, index in ((plane_lo, 0), (plane_hi, self._Nz_fine_total - 1)):
            refined = _refine_plane(np.ascontiguousarray(plane))
            scaled = rescale_nonequilibrium(refined[:, None, :, :], self._neq_c2f)
            self._fine.f[:, index] = scaled[:, 0]

    def _restrict_to_coarse(self) -> None:
        """
        Write the fine solution back onto the coarse grid.

        Averaging 2x2x2 fine cells is a low-pass filter, so applying it over
        the whole block every step damps the coarse solution cumulatively.
        Only the coarse planes the exterior actually reads need to carry fine
        data: streaming moves one cell per step, so a band of
        ``restrict_band`` planes at each face is enough to make everything
        outside the block see the refined solution.  ``restrict_band=0`` means
        restrict the whole block (useful when the coarse field inside the block
        is what gets visualised).
        """
        interior = self._fine.f[:, 1:self._Nz_fine_total - 1]
        coarse_block = rescale_nonequilibrium(
            _restrict_block(np.ascontiguousarray(interior)), self._neq_f2c
        )
        band = self.restrict_band
        if band <= 0 or 2 * band >= self._Nz_refined:
            self._coarse.f[:, self.refine_z_lo:self.refine_z_hi] = coarse_block
            return
        lo, hi = self.refine_z_lo, self.refine_z_hi
        self._coarse.f[:, lo:lo + band] = coarse_block[:, :band]
        self._coarse.f[:, hi - band:hi] = coarse_block[:, -band:]

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self):
        """One coarse timestep: 1 coarse step + 2 fine sub-steps + restriction."""
        lo_pair, hi_pair = self._coarse_interface_planes()
        self._prev_lo = tuple(p.copy() for p in lo_pair)
        self._prev_hi = tuple(p.copy() for p in hi_pair)

        Cd_c, Cly_c, Clz_c = self._coarse._step()

        # Each sub-step streams the populations present at its start, so the
        # ghost carries the coarse state at t and then at t + 1/2.
        self._inject_ghosts(time_frac=0.0)
        self._fine._step()
        self._inject_ghosts(time_frac=0.5)
        result = self._fine._step()

        self._restrict_to_coarse()

        self.step_count += 1
        # Forces come from the fine block when the body is inside it, since
        # that is the better-resolved surface.
        return result if self._body_in_block() else (Cd_c, Cly_c, Clz_c)

    def _body_in_block(self) -> bool:
        return bool(self._fine.surface_links.shape[0] > 0)

    def run(
        self,
        steps: int,
        check_every: int = 500,
        verbose: bool = True,
        callback=None,
        hdf5_path: Optional[str] = None,
        hdf5_every: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run for `steps` coarse timesteps."""
        for i in range(steps):
            Cd, Cly, Clz = self.step()
            self.Cd_history.append(float(Cd))
            self.Cly_history.append(float(Cly))
            self.Clz_history.append(float(Clz))

            if verbose and (i + 1) % check_every == 0:
                print(
                    f"  step {self.step_count:6d}  "
                    f"Cd={Cd:.4f}  Cly={Cly:.4f}  Clz={Clz:.4f}"
                )

            if callback is not None:
                callback(self.step_count, Cd, Cly, Clz)

        return {
            "Cd_history": self.Cd_history,
            "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history,
            "steps_completed": self.step_count,
            "omega_coarse": self.omega_coarse,
            "omega_fine": self.omega_fine,
            "neq_scale_coarse_to_fine": self._neq_c2f,
        }

    # ------------------------------------------------------------------
    # Property forwarding
    # ------------------------------------------------------------------

    @property
    def f(self) -> np.ndarray:
        """The coarse distribution function (the full-domain view)."""
        return self._coarse.f

    @property
    def fine(self):
        """The refined block's solver."""
        return self._fine

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
