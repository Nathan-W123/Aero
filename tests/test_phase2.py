"""
Phase 2 regression tests.

Tests cover:
  - No-slip wall BC: reflection correctness
  - Zou-He pressure inlet/outlet BC: moment invariants
  - Poiseuille flow: parabolic profile against analytic solution (benchmark)
  - Polygon geometry: containment and cell count
  - ImageMask geometry: threshold and invert
  - Checkpoint save/load: round-trip state preservation
  - Strouhal number: detection from synthetic oscillation
"""

import math
import os
import tempfile

import numpy as np
import pytest

from aero.lbm.d2q9 import compute_macroscopic, compute_feq, Q
from aero.lbm.boundary import (
    apply_noslip_walls,
    apply_slip_walls,
    apply_inlet_zou_he_pressure,
    apply_outlet_zou_he_pressure,
)
from aero.lbm.solver import Solver
from aero.geometry.polygon import Polygon
from aero.geometry.image_mask import ImageMask
from aero.diagnostics import compute_strouhal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def equilibrium_f(Ny, Nx, u0=0.05, rho0=1.0):
    rho = np.full((Ny, Nx), rho0)
    ux  = np.full((Ny, Nx), u0)
    uy  = np.zeros((Ny, Nx))
    return compute_feq(rho, ux, uy)


def make_open_solver(
    Ny=60, Nx=120, Re=100.0, u0=0.05,
    wall_bc="slip", inlet_bc="velocity", outlet_bc="convective",
    rho_in=1.0, rho_out=1.0,
):
    solid = np.zeros((Ny, Nx), dtype=bool)
    D     = Ny * 0.2   # arbitrary reference length
    nu    = u0 * D / Re
    tau   = 3.0 * nu + 0.5
    omega = 1.0 / tau
    return Solver(
        Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D,
        wall_bc=wall_bc, inlet_bc=inlet_bc, outlet_bc=outlet_bc,
        rho_in=rho_in, rho_out=rho_out,
    )


# ---------------------------------------------------------------------------
# No-slip wall BC
# ---------------------------------------------------------------------------

class TestNoslipWalls:

    def test_bottom_wall_zero_normal(self):
        """
        After applying no-slip BC, the y-momentum at bottom row must be zero:
        f[2]-f[4] + f[5]+f[6]-f[7]-f[8] = 0  at y=0.
        """
        f = equilibrium_f(20, 40, u0=0.05)
        # Perturb bottom row to break symmetry
        f[:, 0, :] += np.random.default_rng(0).uniform(-0.01, 0.01, size=(Q, 40))
        apply_noslip_walls(f)
        uy_mom = (f[2, 0, :] - f[4, 0, :]
                  + f[5, 0, :] + f[6, 0, :]
                  - f[7, 0, :] - f[8, 0, :])
        assert np.allclose(uy_mom, 0.0, atol=1e-12), "Non-zero y-momentum at bottom wall"

    def test_top_wall_zero_normal(self):
        """Same check at top wall y=Ny-1."""
        f = equilibrium_f(20, 40, u0=0.05)
        f[:, -1, :] += np.random.default_rng(1).uniform(-0.01, 0.01, size=(Q, 40))
        apply_noslip_walls(f)
        uy_mom = (f[2, -1, :] - f[4, -1, :]
                  + f[5, -1, :] + f[6, -1, :]
                  - f[7, -1, :] - f[8, -1, :])
        assert np.allclose(uy_mom, 0.0, atol=1e-12), "Non-zero y-momentum at top wall"

    def test_noslip_differs_from_slip(self):
        """No-slip and slip must produce different diagonal populations."""
        f_slip   = equilibrium_f(20, 40, u0=0.05)
        f_noslip = f_slip.copy()
        # Perturb so diagonal pairs are distinguishable
        f_slip[5, 0, :] += 0.01
        f_noslip[5, 0, :] += 0.01
        f_slip[7, 0, :]   += 0.02
        f_noslip[7, 0, :] += 0.02

        apply_slip_walls(f_slip)
        apply_noslip_walls(f_noslip)

        # At bottom: slip sets f[5]←f[8], no-slip sets f[5]←f[7]
        assert not np.allclose(f_slip[5, 0, :], f_noslip[5, 0, :])


# ---------------------------------------------------------------------------
# Zou-He pressure BCs
# ---------------------------------------------------------------------------

class TestPressureBCs:

    def test_pressure_inlet_sets_correct_density(self):
        """
        After apply_inlet_zou_he_pressure, density at x=0 should equal rho_in.
        """
        Ny, Nx = 20, 40
        f = equilibrium_f(Ny, Nx, u0=0.03)
        rho_in = 1.005
        apply_inlet_zou_he_pressure(f, rho_in)
        rho, _, _ = compute_macroscopic(f)
        assert np.allclose(rho[:, 0], rho_in, atol=1e-12), \
            f"Inlet density not set: max err {np.max(np.abs(rho[:,0]-rho_in)):.2e}"

    def test_pressure_inlet_zero_uy(self):
        """y-momentum at inlet should be zero (uy=0 imposed)."""
        Ny, Nx = 20, 40
        f = equilibrium_f(Ny, Nx, u0=0.03)
        apply_inlet_zou_he_pressure(f, 1.002)
        _, _, uy = compute_macroscopic(f)
        assert np.allclose(uy[:, 0], 0.0, atol=1e-12)

    def test_pressure_outlet_sets_correct_density(self):
        """After apply_outlet_zou_he_pressure, density at x=-1 should equal rho_out."""
        Ny, Nx = 20, 40
        f = equilibrium_f(Ny, Nx, u0=0.03)
        rho_out = 1.0
        apply_outlet_zou_he_pressure(f, rho_out)
        rho, _, _ = compute_macroscopic(f)
        assert np.allclose(rho[:, -1], rho_out, atol=1e-12)

    def test_pressure_outlet_zero_uy(self):
        """y-momentum at outlet should be zero."""
        Ny, Nx = 20, 40
        f = equilibrium_f(Ny, Nx, u0=0.03)
        apply_outlet_zou_he_pressure(f, 1.0)
        _, _, uy = compute_macroscopic(f)
        assert np.allclose(uy[:, -1], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Poiseuille flow benchmark (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPoiseuilleFlow:
    """
    Pressure-driven Poiseuille flow between no-slip walls.

    Setup:
      - Zou-He pressure inlet (rho_in > 1) and pressure outlet (rho_out = 1)
      - No-slip top and bottom walls
      - No obstacle

    Analytic steady-state:
      ux(y) = (rho_in - rho_out) * cs2 / (2 * nu * Nx) * y * (H - y)
    where H = Ny - 1, cs2 = 1/3, nu = (tau - 0.5) / 3.
    """

    @staticmethod
    def analytic_ux(y, Ny, Nx, rho_in, rho_out, tau):
        H   = Ny - 1
        cs2 = 1.0 / 3.0
        nu  = (tau - 0.5) / 3.0
        dp  = (rho_in - rho_out) * cs2
        return dp / (2.0 * nu * Nx) * y * (H - y)

    def test_parabolic_profile(self):
        """
        After 10 000 steps the velocity profile at mid-domain should match the
        analytic parabola to within 5%.
        """
        Ny, Nx   = 50, 200
        tau      = 1.0       # nu = 1/6
        omega    = 1.0 / tau
        rho_in   = 1.002
        rho_out  = 1.0
        u0       = 0.0       # unused (pressure-driven)
        D        = float(Ny)

        solid  = np.zeros((Ny, Nx), dtype=bool)
        solver = Solver(
            Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D,
            wall_bc="noslip", inlet_bc="pressure", outlet_bc="pressure",
            rho_in=rho_in, rho_out=rho_out,
        )
        solver.run(steps=10_000, check_every=10_000, verbose=False)

        _, ux, _ = compute_macroscopic(solver.f)
        y   = np.arange(Ny, dtype=float)
        ux_analytic = self.analytic_ux(y, Ny, Nx, rho_in, rho_out, tau)

        # Compare at interior columns (avoid inlet/outlet boundary effects)
        mid_col = Nx // 2
        ux_sim  = ux[:, mid_col]

        # Relative error vs analytic max velocity
        ux_max = ux_analytic.max()
        rel_err = np.max(np.abs(ux_sim - ux_analytic)) / ux_max
        assert rel_err < 0.05, (
            f"Poiseuille profile error {rel_err*100:.2f}% > 5%\n"
            f"  sim_max={ux_sim.max():.6f}  analytic_max={ux_max:.6f}"
        )

    def test_no_nan_pressure_driven(self):
        """Pressure-driven run with no-slip walls must not produce NaN."""
        Ny, Nx = 30, 120
        tau    = 0.8
        omega  = 1.0 / tau
        solid  = np.zeros((Ny, Nx), dtype=bool)
        solver = Solver(
            Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=0.0, D=float(Ny),
            wall_bc="noslip", inlet_bc="pressure", outlet_bc="pressure",
            rho_in=1.002, rho_out=1.0,
        )
        solver.run(steps=500, check_every=500, verbose=False)
        assert not np.any(np.isnan(solver.f))


# ---------------------------------------------------------------------------
# Polygon geometry
# ---------------------------------------------------------------------------

class TestPolygonGeometry:

    def test_square_cell_count(self):
        """A square polygon spanning [0.4,0.6]^2 of a 100×100 grid."""
        verts = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
        poly  = Polygon(verts)
        solid = poly.mark_solid(100, 100)
        assert solid.dtype == bool
        # Vertices at col/row 40 and 60 → interior+boundary = ~21×21=441 cells
        # Accept 380–460 to allow for boundary-inclusion edge effects
        assert 380 <= solid.sum() <= 460, f"Square cell count = {solid.sum()}"

    def test_triangle_containment(self):
        """A triangle should contain its centroid."""
        # Triangle with centroid near (0.5, 0.5)
        verts = [(0.3, 0.2), (0.7, 0.2), (0.5, 0.8)]
        poly  = Polygon(verts)
        solid = poly.mark_solid(100, 100)
        # Centroid in frac coords: (0.5, 0.4)
        assert solid[40, 50], "Centroid is not solid"

    def test_corners_are_fluid(self):
        """Domain corners should be outside a centre polygon."""
        verts = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
        poly  = Polygon(verts)
        solid = poly.mark_solid(100, 100)
        assert not solid[0, 0]
        assert not solid[0, -1]
        assert not solid[-1, 0]
        assert not solid[-1, -1]

    def test_requires_three_vertices(self):
        with pytest.raises(ValueError):
            Polygon([(0.0, 0.0), (1.0, 0.0)])

    def test_center_inside_square(self):
        verts = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
        poly  = Polygon(verts)
        cx, cy = poly.center(100, 100)
        assert 40 <= cx <= 60
        assert 40 <= cy <= 60


# ---------------------------------------------------------------------------
# ImageMask geometry
# ---------------------------------------------------------------------------

class TestImageMask:

    def _make_png(self, path, shape=(20, 40), value=0):
        """Write a solid or blank grayscale PNG using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(shape[1]/10, shape[0]/10), dpi=10)
        ax.imshow(np.full(shape, value / 255.0), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def test_black_image_all_solid(self, tmp_path):
        """A pure-black image should produce an all-solid mask (pixel=0 <= threshold 0.5)."""
        png = str(tmp_path / "black.png")
        # Use numpy to write a minimal black PNG via matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = np.zeros((20, 40), dtype=np.float64)
        fig, ax = plt.subplots()
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.savefig(png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        mask = ImageMask(png, threshold=0.5)
        solid = mask.mark_solid(20, 40)
        assert solid.dtype == bool
        # At least 80% of cells should be solid (black pixels)
        assert solid.mean() > 0.8, f"Expected mostly solid, got {solid.mean():.2f}"

    def test_invert_flag(self, tmp_path):
        """With invert=True, white regions become solid."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # White image
        arr = np.ones((20, 40), dtype=np.float64)
        png = str(tmp_path / "white.png")
        fig, ax = plt.subplots()
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.savefig(png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        solid_normal = ImageMask(png, threshold=0.5, invert=False).mark_solid(20, 40)
        solid_invert = ImageMask(png, threshold=0.5, invert=True).mark_solid(20, 40)
        # Normal: white pixels > threshold → fluid; invert: white → solid
        assert solid_invert.mean() > solid_normal.mean()

    def test_output_shape(self, tmp_path):
        """Output mask must have shape (Ny, Nx) regardless of image resolution."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = np.zeros((7, 13), dtype=np.float64)
        png = str(tmp_path / "small.png")
        fig, ax = plt.subplots()
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        fig.savefig(png)
        plt.close(fig)

        solid = ImageMask(png).mark_solid(Ny=50, Nx=100)
        assert solid.shape == (50, 100)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def _make_solver(self):
        Ny, Nx = 30, 60
        solid  = np.zeros((Ny, Nx), dtype=bool)
        return Solver(Ny=Ny, Nx=Nx, solid=solid, omega=1.2, u0=0.05, D=6.0)

    def test_round_trip_f_array(self, tmp_path):
        """Load/save must restore the f array exactly."""
        solver = self._make_solver()
        solver.run(steps=50, check_every=50, verbose=False)
        path = str(tmp_path / "checkpoint.npz")
        solver.save_checkpoint(path)

        solver2 = self._make_solver()
        solver2.load_checkpoint(path)
        assert np.allclose(solver.f, solver2.f)

    def test_round_trip_histories(self, tmp_path):
        """Cd and Cl histories must be identical after load."""
        solver = self._make_solver()
        solver.run(steps=50, check_every=50, verbose=False)
        path = str(tmp_path / "checkpoint.npz")
        solver.save_checkpoint(path)

        solver2 = self._make_solver()
        solver2.load_checkpoint(path)
        assert solver.Cd_history == solver2.Cd_history
        assert solver.Cl_history == solver2.Cl_history

    def test_round_trip_step_count(self, tmp_path):
        """Step count must be preserved."""
        solver = self._make_solver()
        solver.run(steps=100, check_every=100, verbose=False)
        path = str(tmp_path / "checkpoint.npz")
        solver.save_checkpoint(path)

        solver2 = self._make_solver()
        solver2.load_checkpoint(path)
        assert solver2.step_count == 100

    def test_continued_run_matches_full_run(self, tmp_path):
        """
        Run 50 steps, checkpoint, reload, run 50 more.
        Should produce same Cd history as a single 100-step run.
        """
        # Full run
        solver_full = self._make_solver()
        solver_full.run(steps=100, check_every=100, verbose=False)

        # Split run: 50 + checkpoint + 50
        solver_a = self._make_solver()
        solver_a.run(steps=50, check_every=50, verbose=False)
        path = str(tmp_path / "mid.npz")
        solver_a.save_checkpoint(path)

        solver_b = self._make_solver()
        solver_b.load_checkpoint(path)
        solver_b.run(steps=50, check_every=50, verbose=False)

        # Histories must match
        assert np.allclose(solver_full.Cd_history, solver_b.Cd_history, atol=1e-15)


# ---------------------------------------------------------------------------
# Strouhal number
# ---------------------------------------------------------------------------

class TestStrouhal:

    def test_detects_correct_frequency(self):
        """
        Inject a synthetic Cl time series with known frequency and verify
        compute_strouhal recovers it to within 5%.
        """
        D  = 40.0
        u0 = 0.05
        St_target = 0.164
        f_target  = St_target * u0 / D   # cycles per timestep

        N  = 5000
        t  = np.arange(N, dtype=float)
        Cl = 0.3 * np.sin(2 * math.pi * f_target * t)

        St_meas = compute_strouhal(list(Cl), D=D, u0=u0)
        assert St_meas is not None
        assert abs(St_meas - St_target) / St_target < 0.05, \
            f"Strouhal={St_meas:.4f}, expected ≈ {St_target:.4f}"

    def test_returns_none_for_short_history(self):
        """Too-short Cl history should return None, not raise."""
        St = compute_strouhal([0.1] * 100, D=40.0, u0=0.05)
        assert St is None

    def test_returns_none_for_zero_amplitude(self):
        """Flat (non-oscillating) Cl should return None."""
        St = compute_strouhal([0.0] * 2000, D=40.0, u0=0.05)
        assert St is None
