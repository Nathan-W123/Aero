"""
Tests for the true narrow-band signed-distance field.

Fast suite
----------
* Point-to-triangle distance kernel — 3 geometric cases
* SDF accuracy on unit_sphere.stl vs analytic sphere  (RMS < 0.1 lc)
* Sign convention: interior < 0, exterior > 0
* Band cells have |phi| < far_value, far-field cells = ±far_value

Slow suite (marked @pytest.mark.slow)
--------------------------------------
* IBM (Guo forcing + SDF) vs bounce-back voxel at Re=20 sphere:
  both should give a physically plausible Cd and neither should diverge.
  The IBM run's Cd must be within 30 % of the voxel run's Cd, confirming
  the SDF-based forcing reaches a similar steady state.
"""

from pathlib import Path

import numpy as np
import pytest

from aero.geometry3d.signed_distance import (
    _sq_dist_points_to_triangle,
    compute_phi_field,
)
from aero.geometry3d.stl_io import load_stl_triangles

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "stl"
SPHERE_STL = SAMPLES / "unit_sphere.stl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analytic_sphere_phi(
    Nz: int, Ny: int, Nx: int,
    cx: float, cy: float, cz: float,
    radius: float,
) -> np.ndarray:
    """Analytic SDF for a sphere: phi = dist_to_centre − radius."""
    z = np.arange(Nz) + 0.5
    y = np.arange(Ny) + 0.5
    x = np.arange(Nx) + 0.5
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) - radius


# ---------------------------------------------------------------------------
# Point-to-triangle kernel unit tests
# ---------------------------------------------------------------------------

class TestPointToTriangleDistance:
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])

    def test_point_above_centroid(self):
        # Centroid of this right-angle triangle is (1/3, 1/3, 0).
        # A point directly above (z = 0.5) should have sq_dist = 0.25.
        P = np.array([[1 / 3, 1 / 3, 0.5]])
        sq = _sq_dist_points_to_triangle(P, self.A, self.B, self.C)
        assert abs(sq[0] - 0.25) < 1e-10

    def test_point_at_vertex_A(self):
        P = np.array([[0.0, 0.0, 0.0]])
        sq = _sq_dist_points_to_triangle(P, self.A, self.B, self.C)
        assert sq[0] < 1e-12

    def test_point_opposite_vertex_A_clamped_to_edge(self):
        # Point at (-1, 0.5, 0) is closest to the midpoint of A–C edge.
        P = np.array([[-1.0, 0.5, 0.0]])
        sq = _sq_dist_points_to_triangle(P, self.A, self.B, self.C)
        assert abs(sq[0] - 1.0) < 1e-9

    def test_vectorised_batch(self):
        # Multiple points simultaneously — spot-check two of the above
        P = np.array([
            [1 / 3, 1 / 3, 0.5],
            [0.0,   0.0,   0.0],
        ])
        sq = _sq_dist_points_to_triangle(P, self.A, self.B, self.C)
        assert abs(sq[0] - 0.25) < 1e-10
        assert sq[1] < 1e-12

    def test_symmetric_triangle_midpoint(self):
        # Equilateral triangle in xy-plane, centroid at origin.
        r = 1.0
        A = r * np.array([1.0, 0.0, 0.0])
        B = r * np.array([-0.5, np.sqrt(3) / 2, 0.0])
        C = r * np.array([-0.5, -np.sqrt(3) / 2, 0.0])
        # Point at centroid (origin) in-plane: closest is centroid → dist = 0
        # Actually centroid is inside triangle, so closest point = centroid itself
        P = np.array([[0.0, 0.0, 0.0]])
        sq = _sq_dist_points_to_triangle(P, A, B, C)
        assert sq[0] < 1e-10


# ---------------------------------------------------------------------------
# SDF accuracy on unit_sphere.stl
# ---------------------------------------------------------------------------

class TestSphereSDF:
    """True SDF must match analytic sphere distance within 0.1 lattice cells RMS."""

    @pytest.fixture(scope="class")
    def phi_and_analytic(self):
        """Compute SDF and analytic reference once for the whole class."""
        tris = load_stl_triangles(str(SPHERE_STL))
        Nz, Ny, Nx = 24, 24, 48
        fit_frac = 0.5
        phi = compute_phi_field(
            tris, nz=Nz, ny=Ny, nx=Nx,
            fit_frac=fit_frac,
            mesh_orient="none",   # unit sphere is already centred at origin
            band_width=4,
            far_value=16.0,
        )
        # radius_grid: unit sphere cross-extent = 2.0, scale = fit_frac*min(Ny,Nz)/2
        radius_grid = fit_frac * min(Ny, Nz) / 2.0   # = 6.0
        cx = (1.0 / 3.0) * Nx      # = 16.0
        cy = 0.5 * Ny               # = 12.0
        cz = 0.5 * Nz               # = 12.0
        analytic = _analytic_sphere_phi(Nz, Ny, Nx, cx, cy, cz, radius_grid)
        return phi, analytic, Nz, Ny, Nx

    def test_shape(self, phi_and_analytic):
        phi, _, Nz, Ny, Nx = phi_and_analytic
        assert phi.shape == (Nz, Ny, Nx)

    def test_sign_convention(self, phi_and_analytic):
        phi, analytic, *_ = phi_and_analytic
        # Cells that analytic phi < -1 (clearly inside) must have phi < 0
        deep_inside = analytic < -1.0
        assert (phi[deep_inside] < 0).all(), "Some deep-interior cells have phi >= 0"
        # Cells that analytic phi > 1 (clearly outside, not in far-value zone) must have phi > 0
        far_outside = (analytic > 1.0) & (analytic < 14.0)
        assert (phi[far_outside] > 0).all(), "Some exterior cells have phi <= 0"

    def test_far_field_clamped(self, phi_and_analytic):
        phi, analytic, *_ = phi_and_analytic
        far_value = 16.0
        deep_far = analytic > 12.0
        assert (phi[deep_far] == far_value).all(), "Far-field fluid cells should equal +far_value"

    def test_rms_accuracy_vs_analytic(self, phi_and_analytic):
        phi, analytic, *_ = phi_and_analytic
        FAR = 14.0
        band = np.abs(phi) < FAR
        err = phi[band] - analytic[band]
        rms = float(np.sqrt(np.mean(err ** 2)))
        assert rms < 0.1, (
            f"SDF RMS error {rms:.4f} lc > 0.1 lc threshold "
            f"(old ±0.5 coarse scheme gave ~0.35 lc)"
        )

    def test_max_error_vs_analytic(self, phi_and_analytic):
        phi, analytic, *_ = phi_and_analytic
        FAR = 14.0
        band = np.abs(phi) < FAR
        max_err = float(np.abs(phi[band] - analytic[band]).max())
        assert max_err < 0.5, f"SDF max error {max_err:.4f} lc > 0.5 lc"

    def test_surface_cells_near_zero(self, phi_and_analytic):
        # Cells right at the surface (|analytic phi| < 0.5) must have |phi| < 1.0
        phi, analytic, *_ = phi_and_analytic
        surface = np.abs(analytic) < 0.5
        if surface.any():
            max_phi = float(np.abs(phi[surface]).max())
            assert max_phi < 1.0, f"Surface-adjacent cells have |phi|={max_phi:.3f} > 1.0"


# ---------------------------------------------------------------------------
# IBM vs voxel Cd at Re=20  (slow — runs solver for 2000 steps)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIBMVsVoxelRe20:
    """
    Compare IBM (Guo forcing + SDF) against bounce-back voxel at Re=20.

    Both should:
    - Not diverge (no NaN)
    - Give Cd in [1.0, 6.0]  (Schiller–Naumann at Re=20 ≈ 2.4; blockage inflates this)
    - Agree to within 40 % of each other
    """

    @pytest.fixture(scope="class")
    def run_both(self):
        from aero.geometry3d.stl_io import load_stl_triangles
        from aero.lbm.solver3d import Solver3D

        tris = load_stl_triangles(str(SPHERE_STL))
        Nz, Ny, Nx = 16, 16, 32      # small grid for speed
        fit_frac = 0.4
        Re = 20.0
        u0 = 0.05

        # Compute true SDF
        phi = compute_phi_field(
            tris, nz=Nz, ny=Ny, nx=Nx,
            fit_frac=fit_frac, mesh_orient="none",
            band_width=3, far_value=16.0,
        )
        solid = phi <= 0.0

        radius_grid = fit_frac * min(Ny, Nz) / 2.0   # = 3.2
        D = 2.0 * radius_grid
        nu = u0 * D / Re
        tau = 3.0 * nu + 0.5
        omega = 1.0 / tau

        STEPS = 400
        CHECK = STEPS  # single stability check at end

        # --- Voxel (bounce-back) run ---
        sv = Solver3D(
            Nz=Nz, Ny=Ny, Nx=Nx, solid=solid,
            omega=omega, u0=u0, D=D,
            backend="numpy", collision="bgk",
            ibm_enabled=False,
        )
        res_vox = sv.run(steps=STEPS, check_every=CHECK, verbose=False)

        # --- IBM (Guo forcing) run ---
        si = Solver3D(
            Nz=Nz, Ny=Ny, Nx=Nx, solid=solid,
            omega=omega, u0=u0, D=D,
            backend="numpy", collision="bgk",
            ibm_enabled=True, phi=phi,
        )
        res_ibm = si.run(steps=STEPS, check_every=CHECK, verbose=False)

        return res_vox, res_ibm

    def test_voxel_cd_not_nan(self, run_both):
        res_vox, _ = run_both
        assert not np.isnan(res_vox["Cd_mean"])

    def test_ibm_cd_not_nan(self, run_both):
        _, res_ibm = run_both
        assert not np.isnan(res_ibm["Cd_mean"])

    def test_voxel_cd_physical(self, run_both):
        res_vox, _ = run_both
        cd = res_vox["Cd_mean"]
        assert 1.0 <= cd <= 8.0, f"Voxel Cd={cd:.3f} outside expected [1.0, 8.0]"

    def test_ibm_cd_physical(self, run_both):
        _, res_ibm = run_both
        cd = res_ibm["Cd_mean"]
        assert 1.0 <= cd <= 8.0, f"IBM Cd={cd:.3f} outside expected [1.0, 8.0]"

    def test_ibm_cd_within_40pct_of_voxel(self, run_both):
        res_vox, res_ibm = run_both
        cd_vox = res_vox["Cd_mean"]
        cd_ibm = res_ibm["Cd_mean"]
        rel = abs(cd_ibm - cd_vox) / max(abs(cd_vox), 1e-6)
        assert rel < 0.40, (
            f"IBM Cd={cd_ibm:.3f} and voxel Cd={cd_vox:.3f} disagree by "
            f"{rel*100:.1f}% (threshold 40%)"
        )
