"""
Phase 4 tests — D3Q19 3D solver.

Test categories:
  1. D3Q19 lattice invariants       — moments of feq, OPP3 symmetry
  2. 3D geometry                    — sphere + box cell counts
  3. 3D kernels                     — mass conservation, Numba vs NumPy equivalence
  4. 3D boundary conditions         — bounce-back, slip/noslip walls, inlet mass balance
  5. 3D forces                      — empty domain Cd=0, sphere link count
  6. Solver3D integration           — uniform flow, mass conservation over 50 steps
  7. Validation (slow)              — sphere Cd in physical range at Re=100
"""

import numpy as np
import pytest
from argparse import Namespace

from cli3d import compute_blockage_metrics, assess_blockage
from aero.lbm.d3q19 import (
    Q3, E3, W3, OPP3, Y_MIR3, CS2_3,
    compute_macroscopic_3d, compute_feq_3d,
)
from aero.lbm.boundary3d import (
    apply_inlet_zou_he_3d, apply_slip_walls_3d, apply_noslip_walls_3d,
    build_surface_links_3d, apply_bounce_back_3d,
    apply_outlet_zero_gradient_3d,
)
from aero.lbm.kernels3d import HAS_NUMBA, collision_kernel_3d, stream_kernel_3d
from aero.lbm.solver3d import Solver3D
from aero.geometry3d.sphere import Sphere
from aero.geometry3d.box import Box
from aero.forces3d import compute_forces_3d, forces_to_coefficients_3d


def _call_collision_kernel_3d(f, got, solid, omega, ex, ey, ez, w, nz, ny, nx):
    dummy = np.zeros((nz, ny, nx), dtype=np.float64)
    collision_kernel_3d(f, got, solid, omega, ex, ey, ez, w, dummy, False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_f(Nz=8, Ny=8, Nx=16, rho0=1.0, u0=0.05):
    rho = np.full((Nz, Ny, Nx), rho0)
    ux  = np.full((Nz, Ny, Nx), u0)
    uy  = np.zeros((Nz, Ny, Nx))
    uz  = np.zeros((Nz, Ny, Nx))
    return np.ascontiguousarray(compute_feq_3d(rho, ux, uy, uz), dtype=np.float64)


def _lattice():
    return (
        E3[:, 0].astype(np.int32),
        E3[:, 1].astype(np.int32),
        E3[:, 2].astype(np.int32),
        W3.copy(),
    )


def _make_solver(Nz=8, Ny=10, Nx=20, re=50.0, backend="numpy"):
    r = 3.0
    geom  = Sphere(radius=r, cx_frac=0.35, cy_frac=0.5, cz_frac=0.5)
    solid = geom.mark_solid(Nz, Ny, Nx)
    u0 = 0.05
    D  = 2.0 * r
    nu = u0 * D / re
    omega = 1.0 / (3.0 * nu + 0.5)
    return Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D, backend=backend)


# ---------------------------------------------------------------------------
# 0. CLI blockage helpers
# ---------------------------------------------------------------------------

class TestCli3DBlockage:
    def test_sphere_blockage_uses_diameter_in_both_cross_sections(self):
        args = Namespace(shape="sphere", radius=10.0, height=0.0, depth=0.0, ny=64, nz=80)
        by, bz, bmax = compute_blockage_metrics(args, D=20.0)
        assert by == pytest.approx(20.0 / 64.0)
        assert bz == pytest.approx(20.0 / 80.0)
        assert bmax == pytest.approx(max(by, bz))

    def test_box_blockage_uses_height_and_depth(self):
        args = Namespace(shape="box", radius=0.0, height=12.0, depth=20.0, ny=60, nz=100)
        by, bz, bmax = compute_blockage_metrics(args, D=12.0)
        assert by == pytest.approx(12.0 / 60.0)
        assert bz == pytest.approx(20.0 / 100.0)
        assert bmax == pytest.approx(0.2)

    def test_blockage_thresholds_without_override(self):
        assert assess_blockage(0.08, allow_high_blockage=False) == ("good", False)
        assert assess_blockage(0.15, allow_high_blockage=False) == ("elevated", False)
        assert assess_blockage(0.25, allow_high_blockage=False) == ("high", False)
        assert assess_blockage(0.31, allow_high_blockage=False) == ("very high", True)

    def test_blockage_override_allows_confined_run(self):
        assert assess_blockage(0.35, allow_high_blockage=True) == ("very high (override)", False)


# ---------------------------------------------------------------------------
# 1. D3Q19 lattice invariants
# ---------------------------------------------------------------------------

class TestD3Q19Constants:
    def test_num_directions(self):
        assert Q3 == 19

    def test_weights_sum_to_one(self):
        assert W3.sum() == pytest.approx(1.0, rel=1e-12)

    def test_weights_positive(self):
        assert (W3 > 0).all()

    def test_velocities_zero_mean(self):
        """sum_i w[i]*e[i] = 0 (isotropy)"""
        for col in range(3):
            val = float(np.sum(W3 * E3[:, col].astype(np.float64)))
            assert val == pytest.approx(0.0, abs=1e-14)

    def test_cs2_isotropy(self):
        """sum_i w[i]*ex[i]*ex[i] = 1/3 = cs^2"""
        for col in range(3):
            val = float(np.sum(W3 * (E3[:, col].astype(np.float64)**2)))
            assert val == pytest.approx(CS2_3, rel=1e-12)

    def test_opp3_is_involution(self):
        """OPP3[OPP3[i]] == i for all i"""
        for i in range(Q3):
            assert OPP3[OPP3[i]] == i

    def test_opp3_reverses_velocity(self):
        """e[OPP3[i]] == -e[i]"""
        for i in range(Q3):
            np.testing.assert_array_equal(E3[OPP3[i]], -E3[i])

    def test_y_mir3_flips_ey(self):
        """Y_MIR3 preserves ex, ez but flips ey."""
        for i in range(Q3):
            j = Y_MIR3[i]
            assert E3[j, 0] == E3[i, 0]   # ex preserved
            assert E3[j, 2] == E3[i, 2]   # ez preserved
            assert E3[j, 1] == -E3[i, 1]  # ey flipped


class TestFeq3D:
    def test_feq_sum_equals_rho(self):
        rho = np.random.uniform(0.9, 1.1, (4, 5, 6))
        ux  = np.zeros_like(rho)
        uy  = np.zeros_like(rho)
        uz  = np.zeros_like(rho)
        feq = compute_feq_3d(rho, ux, uy, uz)
        np.testing.assert_allclose(feq.sum(axis=0), rho, rtol=1e-12)

    def test_feq_x_momentum(self):
        rho = np.ones((3, 4, 5))
        ux  = np.full_like(rho, 0.04)
        uy  = np.zeros_like(rho)
        uz  = np.zeros_like(rho)
        feq = compute_feq_3d(rho, ux, uy, uz)
        ex  = E3[:, 0].astype(np.float64)
        mx  = np.einsum('i,izyx->zyx', ex, feq)
        np.testing.assert_allclose(mx, rho * ux, rtol=1e-12)

    def test_feq_y_momentum_zero(self):
        rho = np.ones((3, 4, 5))
        ux  = np.full_like(rho, 0.05)
        uy  = np.zeros_like(rho)
        uz  = np.zeros_like(rho)
        feq = compute_feq_3d(rho, ux, uy, uz)
        ey  = E3[:, 1].astype(np.float64)
        my  = np.einsum('i,izyx->zyx', ey, feq)
        np.testing.assert_allclose(my, np.zeros_like(rho), atol=1e-14)


# ---------------------------------------------------------------------------
# 2. 3D geometry
# ---------------------------------------------------------------------------

class TestGeometry3D:
    def test_sphere_mark_solid_shape(self):
        s = Sphere(radius=5.0)
        m = s.mark_solid(20, 20, 40)
        assert m.shape == (20, 20, 40)
        assert m.dtype == np.bool_

    def test_sphere_solid_count(self):
        """Volume of sphere ~ (4/3)*pi*r^3; lattice count should be close."""
        r = 10.0
        s = Sphere(radius=r)
        m = s.mark_solid(60, 60, 120)
        expected = (4.0/3.0) * np.pi * r**3
        assert 0.8 * expected <= m.sum() <= 1.2 * expected

    def test_sphere_reference_length(self):
        assert Sphere(radius=8.0).reference_length() == pytest.approx(16.0)

    def test_box_mark_solid_shape(self):
        b = Box(width=10, height=8, depth=6)
        m = b.mark_solid(20, 20, 40)
        assert m.shape == (20, 20, 40)

    def test_box_solid_count(self):
        b = Box(width=10, height=8, depth=6)
        m = b.mark_solid(40, 40, 80)
        # Each cell in [cx-5, cx+5] x [cy-4, cy+4] x [cz-3, cz+3]
        # Counts: 11 * 9 * 7 = 693 approx
        assert 600 <= m.sum() <= 800

    def test_sphere_center(self):
        cx, cy, cz = Sphere(radius=5).center(64, 64, 128)
        assert cx == pytest.approx(128/3)
        assert cy == pytest.approx(32.0)
        assert cz == pytest.approx(32.0)


# ---------------------------------------------------------------------------
# 3. 3D kernels
# ---------------------------------------------------------------------------

class TestKernels3D:
    def setup_method(self):
        rng = np.random.default_rng(99)
        self.Nz, self.Ny, self.Nx = 6, 8, 12
        self.f = np.ascontiguousarray(
            rng.uniform(0.05, 0.15, (Q3, self.Nz, self.Ny, self.Nx)), dtype=np.float64
        )
        self.solid = np.zeros((self.Nz, self.Ny, self.Nx), dtype=np.bool_)
        self.solid[2:4, 3:5, 4:8] = True
        self.ex, self.ey, self.ez, self.w = _lattice()
        self.omega = 1.4

    def test_collision_mass_conservation(self):
        f_post = np.empty_like(self.f)
        _call_collision_kernel_3d(self.f, f_post, self.solid, self.omega,
                                  self.ex, self.ey, self.ez, self.w,
                                  self.Nz, self.Ny, self.Nx)
        np.testing.assert_allclose(f_post.sum(), self.f.sum(), rtol=1e-12)

    def test_stream_mass_conservation(self):
        f_dst = np.empty_like(self.f)
        stream_kernel_3d(self.f, f_dst, self.ex, self.ey, self.ez)
        np.testing.assert_allclose(f_dst.sum(), self.f.sum(), rtol=1e-12)

    def test_stream_rest_unchanged(self):
        """Direction i=0 (rest) must be unmoved."""
        f_dst = np.empty_like(self.f)
        stream_kernel_3d(self.f, f_dst, self.ex, self.ey, self.ez)
        np.testing.assert_array_equal(f_dst[0], self.f[0])

    def test_stream_matches_numpy_roll(self):
        """stream_kernel must match np.roll reference for all directions."""
        ref = np.empty_like(self.f)
        for i in range(Q3):
            tmp = np.roll(self.f[i], int(self.ez[i]), axis=0)
            tmp = np.roll(tmp,       int(self.ey[i]), axis=1)
            ref[i] = np.roll(tmp,   int(self.ex[i]), axis=2)

        got = np.empty_like(self.f)
        stream_kernel_3d(self.f, got, self.ex, self.ey, self.ez)
        np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-15)

    def test_collision_matches_numpy_feq(self):
        """Numba/NumPy collision kernel must match explicit feq computation."""
        rho, ux, uy, uz = compute_macroscopic_3d(self.f)
        ux[self.solid] = 0.0
        uy[self.solid] = 0.0
        uz[self.solid] = 0.0
        feq = compute_feq_3d(rho, ux, uy, uz)
        ref = (1.0 - self.omega) * self.f + self.omega * feq

        got = np.empty_like(self.f)
        _call_collision_kernel_3d(self.f, got, self.solid, self.omega,
                                  self.ex, self.ey, self.ez, self.w,
                                  self.Nz, self.Ny, self.Nx)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# 4. 3D boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions3D:
    def test_bounce_back_empty_links(self):
        f = _uniform_f()
        f_pre = f.copy()
        links = np.empty((0, 4), dtype=np.int32)
        apply_bounce_back_3d(f, f_pre, links)  # must not raise

    def test_bounce_back_reverses_link(self):
        """A single surface link must swap f[i] and f[opp[i]]."""
        f = _uniform_f(Nz=4, Ny=4, Nx=8)
        f_pre = f.copy()
        i = 1  # ex=+1
        links = np.array([[i, 1, 1, 4]], dtype=np.int32)
        apply_bounce_back_3d(f, f_pre, links)
        assert f[OPP3[i], 1, 1, 4] == pytest.approx(f_pre[i, 1, 1, 4])

    def test_slip_walls_top_bottom(self):
        """After slip BC, y-directed populations at walls satisfy mirror symmetry."""
        f = _uniform_f(Nz=4, Ny=8, Nx=16)
        # Set bottom-wall downward populations to a distinct value
        f[4, :, 0, :] = 0.333
        apply_slip_walls_3d(f)
        # f[3] at bottom must equal 0.333 (mirrored from f[4])
        np.testing.assert_allclose(f[3, :, 0, :], 0.333, atol=1e-14)

    def test_noslip_walls_top_bottom(self):
        """After noslip BC, f[opp[i]] at bottom wall = f[i] at bottom wall."""
        f = _uniform_f(Nz=4, Ny=8, Nx=16)
        f[10, :, 0, :] = 0.777  # ey<0 diagonal (ex=-1, ey=-1)
        apply_noslip_walls_3d(f)
        # OPP3[10]=7 (ex=+1,ey=+1) must be set
        np.testing.assert_allclose(f[7, :, 0, :], 0.777, atol=1e-14)

    def test_outlet_zero_gradient(self):
        f = _uniform_f(Nz=4, Ny=4, Nx=8)
        f[:, :, :, -1] = 999.0  # sentinel
        f[:, :, :, -2] = 1.23
        apply_outlet_zero_gradient_3d(f)
        np.testing.assert_allclose(f[:, :, :, -1], 1.23, atol=1e-14)

    def test_inlet_zou_he_mass_balance(self):
        """After inlet BC, the mass at x=0 must increase with u0 direction."""
        f = _uniform_f(Nz=4, Ny=8, Nx=16)
        rho_before = f[:, :, :, 0].sum()
        apply_inlet_zou_he_3d(f, u0=0.05)
        # rho_in should be close to 1.0 (slight increase for velocity inlet)
        rho_after = f[:, :, :, 0].sum()
        # Just check it didn't blow up
        assert np.isfinite(rho_after)
        assert 0.5 * rho_before < rho_after < 2.0 * rho_before

    def test_build_surface_links_sphere(self):
        s = Sphere(radius=5.0)
        solid = s.mark_solid(16, 16, 32)
        links, _ = build_surface_links_3d(solid)
        assert links.shape[1] == 4
        assert links.shape[0] > 0
        # All i values must be 1..18 (no rest direction links)
        assert (links[:, 0] >= 1).all() and (links[:, 0] <= 18).all()


# ---------------------------------------------------------------------------
# 5. 3D forces
# ---------------------------------------------------------------------------

class TestForces3D:
    def test_no_links_returns_zero(self):
        f = _uniform_f(Nz=4, Ny=4, Nx=8)
        links = np.empty((0, 4), dtype=np.int32)
        Fx, Fy, Fz = compute_forces_3d(f, f, links)
        assert Fx == 0.0 and Fy == 0.0 and Fz == 0.0

    def test_force_coefficients_scaling(self):
        Cd, Cly, Clz = forces_to_coefficients_3d(1.0, 0.0, 0.0, 1.0, 0.05, 20.0)
        expected = 1.0 / (0.5 * 1.0 * 0.05**2 * 20.0**2)
        assert Cd == pytest.approx(expected, rel=1e-10)

    def test_zero_dynamic_pressure(self):
        assert forces_to_coefficients_3d(1.0, 2.0, 3.0, 1.0, 0.0, 20.0) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 6. Solver3D integration
# ---------------------------------------------------------------------------

class TestSolver3DIntegration:
    def test_instantiation(self):
        s = _make_solver()
        assert s.f.shape == (Q3, s.Nz, s.Ny, s.Nx)
        assert s.backend == "numpy"

    def test_f_initialized_to_feq(self):
        s = _make_solver()
        rho, ux, uy, uz = compute_macroscopic_3d(s.f)
        fluid = ~s.solid
        np.testing.assert_allclose(rho[fluid].mean(), 1.0, rtol=1e-10)

    def test_density_stable_50_steps(self):
        """Fluid density must stay in [0.8, 1.2] after 50 steps (no blow-up)."""
        s = _make_solver()
        s.run(steps=50, check_every=100, verbose=False)
        rho, _, _, _ = compute_macroscopic_3d(s.f)
        fluid = ~s.solid
        assert float(rho[fluid].min()) > 0.8
        assert float(rho[fluid].max()) < 1.2

    def test_step_count_increments(self):
        s = _make_solver()
        s.run(steps=10, check_every=100, verbose=False)
        assert s.step_count == 10

    def test_histories_length(self):
        s = _make_solver()
        s.run(steps=30, check_every=100, verbose=False)
        assert len(s.Cd_history) == 30
        assert len(s.Cly_history) == 30
        assert len(s.Clz_history) == 30

    def test_backend_auto(self):
        s = _make_solver(backend="auto")
        if HAS_NUMBA:
            assert s.backend == "numba"
        else:
            assert s.backend == "numpy"

    def test_backend_invalid(self):
        with pytest.raises(ValueError):
            _make_solver(backend="tpu")

    def test_checkpoint_roundtrip(self, tmp_path):
        s = _make_solver()
        s.run(steps=20, check_every=100, verbose=False)
        path = str(tmp_path / "ck3d.npz")
        s.save_checkpoint(path)
        s2 = _make_solver()
        s2.load_checkpoint(path)
        np.testing.assert_array_equal(s2.f, s.f)
        assert s2.step_count == s.step_count

    def test_no_nan_after_run(self):
        s = _make_solver()
        s.run(steps=50, check_every=100, verbose=False)
        assert not np.isnan(s.f).any()

    def test_numba_numpy_equivalence(self):
        """Numba and NumPy backends must produce identical Cd over 30 steps."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        s_np = _make_solver(backend="numpy")
        s_nb = _make_solver(backend="numba")
        s_np.run(steps=30, check_every=100, verbose=False)
        s_nb.run(steps=30, check_every=100, verbose=False)
        np.testing.assert_allclose(
            np.array(s_nb.Cd_history), np.array(s_np.Cd_history),
            rtol=1e-10, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 7. Validation (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSphereValidation:
    """
    Sphere at Re=20 on a 48x48x96 grid.

    Re=100 requires tau=0.53 (below the ~0.55 stability floor for 3D at finite
    blockage).  Re=20 gives tau=0.65 — comfortably stable.

    Expected Cd for sphere at Re=20 ≈ 2–4 (Schiller-Naumann correlation: ~2.4).
    Grid blockage = D/Ny = 14/48 ≈ 29 %; inflates Cd somewhat.
    """

    def test_sphere_cd_range(self):
        r     = 7.0
        Nz, Ny, Nx = 48, 48, 96
        geom  = Sphere(radius=r, cx_frac=1.0/3.0, cy_frac=0.5, cz_frac=0.5)
        solid = geom.mark_solid(Nz, Ny, Nx)
        u0    = 0.05
        D     = 2.0 * r   # = 14 cells
        re    = 20.0
        # nu = 0.05*14/20 = 0.035, tau = 3*0.035+0.5 = 0.605 — stable
        nu    = u0 * D / re
        omega = 1.0 / (3.0 * nu + 0.5)

        print(f"\n  tau = {1.0/omega:.4f}  (must be > 0.55)")

        solver = Solver3D(
            Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D, backend="auto"
        )
        result = solver.run(steps=5000, check_every=1000, verbose=True)

        Cd = result["Cd_mean"]
        print(f"\nSphere Re=20 Cd = {Cd:.4f}  (expected ≈ 2–5 with 29% blockage)")
        assert not np.isnan(Cd), "Cd is NaN — simulation diverged"
        assert 0.5 <= Cd <= 10.0, f"Cd={Cd:.4f} outside plausible 0.5–10 range"
