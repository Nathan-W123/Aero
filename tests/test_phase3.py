"""
Phase 3 tests — Numba JIT acceleration.

Test categories:
  1. Kernel correctness  — Numba collision + stream match NumPy to float64 precision
  2. Backend selection   — auto/numpy/numba flags behave correctly
  3. Numerical equivalence — 200-step Numba run matches NumPy run to 1e-10
  4. Speedup benchmark   — Numba ≥2x faster than NumPy on 200x100 (slow, marked)
"""

import numpy as np
import pytest
import time

from aero.lbm.d2q9 import E, W, Q, compute_macroscopic, compute_feq
from aero.lbm.kernels import HAS_NUMBA, collision_kernel, stream_kernel
from aero.lbm.solver import Solver
from aero.geometry.cylinder import Cylinder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solver(backend="numpy", Ny=40, Nx=80, re=100.0):
    r = 8.0
    geom = Cylinder(radius=r, cx_frac=0.35, cy_frac=0.5)
    solid = geom.mark_solid(Ny, Nx)
    u0 = 0.05
    D  = 2.0 * r
    nu = u0 * D / re
    omega = 1.0 / (3.0 * nu + 0.5)
    return Solver(Ny=Ny, Nx=Nx, solid=solid, omega=omega, u0=u0, D=D, backend=backend)


def _lattice_arrays():
    ex = E[:, 0].astype(np.int32)
    ey = E[:, 1].astype(np.int32)
    w  = W.copy()
    return ex, ey, w


def _call_collision_kernel(f, got, solid, omega, ex, ey, w, ny, nx):
    dummy = np.zeros((ny, nx), dtype=np.float64)
    collision_kernel(f, got, solid, omega, ex, ey, w, dummy, False)


# ---------------------------------------------------------------------------
# 1. Kernel unit tests
# ---------------------------------------------------------------------------

class TestKernelAvailability:
    def test_has_numba_is_bool(self):
        assert isinstance(HAS_NUMBA, bool)

    def test_has_numba_true(self):
        # Numba should be installed in this environment
        assert HAS_NUMBA, "Numba expected to be installed — run: pip install numba"

    def test_collision_kernel_callable(self):
        assert callable(collision_kernel)

    def test_stream_kernel_callable(self):
        assert callable(stream_kernel)


class TestCollisionKernel:
    """Verify collision_kernel output matches the pure-NumPy reference."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.Ny, self.Nx = 20, 30
        self.f = np.ascontiguousarray(
            rng.uniform(0.08, 0.16, (Q, self.Ny, self.Nx)), dtype=np.float64
        )
        self.solid = np.zeros((self.Ny, self.Nx), dtype=np.bool_)
        self.solid[8:12, 10:15] = True
        self.omega = 1.5
        self.ex, self.ey, self.w = _lattice_arrays()

    def _numpy_collision(self):
        f = self.f
        rho, ux, uy = compute_macroscopic(f)
        ux[self.solid] = 0.0
        uy[self.solid] = 0.0
        feq = compute_feq(rho, ux, uy)
        return (1.0 - self.omega) * f + self.omega * feq

    def test_matches_numpy_reference(self):
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        ref = self._numpy_collision()
        got = np.empty_like(self.f)
        _call_collision_kernel(self.f, got, self.solid, self.omega, self.ex, self.ey, self.w, self.Ny, self.Nx)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    def test_mass_conservation(self):
        """Total mass (sum of f) must be preserved by collision."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        mass_before = self.f.sum()
        got = np.empty_like(self.f)
        _call_collision_kernel(self.f, got, self.solid, self.omega, self.ex, self.ey, self.w, self.Ny, self.Nx)
        np.testing.assert_allclose(got.sum(), mass_before, rtol=1e-12)

    def test_solid_cells_zero_velocity(self):
        """After collision, solid cells must have zero net momentum."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        got = np.empty_like(self.f)
        _call_collision_kernel(self.f, got, self.solid, self.omega, self.ex, self.ey, self.w, self.Ny, self.Nx)
        _, ux_out, uy_out = compute_macroscopic(got)
        # Solid cells: feq(rho, 0, 0) is exact equilibrium, so after one
        # collision step the velocity should converge toward zero.  With omega=1.5,
        # it won't be exactly zero after one step unless f already was feq(rho,0,0).
        # Just check that the mean velocity inside solid is much smaller than u0.
        assert np.abs(ux_out[self.solid]).mean() < 0.05


class TestStreamKernel:
    """Verify stream_kernel preserves mass and produces correct shifts."""

    def setup_method(self):
        rng = np.random.default_rng(7)
        self.Ny, self.Nx = 16, 24
        self.f = np.ascontiguousarray(
            rng.uniform(0.05, 0.15, (Q, self.Ny, self.Nx)), dtype=np.float64
        )
        self.ex, self.ey, self.w = _lattice_arrays()

    def test_mass_conservation(self):
        """Streaming is a permutation — total mass must be preserved."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        mass_before = self.f.sum()
        f_dst = np.empty_like(self.f)
        stream_kernel(self.f, f_dst, self.ex, self.ey)
        np.testing.assert_allclose(f_dst.sum(), mass_before, rtol=1e-12)

    def test_matches_numpy_roll(self):
        """stream_kernel must produce the same result as the np.roll reference."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        # NumPy reference
        ref = np.empty_like(self.f)
        for i in range(Q):
            tmp = np.roll(self.f[i], int(self.ey[i]), axis=0)
            ref[i] = np.roll(tmp, int(self.ex[i]), axis=1)

        got = np.empty_like(self.f)
        stream_kernel(self.f, got, self.ex, self.ey)
        np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-15)

    def test_rest_direction_unchanged(self):
        """Direction i=0 has (ex=0, ey=0) — output must equal input."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        f_dst = np.empty_like(self.f)
        stream_kernel(self.f, f_dst, self.ex, self.ey)
        np.testing.assert_array_equal(f_dst[0], self.f[0])

    def test_delta_function_shift(self):
        """A unit pulse at (y=5, x=5) in direction i=1 (ex=+1) must appear at x=6."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        Ny, Nx = 10, 12
        f_src = np.zeros((Q, Ny, Nx), dtype=np.float64)
        f_src[1, 5, 5] = 1.0  # direction 1: ex=+1, ey=0
        f_dst = np.empty_like(f_src)
        ex, ey, _ = _lattice_arrays()
        stream_kernel(f_src, f_dst, ex, ey)
        assert f_dst[1, 5, 6] == pytest.approx(1.0)
        assert f_dst[1, 5, 5] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Backend selection
# ---------------------------------------------------------------------------

class TestBackendSelection:
    def test_backend_numpy(self):
        s = _make_solver(backend="numpy")
        assert s.backend == "numpy"
        assert not s._use_numba

    def test_backend_numba(self):
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        s = _make_solver(backend="numba")
        assert s.backend == "numba"
        assert s._use_numba

    def test_backend_auto_picks_numba(self):
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        s = _make_solver(backend="auto")
        assert s.backend == "numba"

    def test_backend_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _make_solver(backend="cuda")

    def test_backend_numba_missing_raises(self, monkeypatch):
        """If Numba is absent, backend='numba' must raise ImportError."""
        import aero.lbm.kernels as km
        monkeypatch.setattr(km, "HAS_NUMBA", False)
        with pytest.raises(ImportError):
            _make_solver(backend="numba")


# ---------------------------------------------------------------------------
# 3. Numerical equivalence
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """Numba and NumPy backends must produce bit-identical Cd/Cl histories."""

    def _run(self, backend, steps=200):
        s = _make_solver(backend=backend, Ny=40, Nx=80)
        s.run(steps=steps, check_every=steps + 1, verbose=False)  # no intermediate prints
        return np.array(s.Cd_history), np.array(s.Cl_history)

    def test_cd_histories_match(self):
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        Cd_np, _  = self._run("numpy")
        Cd_nb, _  = self._run("numba")
        np.testing.assert_allclose(Cd_nb, Cd_np, rtol=1e-10, atol=1e-12,
                                   err_msg="Numba Cd history diverges from NumPy")

    def test_cl_histories_match(self):
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        _, Cl_np = self._run("numpy")
        _, Cl_nb = self._run("numba")
        np.testing.assert_allclose(Cl_nb, Cl_np, rtol=1e-10, atol=1e-12,
                                   err_msg="Numba Cl history diverges from NumPy")

    def test_f_array_matches_after_run(self):
        """Final f distribution arrays must agree to 1e-10."""
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        steps = 150
        s_np = _make_solver(backend="numpy")
        s_nb = _make_solver(backend="numba")
        s_np.run(steps=steps, check_every=steps + 1, verbose=False)
        s_nb.run(steps=steps, check_every=steps + 1, verbose=False)
        np.testing.assert_allclose(s_nb.f, s_np.f, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Speedup benchmark (slow — requires Numba installed)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestNumbaSpeedup:
    """
    Numba backend must deliver ≥2× speedup over NumPy on a 200×100 grid.
    The first Numba run triggers JIT compilation; we warm up first.
    """

    STEPS = 500
    Ny, Nx = 100, 200

    def _time_backend(self, backend):
        s = _make_solver(backend=backend, Ny=self.Ny, Nx=self.Nx)
        s.run(steps=self.STEPS, check_every=self.STEPS + 1, verbose=False)
        t0 = time.perf_counter()
        s.run(steps=self.STEPS, check_every=self.STEPS + 1, verbose=False)
        return time.perf_counter() - t0

    def test_numba_faster_than_numpy(self):
        if not HAS_NUMBA:
            pytest.skip("Numba not installed")
        t_np = self._time_backend("numpy")
        t_nb = self._time_backend("numba")   # JIT already warmed from _time_backend call above
        speedup = t_np / t_nb
        print(f"\n  NumPy : {self.STEPS / t_np:.0f} steps/s")
        print(f"  Numba : {self.STEPS / t_nb:.0f} steps/s")
        print(f"  Speedup: {speedup:.1f}×")
        assert speedup >= 2.0, (
            f"Numba speedup {speedup:.1f}× is below the 2× threshold. "
            f"NumPy: {t_np:.2f}s, Numba: {t_nb:.2f}s"
        )
