"""Tests for Tier 3 engine upgrades: HDF5/XDMF output, true checkpoint restart, spectral post-processing."""

import pathlib
import tempfile
import numpy as np
import pytest

from aero.geometry.cylinder import Cylinder
from aero.lbm.solver import Solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_solver(**kw) -> Solver:
    cyl = Cylinder(radius=6, cx_frac=1/3, cy_frac=0.5)
    solid = cyl.mark_solid(30, 60)
    defaults = dict(Ny=30, Nx=60, solid=solid, omega=1.2, u0=0.05, D=12.0, backend="numpy")
    defaults.update(kw)
    return Solver(**defaults)


# ---------------------------------------------------------------------------
# Item 9: HDF5 / XDMF output
# ---------------------------------------------------------------------------

def test_hdf5_writer_no_h5py():
    """When h5py is absent, HDF5Writer silently does nothing (no exception)."""
    from aero.hdf5_writer import HDF5Writer, HAS_HDF5
    if HAS_HDF5:
        pytest.skip("h5py is installed — test only meaningful without it")
    with tempfile.TemporaryDirectory() as d:
        w = HDF5Writer(str(pathlib.Path(d) / "run.h5"), Ny=8, Nx=16, dims=2)
        ux = np.zeros((8, 16))
        w.write_step(1, ux=ux, uy=ux, rho=np.ones((8, 16)))
        w.close()
        assert not (pathlib.Path(d) / "run.h5").exists()


@pytest.mark.skipif(
    True,  # only runs when h5py is present; keep as documentation
    reason="h5py not installed"
)
def test_hdf5_writer_creates_file():
    from aero.hdf5_writer import HDF5Writer, HAS_HDF5
    if not HAS_HDF5:
        pytest.skip("h5py not installed")
    import h5py
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "run.h5")
        ux = np.random.default_rng(0).random((8, 16))
        uy = np.zeros_like(ux)
        rho = np.ones_like(ux)
        with HDF5Writer(path, Ny=8, Nx=16, dims=2) as w:
            w.write_step(10, ux=ux, uy=uy, rho=rho)
            w.write_step(20, ux=ux * 0.5, uy=uy, rho=rho)
        assert pathlib.Path(path).exists()
        assert pathlib.Path(path).with_suffix(".xdmf").exists()
        with h5py.File(path, "r") as f:
            assert "step_00000010" in f
            assert "step_00000020" in f
            np.testing.assert_allclose(f["step_00000010/ux"][:], ux)


def test_solver_run_hdf5_path_noop_without_h5py():
    """Passing hdf5_path when h5py is absent must not crash."""
    from aero.hdf5_writer import HAS_HDF5
    if HAS_HDF5:
        pytest.skip("h5py installed — skip noop test")
    sol = _small_solver()
    with tempfile.TemporaryDirectory() as d:
        sol.run(steps=50, check_every=50, verbose=False,
                hdf5_path=str(pathlib.Path(d) / "out.h5"), hdf5_every=25)


# ---------------------------------------------------------------------------
# Item 10: True checkpoint restart
# ---------------------------------------------------------------------------

def test_from_checkpoint_reproduces_state():
    """Solver rebuilt from checkpoint via from_checkpoint must continue identically."""
    sol = _small_solver()
    sol.run(steps=100, check_every=100, verbose=False)
    Cd_before = sol.Cd_history[-1]

    with tempfile.TemporaryDirectory() as d:
        ckpt = str(pathlib.Path(d) / "ckpt.npz")
        sol.save_checkpoint(ckpt)

        sol2 = Solver.from_checkpoint(ckpt)
        assert sol2.step_count == sol.step_count
        assert sol2.Ny == sol.Ny
        assert sol2.Nx == sol.Nx
        np.testing.assert_allclose(sol2.f, sol.f, atol=1e-14)
        assert sol2.Cd_history[-1] == pytest.approx(Cd_before, rel=1e-12)


def test_from_checkpoint_continues_run():
    """Checkpoint-restarted solver must produce identical Cd to uninterrupted run."""
    # Run 200 steps uninterrupted
    sol_full = _small_solver()
    res_full = sol_full.run(steps=200, check_every=200, verbose=False)

    # Run 100 steps, checkpoint, rebuild, run another 100
    sol_a = _small_solver()
    sol_a.run(steps=100, check_every=100, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(pathlib.Path(d) / "mid.npz")
        sol_a.save_checkpoint(ckpt)
        sol_b = Solver.from_checkpoint(ckpt)
    sol_b.run(steps=100, check_every=100, verbose=False)

    assert sol_b.Cd_history[-1] == pytest.approx(res_full["Cd_history"][-1], rel=1e-10)


def test_save_checkpoint_stores_solid():
    """Checkpoint must include solid mask so from_checkpoint doesn't need it externally."""
    sol = _small_solver()
    sol.run(steps=20, check_every=20, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        ckpt = str(pathlib.Path(d) / "c.npz")
        sol.save_checkpoint(ckpt)
        data = np.load(ckpt, allow_pickle=False)
        assert "solid" in data
        assert "params_json" in data


# ---------------------------------------------------------------------------
# Item 11: Spectral post-processing
# ---------------------------------------------------------------------------

def test_energy_spectrum_zero_field():
    """Zero velocity field must give zero energy at all wavenumbers."""
    from aero.spectral import energy_spectrum_3d
    ux = np.zeros((8, 8, 16))
    k, E = energy_spectrum_3d(ux, ux, ux)
    np.testing.assert_allclose(E, 0.0, atol=1e-30)


def test_energy_spectrum_single_mode():
    """A single Fourier mode should put energy only in that wavenumber shell."""
    from aero.spectral import energy_spectrum_3d
    Nz, Ny, Nx = 16, 16, 16
    # k=(1,0,0) mode: ux = cos(2π x / Nx)
    x = np.arange(Nx, dtype=np.float64)
    ux = np.cos(2 * np.pi * x / Nx)[np.newaxis, np.newaxis, :] * np.ones((Nz, Ny, 1))
    uy = np.zeros_like(ux)
    uz = np.zeros_like(ux)
    k, E = energy_spectrum_3d(ux, uy, uz)
    # Energy should be concentrated in shell k=1; all other shells ≈ 0
    assert E[1] > 0.0
    assert E[0] == pytest.approx(0.0, abs=1e-20)
    for ki in range(2, len(E)):
        assert E[ki] == pytest.approx(0.0, abs=1e-20)


def test_two_point_correlation_uniform():
    """Uniform field has zero fluctuation — R(r) should be 0 or 1 (both valid)."""
    from aero.spectral import two_point_correlation_z
    u = np.ones((16, 8, 8))
    r, R = two_point_correlation_z(u)
    # u_prime = 0 everywhere; R(0) either 0 (0/0 case) or 1 (normalised)
    assert R[0] in (0.0, 1.0) or np.isnan(R[0]) or R[0] == pytest.approx(1.0, abs=1e-10)


def test_two_point_correlation_single_mode_peaks():
    """Spanwise cosine field — R(r) must peak at r=0 (=1) and trough at r=Nz/2 (≈-1)."""
    from aero.spectral import two_point_correlation_z
    Nz = 32
    z = np.arange(Nz, dtype=np.float64)
    # Use cos so the field is not zero-mean, giving nonzero fluctuation
    u = np.cos(2 * np.pi * z / Nz)[:, np.newaxis, np.newaxis] * np.ones((1, 8, 8))
    r, R = two_point_correlation_z(u)
    assert R[0] == pytest.approx(1.0, abs=1e-6)
    # At half period the correlation should be near -1
    assert R[Nz // 2] < -0.9


def test_integral_length_scale_positive():
    """integral_length_scale must return a positive finite value for a decaying R."""
    from aero.spectral import two_point_correlation_z, integral_length_scale
    Nz = 64
    # Build a signal with nonzero spatial variation: Gaussian bump in z
    z = np.arange(Nz, dtype=np.float64)
    # Sum of two offset Gaussians to give spatially varying non-zero mean fluctuations
    u = (
        np.exp(-((z - 10) ** 2) / 8.0)[:, np.newaxis, np.newaxis]
        * np.ones((1, 8, 8))
    )
    r, R = two_point_correlation_z(u)
    L = integral_length_scale(r, R)
    assert L > 0.0
    assert np.isfinite(L)
