"""Tests for MPI z-slab domain decomposition — Item 15."""

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py", reason="mpi4py not installed")
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid(Nz: int, Ny: int, Nx: int) -> np.ndarray:
    solid = np.zeros((Nz, Ny, Nx), dtype=bool)
    return solid


def _make_comm_size_1() -> MPI.Comm:
    return MPI.COMM_SELF


# ---------------------------------------------------------------------------


def test_mpi_slab_ranges():
    """_slab_ranges must partition [0, Nz_global) without overlap or gap."""
    from aero.lbm.mpi_solver3d import _slab_ranges

    Nz = 24
    size = 5
    ranges = _slab_ranges(Nz, size)
    assert len(ranges) == size
    assert ranges[0][0] == 0
    assert ranges[-1][1] == Nz
    for i in range(size - 1):
        assert ranges[i][1] == ranges[i + 1][0], "slabs must be contiguous"


def test_mpi_single_rank_matches_solver3d():
    """
    1-rank MPISolver3D must produce the same Cd history as a plain Solver3D.
    Both run 20 steps on an empty domain.
    """
    from aero.lbm.mpi_solver3d import MPISolver3D
    from aero.lbm.solver3d import Solver3D

    comm = _make_comm_size_1()
    Nz, Ny, Nx = 8, 10, 16
    solid = _make_solid(Nz, Ny, Nx)
    omega = 1.5
    u0 = 0.05
    D = 4.0
    kw = dict(omega=omega, u0=u0, D=D, backend="numpy", outlet_bc="zerogradient",
              streamwise_bc="periodic")

    # Plain Solver3D — periodic in z, zerogradient outlet
    ref = Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, **kw)
    for _ in range(20):
        ref._step()

    # MPISolver3D with 1 rank
    mpi_sol = MPISolver3D(
        comm=comm, Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid, **kw,
    )
    for _ in range(20):
        mpi_sol.step()

    # Interior cells (exclude ghost rows) should match
    ref_interior   = ref._solver.f[:, :, :, :]    if hasattr(ref, "_solver") else ref.f
    mpi_interior   = mpi_sol._solver.f[:, 1:-1, :, :]

    np.testing.assert_allclose(
        mpi_interior,
        ref.f[:, :, :, :],
        atol=1e-12,
        err_msg="1-rank MPISolver3D diverged from reference Solver3D",
    )


def test_mpi_halo_exchange_populates_ghosts():
    """After halo exchange on 1 rank, ghost cells must retain their values (no-op boundary)."""
    from aero.lbm.mpi_solver3d import MPISolver3D

    comm = _make_comm_size_1()
    Nz, Ny, Nx = 8, 8, 8
    solid = _make_solid(Nz, Ny, Nx)
    mpi_sol = MPISolver3D(
        comm=comm, Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid,
        omega=1.5, u0=0.05, D=4.0, backend="numpy",
        streamwise_bc="periodic",
    )
    # Mark ghost cells with sentinel values so we can track what happens
    mpi_sol._solver.f[:, 0, :, :]  = -999.0
    mpi_sol._solver.f[:, -1, :, :] = -999.0

    # With 1 rank: no neighbours → halo exchange should be a no-op
    mpi_sol._halo_exchange()

    # Ghost cells should still be -999 (no neighbour to overwrite them)
    assert float(mpi_sol._solver.f[0, 0, 0, 0]) == -999.0
    assert float(mpi_sol._solver.f[0, -1, 0, 0]) == -999.0


def test_mpi_run_interface():
    """MPISolver3D.run() must return a dict with expected keys."""
    from aero.lbm.mpi_solver3d import MPISolver3D

    comm = _make_comm_size_1()
    Nz, Ny, Nx = 6, 6, 10
    solid = _make_solid(Nz, Ny, Nx)
    mpi_sol = MPISolver3D(
        comm=comm, Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid,
        omega=1.5, u0=0.05, D=3.0, backend="numpy",
        streamwise_bc="periodic",
    )
    result = mpi_sol.run(steps=10, check_every=5, verbose=False)
    assert "Cd_history" in result
    assert "steps_completed" in result
    assert result["steps_completed"] == 10
    assert len(result["Cd_history"]) == 2  # 10 // 5 = 2 check points


def test_mpi_checkpoint_roundtrip(tmp_path):
    """save_checkpoint / from_checkpoint must restore step_count."""
    from aero.lbm.mpi_solver3d import MPISolver3D

    comm = _make_comm_size_1()
    Nz, Ny, Nx = 6, 6, 8
    solid = _make_solid(Nz, Ny, Nx)
    kw = dict(omega=1.5, u0=0.05, D=3.0, backend="numpy", streamwise_bc="periodic")

    sol = MPISolver3D(comm=comm, Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid, **kw)
    sol.run(steps=5, check_every=100, verbose=False)

    base = str(tmp_path / "ckpt")
    sol.save_checkpoint(base)

    sol2 = MPISolver3D.from_checkpoint(
        comm=comm, base_path=base,
        Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid, **kw,
    )
    assert sol2.step_count == sol.step_count
    np.testing.assert_allclose(sol2._solver.f, sol._solver.f, atol=1e-14)
