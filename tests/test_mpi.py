"""Tests for MPI z-slab domain decomposition."""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py", reason="mpi4py not installed")
from mpi4py import MPI

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid(Nz: int, Ny: int, Nx: int) -> np.ndarray:
    return np.zeros((Nz, Ny, Nx), dtype=bool)


def _sphere_solid(Nz: int, Ny: int, Nx: int, r: float = 3.0) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:Nz, 0:Ny, 0:Nx]
    return ((zz - Nz / 2) ** 2 + (yy - Ny / 2) ** 2 + (xx - Nx / 3) ** 2) <= r * r


def _make_comm_size_1() -> MPI.Comm:
    return MPI.COMM_SELF


def _run_under_mpirun(source: str, ranks: int):
    """Run a snippet under mpirun; skip if no launcher is available."""
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no MPI launcher available")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "case.py"
        script.write_text(source)
        proc = subprocess.run(
            [mpirun, "--allow-run-as-root", "--oversubscribe", "-n", str(ranks),
             sys.executable, str(script)],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
        )
    if proc.returncode != 0:
        pytest.skip(f"mpirun could not launch {ranks} ranks: {proc.stderr[-400:]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# Decomposition bookkeeping
# ---------------------------------------------------------------------------

def test_mpi_slab_ranges():
    """_slab_ranges must partition [0, Nz_global) without overlap or gap."""
    from aero.lbm.mpi_solver3d import _slab_ranges

    Nz, size = 24, 5
    ranges = _slab_ranges(Nz, size)
    assert len(ranges) == size
    assert ranges[0][0] == 0
    assert ranges[-1][1] == Nz
    for i in range(size - 1):
        assert ranges[i][1] == ranges[i + 1][0], "slabs must be contiguous"


def test_ghost_links_are_dropped():
    """
    Surface links in a ghost layer belong to a neighbouring rank.

    Summing forces over them as well as over the owner's copy counts those
    cells twice in the Allreduce.
    """
    from aero.lbm.mpi_solver3d import restrict_links_to_interior

    nz_local = 4
    links = np.array(
        [[1, 0, 2, 3],                      # bottom ghost — belongs to rank-1
         [2, 1, 2, 3],                      # owned
         [3, 4, 2, 3],                      # owned (last interior row)
         [4, 5, 2, 3]],                     # top ghost — belongs to rank+1
        dtype=np.int32,
    )
    q = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    kept, kept_q = restrict_links_to_interior(links, q, nz_local)
    assert kept.shape[0] == 2
    assert set(kept[:, 1].tolist()) == {1, 4}
    np.testing.assert_allclose(kept_q, [0.2, 0.3])


def test_ghost_links_dropped_handles_empty_table():
    from aero.lbm.mpi_solver3d import restrict_links_to_interior

    links = np.empty((0, 4), dtype=np.int32)
    q = np.empty(0, dtype=np.float32)
    kept, kept_q = restrict_links_to_interior(links, q, 4)
    assert kept.shape == (0, 4) and kept_q.shape == (0,)


def test_streamwise_bc_is_not_overridden():
    """
    Only z is decomposed, so x keeps the caller's BCs on every rank.

    Forcing the inner solver streamwise-periodic silently removed the inlet
    and the outlet, turning an open wind tunnel into a closed duct.
    """
    from aero.lbm.mpi_solver3d import MPISolver3D

    Nz, Ny, Nx = 8, 8, 12
    sol = MPISolver3D(
        comm=_make_comm_size_1(), Nz_global=Nz, Ny=Ny, Nx=Nx,
        solid_global=_make_solid(Nz, Ny, Nx),
        omega=1.5, u0=0.05, D=4.0, backend="numpy",
        streamwise_bc="open", inlet_bc="velocity", outlet_bc="convective",
    )
    assert sol._solver.streamwise_bc == "open"
    assert sol._solver.inlet_bc == "velocity"
    assert sol._solver.outlet_bc == "convective"


# ---------------------------------------------------------------------------
# Single-rank equivalence
# ---------------------------------------------------------------------------

def test_mpi_single_rank_matches_solver3d():
    """A 1-rank decomposition must reproduce the plain solver exactly."""
    from aero.lbm.mpi_solver3d import MPISolver3D
    from aero.lbm.solver3d import Solver3D

    Nz, Ny, Nx = 8, 10, 16
    solid = _sphere_solid(Nz, Ny, Nx)
    kw = dict(omega=1.5, u0=0.05, D=4.0, backend="numpy", wall_bc="noslip")

    ref = Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, **kw)
    for _ in range(20):
        ref._step()

    sol = MPISolver3D(
        comm=_make_comm_size_1(), Nz_global=Nz, Ny=Ny, Nx=Nx,
        solid_global=solid, **kw,
    )
    for _ in range(20):
        sol.step()

    np.testing.assert_allclose(
        sol._solver.f[:, sol.owned_slice], ref.f, atol=1e-12,
        err_msg="1-rank MPISolver3D diverged from reference Solver3D",
    )
    assert sol._solver.surface_links.shape[0] == ref.surface_links.shape[0]


def test_mpi_halo_exchange_wraps_z_periodically():
    """z is periodic globally, so the ghosts mirror the opposite interior row."""
    from aero.lbm.mpi_solver3d import MPISolver3D

    Nz, Ny, Nx = 8, 8, 8
    sol = MPISolver3D(
        comm=_make_comm_size_1(), Nz_global=Nz, Ny=Ny, Nx=Nx,
        solid_global=_make_solid(Nz, Ny, Nx),
        omega=1.5, u0=0.05, D=4.0, backend="numpy",
    )
    f = sol._solver.f
    f[:, 0, :, :] = -999.0
    f[:, -1, :, :] = -999.0
    sol._halo_exchange()

    np.testing.assert_allclose(f[:, -1], f[:, 1])
    np.testing.assert_allclose(f[:, 0], f[:, -2])


def test_mpi_run_records_history_every_step_by_default():
    """
    The history must be sampled per step, like the serial solvers.

    A history recorded every `check_every` steps silently rescales any
    frequency derived from it.
    """
    from aero.lbm.mpi_solver3d import MPISolver3D

    Nz, Ny, Nx = 6, 6, 10
    sol = MPISolver3D(
        comm=_make_comm_size_1(), Nz_global=Nz, Ny=Ny, Nx=Nx,
        solid_global=_make_solid(Nz, Ny, Nx),
        omega=1.5, u0=0.05, D=3.0, backend="numpy",
    )
    result = sol.run(steps=10, check_every=5, verbose=False)
    assert result["steps_completed"] == 10
    assert result["sample_interval"] == 1
    assert len(result["Cd_history"]) == 10


def test_mpi_reduce_every_thins_the_history():
    from aero.lbm.mpi_solver3d import MPISolver3D

    Nz, Ny, Nx = 6, 6, 10
    sol = MPISolver3D(
        comm=_make_comm_size_1(), Nz_global=Nz, Ny=Ny, Nx=Nx,
        solid_global=_make_solid(Nz, Ny, Nx),
        omega=1.5, u0=0.05, D=3.0, backend="numpy", reduce_every=5,
    )
    result = sol.run(steps=10, check_every=100, verbose=False)
    assert len(result["Cd_history"]) == 2
    assert result["sample_interval"] == 5


def test_mpi_checkpoint_roundtrip(tmp_path):
    from aero.lbm.mpi_solver3d import MPISolver3D

    Nz, Ny, Nx = 6, 6, 8
    solid = _make_solid(Nz, Ny, Nx)
    kw = dict(omega=1.5, u0=0.05, D=3.0, backend="numpy")

    sol = MPISolver3D(
        comm=_make_comm_size_1(), Nz_global=Nz, Ny=Ny, Nx=Nx,
        solid_global=solid, **kw,
    )
    sol.run(steps=5, check_every=100, verbose=False)
    base = str(tmp_path / "ckpt")
    sol.save_checkpoint(base)

    sol2 = MPISolver3D.from_checkpoint(
        comm=_make_comm_size_1(), base_path=base,
        Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid, **kw,
    )
    assert sol2.step_count == sol.step_count
    np.testing.assert_allclose(sol2._solver.f, sol._solver.f, atol=1e-14)


# ---------------------------------------------------------------------------
# Real multi-rank run
# ---------------------------------------------------------------------------

_MULTIRANK_CASE = textwrap.dedent(
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    from mpi4py import MPI
    from aero.lbm.mpi_solver3d import MPISolver3D
    from aero.lbm.solver3d import Solver3D

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    Nz, Ny, Nx = 16, 12, 24
    zz, yy, xx = np.mgrid[0:Nz, 0:Ny, 0:Nx]
    solid = ((zz - 8) ** 2 + (yy - 6) ** 2 + (xx - 8) ** 2) <= 9
    kw = dict(omega=1.5, u0=0.05, D=6.0, backend="numpy", wall_bc="noslip")

    m = MPISolver3D(comm=comm, Nz_global=Nz, Ny=Ny, Nx=Nx, solid_global=solid, **kw)
    for _ in range(25):
        local = m.step()
    Cd, Cly, Clz = m.reduce_forces(local)

    ref = Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=solid, **kw)
    for _ in range(25):
        ref_c = ref._step()

    own = np.ascontiguousarray(m._solver.f[:, m.owned_slice])
    gathered = comm.gather(own, root=0)
    nlinks = comm.gather(m._solver.surface_links.shape[0], root=0)
    if rank == 0:
        full = np.concatenate(gathered, axis=1)
        print("FIELD_ERR", float(np.max(np.abs(full - ref.f))))
        print("CD_ERR", abs(Cd - ref_c[0]))
        print("LINKS", sum(nlinks), ref.surface_links.shape[0])
    """
)


@pytest.mark.parametrize("ranks", [2, 4])
def test_multirank_matches_serial(ranks):
    """
    The decomposed run must reproduce the undecomposed one.

    This is the test that catches ghost-layer double counting: the summed link
    count has to equal the serial link count exactly.
    """
    out = _run_under_mpirun(_MULTIRANK_CASE, ranks)
    values = dict(
        (line.split()[0], line.split()[1:])
        for line in out.strip().splitlines()
        if line.split() and line.split()[0] in {"FIELD_ERR", "CD_ERR", "LINKS"}
    )
    assert float(values["FIELD_ERR"][0]) < 1e-12
    assert float(values["CD_ERR"][0]) < 1e-10
    summed, serial = (int(v) for v in values["LINKS"])
    assert summed == serial, "ghost-layer links were counted more than once"
