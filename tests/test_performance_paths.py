"""
The fast paths must compute exactly what the straightforward ones do.

Each optimisation here -- tiled collision kernels, collision fused with
streaming, recycled work arrays, compiled inlet/outlet -- was written to be
bit-identical to the code it replaced, so these compare with array_equal
rather than a tolerance wherever the operation order is the same.
"""

import numpy as np
import pytest

from aero.lbm import boundary3d as b3
from aero.lbm import kernels3d as k3
from aero.lbm import kernels3d_reg as kr3
from aero.lbm.lattice3d import D3Q19, D3Q27, compute_feq
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D

numba = pytest.importorskip("numba")


def _sphere(nz, ny, nx, r, cz=0.5):
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    return ((zz - cz * nz) ** 2 + (yy - ny / 2) ** 2 + (xx - nx / 3) ** 2) <= r * r


def _perturbed_state(lat, shape, seed=0):
    """An equilibrium state with a random, non-trivial velocity field."""
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    u = [0.05 * rng.standard_normal(shape) for _ in range(3)]
    f = compute_feq(rho, *u, lat)
    f += 1e-4 * rng.standard_normal(f.shape)       # some non-equilibrium
    return np.ascontiguousarray(f)


def _kernel_args(lat, solid, force_field):
    ex = lat.E[:, 0].astype(np.int32)
    ey = lat.E[:, 1].astype(np.int32)
    ez = lat.E[:, 2].astype(np.int32)
    omega_field = np.full(solid.shape, 1.7)
    return (solid, 1.8, ex, ey, ez, lat.W.copy(), omega_field, True,
            np.zeros(3), force_field, 2, lat.h3_factor)


# ---------------------------------------------------------------------------
# Collision fused with streaming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat", [D3Q19, D3Q27], ids=lambda l: l.name)
@pytest.mark.parametrize("kernel", [k3.collision_kernel_3d, kr3.regularized_collision_kernel_3d],
                         ids=["bgk", "regularized"])
def test_fused_collide_stream_equals_collide_then_stream(lat, kernel):
    # nx = 70 is not a multiple of the 64-cell tile, so the wrap at both ends
    # lands in a partial tile as well as a full one.
    nz, ny, nx = 6, 7, 70
    f = _perturbed_state(lat, (nz, ny, nx))
    solid = _sphere(nz, ny, nx, 2.5, cz=0.0)        # straddles the z wrap
    acc = 1e-5 * np.random.default_rng(1).standard_normal((3, nz, ny, nx))
    args = _kernel_args(lat, solid, acc)

    pre = np.empty_like(f)
    two_pass = np.empty_like(f)
    kernel(f, pre, *args, False)
    k3.stream_kernel_3d(pre, two_pass, args[2], args[3], args[4])

    fused = np.empty_like(f)
    kernel(f, fused, *args, True)
    assert np.array_equal(fused, two_pass)

    # and the pre-streaming values can be read back from where they landed
    nodes = np.argwhere(np.ones((nz, ny, nx), dtype=bool))
    recovered = np.full_like(f, np.nan)
    k3.gather_prestream_3d(fused, recovered, *(np.ascontiguousarray(nodes[:, j]) for j in range(3)),
                           args[2], args[3], args[4])
    assert np.array_equal(recovered, pre)


# ---------------------------------------------------------------------------
# Compiled boundary conditions
# ---------------------------------------------------------------------------

def _without_numba(monkeypatch):
    monkeypatch.setattr(b3, "_HAS_NUMBA", False)


@pytest.mark.parametrize("lat", [D3Q19, D3Q27], ids=lambda l: l.name)
def test_compiled_zou_he_inlet_is_bit_identical(lat, monkeypatch):
    f = _perturbed_state(lat, (5, 6, 8), seed=3)
    rng = np.random.default_rng(4)
    ux = 0.05 + 0.01 * rng.standard_normal((5, 6))
    uy = 0.01 * rng.standard_normal((5, 6))
    uz = 0.01 * rng.standard_normal((5, 6))

    fast = f.copy()
    rho_fast = b3._zou_he_x_inlet(fast, ux, uy, uz, lat)
    _without_numba(monkeypatch)
    slow = f.copy()
    rho_slow = b3._zou_he_x_inlet(slow, ux, uy, uz, lat)

    assert np.array_equal(fast, slow)
    assert np.array_equal(rho_fast, rho_slow)


def test_compiled_convective_outlet_is_bit_identical(monkeypatch):
    f = _perturbed_state(D3Q19, (5, 6, 8), seed=5)
    prev = np.ascontiguousarray(f[:, :, :, -1] * 1.01)

    fast, prev_fast = f.copy(), prev.copy()
    b3.apply_outlet_convective_3d(fast, prev_fast, 0.05)
    _without_numba(monkeypatch)
    slow, prev_slow = f.copy(), prev.copy()
    b3.apply_outlet_convective_3d(slow, prev_slow, 0.05)

    assert np.array_equal(fast, slow)
    assert np.array_equal(prev_fast, prev_slow)


# ---------------------------------------------------------------------------
# Solver-level: recycled work arrays and the fused path
# ---------------------------------------------------------------------------

def _solver3d(**kw):
    nz, ny, nx = 12, 12, 24
    args = dict(Nz=nz, Ny=ny, Nx=nx, solid=_sphere(nz, ny, nx, 3.0), omega=1.8,
                u0=0.05, D=6.0, backend="numba")
    args.update(kw)
    return Solver3D(**args)


@pytest.mark.parametrize("collision", ["bgk", "regularized"])
def test_fused_solver_matches_the_unfused_two_pass_step(collision):
    """
    Bouzidi with no distance field is plain halfway bounce-back, but it keeps
    the two-pass collide-then-stream path: the two runs differ only in whether
    streaming is fused.
    """
    fused = _solver3d(collision=collision)
    two_pass = _solver3d(collision=collision, bouzidi=True)
    cd_f = [fused._step()[0] for _ in range(25)]
    cd_t = [two_pass._step()[0] for _ in range(25)]
    assert np.array_equal(fused.f, two_pass.f)
    assert cd_f == cd_t


def test_work_arrays_never_overwrite_what_a_caller_holds():
    s = _solver3d(collision="regularized")
    mine = s.f.copy()
    s.f = mine                                        # set from outside
    snapshot = mine.copy()
    for _ in range(4):
        s._step()
    assert np.array_equal(mine, snapshot)             # never taken into the pool

    held = s.f
    kept = held.copy()
    s._step()
    assert np.array_equal(held, kept)                 # still intact one step on


def test_work_arrays_are_recycled_not_reallocated():
    s = _solver3d()
    s._step()
    s._step()
    ids = {id(b) for b in s._pool}
    for _ in range(6):
        s._step()
    assert {id(b) for b in s._pool} == ids
    assert len(s._pool) == 3


def test_2d_work_arrays_are_recycled():
    ny, nx = 20, 40
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - ny / 2) ** 2 + (xx - nx / 4) ** 2) <= 9
    s = Solver(Ny=ny, Nx=nx, solid=solid, omega=1.8, u0=0.05, D=6.0, backend="numba",
               collision="regularized")
    s._step()
    s._step()
    ids = {id(b) for b in s._pool}
    for _ in range(5):
        s._step()
    assert {id(b) for b in s._pool} == ids


# ---------------------------------------------------------------------------
# LES + wall model reach the end of a run
# ---------------------------------------------------------------------------

def test_3d_les_with_wall_model_runs_to_completion():
    """
    Regression: the surface observables at the end of ``run()`` passed the
    wall-model arguments to the subgrid-model builder and raised TypeError,
    so every 3D LES run crashed on its last step.
    """
    s = _solver3d(collision="regularized", les=True, wall_model=True)
    result = s.run(steps=6, check_every=10 ** 9, verbose=False)
    assert np.isfinite(result["Cd_mean"])
    fields = s.surface_fields()
    assert np.isfinite(np.asarray(fields["tau_wall"], dtype=float)).all()
    assert s._relaxation_field(s.f)[1]                  # the field the step used


def test_2d_surface_fields_use_the_wall_modelled_viscosity():
    ny, nx = 20, 40
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - ny / 2) ** 2 + (xx - nx / 4) ** 2) <= 9
    s = Solver(Ny=ny, Nx=nx, solid=solid, omega=1.8, u0=0.05, D=6.0, backend="numba",
               les=True, wall_model=True)
    for _ in range(5):
        s._step()
    assert s._relaxation_field(s.f)[1]
    fields = s.surface_fields()
    assert np.isfinite(np.asarray(fields["tau_wall"], dtype=float)).all()
