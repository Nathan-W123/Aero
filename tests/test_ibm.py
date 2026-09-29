"""Tests for Guo IBM and signed distance."""

import struct
from pathlib import Path

import numpy as np
import pytest

from aero.geometry3d.signed_distance import compute_phi_field
from aero.geometry3d.stl_io import load_stl_triangles
from aero.lbm.solver3d import Solver3D


def _write_binary_cube_stl(path: Path) -> None:
    tris = [
        [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        [(0, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(0, 0, 1), (1, 1, 1), (1, 0, 1)],
        [(0, 0, 1), (0, 1, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (0, 1, 1)],
        [(0, 0, 0), (0, 1, 1), (0, 0, 1)],
        [(1, 0, 0), (1, 0, 1), (1, 1, 1)],
        [(1, 0, 0), (1, 1, 1), (1, 1, 0)],
        [(0, 0, 0), (0, 0, 1), (1, 0, 1)],
        [(0, 0, 0), (1, 0, 1), (1, 0, 0)],
        [(0, 1, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 1, 0), (1, 1, 1), (0, 1, 1)],
    ]
    with path.open("wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for tri in tris:
            fh.write(struct.pack("<12f", 0, 0, 0, *tri[0], *tri[1], *tri[2]))
            fh.write(struct.pack("<H", 0))


def test_phi_field_shape(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_cube_stl(stl)
    tris = load_stl_triangles(str(stl))
    phi = compute_phi_field(tris, 16, 16, 32)
    assert phi.shape == (16, 16, 32)
    assert np.any(phi < 0)


def test_ibm_solver_short_run(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_cube_stl(stl)
    tris = load_stl_triangles(str(stl))
    phi = compute_phi_field(tris, 12, 12, 24)
    solid = phi <= 0
    solver = Solver3D(
        Nz=12, Ny=12, Nx=24, solid=solid, omega=1.0, u0=0.05, D=4.0,
        backend="numpy", ibm_enabled=True, phi=phi,
    )
    result = solver.run(steps=20, check_every=20, verbose=False)
    assert not np.isnan(result["Cd_mean"])


# ---------------------------------------------------------------------------
# The immersed body is a force, and its drag is the reaction to that force
# ---------------------------------------------------------------------------

def _cylinder_case(ny=120, nx=300, r=10.0, re=20.0):
    from aero.geometry.cylinder import Cylinder
    u0 = 0.05
    d = 2.0 * r
    omega = 1.0 / (3.0 * u0 * d / re + 0.5)
    geom = Cylinder(radius=r)
    return geom.mark_solid(ny, nx), geom.sdf_field(ny, nx), dict(omega=omega, u0=u0, D=d)


def test_ibm_body_has_no_bounce_back_links_and_collides_as_fluid():
    from aero.lbm.solver import Solver
    solid, phi, kw = _cylinder_case(ny=40, nx=80, r=5.0)
    s = Solver(Ny=40, Nx=80, solid=solid, phi=phi, ibm_enabled=True, backend="numpy", **kw)
    assert s.surface_links.shape[0] == 0
    assert not s._collide_solid.any()
    assert s.solid.any()                               # the geometry is still there


def test_ibm_needs_a_distance_field():
    from aero.lbm.solver import Solver
    solid, _, kw = _cylinder_case(ny=40, nx=80, r=5.0)
    with pytest.raises(ValueError, match="phi"):
        Solver(Ny=40, Nx=80, solid=solid, ibm_enabled=True, backend="numpy", **kw)
    with pytest.raises(ValueError, match="phi"):
        Solver3D(Nz=8, Ny=8, Nx=16, solid=np.zeros((8, 8, 16), bool), omega=1.5,
                 u0=0.05, D=4.0, backend="numpy", ibm_enabled=True)


def test_ibm_drag_matches_the_voxel_body_when_resolved():
    """
    A 20-cell cylinder at Re=20: the immersed body and the bounce-back body
    must agree to a few percent (measured -0.7%).  Before the fix the
    immersed one held a 1.5-cell band of fluid still and read its drag from
    momentum exchange inside that band, and came out ~40% low.
    """
    pytest.importorskip("numba")
    from aero.lbm.solver import Solver
    solid, phi, kw = _cylinder_case()
    common = dict(Ny=120, Nx=300, solid=solid, backend="numba", **kw)
    voxel = Solver(**common).run(steps=5000, check_every=10 ** 9, verbose=False)
    ibm = Solver(phi=phi, ibm_enabled=True, **common).run(steps=5000, check_every=10 ** 9, verbose=False)
    assert ibm["Cd_mean"] == pytest.approx(voxel["Cd_mean"], rel=0.05)
    assert np.isnan(ibm["Cd_p_mean"])                  # no surface to split it on
