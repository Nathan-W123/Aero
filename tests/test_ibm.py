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
