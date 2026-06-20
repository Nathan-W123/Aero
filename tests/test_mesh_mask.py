"""Tests for STL mesh voxelization."""

import struct
from pathlib import Path

import numpy as np
import pytest

from aero.geometry3d.mesh_mask import MeshMask, points_inside_mesh
from aero.geometry3d.stl_io import load_stl_triangles


def _write_binary_cube_stl(path: Path) -> None:
    """Unit cube [0,1]^3 as binary STL."""
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


def test_load_binary_stl(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_cube_stl(stl)
    tris = load_stl_triangles(str(stl))
    assert tris.shape == (12, 3, 3)


def test_point_inside_unit_cube(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_cube_stl(stl)
    tris = load_stl_triangles(str(stl))
    inside = points_inside_mesh(np.array([[0.5, 0.5, 0.5]]), tris)
    outside = points_inside_mesh(np.array([[1.5, 0.5, 0.5]]), tris)
    assert inside[0]
    assert not outside[0]


def test_mesh_mask_voxelizes_solid_cells(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_cube_stl(stl)
    mesh = MeshMask(str(stl), fit_frac=0.5)
    solid = mesh.mark_solid(Nz=24, Ny=24, Nx=48)
    assert solid.dtype == bool
    assert solid.shape == (24, 24, 48)
    assert 50 < solid.sum() < 5000
    assert mesh.reference_length() > 0
