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


# ---------------------------------------------------------------------------
# The column voxelizer
# ---------------------------------------------------------------------------

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "stl"


def _cell_centres(nz, ny, nx):
    z, y, x = (np.arange(n) + 0.5 for n in (nz, ny, nx))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


@pytest.mark.parametrize("name", ["unit_cube", "cube_20mm", "cylinder", "sphere", "unit_sphere"])
def test_voxelizer_agrees_with_the_point_test_on_every_sample(name):
    from aero.geometry3d.mesh_mask import voxelize_mesh
    from aero.geometry3d.stl_prep import prepare_mesh_triangles

    # (a fit that keeps faces off the cell centres: a centre exactly on a face
    # is a tie, which the point test resolves by where its skewed ray lands)
    grid = (14, 16, 30)
    tris, _ = prepare_mesh_triangles(load_stl_triangles(str(SAMPLES / f"{name}.stl")), *grid, fit_frac=0.47)
    fast = voxelize_mesh(tris, *grid)
    slow = points_inside_mesh(_cell_centres(*grid), tris).reshape(grid)
    assert fast.sum() > 0
    assert np.array_equal(fast, slow)


def test_voxelizer_fills_exactly_the_cells_of_a_box_on_cell_faces():
    """Faces on whole numbers put the rays' columns exactly between two cells' centres."""
    from aero.geometry3d.mesh_mask import voxelize_mesh

    tris = []
    lo, hi = np.array([3.0, 2.0, 4.0]), np.array([9.0, 7.0, 6.0])        # x, y, z
    c = [(lo * (1 - np.array(bits)) + hi * np.array(bits)) for bits in
         [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    for f in [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 7), (0, 7, 3),
              (1, 5, 6), (1, 6, 2), (0, 1, 5), (0, 5, 4), (3, 2, 6), (3, 6, 7)]:
        tris.append([c[i] for i in f])
    solid = voxelize_mesh(np.array(tris), 10, 10, 12)
    want = np.zeros((10, 10, 12), dtype=bool)
    want[4:6, 2:7, 3:9] = True
    assert np.array_equal(solid, want)


def test_voxelizer_breaks_ties_on_cell_centres_the_same_way_every_time():
    """Faces through cell centres: the low face's cells are in, the high face's out, on every axis."""
    from aero.geometry3d.mesh_mask import voxelize_mesh

    lo, hi = np.array([3.5, 2.5, 4.5]), np.array([9.5, 7.5, 6.5])
    c = [(lo * (1 - np.array(bits)) + hi * np.array(bits)) for bits in
         [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    tris = [[c[i] for i in f] for f in [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 7), (0, 7, 3),
                                         (1, 5, 6), (1, 6, 2), (0, 1, 5), (0, 5, 4), (3, 2, 6), (3, 6, 7)]]
    solid = voxelize_mesh(np.array(tris), 10, 10, 12)
    want = np.zeros((10, 10, 12), dtype=bool)
    want[4:6, 2:7, 3:9] = True
    assert np.array_equal(solid, want)


def test_voxelizer_ignores_an_open_mesh_rather_than_flooding():
    """One stray triangle: an odd crossing count must not fill to the tunnel's end."""
    from aero.geometry3d.mesh_mask import voxelize_mesh

    tri = np.array([[[5.0, 0.0, 0.0], [5.0, 10.0, 0.0], [5.0, 0.0, 10.0]]])
    assert not voxelize_mesh(tri, 8, 8, 12).any()


def test_mesh_mask_uses_the_fast_voxelizer():
    import time

    mask = MeshMask(str(SAMPLES / "unit_sphere.stl"), fit_frac=0.3)
    t = time.perf_counter()
    solid = mask.mark_solid(48, 48, 96)
    assert time.perf_counter() - t < 2.0          # was ~13 s with a point test per cell
    assert solid.sum() > 1000
