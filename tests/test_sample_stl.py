"""Load bundled sample STL fixtures."""

from pathlib import Path

import pytest

from aero.geometry3d.stl_io import load_stl_triangles, triangle_bounds
from aero.geometry3d.mesh_mask import MeshMask

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples" / "stl"


@pytest.mark.parametrize("name", [
    "unit_sphere.stl",
    "unit_cube.stl",
    "cylinder.stl",
    "sphere.stl",
    "simple_plane.stl",
    "tunnel_plane.stl",
])
def test_sample_stl_loads(name):
    path = SAMPLES_DIR / name
    assert path.is_file(), f"Missing sample STL: {path}"
    tris = load_stl_triangles(str(path))
    assert tris.shape[1:] == (3, 3)
    assert len(tris) > 0
    lo, hi = triangle_bounds(tris)
    assert (hi > lo).all()


def test_sample_sphere_mesh_mask():
    path = SAMPLES_DIR / "unit_sphere.stl"
    mask = MeshMask(str(path), fit_frac=0.25)
    solid = mask.mark_solid(16, 16, 32)
    assert solid.sum() > 0
    assert solid.sum() < solid.size


def test_tunnel_plane_is_one_closed_outward_surface_that_survives_the_grid():
    """It was built to: wings thick enough to exist ~48 cells across."""
    import numpy as np
    from aero.geometry3d.stl_prep import prepare_mesh_triangles
    from aero.geometry3d.mesh_mask import voxelize_mesh
    from aero.web.server import _open_edges

    tris = load_stl_triangles(str(SAMPLES_DIR / "tunnel_plane.stl"))
    assert _open_edges(tris) == 0
    volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    assert volume > 0                                     # normals point out
    # as in the file: x downstream, y up, z across; 48 cells across the span
    placed, _ = prepare_mesh_triangles(tris, 96, 72, 144, fit_frac=48 / 72, mesh_orient="none")
    solid = voxelize_mesh(placed, 96, 72, 144)
    zs = np.nonzero(solid.any(axis=(1, 2)))[0]
    assert zs.max() - zs.min() + 1 >= 46                  # the wings reach their tips
