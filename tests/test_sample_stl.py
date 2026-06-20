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
