"""Tests for STL orientation and blockage metrics."""

import numpy as np

from aero.geometry3d.stl_prep import (
    orient_mesh_for_tunnel,
    compute_frontal_blockage,
    prepare_mesh_triangles,
)
from aero.geometry3d.mesh_mask import MeshMask


def _axis_aligned_box(dx: float, dy: float, dz: float) -> np.ndarray:
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = dx, dy, dz
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 7), (0, 7, 3), (1, 5, 6), (1, 6, 2),
        (0, 1, 5), (0, 5, 4), (3, 2, 6), (3, 6, 7),
    ]
    tris = np.array([[corners[a], corners[b], corners[c]] for a, b, c in faces], dtype=np.float64)
    return tris


def test_orient_maps_long_wingspan_to_y():
    # Thin in z, long in y, medium in x
    tris = _axis_aligned_box(5.0, 20.0, 2.0)
    oriented = orient_mesh_for_tunnel(tris)
    verts = oriented.reshape(-1, 3)
    extent = verts.max(axis=0) - verts.min(axis=0)
    order = np.argsort(extent)
    assert order[2] == 1  # largest extent → y (span)
    assert order[0] == 2  # smallest → z (thin)


def test_frontal_blockage_from_sphere_voxel():
    from aero.geometry3d.sphere import Sphere

    solid = Sphere(radius=8, cx_frac=0.33, cy_frac=0.5, cz_frac=0.5).mark_solid(32, 32, 64)
    by, bz, blk = compute_frontal_blockage(solid)
    assert 0.2 < blk < 0.6
    assert blk >= by and blk >= bz


def test_manual_rotation_about_z_swaps_xy_extent():
    from aero.geometry3d.stl_prep import apply_mesh_rotation

    tris = _axis_aligned_box(10.0, 30.0, 4.0)
    verts = tris.reshape(-1, 3)
    centred = (verts - verts.mean(axis=0)).reshape(-1, 3, 3)
    rotated = apply_mesh_rotation(centred, 0.0, 0.0, 90.0)
    ext0 = centred.reshape(-1, 3).max(axis=0) - centred.reshape(-1, 3).min(axis=0)
    ext90 = rotated.reshape(-1, 3).max(axis=0) - rotated.reshape(-1, 3).min(axis=0)
    assert abs(ext90[0] - ext0[1]) < 0.5
    assert abs(ext90[1] - ext0[0]) < 0.5


def test_mesh_mask_orient_auto(tmp_path):
    stl = tmp_path / "box.stl"
    tris = _axis_aligned_box(10.0, 30.0, 4.0)
    # write minimal ascii stl
    lines = ["solid box"]
    for tri in tris:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for v in tri:
            lines.append(f"      vertex {v[0]} {v[1]} {v[2]}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid")
    stl.write_text("\n".join(lines))

    mask = MeshMask(str(stl), fit_frac=0.4, mesh_orient="auto")
    solid = mask.mark_solid(24, 24, 48)
    assert solid.sum() > 0
    _, _, blk = compute_frontal_blockage(solid)
    assert 0.05 < blk < 0.95
