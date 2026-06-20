"""STL orientation and tunnel blockage metrics for mesh geometry."""

from __future__ import annotations

import numpy as np


def orient_mesh_for_tunnel(triangles: np.ndarray) -> np.ndarray:
    """
    Center mesh at origin and rotate into tunnel axes: stream +x, span +y, thin +z.

    Uses PCA: smallest variance → vertical (z), largest → span (y), middle → stream (x).
    Works well for aircraft-like shapes (fuselage along x, wings along y).
    """
    verts = triangles.reshape(-1, 3)
    centre = verts.mean(axis=0)
    xc = verts - centre
    if len(xc) < 3:
        return triangles - centre

    _, evecs = np.linalg.eigh(np.cov(xc.T))
    thin = evecs[:, 0].copy()
    stream = evecs[:, 1].copy()
    span = evecs[:, 2].copy()
    if np.dot(np.cross(stream, span), thin) < 0.0:
        thin = -thin
    R = np.stack([stream, span, thin], axis=1)
    rotated = xc @ R
    return rotated.reshape(-1, 3, 3)


def scale_and_place_mesh(
    triangles: np.ndarray,
    Nz: int,
    Ny: int,
    Nx: int,
    cx_frac: float,
    cy_frac: float,
    cz_frac: float,
    fit_frac: float,
    manual_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Scale to cross-stream fit and translate centre to tunnel placement."""
    mins = triangles.reshape(-1, 3).min(axis=0)
    maxs = triangles.reshape(-1, 3).max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)

    target = fit_frac * min(Ny, Nz)
    cross = max(float(extent[1]), float(extent[2]))
    auto_scale = target / cross
    scale = auto_scale if manual_scale is None else auto_scale * manual_scale

    cx = cx_frac * Nx
    cy = cy_frac * Ny
    cz = cz_frac * Nz

    tris = triangles * scale
    tris[:, :, 0] += cx
    tris[:, :, 1] += cy
    tris[:, :, 2] += cz

    ref_L = max(float(extent[1]), float(extent[2])) * scale
    ref_L = max(ref_L, 1.0)
    return tris, ref_L


def prepare_mesh_triangles(
    triangles: np.ndarray,
    Nz: int,
    Ny: int,
    Nx: int,
    cx_frac: float = 1.0 / 3.0,
    cy_frac: float = 0.5,
    cz_frac: float = 0.5,
    fit_frac: float = 0.35,
    mesh_orient: str = "auto",
    manual_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Orient (optional), scale, and place triangles on the LBM grid."""
    tris = triangles
    if mesh_orient == "auto":
        tris = orient_mesh_for_tunnel(tris)
    elif mesh_orient not in ("none", ""):
        raise ValueError(f"Unknown mesh_orient '{mesh_orient}'. Use 'auto' or 'none'.")
    return scale_and_place_mesh(
        tris, Nz, Ny, Nx, cx_frac, cy_frac, cz_frac, fit_frac, manual_scale
    )


def compute_frontal_blockage(solid: np.ndarray) -> tuple[float, float, float]:
    """
    Frontal blockage from a voxelized solid mask (Nz, Ny, Nx).

    Returns (blockage_y, blockage_z, max_blockage) using the largest y/z
    projected extent at any streamwise x station.
    """
    nz, ny, nx = solid.shape
    max_by = 0.0
    max_bz = 0.0
    for xi in range(nx):
        slab = solid[:, :, xi]
        if not slab.any():
            continue
        zs, ys = np.where(slab)
        y_ext = (ys.max() - ys.min() + 1) / max(ny, 1)
        z_ext = (zs.max() - zs.min() + 1) / max(nz, 1)
        max_by = max(max_by, float(y_ext))
        max_bz = max(max_bz, float(z_ext))
    return max_by, max_bz, max(max_by, max_bz)


def mesh_midplane_slice(solid: np.ndarray, axis: str = "y") -> np.ndarray:
    """2D slice of solid mask for preview (streamwise x on columns)."""
    nz, ny, nx = solid.shape
    if axis == "y":
        mid = ny // 2
        return solid[:, mid, :].astype(np.float64)
    if axis == "z":
        mid = nz // 2
        return solid[mid, :, :].astype(np.float64)
    mid = nx // 2
    return solid[:, :, mid].astype(np.float64)
