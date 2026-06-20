"""3D obstacle from an STL mesh voxelized onto the LBM grid."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import Geometry3D
from .stl_io import load_stl_triangles
from .stl_prep import prepare_mesh_triangles


def _ray_triangle_hits(points: np.ndarray, tri: np.ndarray, ray: np.ndarray) -> np.ndarray:
    """
    Möller–Trumbore ray/triangle test for many origins.

    points : (M, 3)
    tri    : (3, 3)
    ray    : (3,) unit direction
    Returns bool (M,) — True when a forward hit exists.
    """
    eps = 1e-9
    v0, v1, v2 = tri
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(ray, edge2)
    det = edge1 @ pvec
    if abs(det) < eps:
        return np.zeros(len(points), dtype=bool)

    inv_det = 1.0 / det
    tvec = points - v0
    u = (tvec @ pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,j->i", qvec, ray) * inv_det
    t = np.einsum("ij,j->i", qvec, edge2) * inv_det
    return (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps)


def points_inside_mesh(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Parity ray cast; use a non-axis-aligned ray to avoid symmetric hits."""
    ray = np.array([1.0, 0.123456789, 0.987654321], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    inside = np.zeros(len(points), dtype=bool)
    for tri in triangles:
        hits = _ray_triangle_hits(points, tri, ray)
        inside ^= hits
    return inside


class MeshMask(Geometry3D):
    """
    Voxelize a watertight STL mesh onto the LBM grid.

    The mesh is auto-oriented (PCA → stream/span/thin), scaled to fit the
    tunnel cross-section, and placed with its bounding-box centre at
    (cx_frac·Nx, cy_frac·Ny, cz_frac·Nz).

    Parameters
    ----------
    path      : STL file path
    cx_frac, cy_frac, cz_frac : placement fractions
    fit_frac  : max cross-stream extent as fraction of min(Ny, Nz)
    mesh_orient : "auto" (PCA align to tunnel) or "none"
    scale     : optional manual scale multiplier applied after auto-fit
    """

    def __init__(
        self,
        path: str,
        cx_frac: float = 1.0 / 3.0,
        cy_frac: float = 0.5,
        cz_frac: float = 0.5,
        fit_frac: float = 0.35,
        mesh_orient: str = "auto",
        scale: Optional[float] = None,
    ):
        self.path = path
        self.cx_frac = cx_frac
        self.cy_frac = cy_frac
        self.cz_frac = cz_frac
        self.fit_frac = max(float(fit_frac), 0.05)
        self.mesh_orient = mesh_orient
        self.manual_scale = float(scale) if scale is not None else None
        self._triangles = load_stl_triangles(path)
        self._reference_length: Optional[float] = None

    def _transform_to_grid(
        self, Nz: int, Ny: int, Nx: int
    ) -> Tuple[np.ndarray, float]:
        return prepare_mesh_triangles(
            self._triangles,
            Nz,
            Ny,
            Nx,
            self.cx_frac,
            self.cy_frac,
            self.cz_frac,
            self.fit_frac,
            self.mesh_orient,
            self.manual_scale,
        )

    def mark_solid(self, Nz: int, Ny: int, Nx: int) -> np.ndarray:
        tris, ref_L = self._transform_to_grid(Nz, Ny, Nx)
        self._reference_length = ref_L

        z = np.arange(Nz, dtype=np.float64) + 0.5
        y = np.arange(Ny, dtype=np.float64) + 0.5
        x = np.arange(Nx, dtype=np.float64) + 0.5
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        inside = points_inside_mesh(points, tris)
        return inside.reshape(Nz, Ny, Nx)

    def center(self, Nz: int, Ny: int, Nx: int) -> Tuple[float, float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny, self.cz_frac * Nz

    def reference_length(self) -> float:
        if self._reference_length is None:
            raise RuntimeError("Call mark_solid() before reference_length()")
        return self._reference_length
