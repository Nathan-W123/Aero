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


#: Where the voxelizer samples each cell: its centre, shifted by these few
#: ten-millionths of a cell.  Far too little to move a surface, but a mesh
#: whose coordinates land on half-integers (an axis-aligned box of whole-cell
#: size does) would otherwise put rays exactly on its edges, where a crossing
#: can count twice or not at all, and cell centres exactly on its faces,
#: where round-off decides the side.  With the shift a centre on a face is
#: always on the same side of it.
_RAY_JITTER_X = 3.71e-7
_RAY_JITTER_Y = 1.37e-7
_RAY_JITTER_Z = 2.91e-7


def voxelize_mesh(triangles: np.ndarray, nz: int, ny: int, nx: int) -> np.ndarray:
    """
    Solid mask of a closed triangle mesh in grid coordinates, shape (nz, ny, nx).

    Cell (k, j, i) is solid when its centre (i + 1/2, j + 1/2, k + 1/2), in
    (x, y, z) -- shifted by a hair, see ``_RAY_JITTER_X`` -- is inside the mesh.  One ray per (j, k) column, along +x: the
    surface crossings in that column, sorted, pair up into inside spans
    (parity), and each span fills the cells whose centres it covers.  Each
    triangle is tested only against the columns under its own y-z footprint,
    so the cost goes with the mesh's surface rather than with triangles x
    cells; it gives what :func:`points_inside_mesh` gives at the cell
    centres, a thousand times faster on the sample meshes.

    A column with an odd number of crossings (a mesh that is not closed)
    drops its last one rather than filling to the end of the tunnel.
    """
    tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    solid = np.zeros((nz, ny, nx), dtype=bool)
    if tri.size == 0:
        return solid
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    # which columns each triangle's y-z footprint can cover: centres j + 1/2 + jitter
    ys, zs = tri[:, :, 1], tri[:, :, 2]
    j0 = np.clip(np.ceil(ys.min(1) - 0.5 - _RAY_JITTER_Y), 0, ny).astype(np.int64)
    j1 = np.clip(np.floor(ys.max(1) - 0.5 - _RAY_JITTER_Y), -1, ny - 1).astype(np.int64)
    k0 = np.clip(np.ceil(zs.min(1) - 0.5 - _RAY_JITTER_Z), 0, nz).astype(np.int64)
    k1 = np.clip(np.floor(zs.max(1) - 0.5 - _RAY_JITTER_Z), -1, nz - 1).astype(np.int64)
    wj, wk = np.maximum(j1 - j0 + 1, 0), np.maximum(k1 - k0 + 1, 0)
    count = wj * wk
    # twice the signed footprint area; a triangle edge-on to the rays has none
    area = (b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2])
    count[np.abs(area) < 1e-14] = 0
    total = int(count.sum())
    if total == 0:
        return solid

    # one entry per (triangle, candidate column)
    t = np.repeat(np.arange(len(tri)), count)
    local = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    jj = j0[t] + local % wj[t]
    kk = k0[t] + local // wj[t]
    py = jj + 0.5 + _RAY_JITTER_Y
    pz = kk + 0.5 + _RAY_JITTER_Z
    at, bt, ct, d = a[t], b[t], c[t], area[t]
    w1 = ((py - at[:, 1]) * (ct[:, 2] - at[:, 2]) - (ct[:, 1] - at[:, 1]) * (pz - at[:, 2])) / d
    w2 = ((bt[:, 1] - at[:, 1]) * (pz - at[:, 2]) - (py - at[:, 1]) * (bt[:, 2] - at[:, 2])) / d
    w0 = 1.0 - w1 - w2
    hit = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    col = (kk * ny + jj)[hit]
    x = (w0 * at[:, 0] + w1 * bt[:, 0] + w2 * ct[:, 0])[hit]
    if col.size == 0:
        return solid

    # sort the crossings along each column and pair them up
    order = np.lexsort((x, col))
    col, x = col[order], x[order]
    first = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
    run = np.diff(np.r_[first, col.size])
    rank = np.arange(col.size) - np.repeat(first, run)
    enter = (rank % 2 == 0) & (rank + 1 < np.repeat(run, run))          # an exit follows
    starts = np.flatnonzero(enter)
    i0 = np.clip(np.ceil(x[starts] - 0.5 - _RAY_JITTER_X), 0, nx).astype(np.int64)
    i1 = np.clip(np.ceil(x[starts + 1] - 0.5 - _RAY_JITTER_X), 0, nx).astype(np.int64)
    fill = np.zeros((nz * ny, nx + 1), dtype=np.int32)
    np.add.at(fill, (col[starts], i0), 1)
    np.add.at(fill, (col[starts], i1), -1)
    return (np.cumsum(fill, axis=1)[:, :nx] > 0).reshape(nz, ny, nx)


class MeshMask(Geometry3D):
    """
    Voxelize a watertight STL mesh onto the LBM grid.

    The mesh is auto-oriented (PCA → stream/span/thin), scaled to fit the
    tunnel cross-section, and placed with its bounding-box centre at
    (cx_frac·Nx, cy_frac·Ny, cz_frac·Nz).

    Parameters
    ----------
    path      : STL file path (or None, with ``triangles``)
    cx_frac, cy_frac, cz_frac : placement fractions
    fit_frac  : max cross-stream extent as fraction of min(Ny, Nz)
    mesh_orient : "auto" (PCA align to tunnel) or "none"
    scale     : optional manual scale multiplier applied after auto-fit
    triangles : the mesh itself, (T, 3, 3), instead of a file -- an upload
    """

    def __init__(
        self,
        path: Optional[str] = None,
        cx_frac: float = 1.0 / 3.0,
        cy_frac: float = 0.5,
        cz_frac: float = 0.5,
        fit_frac: float = 0.35,
        mesh_orient: str = "auto",
        scale: Optional[float] = None,
        mesh_rot_x: float = 0.0,
        mesh_rot_y: float = 0.0,
        mesh_rot_z: float = 0.0,
        triangles: Optional[np.ndarray] = None,
    ):
        if (path is None) == (triangles is None):
            raise ValueError("MeshMask needs exactly one of path and triangles")
        self.path = path
        self.cx_frac = cx_frac
        self.cy_frac = cy_frac
        self.cz_frac = cz_frac
        self.fit_frac = max(float(fit_frac), 0.05)
        self.mesh_orient = mesh_orient
        self.mesh_rot_x = float(mesh_rot_x)
        self.mesh_rot_y = float(mesh_rot_y)
        self.mesh_rot_z = float(mesh_rot_z)
        self.manual_scale = float(scale) if scale is not None else None
        self._triangles = (load_stl_triangles(path) if triangles is None
                           else np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3))
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
            self.mesh_rot_x,
            self.mesh_rot_y,
            self.mesh_rot_z,
        )

    def mark_solid(self, Nz: int, Ny: int, Nx: int) -> np.ndarray:
        tris, ref_L = self._transform_to_grid(Nz, Ny, Nx)
        self._reference_length = ref_L
        return voxelize_mesh(tris, Nz, Ny, Nx)

    def center(self, Nz: int, Ny: int, Nx: int) -> Tuple[float, float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny, self.cz_frac * Nz

    def reference_length(self) -> float:
        if self._reference_length is None:
            raise RuntimeError("Call mark_solid() before reference_length()")
        return self._reference_length
