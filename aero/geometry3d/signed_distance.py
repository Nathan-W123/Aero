"""
True narrow-band signed-distance field from an STL triangle mesh.

Replaces the earlier coarse ±0.5 approximation with exact Euclidean distance
to the nearest triangle surface for cells within *band_width* voxels of the
obstacle.  Far-field cells receive ±far_value so the IBM forcing is never
activated outside the band.

Math
----
Closest point on triangle (A, B, C) to query P is found by parameterising
the triangle as  Q(s,t) = A + s·e0 + t·e1  (e0 = B−A, e1 = C−A) and
minimising |P−Q|² over the feasible region {s≥0, t≥0, s+t≤1}.  The
constrained optimum lies in one of seven Voronoi regions (3 vertices,
3 edges, 1 interior). All N query points are handled in one vectorised
NumPy pass per triangle; triangles are iterated to find the global minimum.

Sign convention: phi < 0 inside mesh (solid), phi > 0 outside (fluid).
|phi| = Euclidean distance in lattice cells to the nearest surface.

API
---
compute_phi_field(triangles, nz, ny, nx, ...)  →  ndarray (nz, ny, nx)
"""

from __future__ import annotations

import numpy as np

from .mesh_mask import points_inside_mesh
from .stl_prep import prepare_mesh_triangles


# ---------------------------------------------------------------------------
# Point-to-triangle squared distance (vectorised over N query points)
# ---------------------------------------------------------------------------

def _sq_dist_points_to_triangle(
    P: np.ndarray,   # (N, 3)
    A: np.ndarray,   # (3,)
    B: np.ndarray,   # (3,)
    C: np.ndarray,   # (3,)
) -> np.ndarray:
    """
    Squared Euclidean distance from each of N points to triangle (A, B, C).

    Uses the Eberly (2006) parameterised-quadratic / Voronoi-region method.
    All N points are processed simultaneously via NumPy broadcasting.

    Returns
    -------
    sq_dist : (N,) float64
    """
    e0 = B - A          # (3,)
    e1 = C - A          # (3,)
    diff = A - P        # (N, 3)   i.e. A − P for each point

    a  = float(e0 @ e0)
    b  = float(e0 @ e1)
    c  = float(e1 @ e1)
    d  = diff @ e0      # (N,)  = (A−P)·e0
    e  = diff @ e1      # (N,)  = (A−P)·e1

    det = a * c - b * b

    if det < 1e-14:
        # Degenerate triangle: distance to the longer of the two main edges
        if a >= c:
            t = np.clip(-d / max(a, 1e-30), 0.0, 1.0)
            closest = A + t[:, None] * e0
        else:
            t = np.clip(-e / max(c, 1e-30), 0.0, 1.0)
            closest = A + t[:, None] * e1
        dv = P - closest
        return (dv * dv).sum(axis=1)

    # Unnormalised barycentric numerators for the interior minimum
    s = b * e - c * d   # (N,)
    t = b * d - a * e   # (N,)

    # ---- Voronoi region classification (7 regions) ----
    # Default: interior solution Q = A + (s/det)*e0 + (t/det)*e1
    inv_det = 1.0 / det
    closest = A + (s * inv_det)[:, None] * e0 + (t * inv_det)[:, None] * e1

    # Region flags
    beyond  = (s + t) > det   # beyond edge B−C
    neg_s   = s < 0.0
    neg_t   = t < 0.0

    # Region 6: s<0 and t<0  → vertex A
    m6 = neg_s & neg_t
    closest[m6] = A

    # Region 5: beyond B−C and t<0  → edge A−B (t=0)
    m5 = beyond & neg_t
    s5 = np.clip(-d[m5] / max(a, 1e-30), 0.0, 1.0)
    closest[m5] = A + s5[:, None] * e0

    # Region 4: beyond B−C and s<0  → edge A−C (s=0)
    m4 = beyond & neg_s
    t4 = np.clip(-e[m4] / max(c, 1e-30), 0.0, 1.0)
    closest[m4] = A + t4[:, None] * e1

    # Region 3: t<0 only  → edge A−B (t=0, s in [0,1])
    m3 = neg_t & ~neg_s & ~beyond
    s3 = np.clip(-d[m3] / max(a, 1e-30), 0.0, 1.0)
    closest[m3] = A + s3[:, None] * e0

    # Region 2: s<0 only  → edge A−C (s=0, t in [0,1])
    m2 = neg_s & ~neg_t & ~beyond
    t2 = np.clip(-e[m2] / max(c, 1e-30), 0.0, 1.0)
    closest[m2] = A + t2[:, None] * e1

    # Region 1: beyond B−C only  → edge B−C
    m1 = beyond & ~neg_s & ~neg_t
    if m1.any():
        BC   = C - B
        bc2  = float(BC @ BC)
        if bc2 > 1e-14:
            PB   = P[m1] - B
            u1   = np.clip((PB @ BC) / bc2, 0.0, 1.0)
            closest[m1] = B + u1[:, None] * BC
        else:
            closest[m1] = B

    dv = P - closest
    return (dv * dv).sum(axis=1)


# ---------------------------------------------------------------------------
# Narrow-band dilation helper
# ---------------------------------------------------------------------------

def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    """Expand a 3-D boolean mask by `steps` voxels in all 6 face-neighbours."""
    result = mask.copy()
    for _ in range(steps):
        nxt = result.copy()
        nxt[1:, :, :] |= result[:-1, :, :]
        nxt[:-1, :, :] |= result[1:, :, :]
        nxt[:, 1:, :] |= result[:, :-1, :]
        nxt[:, :-1, :] |= result[:, 1:, :]
        nxt[:, :, 1:] |= result[:, :, :-1]
        nxt[:, :, :-1] |= result[:, :, 1:]
        result = nxt
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_phi_field(
    triangles: np.ndarray,
    nz: int,
    ny: int,
    nx: int,
    cx_frac: float = 1.0 / 3.0,
    cy_frac: float = 0.5,
    cz_frac: float = 0.5,
    fit_frac: float = 0.35,
    mesh_orient: str = "auto",
    band_width: int = 3,
    far_value: float = 16.0,
) -> np.ndarray:
    """
    Compute a true narrow-band signed distance field on cell centres.

    phi < 0  →  inside mesh (solid)
    phi > 0  →  outside mesh (fluid)
    |phi|    →  Euclidean distance to nearest triangle (lattice cells)

    Cells farther than `band_width` voxels from the surface receive ±far_value
    and are never activated by the IBM band test `0 < phi <= delta`.

    Parameters
    ----------
    triangles   : (T, 3, 3) triangle vertex arrays in model coordinates
    nz, ny, nx  : grid dimensions
    cx_frac, cy_frac, cz_frac : obstacle centre as fractions of grid size
    fit_frac    : cross-stream extent as fraction of min(ny, nz)
    mesh_orient : "auto" (PCA) or "none"
    band_width  : narrow-band half-width in voxels (default 3)
    far_value   : phi magnitude for far-field cells (default 16.0)

    Returns
    -------
    phi : (nz, ny, nx) float64
    """
    # 1. Orient, scale, place triangles in grid space
    tris, _ = prepare_mesh_triangles(
        triangles, nz, ny, nx, cx_frac, cy_frac, cz_frac, fit_frac, mesh_orient
    )

    # 2. Build grid cell-centre coordinates  (N, 3) in (x, y, z) order
    z = np.arange(nz, dtype=np.float64) + 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    x = np.arange(nx, dtype=np.float64) + 0.5
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")          # (nz, ny, nx)
    all_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])  # (N, 3)

    # 3. Inside/outside classification via parity ray cast
    inside = points_inside_mesh(all_pts, tris).reshape(nz, ny, nx)

    # 4. Narrow-band: cells within band_width of the surface boundary
    near_surface = _dilate(inside, band_width) & _dilate(~inside, band_width)

    # 5. Initialise far-field values
    phi = np.where(inside, -far_value, far_value).astype(np.float64)

    # 6. Exact distance for narrow-band cells only
    band_idx = np.argwhere(near_surface)     # (M, 3): indices [iz, iy, ix]
    if len(band_idx) == 0:
        return phi

    iz, iy, ix = band_idx[:, 0], band_idx[:, 1], band_idx[:, 2]
    band_pts = np.column_stack([
        xx[iz, iy, ix],
        yy[iz, iy, ix],
        zz[iz, iy, ix],
    ])                                        # (M, 3) in (x, y, z)

    min_sq = np.full(len(band_pts), np.inf, dtype=np.float64)

    for tri in tris:
        # tri[v] = (x, y, z) vertex in grid coords
        sq = _sq_dist_points_to_triangle(band_pts, tri[0], tri[1], tri[2])
        np.minimum(min_sq, sq, out=min_sq)

    dist = np.sqrt(np.maximum(min_sq, 0.0))
    sign = np.where(inside[iz, iy, ix], -1.0, 1.0)
    phi[iz, iy, ix] = sign * dist

    return phi
