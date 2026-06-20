"""
Convex or concave polygon obstacle, defined by vertices in fractional grid coordinates.

Uses matplotlib.path.Path for point-in-polygon testing — no scipy dependency.
Vertices are given as (x_frac, y_frac) pairs where 0.0 maps to the left/bottom
edge and 1.0 maps to the right/top edge of the domain.
"""

import numpy as np
from matplotlib.path import Path
from .base import Geometry
from typing import List, Tuple


class Polygon(Geometry):
    """
    Arbitrary 2D polygon obstacle.

    Parameters
    ----------
    vertices : list of (x_frac, y_frac) tuples — vertices in [0,1]^2 coords,
               given in order (CW or CCW both work).
    """

    def __init__(self, vertices: List[Tuple[float, float]]):
        if len(vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices.")
        self.vertices = list(vertices)

    def mark_solid(self, Ny: int, Nx: int) -> np.ndarray:
        # Convert fractional coords to lattice cell coords
        verts_px = [(x * Nx, y * Ny) for x, y in self.vertices]

        # Explicit MOVETO/LINETO/CLOSEPOLY codes — required for correct
        # contains_points() results; Path(verts, closed=True) misses ~half
        # of boundary-adjacent cells due to a matplotlib edge-detection quirk.
        n = len(verts_px)
        verts_closed = list(verts_px) + [verts_px[0]]
        codes = [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
        path = Path(verts_closed, codes)

        x = np.arange(Nx)
        y = np.arange(Ny)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
        solid = path.contains_points(points).reshape(Ny, Nx)
        return solid

    def center(self, Ny: int, Nx: int) -> Tuple[float, float]:
        xs = [v[0] * Nx for v in self.vertices]
        ys = [v[1] * Ny for v in self.vertices]
        return float(np.mean(xs)), float(np.mean(ys))

    def reference_length(self) -> float:
        """Bounding-box diagonal in fractional units — caller must scale by Nx or Ny."""
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return float(max(max(xs) - min(xs), max(ys) - min(ys)))
