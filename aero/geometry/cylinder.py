"""Circular cylinder obstacle."""

import numpy as np
from typing import Tuple
from .base import Geometry


class Cylinder(Geometry):
    """
    Circular cylinder centered at (cx_frac*Nx, cy_frac*Ny).

    Parameters
    ----------
    radius : float   — radius in lattice cells
    cx_frac : float  — x center as fraction of Nx (default 1/3)
    cy_frac : float  — y center as fraction of Ny (default 1/2)
    """

    def __init__(self, radius: float = 20.0, cx_frac: float = 1/3, cy_frac: float = 0.5):
        self.radius = radius
        self.cx_frac = cx_frac
        self.cy_frac = cy_frac

    def mark_solid(self, Ny: int, Nx: int) -> np.ndarray:
        cx, cy = self.center(Ny, Nx)
        x = np.arange(Nx, dtype=np.float64)
        y = np.arange(Ny, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= self.radius ** 2

    def center(self, Ny: int, Nx: int) -> Tuple[float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny

    def reference_length(self) -> float:
        return 2.0 * self.radius
