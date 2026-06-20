"""Rectangular obstacle."""

import numpy as np
from typing import Optional, Tuple
from .base import Geometry


class Rectangle(Geometry):
    """
    Axis-aligned rectangle centered at (cx_frac*Nx, cy_frac*Ny).

    Parameters
    ----------
    width  : float — full width in lattice cells (streamwise)
    height : float — full height in lattice cells (cross-stream)
    cx_frac, cy_frac : float — center as fractions of Nx, Ny
    """

    def __init__(
        self,
        width: float = 40.0,
        height: float = 20.0,
        cx_frac: float = 1/3,
        cy_frac: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.cx_frac = cx_frac
        self.cy_frac = cy_frac

    def mark_solid(self, Ny: int, Nx: int) -> np.ndarray:
        cx, cy = self.center(Ny, Nx)
        x = np.arange(Nx, dtype=np.float64)
        y = np.arange(Ny, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        return (np.abs(xx - cx) <= self.width / 2) & (np.abs(yy - cy) <= self.height / 2)

    def center(self, Ny: int, Nx: int) -> Tuple[float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny

    def reference_length(self) -> float:
        return self.height  # cross-stream dimension is the aerodynamic reference

    def sdf_field(self, Ny: int, Nx: int) -> Optional[np.ndarray]:
        cx, cy = self.center(Ny, Nx)
        x = np.arange(Nx, dtype=np.float64) + 0.5
        y = np.arange(Ny, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(x, y)
        dx = np.abs(xx - cx) - self.width / 2.0
        dy = np.abs(yy - cy) - self.height / 2.0
        # Exact SDF of axis-aligned box
        return (
            np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
            + np.minimum(np.maximum(dx, dy), 0.0)
        )
