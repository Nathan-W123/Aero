"""Axis-aligned rectangular box obstacle."""
import numpy as np
from typing import Optional, Tuple
from .base import Geometry3D


class Box(Geometry3D):
    """
    Axis-aligned box (rectangular prism) centered at fractional coordinates.

    Parameters
    ----------
    width   : float — streamwise extent (x) in lattice cells
    height  : float — vertical extent (y) in lattice cells
    depth   : float — spanwise extent (z) in lattice cells
    cx_frac, cy_frac, cz_frac : float — centre as fractions of Nx, Ny, Nz
    """

    def __init__(
        self,
        width:   float = 10.0,
        height:  float = 10.0,
        depth:   float = 10.0,
        cx_frac: float = 1.0/3.0,
        cy_frac: float = 0.5,
        cz_frac: float = 0.5,
    ):
        self.width   = width
        self.height  = height
        self.depth   = depth
        self.cx_frac = cx_frac
        self.cy_frac = cy_frac
        self.cz_frac = cz_frac

    def mark_solid(self, Nz: int, Ny: int, Nx: int) -> np.ndarray:
        cx, cy, cz = self.center(Nz, Ny, Nx)
        x = np.arange(Nx, dtype=np.float64)
        y = np.arange(Ny, dtype=np.float64)
        z = np.arange(Nz, dtype=np.float64)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        return (
            (np.abs(xx - cx) <= self.width  / 2.0)
          & (np.abs(yy - cy) <= self.height / 2.0)
          & (np.abs(zz - cz) <= self.depth  / 2.0)
        )

    def center(self, Nz: int, Ny: int, Nx: int) -> Tuple[float, float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny, self.cz_frac * Nz

    def reference_length(self) -> float:
        return self.height  # cross-stream dimension

    def sdf_field(self, Nz: int, Ny: int, Nx: int) -> Optional[np.ndarray]:
        cx, cy, cz = self.center(Nz, Ny, Nx)
        x = np.arange(Nx, dtype=np.float64) + 0.5
        y = np.arange(Ny, dtype=np.float64) + 0.5
        z = np.arange(Nz, dtype=np.float64) + 0.5
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        dx = np.abs(xx - cx) - self.width  / 2.0
        dy = np.abs(yy - cy) - self.height / 2.0
        dz = np.abs(zz - cz) - self.depth  / 2.0
        return (
            np.sqrt(
                np.maximum(dx, 0.0) ** 2
                + np.maximum(dy, 0.0) ** 2
                + np.maximum(dz, 0.0) ** 2
            )
            + np.minimum(np.maximum(np.maximum(dx, dy), dz), 0.0)
        )
