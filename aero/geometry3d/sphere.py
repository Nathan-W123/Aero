"""Spherical obstacle (canonical 3D validation shape)."""
import numpy as np
from typing import Tuple
from .base import Geometry3D


class Sphere(Geometry3D):
    """
    Sphere centered at (cx_frac*Nx, cy_frac*Ny, cz_frac*Nz) with given radius.

    Reference Cd for sphere at Re=100 ≈ 1.0–1.1 (literature).

    Parameters
    ----------
    radius  : float — radius in lattice cells
    cx_frac, cy_frac, cz_frac : float — centre as fractions of Nx, Ny, Nz
    """

    def __init__(
        self,
        radius: float = 10.0,
        cx_frac: float = 1.0/3.0,
        cy_frac: float = 0.5,
        cz_frac: float = 0.5,
    ):
        self.radius   = radius
        self.cx_frac  = cx_frac
        self.cy_frac  = cy_frac
        self.cz_frac  = cz_frac

    def mark_solid(self, Nz: int, Ny: int, Nx: int) -> np.ndarray:
        cx, cy, cz = self.center(Nz, Ny, Nx)
        x = np.arange(Nx, dtype=np.float64)
        y = np.arange(Ny, dtype=np.float64)
        z = np.arange(Nz, dtype=np.float64)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        return (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 <= self.radius**2

    def center(self, Nz: int, Ny: int, Nx: int) -> Tuple[float, float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny, self.cz_frac * Nz

    def reference_length(self) -> float:
        return 2.0 * self.radius  # diameter
