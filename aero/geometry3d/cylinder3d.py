"""Finite circular cylinder aligned with the spanwise z-axis."""
import numpy as np
from typing import Optional, Tuple
from .base import Geometry3D


class Cylinder3D(Geometry3D):
    """
    Finite cylinder centered at fractional coordinates.

    The cylinder axis is aligned with z, so the crossflow plane is x-y and the
    spanwise extent is controlled by `length`.

    Parameters
    ----------
    radius  : float — cylinder radius in lattice cells
    length  : float — spanwise length in lattice cells
    cx_frac, cy_frac, cz_frac : float — centre as fractions of Nx, Ny, Nz
    """

    def __init__(
        self,
        radius: float = 10.0,
        length: float = 20.0,
        cx_frac: float = 1.0 / 3.0,
        cy_frac: float = 0.5,
        cz_frac: float = 0.5,
    ):
        self.radius = radius
        self.length = length
        self.cx_frac = cx_frac
        self.cy_frac = cy_frac
        self.cz_frac = cz_frac

    def mark_solid(self, Nz: int, Ny: int, Nx: int) -> np.ndarray:
        cx, cy, cz = self.center(Nz, Ny, Nx)
        x = np.arange(Nx, dtype=np.float64)
        y = np.arange(Ny, dtype=np.float64)
        z = np.arange(Nz, dtype=np.float64)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        radial = (xx - cx) ** 2 + (yy - cy) ** 2 <= self.radius ** 2
        span = np.abs(zz - cz) <= self.length / 2.0
        return radial & span

    def center(self, Nz: int, Ny: int, Nx: int) -> Tuple[float, float, float]:
        return self.cx_frac * Nx, self.cy_frac * Ny, self.cz_frac * Nz

    def reference_length(self) -> float:
        return 2.0 * self.radius

    def sdf_field(self, Nz: int, Ny: int, Nx: int) -> Optional[np.ndarray]:
        cx, cy, cz = self.center(Nz, Ny, Nx)
        x = np.arange(Nx, dtype=np.float64) + 0.5
        y = np.arange(Ny, dtype=np.float64) + 0.5
        z = np.arange(Nz, dtype=np.float64) + 0.5
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        dr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - self.radius
        dz = np.abs(zz - cz) - self.length / 2.0
        # Exact SDF of finite cylinder (capped)
        return (
            np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
            + np.minimum(np.maximum(dr, dz), 0.0)
        )
