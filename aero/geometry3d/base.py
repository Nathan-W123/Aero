"""Abstract base class for 3D geometry obstacles."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class Geometry3D(ABC):
    @abstractmethod
    def mark_solid(self, Nz: int, Ny: int, Nx: int) -> np.ndarray:
        """Return bool array (Nz, Ny, Nx): True = solid cell."""

    @abstractmethod
    def center(self, Nz: int, Ny: int, Nx: int) -> Tuple[float, float, float]:
        """Return (cx, cy, cz) in lattice coordinates."""

    @abstractmethod
    def reference_length(self) -> float:
        """Characteristic length D used to compute Re and force coefficients."""

    def sdf_field(self, Nz: int, Ny: int, Nx: int) -> Optional[np.ndarray]:
        """
        Signed-distance field on cell centres.  Returns None if not implemented.
        phi < 0 inside obstacle, phi > 0 outside, |phi| = distance in lattice cells.
        """
        return None
