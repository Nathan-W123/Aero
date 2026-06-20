"""Abstract base class for 3D geometry obstacles."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


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
