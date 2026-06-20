"""Abstract base class for 2D obstacle geometry."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Geometry(ABC):
    @abstractmethod
    def mark_solid(self, Ny: int, Nx: int) -> np.ndarray:
        """
        Return boolean mask where True = solid obstacle cell.

        Parameters
        ----------
        Ny, Nx : int — grid dimensions

        Returns
        -------
        solid : ndarray bool, shape (Ny, Nx)
        """
        ...

    @abstractmethod
    def center(self, Ny: int, Nx: int) -> Tuple[float, float]:
        """Return (cx, cy) obstacle center in lattice cells."""
        ...

    @abstractmethod
    def reference_length(self) -> float:
        """Characteristic length in lattice cells (diameter, chord, etc.)."""
        ...
