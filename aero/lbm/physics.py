"""Shared solver physics options for 2D/3D LBM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PhysicsOptions:
    collision: str = "bgk"
    trt_lambda: float = 0.25
    sponge_thickness: int = 0
    sponge_strength: float = 0.1
    les: bool = False
    les_cs: float = 0.16
    ibm_enabled: bool = False
    inlet_perturbation: float = 0.0

    def validate_collision(self) -> None:
        if self.collision not in ("bgk", "mrt", "trt"):
            raise ValueError(f"Unknown collision '{self.collision}'. Choose 'bgk', 'mrt', or 'trt'.")


def base_nu_from_omega(omega: float) -> float:
    tau = 1.0 / omega
    return (tau - 0.5) / 3.0


def inv_positive(rho: np.ndarray) -> np.ndarray:
    """
    1/rho where rho > 0, else 0 — without dividing by the masked entries.

    ``np.where(rho > 0, 1.0 / rho, 0.0)`` evaluates the reciprocal for every
    cell first, so a single empty cell raises a divide-by-zero RuntimeWarning
    and every call pays for a full extra pass.  ``where=`` skips those lanes.
    """
    out = np.zeros_like(rho, dtype=np.float64)
    return np.divide(1.0, rho, out=out, where=rho > 0.0)
