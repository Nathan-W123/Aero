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
