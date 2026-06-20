"""Voxelize an STL for GUI/CLI preview without running a simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .mesh_mask import MeshMask
from .stl_prep import compute_frontal_blockage, mesh_midplane_slice


@dataclass
class MeshPreviewResult:
    solid: np.ndarray
    blockage_y: float
    blockage_z: float
    blockage: float
    reference_length: float
    solid_cells: int
    midplane: np.ndarray


def build_mesh_preview(
    stl_path: str,
    *,
    nz: int,
    ny: int,
    nx: int,
    cx_frac: float = 1.0 / 3.0,
    cy_frac: float = 0.5,
    cz_frac: float = 0.5,
    fit_frac: float = 0.35,
    mesh_orient: str = "auto",
) -> Optional[MeshPreviewResult]:
    path = Path(stl_path)
    if not path.is_file():
        return None
    geom = MeshMask(
        str(path),
        cx_frac=cx_frac,
        cy_frac=cy_frac,
        cz_frac=cz_frac,
        fit_frac=fit_frac,
        mesh_orient=mesh_orient,
    )
    solid = geom.mark_solid(nz, ny, nx)
    by, bz, blk = compute_frontal_blockage(solid)
    return MeshPreviewResult(
        solid=solid,
        blockage_y=by,
        blockage_z=bz,
        blockage=blk,
        reference_length=geom.reference_length(),
        solid_cells=int(solid.sum()),
        midplane=mesh_midplane_slice(solid, axis="y"),
    )
