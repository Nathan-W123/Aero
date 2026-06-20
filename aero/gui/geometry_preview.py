"""Build 3D solid masks from GUI settings for pre-run viewport preview."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .state import GuiConfig


def solid_mask_for_3d_config(config: GuiConfig) -> Optional[np.ndarray]:
    """
    Voxelize the configured 3D obstacle on the current grid.

    Returns None when preview is unavailable (2D mode, missing STL, etc.).
    """
    if config.mode != "3d":
        return None

    params = config.params_3d
    try:
        nx = int(params.get("nx", "64"))
        ny = int(params.get("ny", "48"))
        nz = int(params.get("nz", "48"))
    except (TypeError, ValueError):
        return None

    shape = config.shape_3d
    cx_frac, cy_frac, cz_frac = 1.0 / 3.0, 0.5, 0.5

    if shape == "sphere":
        from aero.geometry3d.sphere import Sphere

        radius = float(params.get("radius", "10") or "10")
        geom = Sphere(radius=radius, cx_frac=cx_frac, cy_frac=cy_frac, cz_frac=cz_frac)
        return geom.mark_solid(nz, ny, nx)

    if shape == "box":
        from aero.geometry3d.box import Box

        geom = Box(
            width=float(params.get("width", "10") or "10"),
            height=float(params.get("height", "10") or "10"),
            depth=float(params.get("depth", "10") or "10"),
            cx_frac=cx_frac,
            cy_frac=cy_frac,
            cz_frac=cz_frac,
        )
        return geom.mark_solid(nz, ny, nx)

    if shape == "cylinder":
        from aero.geometry3d.cylinder3d import Cylinder3D

        geom = Cylinder3D(
            radius=float(params.get("radius", "10") or "10"),
            length=float(params.get("length", "20") or "20"),
            cx_frac=cx_frac,
            cy_frac=cy_frac,
            cz_frac=cz_frac,
        )
        return geom.mark_solid(nz, ny, nx)

    if shape == "mesh":
        from aero.geometry3d.mesh_preview import build_mesh_preview

        stl_path = params.get("stl_path", "")
        if not stl_path:
            return None
        preview = build_mesh_preview(
            stl_path,
            nz=nz,
            ny=ny,
            nx=nx,
            fit_frac=float(params.get("stl_fit", "0.35") or "0.35"),
            mesh_orient=params.get("mesh_orient", "auto") or "auto",
            mesh_rot_x=float(params.get("mesh_rot_x", "0") or "0"),
            mesh_rot_y=float(params.get("mesh_rot_y", "0") or "0"),
            mesh_rot_z=float(params.get("mesh_rot_z", "0") or "0"),
        )
        if preview is None:
            return None
        return preview.solid

    return None
