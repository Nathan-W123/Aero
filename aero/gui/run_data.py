"""Load live chart and preview data for the GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from aero.run_manifest import load_run_manifest


def resolve_output_dir(config, repo_root: Path) -> Path:
    raw = config.output_dir_3d if config.mode == "3d" else config.output_dir_2d
    path = Path(raw)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def latest_run_manifest(config, repo_root: Path) -> Optional[Dict[str, Any]]:
    return load_run_manifest(resolve_output_dir(config, repo_root))


def latest_volume_file(config, repo_root: Path) -> Optional[Path]:
    output_dir = resolve_output_dir(config, repo_root)
    if config.mode != "3d" or not output_dir.is_dir():
        return None
    volumes = sorted(output_dir.glob("*.vti"), key=lambda p: p.stat().st_mtime)
    return volumes[-1] if volumes else None


def _midplane_3d(arr: np.ndarray) -> np.ndarray:
    return arr[arr.shape[0] // 2]


def load_vti_midplane_fields(volume_path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        import pyvista as pv
    except ImportError:
        return None

    grid = pv.read(volume_path)
    nx, ny, nz = (int(grid.dimensions[i] - 1) for i in range(3))
    rho = np.asarray(grid.cell_data["rho"], dtype=np.float64).reshape((nz, ny, nx), order="C")
    ux = np.asarray(grid.cell_data["ux"], dtype=np.float64).reshape((nz, ny, nx), order="C")
    uy = np.asarray(grid.cell_data["uy"], dtype=np.float64).reshape((nz, ny, nx), order="C")
    uz = np.asarray(grid.cell_data["uz"], dtype=np.float64).reshape((nz, ny, nx), order="C")
    solid = np.asarray(grid.cell_data["solid"]).reshape((nz, ny, nx), order="C").astype(bool)
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    pressure = rho / 3.0
    return {
        "velocity": _midplane_3d(umag),
        "pressure": _midplane_3d(pressure),
        "solid": _midplane_3d(solid),
    }


def chart_data_from_run(config, repo_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    manifest = latest_run_manifest(config, repo_root)
    fields = None
    if config.mode == "3d":
        volume = latest_volume_file(config, repo_root)
        if volume is not None:
            fields = load_vti_midplane_fields(volume)
    return manifest, fields
