"""Saved simulation case helpers for the Aero CFD GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero.run_manifest import load_run_manifest

from .state import GuiConfig


@dataclass
class CaseEntry:
    name: str
    path: Path
    mode: str
    shape: str
    re: float
    steps: int
    cd_mean: Optional[float]
    modified_ts: float

    @property
    def label(self) -> str:
        cd_text = f"Cd={self.cd_mean:.3f}" if self.cd_mean is not None else "Cd=—"
        mode_tag = self.mode.upper()
        return f"{mode_tag} · {self.shape} · Re={self.re:.0f} · {cd_text} · {self.name}"


def cases_root_path(repo_root: Path, cases_root: str = "./cases") -> Path:
    path = Path(cases_root)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def list_saved_cases(repo_root: Path, cases_root: str = "./cases") -> List[CaseEntry]:
    root = cases_root_path(repo_root, cases_root)
    if not root.is_dir():
        return []

    entries: List[CaseEntry] = []
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        config_path = case_dir / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        results_path = case_dir / "results.json"
        cd_mean = None
        if results_path.is_file():
            try:
                results = json.loads(results_path.read_text())
                cd_mean = results.get("Cd_mean")
            except (json.JSONDecodeError, OSError):
                pass

        mode = str(config.get("mode") or ("3d" if config.get("nz") is not None else "2d"))
        entries.append(
            CaseEntry(
                name=case_dir.name,
                path=case_dir.resolve(),
                mode=mode,
                shape=str(config.get("shape", "?")),
                re=float(config.get("re", 0.0)),
                steps=int(config.get("steps", 0)),
                cd_mean=float(cd_mean) if cd_mean is not None else None,
                modified_ts=case_dir.stat().st_mtime,
            )
        )

    entries.sort(key=lambda item: item.modified_ts, reverse=True)
    return entries


def relative_case_output_dir(case_dir: Path, repo_root: Path) -> str:
    try:
        return f"./{case_dir.resolve().relative_to(repo_root.resolve())}"
    except ValueError:
        return str(case_dir.resolve())


def apply_case_config(gui: GuiConfig, case_config: Dict[str, Any]) -> None:
    """Populate GuiConfig from a saved case config.json."""
    mode = str(case_config.get("mode") or ("3d" if case_config.get("nz") is not None else "2d"))
    gui.mode = mode
    shape = str(case_config.get("shape", "cylinder"))
    params = gui.params_2d if mode == "2d" else gui.params_3d

    if mode == "2d":
        gui.shape_2d = shape
    else:
        gui.shape_3d = shape

    field_map = {
        "re": "re",
        "steps": "steps",
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
        "u0": "u0",
        "radius": "radius",
        "width": "width",
        "height": "height",
        "depth": "depth",
        "length": "length",
        "backend": "backend",
        "collision": "collision",
        "wall_bc": "wall_bc",
        "inlet_bc": "inlet_bc",
        "outlet_bc": "outlet_bc",
        "viz3d": "viz3d",
        "inlet_perturbation": "inlet_perturbation",
        "trt_lambda": "trt_lambda",
        "sponge_cells": "sponge_cells",
        "sponge_strength": "sponge_strength",
        "les": "les",
        "les_cs": "les_cs",
        "thermal": "thermal",
        "T_hot": "T_hot",
        "T_cold": "T_cold",
        "alpha_T": "alpha_T",
        "buoyancy": "buoyancy",
        "g_gravity": "g_gravity",
        "beta": "beta",
        "T_ref": "T_ref",
        "mesh_bc": "mesh_bc",
        "mesh_orient": "mesh_orient",
        "mesh_rot_x": "mesh_rot_x",
        "mesh_rot_y": "mesh_rot_y",
        "mesh_rot_z": "mesh_rot_z",
        "stl_path": "stl_path",
        "stl_fit": "stl_fit",
    }
    for src, dst in field_map.items():
        if src not in case_config or case_config[src] is None:
            continue
        if dst in params:
            params[dst] = str(case_config[src])


def set_case_output_dir(gui: GuiConfig, case_dir: Path, repo_root: Path) -> None:
    rel = relative_case_output_dir(case_dir, repo_root)
    if gui.mode == "2d":
        gui.output_dir_2d = rel
    else:
        gui.output_dir_3d = rel


def load_case_results(case_dir: Path) -> Dict[str, str]:
    """Load saved scalar results for the results summary panel."""
    results_path = case_dir / "results.json"
    if not results_path.is_file():
        return {}
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    mode = str(data.get("mode", "3d" if "Cly_mean" in data else "2d"))
    metrics: Dict[str, str] = {
        "cd_mean": f"{data.get('Cd_mean', '—')}",
        "cd_std": f"{data.get('Cd_std', '')}",
        "cd_p_mean": f"{data.get('Cd_p_mean', '—')}",
        "cd_v_mean": f"{data.get('Cd_v_mean', '—')}",
        "elapsed": f"{data.get('elapsed_seconds', '—')}",
    }
    if mode == "3d":
        metrics["cl_y_mean"] = f"{data.get('Cly_mean', '—')}"
        metrics["cl_y_std"] = f"{data.get('Cly_std', '')}"
        metrics["cl_z_mean"] = f"{data.get('Clz_mean', '—')}"
        metrics["cl_z_std"] = f"{data.get('Clz_std', '')}"
    else:
        metrics["cl_y_mean"] = f"{data.get('Cl_mean', '—')}"
        metrics["cl_y_std"] = f"{data.get('Cl_std', '')}"
        metrics["cl_z_mean"] = "—"
        metrics["cl_z_std"] = ""
    steps_run = data.get("steps_run")
    if steps_run is not None and data.get("elapsed_seconds"):
        try:
            metrics["steps_per_sec"] = str(int(float(steps_run) / float(data["elapsed_seconds"])))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    scalar_stats = data.get("scalar_stats")
    if isinstance(scalar_stats, dict):
        metrics["scalar_mean"] = f"{scalar_stats.get('mean', '—')}"
        metrics["scalar_min"] = f"{scalar_stats.get('min', '—')}"
        metrics["scalar_max"] = f"{scalar_stats.get('max', '—')}"
    return metrics


def chart_manifest_for_case(case_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = load_run_manifest(case_dir)
    if manifest:
        return manifest

    history_path = case_dir / "history.json"
    config_path = case_dir / "config.json"
    results_path = case_dir / "results.json"
    if not history_path.is_file():
        return None

    try:
        history = json.loads(history_path.read_text())
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        return None

    mode = str(config.get("mode") or ("3d" if config.get("nz") is not None else "2d"))
    manifest = {
        "mode": mode,
        "shape": config.get("shape"),
        "re": config.get("re"),
        "steps": config.get("steps"),
        "Cd_history": history.get("Cd_history", []),
    }
    if mode == "3d":
        manifest["Cly_history"] = history.get("Cly_history", [])
        manifest["Clz_history"] = history.get("Clz_history", [])
    else:
        manifest["Cl_history"] = history.get("Cl_history", [])
    if results:
        manifest["Cd_mean"] = results.get("Cd_mean")
    return manifest
