"""State and command-building helpers for the Aero desktop GUI."""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GuiConfig:
    mode: str = "3d"
    shape_2d: str = "cylinder"
    shape_3d: str = "sphere"
    output_dir_2d: str = "./outputs"
    output_dir_3d: str = "./outputs3d"
    params_2d: Dict[str, str] = field(default_factory=lambda: {
        "re": "100",
        "steps": "5000",
        "nx": "400",
        "ny": "200",
        "u0": "0.05",
        "radius": "20",
        "width": "40",
        "height": "20",
        "backend": "auto",
        "collision": "bgk",
        "wall_bc": "slip",
        "inlet_bc": "velocity",
        "outlet_bc": "convective",
        "inlet_perturbation": "0",
    })
    params_3d: Dict[str, str] = field(default_factory=lambda: {
        "re": "20",
        "steps": "500",
        "nx": "64",
        "ny": "48",
        "nz": "48",
        "u0": "0.05",
        "radius": "10",
        "width": "10",
        "height": "10",
        "depth": "10",
        "length": "20",
        "backend": "auto",
        "collision": "bgk",
        "wall_bc": "slip",
        "outlet_bc": "convective",
        "viz3d": "auto",
    })
    cases_root: str = "./cases"
    save_cases: bool = True


GUI_COMBO_FIELDS: Dict[str, List[str]] = {
    "wall_bc": ["slip", "noslip"],
    "outlet_bc": ["convective", "zerogradient"],
    "inlet_bc": ["velocity", "pressure"],
    "inlet_perturbation": ["0", "0.02", "0.05"],
    "collision": ["bgk", "mrt"],
    "backend": ["auto", "numpy", "numba"],
    "viz3d": ["auto", "slices", "pyvista", "all"],
}


def build_grid_study_commands(config: GuiConfig, factors: Optional[List[float]] = None) -> List[List[str]]:
    """Build CLI commands for a grid-refinement study at relative factors."""
    from aero.benchmarks import scale_params_for_grid

    factors = factors or [0.5, 1.0, 2.0]
    commands: List[List[str]] = []
    base_params = config.params_2d if config.mode == "2d" else config.params_3d
    shape = config.shape_2d if config.mode == "2d" else config.shape_3d

    for factor in factors:
        scaled = scale_params_for_grid(base_params, factor, config.mode)
        temp = GuiConfig(
            mode=config.mode,
            shape_2d=config.shape_2d,
            shape_3d=config.shape_3d,
            cases_root=config.cases_root,
            save_cases=config.save_cases,
        )
        if config.mode == "2d":
            temp.params_2d.update(scaled)
        else:
            temp.params_3d.update(scaled)
        commands.append(build_command(temp))
    return commands


def _append_flag(cmd: List[str], flag: str, value: str) -> None:
    if value != "":
        cmd.extend([flag, value])


def progress_check_every(steps: int) -> int:
    """Reporting interval for live GUI charts (more frequent on short runs)."""
    steps = max(int(steps), 1)
    if steps <= 500:
        return max(25, steps // 8)
    if steps <= 5000:
        return max(50, steps // 15)
    return min(1000, max(200, steps // 20))


def build_command(config: GuiConfig) -> List[str]:
    python = sys.executable
    if config.mode == "2d":
        params = config.params_2d
        cmd = [python, "cli.py", "--shape", config.shape_2d]
        for key in ("re", "steps", "nx", "ny", "u0", "backend", "collision", "wall_bc", "inlet_bc", "outlet_bc", "inlet_perturbation"):
            _append_flag(cmd, f"--{key.replace('_', '-')}", params[key])
        _append_flag(cmd, "--check-every", str(progress_check_every(int(params.get("steps", "5000")))))
        if config.shape_2d == "cylinder":
            _append_flag(cmd, "--radius", params["radius"])
        elif config.shape_2d == "rectangle":
            _append_flag(cmd, "--width", params["width"])
            _append_flag(cmd, "--height", params["height"])
        if config.save_cases:
            cmd.append("--save-case")
            _append_flag(cmd, "--cases-root", config.cases_root)
        else:
            _append_flag(cmd, "--output", config.output_dir_2d)
        return cmd

    params = config.params_3d
    cmd = [python, "cli3d.py", "--shape", config.shape_3d]
    for key in ("re", "steps", "nx", "ny", "nz", "u0", "backend", "collision", "wall_bc", "outlet_bc", "viz3d"):
        _append_flag(cmd, f"--{key.replace('_', '-')}", params[key])
    _append_flag(cmd, "--check-every", str(progress_check_every(int(params.get("steps", "500")))))
    if config.shape_3d == "sphere":
        _append_flag(cmd, "--radius", params["radius"])
    elif config.shape_3d == "box":
        _append_flag(cmd, "--width", params["width"])
        _append_flag(cmd, "--height", params["height"])
        _append_flag(cmd, "--depth", params["depth"])
    elif config.shape_3d == "cylinder":
        _append_flag(cmd, "--radius", params["radius"])
        _append_flag(cmd, "--length", params["length"])
    cmd.append("--export-vtk")
    if config.save_cases:
        cmd.append("--save-case")
        _append_flag(cmd, "--cases-root", config.cases_root)
    else:
        _append_flag(cmd, "--output", config.output_dir_3d)
    return cmd


def list_preview_images(config: GuiConfig) -> List[Path]:
    output_dir = Path(config.output_dir_2d if config.mode == "2d" else config.output_dir_3d)
    if not output_dir.exists():
        return []
    return sorted(path for path in output_dir.iterdir() if path.suffix.lower() == ".png")


def list_volume_files(config: GuiConfig) -> List[Path]:
    output_dir = Path(config.output_dir_3d)
    if config.mode != "3d" or not output_dir.exists():
        return []
    return sorted(path for path in output_dir.iterdir() if path.suffix.lower() == ".vti")


def latest_volume_file(config: GuiConfig, repo_root: Optional[Path] = None) -> Optional[Path]:
    from .run_data import latest_volume_file as _latest

    root = Path(".") if repo_root is None else Path(repo_root)
    return _latest(config, root)


def pick_preview_image(paths: List[Path], keywords: Tuple[str, ...]) -> Optional[Path]:
    for path in paths:
        name = path.name.lower()
        if all(keyword in name for keyword in keywords):
            return path
    for path in paths:
        name = path.name.lower()
        if any(keyword in name for keyword in keywords):
            return path
    return None


def parse_run_results(log_text: str) -> Dict[str, str]:
    """Extract headline CFD metrics from simulator stdout."""
    patterns = {
        "cd_mean": r"Cd\s+\(mean\)\s*=\s*([-\d.]+)",
        "cd_std": r"Cd\s+\(mean\)\s*=\s*[-\d.]+\s*±\s*([-\d.]+)",
        "cl_y_mean": r"Cl_y\s+\(mean\)\s*=\s*([-\d.]+)",
        "cl_y_std": r"Cl_y\s+\(mean\)\s*=\s*[-\d.]+\s*±\s*([-\d.]+)",
        "cl_z_mean": r"Cl_z\s+\(mean\)\s*=\s*([-\d.]+)",
        "cl_z_std": r"Cl_z\s+\(mean\)\s*=\s*[-\d.]+\s*±\s*([-\d.]+)",
        "elapsed": r"Elapsed\s*=\s*([-\d.]+)\s*s",
        "steps_per_sec": r"Elapsed\s*=\s*[-\d.]+\s*s\s*\((\d+)\s*steps/s\)",
        "re": r"Re\s*:\s*([-\d.]+)",
        "blockage": r"Blockage\s*:\s*([-\d.]+%)",
        "tau": r"tau\s*:\s*([-\d.]+)",
    }
    found: Dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            found[key] = match.group(1)
    return found


_PROGRESS_2D = re.compile(
    r"step\s+(\d+)/(\d+)\s+Cd=([+-][\d.]+)\s+Cl=([+-][\d.]+)\s+\[(\d+)\s*steps/s\]"
)
_PROGRESS_3D = re.compile(
    r"step\s+(\d+)/(\d+)\s+Cd=([+-][\d.]+)\s+Cl_y=([+-][\d.]+)\s+\[(\d+)\s*steps/s\]"
)


@dataclass
class ProgressSnapshot:
    step: int
    total_steps: int
    cd: float
    cl: float
    steps_per_sec: int


def parse_progress_line(line: str) -> Optional[ProgressSnapshot]:
    """Parse a solver progress line for live convergence charts."""
    for pattern in (_PROGRESS_2D, _PROGRESS_3D):
        match = pattern.search(line)
        if match:
            return ProgressSnapshot(
                step=int(match.group(1)),
                total_steps=int(match.group(2)),
                cd=float(match.group(3)),
                cl=float(match.group(4)),
                steps_per_sec=int(match.group(5)),
            )
    return None
