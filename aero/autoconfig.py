"""Shared preflight autoconfiguration helpers for recoverable run settings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AutoconfigChange:
    field: str
    old: Any
    new: Any
    reason: str


@dataclass
class AutoconfigReport:
    mode: str
    policy: str
    applied: bool
    changes: List[AutoconfigChange] = field(default_factory=list)
    final_summary: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "policy": self.policy,
            "applied": self.applied,
            "changes": [
                {
                    "field": item.field,
                    "old": item.old,
                    "new": item.new,
                    "reason": item.reason,
                }
                for item in self.changes
            ],
            "final_summary": self.final_summary,
        }


def _set(args: Any, report: AutoconfigReport, field_name: str, new_value: Any, reason: str) -> None:
    old_value = getattr(args, field_name)
    if old_value == new_value:
        return
    setattr(args, field_name, new_value)
    report.changes.append(AutoconfigChange(field_name, old_value, new_value, reason))


def _reference_length_2d(args: Any) -> float:
    if args.shape == "cylinder":
        return 2.0 * float(args.radius)
    if args.shape == "rectangle":
        return float(args.height)
    return float(getattr(args, "height", args.ny * 0.2))


def _reference_length_3d(args: Any) -> float:
    if args.shape == "sphere":
        return 2.0 * float(args.radius)
    if args.shape == "box":
        return float(args.height)
    if args.shape == "cylinder":
        return 2.0 * float(args.radius)
    if args.shape == "mesh":
        return max(float(min(args.ny, args.nz) * getattr(args, "stl_fit", 0.35)), 1.0)
    return float(args.ny * 0.2)


def _blockage_2d(args: Any) -> float:
    return _reference_length_2d(args) / max(float(args.ny), 1.0)


def _blockage_3d(args: Any) -> float:
    shape = str(args.shape)
    if shape == "sphere":
        frontal_y = 2.0 * float(args.radius)
        frontal_z = frontal_y
    elif shape == "box":
        frontal_y = float(args.height)
        frontal_z = float(args.depth)
    elif shape == "cylinder":
        frontal_y = 2.0 * float(args.radius)
        frontal_z = float(args.length)
    elif shape == "mesh":
        fit = float(getattr(args, "stl_fit", 0.35))
        frontal_y = fit * float(args.ny)
        frontal_z = fit * float(args.nz)
    else:
        frontal_y = _reference_length_3d(args)
        frontal_z = frontal_y
    by = frontal_y / max(float(args.ny), 1.0)
    bz = frontal_z / max(float(args.nz), 1.0)
    return max(by, bz)


def _tau(re: float, u0: float, d_cells: float) -> float:
    return 3.0 * (u0 * d_cells / max(re, 1e-12)) + 0.5


def _ma(u0: float) -> float:
    return float(u0) * math.sqrt(3.0)


def _finalize_report(report: AutoconfigReport, args: Any, mode: str) -> AutoconfigReport:
    if mode == "2d":
        d_cells = _reference_length_2d(args)
        blockage = _blockage_2d(args)
    else:
        d_cells = _reference_length_3d(args)
        blockage = _blockage_3d(args)
    report.applied = bool(report.changes)
    report.final_summary = {
        "u0": float(args.u0),
        "re": float(args.re),
        "tau": _tau(float(args.re), float(args.u0), d_cells),
        "Ma": _ma(float(args.u0)),
        "blockage": blockage,
        "collision": str(getattr(args, "collision", "bgk")),
        "nx": int(args.nx),
        "ny": int(args.ny),
    }
    if mode == "3d":
        report.final_summary["nz"] = int(args.nz)
    return report


def _scale_2d_geometry(args: Any, report: AutoconfigReport, scale: float, reason: str) -> None:
    for field_name in ("radius", "width", "height"):
        if hasattr(args, field_name):
            new_value = max(float(getattr(args, field_name)) * scale, 1.0)
            _set(args, report, field_name, new_value, reason)


def _scale_3d_geometry(args: Any, report: AutoconfigReport, scale: float, reason: str) -> None:
    for field_name in ("radius", "width", "height", "depth", "length"):
        if hasattr(args, field_name):
            new_value = max(float(getattr(args, field_name)) * scale, 1.0)
            _set(args, report, field_name, new_value, reason)


def autoconfigure_2d(args: Any, policy: str = "off") -> Optional[AutoconfigReport]:
    policy = str(policy).lower()
    if policy == "off":
        return None
    report = AutoconfigReport(mode="2d", policy=policy, applied=False)

    for _ in range(4):
        changed = False
        d_cells = _reference_length_2d(args)
        tau = _tau(float(args.re), float(args.u0), d_cells)
        ma = _ma(float(args.u0))
        blockage = _blockage_2d(args)

        if tau < 0.55:
            target_tau = 0.58
            current_excess = max(tau - 0.5, 1e-12)
            target_scale = max((target_tau - 0.5) / current_excess, 1.0)
            target_ny = max(int(math.ceil(float(args.ny) * target_scale)), int(args.ny))
            target_nx = max(int(math.ceil(float(args.nx) * target_scale)), int(args.nx))
            if target_ny > int(args.ny) or target_nx > int(args.nx):
                scale = max(target_ny / max(float(args.ny), 1.0), target_nx / max(float(args.nx), 1.0))
                _scale_2d_geometry(args, report, scale, "Scaled obstacle dimensions consistently with tau-recovery lattice refinement.")
                if target_ny > int(args.ny):
                    _set(args, report, "ny", int(target_ny), "Increased lattice resolution to move tau away from instability.")
                    changed = True
                if target_nx > int(args.nx):
                    _set(args, report, "nx", int(target_nx), "Scaled streamwise resolution during tau recovery.")
                    changed = True
            if str(args.collision).lower() == "bgk":
                _set(args, report, "collision", "mrt", "Switched collision model for improved stability at low tau.")
                changed = True

        if ma > 0.15:
            target_u0 = min(float(args.u0), 0.12 / math.sqrt(3.0))
            if target_u0 < float(args.u0):
                _set(args, report, "u0", float(target_u0), "Reduced inlet velocity to keep Mach number in a safer low-Mach regime.")
                changed = True

        if blockage > 0.40:
            target_ny = max(int(math.ceil(d_cells / 0.20)), int(args.ny))
            if target_ny > int(args.ny):
                scale = target_ny / max(float(args.ny), 1.0)
                target_nx = max(int(math.ceil(float(args.nx) * scale)), int(args.nx))
                _set(args, report, "ny", int(target_ny), "Increased cross-stream resolution to reduce excessive blockage.")
                _set(args, report, "nx", int(target_nx), "Scaled streamwise resolution with Ny during blockage recovery.")
                changed = True

        if float(args.re) > 47.0 and float(getattr(args, "inlet_perturbation", 0.0)) <= 0.0 and str(args.wall_bc) == "slip":
            _set(args, report, "inlet_perturbation", 0.02, "Added a small inlet perturbation to help trigger expected vortex shedding.")
            changed = True

        if (float(args.re) > 250.0 or tau < 0.58) and str(args.collision).lower() == "bgk":
            _set(args, report, "collision", "mrt", "Preferred MRT over BGK for a stiffer operating point.")
            changed = True

        if not changed:
            break

    return _finalize_report(report, args, "2d")


def autoconfigure_3d(args: Any, policy: str = "off") -> Optional[AutoconfigReport]:
    policy = str(policy).lower()
    if policy == "off":
        return None
    report = AutoconfigReport(mode="3d", policy=policy, applied=False)

    for _ in range(4):
        changed = False
        d_cells = _reference_length_3d(args)
        tau = _tau(float(args.re), float(args.u0), d_cells)
        ma = _ma(float(args.u0))
        blockage = _blockage_3d(args)

        if tau < 0.55:
            target_tau = 0.58
            current_excess = max(tau - 0.5, 1e-12)
            target_scale = max((target_tau - 0.5) / current_excess, 1.0)
            target_ny = max(int(math.ceil(float(args.ny) * target_scale)), int(args.ny))
            target_nz = max(int(math.ceil(float(args.nz) * target_scale)), int(args.nz))
            target_nx = max(int(math.ceil(float(args.nx) * target_scale)), int(args.nx))
            if target_ny > int(args.ny) or target_nz > int(args.nz) or target_nx > int(args.nx):
                scale = max(
                    target_nx / max(float(args.nx), 1.0),
                    target_ny / max(float(args.ny), 1.0),
                    target_nz / max(float(args.nz), 1.0),
                )
                _scale_3d_geometry(args, report, scale, "Scaled obstacle dimensions consistently with tau-recovery lattice refinement.")
                if target_ny > int(args.ny):
                    _set(args, report, "ny", int(target_ny), "Increased lattice resolution to move tau away from instability.")
                    changed = True
                if target_nz > int(args.nz):
                    _set(args, report, "nz", int(target_nz), "Increased spanwise resolution during tau recovery.")
                    changed = True
                if target_nx > int(args.nx):
                    _set(args, report, "nx", int(target_nx), "Scaled streamwise resolution during tau recovery.")
                    changed = True
            if str(args.collision).lower() == "bgk":
                _set(args, report, "collision", "mrt", "Switched collision model for improved stability at low tau.")
                changed = True

        if ma > 0.15:
            target_u0 = min(float(args.u0), 0.12 / math.sqrt(3.0))
            if target_u0 < float(args.u0):
                _set(args, report, "u0", float(target_u0), "Reduced inlet velocity to keep Mach number in a safer low-Mach regime.")
                changed = True

        if blockage > 0.30:
            shape = str(args.shape)
            if shape == "sphere":
                frontal_y = 2.0 * float(args.radius)
                frontal_z = frontal_y
            elif shape == "box":
                frontal_y = float(args.height)
                frontal_z = float(args.depth)
            elif shape == "cylinder":
                frontal_y = 2.0 * float(args.radius)
                frontal_z = float(args.length)
            else:
                fit = float(getattr(args, "stl_fit", 0.35))
                frontal_y = fit * float(args.ny)
                frontal_z = fit * float(args.nz)
            target_ny = max(int(math.ceil(frontal_y / 0.20)), int(args.ny))
            target_nz = max(int(math.ceil(frontal_z / 0.20)), int(args.nz))
            scale = max(target_ny / max(float(args.ny), 1.0), target_nz / max(float(args.nz), 1.0), 1.0)
            target_nx = max(int(math.ceil(float(args.nx) * scale)), int(args.nx))
            if target_ny > int(args.ny):
                _set(args, report, "ny", int(target_ny), "Increased vertical resolution to reduce excessive frontal blockage.")
                changed = True
            if target_nz > int(args.nz):
                _set(args, report, "nz", int(target_nz), "Increased spanwise resolution to reduce excessive frontal blockage.")
                changed = True
            if target_nx > int(args.nx):
                _set(args, report, "nx", int(target_nx), "Scaled streamwise resolution with cross-stream blockage recovery.")
                changed = True

        if float(args.re) > 47.0 and float(getattr(args, "inlet_perturbation", 0.0)) <= 0.0 and str(args.wall_bc) == "slip":
            _set(args, report, "inlet_perturbation", 0.02, "Added a small inlet perturbation to help trigger expected 3D shedding.")
            changed = True

        if (float(args.re) > 250.0 or tau < 0.58) and str(args.collision).lower() == "bgk":
            _set(args, report, "collision", "mrt", "Preferred MRT over BGK for a stiffer operating point.")
            changed = True

        if not changed:
            break

    return _finalize_report(report, args, "3d")
