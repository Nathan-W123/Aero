"""
Literature benchmarks, grid-convergence checks, and physics validation helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationReport:
    benchmark_status: str = "n/a"
    benchmark_message: str = "No reference benchmark for this case."
    grid_status: str = "n/a"
    grid_message: str = "Run Grid Study to check resolution independence."
    bc_warnings: List[str] = field(default_factory=list)
    collision_status: str = "n/a"
    collision_message: str = ""

    @property
    def overall_ok(self) -> bool:
        if self.benchmark_status == "fail" or self.collision_status == "fail":
            return False
        if self.grid_status == "fail":
            return False
        return True


@dataclass
class GridStudyResult:
    resolutions: List[float]
    cd_values: List[float]
    observed_order: Optional[float]
    extrapolated_cd: Optional[float]
    relative_errors: List[float]
    status: str
    message: str


def schiller_naumann_cd(re: float) -> float:
    """Drag coefficient for a sphere (Schiller–Naumann correlation)."""
    re = max(float(re), 1e-6)
    return (24.0 / re) * (1.0 + 0.15 * (re ** 0.687))


def literature_cd_range(mode: str, shape: str, re: float) -> Optional[Tuple[float, float, str]]:
    """Return (cd_min, cd_max, reference_note) if a literature band exists."""
    shape = shape.lower()
    mode = mode.lower()
    re = float(re)

    if mode == "2d" and shape == "cylinder" and abs(re - 100.0) < 5.0:
        return (
            1.2,
            2.2,
            "2D cylinder Re≈100 with ~20% blockage (Tritton/Fornberg + confinement)",
        )

    if mode == "3d" and shape == "sphere":
        if abs(re - 20.0) < 3.0:
            cd_ref = schiller_naumann_cd(re)
            return (
                max(0.5, cd_ref * 0.6),
                cd_ref * 2.5,
                f"Sphere Re≈20 — Schiller–Naumann Cd≈{cd_ref:.2f} (confinement widens band)",
            )
        if abs(re - 100.0) < 10.0:
            return (0.85, 1.25, "Sphere Re≈100 — literature Cd≈1.0 (confined tunnel)")

    if mode == "3d" and shape == "box" and re <= 150.0:
        return (0.8, 3.5, "Box Re≤150 — qualitative band (geometry-dependent)")

    return None


def assess_literature(
    *,
    mode: str,
    shape: str,
    re: float,
    cd: Optional[float],
) -> Tuple[str, str]:
    if cd is None or math.isnan(cd):
        return "fail", "Cd unavailable — simulation may have diverged."

    band = literature_cd_range(mode, shape, re)
    if band is None:
        return "n/a", "No literature benchmark configured for this case."

    cd_min, cd_max, note = band
    if cd_min <= cd <= cd_max:
        return "pass", f"Cd={cd:.4f} within [{cd_min:.2f}, {cd_max:.2f}] — {note}"
    margin = min(abs(cd - cd_min), abs(cd - cd_max))
    if margin / max(cd, 1e-6) < 0.15:
        return "warn", f"Cd={cd:.4f} near band [{cd_min:.2f}, {cd_max:.2f}] — {note}"
    return "fail", f"Cd={cd:.4f} outside [{cd_min:.2f}, {cd_max:.2f}] — {note}"


def assess_grid_convergence(cd_values: List[float]) -> Tuple[str, str]:
    """Assess grid independence from coarse→fine Cd values (≥2 levels)."""
    if len(cd_values) < 2:
        return "n/a", "Need at least two grid levels."

    labels = []
    for i in range(1, len(cd_values)):
        prev, curr = cd_values[i - 1], cd_values[i]
        if abs(prev) < 1e-12:
            pct = float("inf")
        else:
            pct = abs(curr - prev) / abs(prev) * 100.0
        labels.append(f"L{i}→L{i+1}: {pct:.1f}%")

    finest_change = abs(cd_values[-1] - cd_values[-2]) / max(abs(cd_values[-2]), 1e-12)
    summary = " · ".join(labels)
    if finest_change < 0.05:
        return "pass", f"Grid independent (<5% finest step) — {summary}"
    if finest_change < 0.10:
        return "warn", f"Nearly converged (5–10%) — {summary}"
    return "fail", f"Not grid independent (>10%) — {summary}"


def observed_order_of_convergence(
    cd_values: List[float],
    refinement_ratio: float = 2.0,
) -> Optional[float]:
    """Observed order of convergence from coarse→medium→fine values."""
    if len(cd_values) < 3:
        return None
    coarse, medium, fine = map(float, cd_values[-3:])
    denom = medium - coarse
    numer = fine - medium
    if abs(denom) < 1e-12 or abs(numer) < 1e-12:
        return None
    ratio = abs(numer / denom)
    if ratio <= 0.0:
        return None
    return float(abs(math.log(abs(denom / numer)) / math.log(float(refinement_ratio))))


def richardson_extrapolation(
    cd_values: List[float],
    refinement_ratio: float = 2.0,
    observed_order: Optional[float] = None,
) -> Optional[float]:
    """Richardson extrapolation using the two finest solutions."""
    if len(cd_values) < 2:
        return None
    fine = float(cd_values[-1])
    medium = float(cd_values[-2])
    p = observed_order if observed_order is not None else observed_order_of_convergence(cd_values, refinement_ratio)
    if p is None:
        return None
    denom = float(refinement_ratio) ** p - 1.0
    if abs(denom) < 1e-12:
        return None
    return fine + (fine - medium) / denom


def grid_study(
    run_case: Callable[[float], Any],
    resolutions: List[float],
    *,
    refinement_ratio: float = 2.0,
) -> GridStudyResult:
    """
    Run a coarse→fine grid study via a user-supplied case runner.

    `run_case(resolution)` may return a scalar Cd or a dict containing `Cd_mean`.
    """
    cds: List[float] = []
    for resolution in resolutions:
        result = run_case(float(resolution))
        cd = float(result["Cd_mean"]) if isinstance(result, dict) else float(result)
        cds.append(cd)

    status, message = assess_grid_convergence(cds)
    observed_order = observed_order_of_convergence(cds, refinement_ratio=refinement_ratio)
    extrapolated_cd = richardson_extrapolation(
        cds,
        refinement_ratio=refinement_ratio,
        observed_order=observed_order,
    )
    rel_errors: List[float] = []
    if extrapolated_cd is not None:
        for cd in cds:
            rel_errors.append(abs(cd - extrapolated_cd) / max(abs(extrapolated_cd), 1e-12))
    return GridStudyResult(
        resolutions=[float(r) for r in resolutions],
        cd_values=cds,
        observed_order=observed_order,
        extrapolated_cd=extrapolated_cd,
        relative_errors=rel_errors,
        status=status,
        message=message,
    )


def validate_bc_config(
    *,
    mode: str,
    wall_bc: str,
    outlet_bc: str,
    inlet_bc: str = "velocity",
    re: float = 100.0,
    inlet_perturbation: float = 0.0,
) -> List[str]:
    warnings: List[str] = []
    wall_bc = wall_bc.lower()
    outlet_bc = outlet_bc.lower()
    inlet_bc = inlet_bc.lower()

    if outlet_bc not in {"convective", "zerogradient"}:
        warnings.append(f"Unknown outlet_bc '{outlet_bc}' — use convective or zerogradient.")

    if mode == "2d":
        if wall_bc not in {"slip", "noslip"}:
            warnings.append(f"Unknown wall_bc '{wall_bc}' — use slip or noslip.")
        if inlet_bc not in {"velocity", "pressure"}:
            warnings.append(f"Unknown inlet_bc '{inlet_bc}' — use velocity or pressure.")
        if re > 47 and inlet_perturbation <= 0 and wall_bc == "slip":
            warnings.append(
                "Re > 47 with slip walls and no inlet perturbation — "
                "vortex shedding may not trigger; try inlet_perturbation ≥ 0.02 or noslip walls."
            )
    else:
        if wall_bc not in {"slip", "noslip"}:
            warnings.append(f"Unknown wall_bc '{wall_bc}' — use slip or noslip.")
        if re > 47 and inlet_perturbation <= 0 and wall_bc == "slip":
            warnings.append(
                "Re > 47 with slip walls and no inlet perturbation — "
                "3D shedding may not trigger; try inlet_perturbation ≥ 0.02."
            )

    if outlet_bc == "zerogradient" and re > 200:
        warnings.append("Zero-gradient outlet at high Re may reflect spurious waves — prefer convective.")

    return warnings


def assess_collision(
    *,
    re: float,
    tau: Optional[float],
    collision: str,
) -> Tuple[str, str]:
    collision = collision.lower()
    if tau is None:
        return "n/a", "tau unknown — run simulation to evaluate stability."

    if tau < 0.55:
        return "fail", f"tau={tau:.3f} < 0.55 — simulation unstable; reduce Re or refine grid."

    if collision == "mrt":
        if tau < 0.58 and re > 100:
            return "warn", f"MRT with tau={tau:.3f} — monitor stability at Re={re:.0f}."
        return "pass", f"MRT collision, tau={tau:.3f} — suitable for Re={re:.0f}."

    if collision == "trt":
        if re > 300:
            return "warn", f"TRT at Re={re:.0f} — consider LES for turbulent regimes."
        return "pass", f"TRT collision, tau={tau:.3f} — enhanced stability for Re={re:.0f}."

    if re > 250:
        return "warn", f"BGK at Re={re:.0f} may be stiff — try collision=mrt or trt."
    if tau < 0.58:
        return "warn", f"BGK with tau={tau:.3f} near stability limit — consider MRT or lower Re."
    return "pass", f"BGK collision, tau={tau:.3f} — stable for Re={re:.0f}."


def derive_tau(re: float, u0: float, d_cells: float) -> float:
    """Lattice relaxation time from Re, u0, and reference length in cells."""
    nu = u0 * d_cells / max(re, 1e-6)
    return 3.0 * nu + 0.5


def reference_length_cells(mode: str, shape: str, params: Dict[str, str]) -> float:
    shape = shape.lower()
    if shape in {"cylinder", "sphere"}:
        return 2.0 * float(params.get("radius", "10"))
    if shape == "rectangle":
        return float(params.get("height", "20"))
    if shape == "box":
        return float(params.get("height", "10"))
    return float(params.get("height", params.get("radius", "10")))


def build_validation_report(
    *,
    mode: str,
    shape: str,
    params: Dict[str, str],
    cd: Optional[float] = None,
    grid_cd_values: Optional[List[float]] = None,
) -> ValidationReport:
    re = float(params.get("re", "100"))
    u0 = float(params.get("u0", "0.05"))
    d = reference_length_cells(mode, shape, params)
    tau = derive_tau(re, u0, d)

    b_status, b_msg = assess_literature(mode=mode, shape=shape, re=re, cd=cd)
    if grid_cd_values:
        g_status, g_msg = assess_grid_convergence(grid_cd_values)
    else:
        g_status, g_msg = "n/a", "Run Grid Study to check resolution independence."

    bc_warnings = validate_bc_config(
        mode=mode,
        wall_bc=params.get("wall_bc", "slip"),
        outlet_bc=params.get("outlet_bc", "convective"),
        inlet_bc=params.get("inlet_bc", "velocity"),
        re=re,
        inlet_perturbation=float(params.get("inlet_perturbation", "0") or "0"),
    )
    c_status, c_msg = assess_collision(re=re, tau=tau, collision=params.get("collision", "bgk"))

    bc_warnings = list(bc_warnings)
    les_enabled = str(params.get("les", "false")).lower() in ("true", "1", "yes")
    if re > 300 and not les_enabled and params.get("collision", "bgk") == "bgk":
        bc_warnings.append(
            "Re > 300 with BGK and no LES — consider collision=trt/mrt or enable LES."
        )
    if params.get("mesh_bc") == "ibm":
        bc_warnings.append("IBM boundary active — compare against voxel mode for validation.")

    return ValidationReport(
        benchmark_status=b_status,
        benchmark_message=b_msg,
        grid_status=g_status,
        grid_message=g_msg,
        bc_warnings=bc_warnings,
        collision_status=c_status,
        collision_message=c_msg,
    )


def scale_params_for_grid(params: Dict[str, str], factor: float, mode: str) -> Dict[str, str]:
    """Scale grid and geometry for a grid-refinement study."""
    scaled = dict(params)
    for key in ("nx", "ny", "nz"):
        if key in scaled and scaled[key]:
            scaled[key] = str(max(int(int(scaled[key]) * factor), 12))
    for key in ("radius", "width", "height", "depth", "length"):
        if key in scaled and scaled[key]:
            val = max(int(round(float(scaled[key]) * factor)), 2)
            scaled[key] = str(val)
    return scaled


def dean_correlation_cf(re_bulk: float) -> float:
    """Dean correlation for turbulent internal-flow skin-friction coefficient."""
    re_bulk = max(float(re_bulk), 1.0)
    return 0.073 * (re_bulk ** -0.25)


def channel_friction_coefficient(
    *,
    body_force_x: float,
    hydraulic_diameter: float,
    bulk_velocity: float,
    rho: float = 1.0,
) -> float:
    """Estimate Cf from a uniform streamwise body force in a periodic channel."""
    bulk_velocity = max(abs(float(bulk_velocity)), 1e-12)
    tau_w = abs(float(body_force_x)) * float(hydraulic_diameter) / 4.0
    return 2.0 * tau_w / (float(rho) * bulk_velocity * bulk_velocity)


def backward_facing_step_reattachment_length(
    ux: np.ndarray,
    *,
    step_index: int,
    wall_offset: int = 1,
) -> Optional[float]:
    """
    Estimate reattachment length from the first downstream sign change of near-wall ux.

    Accepts either `(Ny, Nx)` or `(Nz, Ny, Nx)` streamwise velocity fields.
    """
    field = np.asarray(ux, dtype=np.float64)
    if field.ndim == 3:
        field = np.mean(field, axis=0)
    if field.ndim != 2:
        raise ValueError("ux must be 2D or 3D.")
    y = min(max(int(wall_offset), 0), field.shape[0] - 1)
    line = field[y, :]
    if step_index >= line.size - 1:
        return None
    for x in range(int(step_index) + 1, line.size):
        if line[x] > 0.0:
            return float(x - step_index)
    return None
