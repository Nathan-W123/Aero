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


@dataclass
class UncertaintyReport:
    overall_status: str
    summary: str
    components: Dict[str, Dict[str, Any]]


@dataclass
class ScalarValidationResult:
    status: str
    message: str
    rmse: float
    max_error: float
    mean_profile: List[float]
    reference_profile: List[float]


def schiller_naumann_cd(re: float) -> float:
    """Drag coefficient for a sphere (Schiller–Naumann correlation)."""
    re = max(float(re), 1e-6)
    return (24.0 / re) * (1.0 + 0.15 * (re ** 0.687))


#: Drag inflation per unit blockage (b = D / tunnel span) for a sphere in this
#: tunnel -- slip walls in y, periodic in z, so effectively a square array of
#: spheres -- as (Re, slope) points of Cd(b) = Cd(0) * (1 + slope * b).  Both
#: were measured with r = 7 (14 cells across the sphere):
#:
#:   Re=20   +32.0 / +15.2 / +12.9% over Schiller-Naumann at 29.2 / 14.6 / 9.7%
#:           blockage (tunnel length grown with the span; Cd on the voxel
#:           frontal area).  Straight line: intercept +1.9%, slope 1.02 --
#:           1.00 per unit blockage relative to the intercept.
#:   Re=100  1.3607 at 29.2% and 1.2629 at 14.6% (tunnel length fixed at 96;
#:           Cd on pi r^2): slope 0.58, intercept 1.165.
#:
#: Confinement weakens as Re rises -- a thinner viscous region reaches the
#: walls less -- so using the Re=20 slope at Re=100 overstated it almost 2x.
#: Between the points the slope is interpolated in log Re; outside them it
#: is held at the nearest one.  A calibration, not a law: the bands below keep
#: a +/-15% allowance around it.
SPHERE_CONFINEMENT_SLOPES: Tuple[Tuple[float, float], ...] = ((20.0, 1.00), (100.0, 0.58))


def sphere_confinement_slope(re: float) -> float:
    """Drag inflation per unit blockage at this Re (see SPHERE_CONFINEMENT_SLOPES)."""
    pts = SPHERE_CONFINEMENT_SLOPES
    re = float(re)
    if re <= pts[0][0]:
        return pts[0][1]
    for (re0, k0), (re1, k1) in zip(pts, pts[1:]):
        if re <= re1:
            t = (math.log(re) - math.log(re0)) / (math.log(re1) - math.log(re0))
            return k0 + t * (k1 - k0)
    return pts[-1][1]


def sphere_confinement_factor(blockage: Optional[float], re: float) -> float:
    """Multiplier on the unconfined sphere Cd for a given frontal blockage at Re."""
    if blockage is None:
        return 1.0
    return 1.0 + sphere_confinement_slope(re) * max(float(blockage), 0.0)


def sphere_expected_cd(re: float, blockage: Optional[float] = None) -> Tuple[float, float, str]:
    """
    (unconfined Cd, confined estimate, one-line note) for a sphere.

    The unconfined value is Schiller-Naumann; the confined one applies the
    measured blockage inflation.  Compare a run against the *confined* number
    unless the tunnel is wide enough (blockage under ~5%) for the two to agree.
    """
    sn = schiller_naumann_cd(re)
    if blockage is None:
        return sn, sn, f"Schiller-Naumann Cd={sn:.2f} at Re={re:g} (unconfined; blockage unknown)"
    factor = sphere_confinement_factor(blockage, re)
    conf = sn * factor
    return sn, conf, (
        f"Schiller-Naumann Cd={sn:.2f} at Re={re:g}, x{factor:.2f} "
        f"for {blockage*100:.0f}% blockage -> expect ~{conf:.2f}"
    )


def literature_cd_range(
    mode: str, shape: str, re: float, blockage: Optional[float] = None
) -> Optional[Tuple[float, float, str]]:
    """
    Return (cd_min, cd_max, reference_note) if a literature band exists.

    For the sphere the band is the confinement-corrected Schiller-Naumann value
    +/-15% when the blockage is known, and a wide envelope (covering blockage
    up to ~45%) when it is not.  Before the frontal-area fix the sphere bands
    were tuned around coefficients that read pi/4 low, which is why they
    looked so generous.
    """
    shape = shape.lower()
    mode = mode.lower()
    re = float(re)

    if mode == "2d" and shape == "cylinder" and abs(re - 100.0) < 5.0:
        return (
            1.2,
            2.2,
            "2D cylinder Re≈100 with ~20% blockage (Tritton/Fornberg + confinement)",
        )

    if mode == "3d" and shape == "sphere" and re <= 300.0:
        sn, conf, note = sphere_expected_cd(re, blockage)
        if blockage is None:
            return (sn * 0.85, sn * 1.6, note + "; band spans blockage up to ~45%")
        return (conf * 0.85, conf * 1.15, note)

    if mode == "3d" and shape == "box" and re <= 150.0:
        return (0.8, 3.5, "Box Re≤150 — qualitative band (geometry-dependent)")

    return None


def assess_literature(
    *,
    mode: str,
    shape: str,
    re: float,
    cd: Optional[float],
    blockage: Optional[float] = None,
) -> Tuple[str, str]:
    if cd is None or math.isnan(cd):
        return "fail", "Cd unavailable — simulation may have diverged."

    band = literature_cd_range(mode, shape, re, blockage)
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


#: Reynolds number at which the wake stops being steady, by body.  47 is the
#: circular cylinder's (2D, or spanwise-periodic 3D); a square cylinder's is
#: close to it.  A sphere's wake stays steady and axisymmetric to Re ~ 210,
#: steady but asymmetric to ~ 270, and only then sheds (Johnson & Patel 1999);
#: a cube's sequence is similar, ~216 then ~270 (Saha 2004).  Using 47 for a
#: sphere told users to perturb a flow that should be steady, which only adds
#: forcing noise to the drag.
SHEDDING_ONSET_RE = {
    ("2d", "cylinder"): 47.0,
    ("2d", "rectangle"): 47.0,
    ("3d", "cylinder"): 47.0,
    ("3d", "sphere"): 270.0,
    ("3d", "box"): 270.0,
}


def shedding_onset_re(mode: str, shape: Optional[str]) -> Optional[float]:
    """Onset of unsteady shedding for a known body, or None if not tabulated."""
    if not shape:
        return None
    return SHEDDING_ONSET_RE.get((str(mode).lower(), str(shape).lower()))


def validate_bc_config(
    *,
    mode: str,
    wall_bc: str,
    outlet_bc: str,
    inlet_bc: str = "velocity",
    re: float = 100.0,
    inlet_perturbation: float = 0.0,
    shape: Optional[str] = None,
) -> List[str]:
    warnings: List[str] = []
    onset = shedding_onset_re(mode, shape)
    if onset is None:
        onset = 47.0     # unknown body: keep the historical default
    wall_bc = (wall_bc or "slip").lower()
    outlet_bc = (outlet_bc or "convective").lower()
    inlet_bc = (inlet_bc or "velocity").lower()

    if outlet_bc not in {"convective", "zerogradient"}:
        warnings.append(f"Unknown outlet_bc '{outlet_bc}' — use convective or zerogradient.")

    if mode == "2d":
        if wall_bc not in {"slip", "noslip", "moving"}:
            warnings.append(f"Unknown wall_bc '{wall_bc}' — use slip, noslip, or moving.")
        if inlet_bc not in {"velocity", "pressure"}:
            warnings.append(f"Unknown inlet_bc '{inlet_bc}' — use velocity or pressure.")
        if re > onset and inlet_perturbation <= 0 and wall_bc == "slip":
            warnings.append(
                f"Re > {onset:g} with slip walls and no inlet perturbation — "
                "vortex shedding may not trigger; try inlet_perturbation ≥ 0.02 or noslip walls."
            )
    else:
        if wall_bc not in {"slip", "noslip", "moving"}:
            warnings.append(f"Unknown wall_bc '{wall_bc}' — use slip, noslip, or moving.")
        if re > onset and inlet_perturbation <= 0 and wall_bc == "slip":
            warnings.append(
                f"Re > {onset:g} with slip walls and no inlet perturbation — "
                "3D shedding may not trigger; try inlet_perturbation ≥ 0.02."
            )

    if re <= onset and inlet_perturbation > 0 and shape:
        warnings.append(
            f"The wake is steady at Re={re:g} (shedding starts near Re≈{onset:g} for a "
            f"{shape}); the inlet perturbation only forces noise into Cd — set it to 0."
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
    blockage: Optional[float] = None,
) -> ValidationReport:
    re = float(params.get("re", "100"))
    u0 = float(params.get("u0", "0.05"))
    d = reference_length_cells(mode, shape, params)
    tau = derive_tau(re, u0, d)

    b_status, b_msg = assess_literature(mode=mode, shape=shape, re=re, cd=cd, blockage=blockage)
    if grid_cd_values:
        g_status, g_msg = assess_grid_convergence(grid_cd_values)
    else:
        g_status, g_msg = "n/a", "Run Grid Study to check resolution independence."

    bc_warnings = validate_bc_config(
        mode=mode,
        wall_bc=params.get("wall_bc") or "slip",
        outlet_bc=params.get("outlet_bc") or "convective",
        inlet_bc=params.get("inlet_bc") or "velocity",
        re=re,
        inlet_perturbation=float(params.get("inlet_perturbation", "0") or "0"),
        shape=shape,
    )
    c_status, c_msg = assess_collision(
        re=re, tau=tau, collision=params.get("collision") or "bgk",
    )

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


def validate_scalar_diffusion_profile(
    scalar: np.ndarray,
    *,
    T_hot: float,
    T_cold: float,
    wall_axis: int = -2,
    solid: Optional[np.ndarray] = None,
    pass_rmse: float = 0.03,
    warn_rmse: float = 0.06,
) -> ScalarValidationResult:
    """
    Validate a wall-bounded passive-scalar diffusion profile against the linear steady solution.

    This benchmark corresponds to pure diffusion between two isothermal/isoconcentration walls.
    For 2D fields `(Ny, Nx)`, use `wall_axis=0`. For 3D fields `(Nz, Ny, Nx)`, use `wall_axis=1`.
    """
    field = np.asarray(scalar, dtype=np.float64)
    if field.ndim not in (2, 3):
        raise ValueError("scalar must be a 2D or 3D field.")
    axis = wall_axis if wall_axis >= 0 else field.ndim + wall_axis
    if axis < 0 or axis >= field.ndim:
        raise ValueError("wall_axis out of bounds for scalar field.")

    if solid is not None:
        mask = np.asarray(solid, dtype=bool)
        if mask.shape != field.shape:
            raise ValueError("solid mask must match scalar field shape.")
        field = field.copy()
        field[mask] = np.nan

    reduce_axes = tuple(i for i in range(field.ndim) if i != axis)
    profile = np.nanmean(field, axis=reduce_axes)
    coords = np.arange(profile.size, dtype=np.float64)
    reference = float(T_hot) + (float(T_cold) - float(T_hot)) * coords / max(profile.size - 1, 1)
    interior = slice(1, -1) if profile.size > 2 else slice(0, profile.size)
    error = np.asarray(profile[interior] - reference[interior], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(error * error))) if error.size else 0.0
    max_error = float(np.max(np.abs(error))) if error.size else 0.0

    if rmse <= pass_rmse:
        status = "pass"
    elif rmse <= warn_rmse:
        status = "warn"
    else:
        status = "fail"

    return ScalarValidationResult(
        status=status,
        message=f"Scalar diffusion RMSE={rmse:.4f}, max error={max_error:.4f} against linear steady profile.",
        rmse=rmse,
        max_error=max_error,
        mean_profile=[float(x) for x in profile],
        reference_profile=[float(x) for x in reference],
    )


def run_scalar_diffusion_benchmark_2d(
    *,
    Ny: int = 32,
    Nx: int = 8,
    omega: float = 1.5,
    alpha_T: float = 0.1,
    steps: int = 3000,
    T_hot: float = 1.0,
    T_cold: float = 0.0,
    backend: str = "numpy",
) -> Dict[str, Any]:
    """
    Run the canonical 2D passive-scalar diffusion benchmark.

    The reference solution is linear in the wall-normal direction for zero flow,
    with prescribed bottom/top scalar values.
    """
    from .lbm.solver import Solver

    solid = np.zeros((int(Ny), int(Nx)), dtype=bool)
    solver = Solver(
        Ny=int(Ny),
        Nx=int(Nx),
        solid=solid,
        omega=float(omega),
        u0=0.0,
        D=max(float(Ny) / 2.0, 1.0),
        backend=backend,
        inlet_bc="pressure",
        outlet_bc="pressure",
        thermal=True,
        T_hot=float(T_hot),
        T_cold=float(T_cold),
        alpha_T=float(alpha_T),
        buoyancy=False,
    )
    result = solver.run(steps=int(steps), check_every=max(int(steps), 1), verbose=False)
    scalar_validation = validate_scalar_diffusion_profile(
        result["scalar"],
        T_hot=T_hot,
        T_cold=T_cold,
        wall_axis=0,
        solid=solid,
    )
    result["scalar_validation"] = _json_ready_report(scalar_validation)
    return result


def _status_rank(status: str) -> int:
    return {"pass": 0, "n/a": 1, "warn": 2, "fail": 3}.get(str(status).lower(), 2)


def _combine_statuses(statuses: List[str]) -> str:
    if not statuses:
        return "n/a"
    return max(statuses, key=_status_rank)


def _json_ready_report(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready_report(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _json_ready_report(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_report(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _blockage_component(mode: str, shape: str, params: Dict[str, Any]) -> Dict[str, Any]:
    d = reference_length_cells(mode, shape, params)
    ny = max(float(params.get("ny", 1) or 1), 1.0)
    if mode == "3d":
        nz = max(float(params.get("nz", ny) or ny), 1.0)
        ratio_y = d / ny
        ratio_z = d / nz
        ratio = max(ratio_y, ratio_z)
        detail = f"Cross-stream blockage y={ratio_y*100:.1f}%, z={ratio_z*100:.1f}%."
    else:
        ratio = d / ny
        detail = f"Blockage ratio D/Ny={ratio*100:.1f}%."
    if ratio < 0.10:
        status = "pass"
    elif ratio < 0.20:
        status = "warn"
    else:
        status = "fail"
    if mode == "3d" and str(shape).lower() == "sphere":
        re = max(float(params.get("re", 100.0) or 100.0), 1e-6)
        detail += (f" Expected to raise a sphere's Cd by ~{(sphere_confinement_factor(ratio, re)-1)*100:.0f}%"
                   " (measured blockage sweep).")
    return {
        "status": status,
        "value": float(ratio),
        "message": detail,
    }


def _domain_length_component(mode: str, shape: str, params: Dict[str, Any]) -> Dict[str, Any]:
    nx = max(float(params.get("nx", 1) or 1), 1.0)
    d = max(reference_length_cells(mode, shape, params), 1.0)
    cx_frac = float(params.get("cx_frac", 1.0 / 3.0) or (1.0 / 3.0))
    center_x = cx_frac * nx
    upstream = center_x / d
    downstream = max(nx - center_x, 0.0) / d
    if upstream >= 5.0 and downstream >= 15.0:
        status = "pass"
    elif upstream >= 3.0 and downstream >= 8.0:
        status = "warn"
    else:
        status = "fail"
    return {
        "status": status,
        "value": {"upstream_D": float(upstream), "downstream_D": float(downstream)},
        "message": f"Upstream length={upstream:.1f}D, downstream length={downstream:.1f}D.",
    }


# ---------------------------------------------------------------------------
# Statistics of a correlated time series
# ---------------------------------------------------------------------------
#
# A force coefficient sampled every LBM step is not a set of independent
# measurements.  The trace carries acoustic modes of the tunnel cross-section
# (period ~ span / c_s: 83 and 59 steps in a 48-cell tunnel, barely damped at
# low viscosity), whatever the inlet perturbation forces (a 37-step travelling
# wave), and above the shedding threshold the shedding cycle itself.
# Neighbouring samples are nearly identical, so sigma / sqrt(N) over-counts the
# information in the run: by about 3x on a Re=100 sphere trace, more on slow
# shedding.  Everything below works in terms of the *effective* sample size.

def integrated_autocorr_time(x: Any, c: float = 5.0) -> float:
    """
    Integrated autocorrelation time, in samples, with Sokal's automatic window.

    tau = 1/2 + sum_{t=1..M} rho(t), with M the smallest lag satisfying
    M >= c * tau(M).  Independent samples give 1/2; the effective sample size
    is N / (2 tau).  Clamped below at 1/2: an oscillating trace can make the
    raw sum smaller, but claiming better-than-independent samples is never the
    safe direction for an uncertainty.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0.5
    x = x - x.mean()
    if not np.any(x):
        return 0.5
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    taus = 0.5 + np.cumsum(acf[1:])
    lags = np.arange(1, n)
    ok = lags >= c * taus
    tau = float(taus[int(np.argmax(ok))]) if ok.any() else float(taus[-1])
    return max(tau, 0.5)


def _t_quantile_975(dof: float) -> float:
    """Student-t 97.5% quantile, Cornish-Fisher to O(1/dof^2); 1.96 as dof -> inf."""
    z = 1.959963984540054
    if not np.isfinite(dof) or dof > 1e6:
        return z
    v = max(float(dof), 1.0)
    return (z + (z ** 3 + z) / (4 * v)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v * v))


def mean_uncertainty(x: Any, n_batches: int = 10) -> Dict[str, Any]:
    """
    Mean of a correlated series and an honest 95% interval for it.

    Two estimates of the standard error, and the larger is kept:
      * sigma / sqrt(N_eff) with N_eff = N / (2 tau_int);
      * batch means -- the scatter of the means of ``n_batches`` equal blocks,
        which also catches slow drift that the autocorrelation sum can miss.
    The 95% half-width uses a Student-t quantile on the effective degrees of
    freedom, which matters when N_eff is ten rather than ten thousand.

    Also reports a stationarity check: the difference between the means of
    the window's two halves, against its own uncertainty.  A run still
    relaxing from its start-up shows up here even when every other number
    looks tidy.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": None, "sigma": None, "sem95": None, "n_eff": 0.0,
                "tau_int": None, "drift": None, "drift_sigma": None, "stationary": None}
    mean = float(x.mean())
    sigma = float(x.std(ddof=1)) if n > 1 else 0.0
    tau = integrated_autocorr_time(x)
    n_eff = max(n / (2.0 * tau), 1.0)
    sem_acf = sigma / np.sqrt(n_eff)

    sem_batch = 0.0
    b = min(n_batches, n // 2)
    if b >= 2:
        L = n // b
        means = x[: L * b].reshape(b, L).mean(axis=1)
        sem_batch = float(means.std(ddof=1) / np.sqrt(b))
    if sem_batch > sem_acf:
        sem, dof = sem_batch, b - 1
    else:
        sem, dof = sem_acf, n_eff - 1
    sem95 = float(_t_quantile_975(dof) * sem)

    drift = drift_sigma = None
    stationary = None
    if n >= 8:
        h1, h2 = x[: n // 2], x[n // 2:]
        s1 = h1.std(ddof=1) / np.sqrt(max(h1.size / (2 * integrated_autocorr_time(h1)), 1.0))
        s2 = h2.std(ddof=1) / np.sqrt(max(h2.size / (2 * integrated_autocorr_time(h2)), 1.0))
        drift = float(h2.mean() - h1.mean())
        drift_sigma = float(np.hypot(s1, s2))
        stationary = bool(abs(drift) <= 2.0 * drift_sigma + 1e-12 * max(abs(mean), 1.0))
    return {"n": n, "mean": mean, "sigma": sigma, "tau_int": float(tau), "n_eff": float(n_eff),
            "sem": float(sem), "sem95": sem95, "drift": drift, "drift_sigma": drift_sigma,
            "stationary": stationary}


def _statistical_component(mode: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Statistical uncertainty of the mean Cd over the averaging window.

    ``result["analysis_window"]`` sets the window (samples from the end); the
    default is the last fifth of the run, matching the CLI's printed mean.

    Before this used 1.96 sigma / sqrt(N) with N the raw sample count, which
    treats every step as an independent measurement and understated the
    uncertainty by roughly sqrt(2 tau_int) -- about 3x on a Re=100 sphere.
    """
    cd_history = np.asarray(result.get("Cd_history", []), dtype=np.float64)
    if cd_history.size == 0:
        return {"status": "n/a", "value": None, "message": "No coefficient history available."}
    window = result.get("analysis_window")
    if not window:
        window = max(int(min(cd_history.size, max(cd_history.size // 5, 20))), 1)
    window = int(min(max(int(window), 1), cd_history.size))
    st = mean_uncertainty(cd_history[-window:])
    lift_key = "Cl_history" if mode == "2d" else "Cly_history"
    lift_history = np.asarray(result.get(lift_key, []), dtype=np.float64)
    lift = mean_uncertainty(lift_history[-window:]) if lift_history.size else {"sem95": None}
    rel_cd = abs(st["sem95"] or 0.0) / max(abs(st["mean"] or 0.0), 1e-12)
    if rel_cd < 0.01:
        status = "pass"
    elif rel_cd < 0.05:
        status = "warn"
    else:
        status = "fail"
    message = (f"95% statistical uncertainty of the mean Cd is {rel_cd*100:.2f}% "
               f"({st['n_eff']:.0f} effective samples of {window}; "
               f"correlation time {st['tau_int']:.0f} steps).")
    if st["stationary"] is False:
        status = "fail" if status == "fail" else "warn"
        message += (f" The two halves of the window differ by {st['drift']:+.4f} "
                    f"(> 2 x {st['drift_sigma']:.4f}): still drifting, run longer.")
    return {
        "status": status,
        "value": {
            "window": int(window),
            "cd_mean": st["mean"],
            "cd_sigma": st["sigma"],
            "cd_sem95": st["sem95"],
            "cd_relative_sem95": float(rel_cd),
            "n_eff": st["n_eff"],
            "tau_int": st["tau_int"],
            "drift": st["drift"],
            "stationary": st["stationary"],
            "lift_sem95": lift.get("sem95"),
        },
        "message": message,
    }


def _resolution_estimate(mode: str, shape: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A-priori resolution check when no grid study is attached.

    Counts cells across the body and across the laminar boundary layer, whose
    thickness scales as D / sqrt(Re).  It is a heuristic -- only a grid study
    measures the discretisation error -- but it catches the common case of a
    body a dozen cells across at Re ~ 100, where the boundary layer is barely
    one cell thick.
    """
    d = reference_length_cells(mode, shape, params)
    re = max(float(params.get("re", 100.0) or 100.0), 1e-6)
    bl = d / np.sqrt(re)
    if d >= 30 and bl >= 3:
        status = "pass"
    elif d >= 10 and bl >= 1:
        status = "warn"
    else:
        status = "fail"
    return {
        "status": status,
        "value": {"cells_across_body": float(d), "cells_across_boundary_layer": float(bl)},
        "message": (f"No grid study; estimate only: {d:.0f} cells across the body, "
                    f"boundary layer ≈ D/√Re = {bl:.1f} cells. Run Grid Study to measure."),
    }


def _discretization_component(
    result: Dict[str, Any], mode: Optional[str] = None,
    shape: Optional[str] = None, params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    grid_cd_values = result.get("grid_cd_values")
    if not grid_cd_values:
        if mode and shape and params:
            return _resolution_estimate(mode, shape, params)
        return {
            "status": "n/a",
            "value": None,
            "message": "No grid study attached to this run.",
        }
    values = [float(value) for value in grid_cd_values]
    status, message = assess_grid_convergence(values)
    observed_order = observed_order_of_convergence(values)
    extrapolated = richardson_extrapolation(values, observed_order=observed_order)
    fine_error = None
    if extrapolated is not None and values:
        fine_error = abs(values[-1] - extrapolated) / max(abs(extrapolated), 1e-12)
    return {
        "status": status,
        "value": {
            "cd_values": values,
            "observed_order": observed_order,
            "extrapolated_cd": extrapolated,
            "fine_relative_error": fine_error,
        },
        "message": message,
    }


def _convergence_component(result: Dict[str, Any]) -> Dict[str, Any]:
    convergence = _json_ready_report(result.get("convergence_report"))
    strouhal = _json_ready_report(result.get("strouhal_report"))
    stop_reason = str(result.get("stop_reason", "max_steps"))
    if convergence is None and strouhal is None:
        return {
            "status": "n/a",
            "value": {"stop_reason": stop_reason},
            "message": "No convergence metadata recorded.",
        }
    status = "pass" if stop_reason == "auto_converged" else "warn"
    return {
        "status": status,
        "value": {
            "stop_reason": stop_reason,
            "convergence_report": convergence,
            "strouhal_report": strouhal,
        },
        "message": f"Run stopped with reason '{stop_reason}'.",
    }


def _bc_sensitivity_component(
    mode: str, params: Dict[str, Any], shape: Optional[str] = None
) -> Dict[str, Any]:
    warnings = validate_bc_config(
        mode=mode,
        wall_bc=str(params.get("wall_bc", "slip")),
        outlet_bc=str(params.get("outlet_bc", "convective")),
        inlet_bc=str(params.get("inlet_bc", "velocity")),
        re=float(params.get("re", 100.0) or 100.0),
        inlet_perturbation=float(params.get("inlet_perturbation", 0.0) or 0.0),
        shape=shape,
    )
    streamwise_bc = str(params.get("streamwise_bc", "") or "")
    if streamwise_bc in {"periodic", "recycling"}:
        warnings.append(
            f"Streamwise BC '{streamwise_bc}' is configured; compare against alternate inlet/outlet treatments for sensitivity."
        )
    status = "pass" if not warnings else "warn"
    return {
        "status": status,
        "value": {"warnings": warnings, "streamwise_bc": streamwise_bc or None},
        "message": "No obvious BC sensitivity concerns." if not warnings else "Boundary-condition sensitivity review recommended.",
    }


def build_uncertainty_report(
    *,
    mode: str,
    shape: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
) -> UncertaintyReport:
    """
    Build a JSON-ready uncertainty budget from one completed run.

    This focuses on the major research-review axes already visible in the codebase:
    discretization, blockage, domain extent, statistical uncertainty, convergence,
    and boundary-condition sensitivity.
    """
    components = {
        "discretization": _discretization_component(result, mode, shape, params),
        "blockage": _blockage_component(mode, shape, params),
        "domain_length": _domain_length_component(mode, shape, params),
        "statistical": _statistical_component(mode, result),
        "convergence": _convergence_component(result),
        "bc_sensitivity": _bc_sensitivity_component(mode, params, shape),
    }
    overall_status = _combine_statuses([component["status"] for component in components.values()])
    summary = "; ".join(
        f"{name}={component['status']}" for name, component in components.items()
    )
    return UncertaintyReport(
        overall_status=overall_status,
        summary=summary,
        components=components,
    )
