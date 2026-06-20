"""Unit tests for literature / grid / BC validation helpers."""

import math

import numpy as np
import pytest

from aero.benchmarks import (
    assess_collision,
    assess_grid_convergence,
    assess_literature,
    backward_facing_step_reattachment_length,
    build_uncertainty_report,
    build_validation_report,
    channel_friction_coefficient,
    dean_correlation_cf,
    grid_study,
    observed_order_of_convergence,
    richardson_extrapolation,
    run_scalar_diffusion_benchmark_2d,
    schiller_naumann_cd,
    scale_params_for_grid,
    validate_scalar_diffusion_profile,
    validate_bc_config,
)


def test_schiller_naumann_re20():
    cd = schiller_naumann_cd(20.0)
    assert 2.0 < cd < 3.0


def test_literature_sphere_re20_pass():
    status, _ = assess_literature(mode="3d", shape="sphere", re=20.0, cd=2.5)
    assert status == "pass"


def test_literature_cylinder_re100_fail():
    status, _ = assess_literature(mode="2d", shape="cylinder", re=100.0, cd=5.0)
    assert status == "fail"


def test_grid_convergence_independent():
    status, msg = assess_grid_convergence([2.0, 2.05, 2.06])
    assert status == "pass"
    assert "5%" in msg or "independent" in msg.lower()


def test_grid_convergence_not_independent():
    status, _ = assess_grid_convergence([2.0, 2.5, 3.0])
    assert status == "fail"


def test_bc_warning_shedding():
    warnings = validate_bc_config(
        mode="2d",
        wall_bc="slip",
        outlet_bc="convective",
        inlet_bc="velocity",
        re=100.0,
        inlet_perturbation=0.0,
    )
    assert any("perturbation" in w.lower() or "shedding" in w.lower() for w in warnings)


def test_collision_mrt_stable():
    status, _ = assess_collision(re=100.0, tau=0.65, collision="mrt")
    assert status == "pass"


def test_collision_bgk_high_re_warn():
    status, _ = assess_collision(re=300.0, tau=0.62, collision="bgk")
    assert status == "warn"


def test_scale_params_doubles_grid():
    scaled = scale_params_for_grid({"nx": "64", "ny": "48", "radius": "10"}, 2.0, "3d")
    assert scaled["nx"] == "128"
    assert scaled["ny"] == "96"
    assert scaled["radius"] == "20"


def test_build_validation_report_shape():
    report = build_validation_report(
        mode="3d",
        shape="sphere",
        params={"re": "20", "u0": "0.05", "radius": "10", "wall_bc": "slip", "outlet_bc": "convective", "collision": "bgk"},
        cd=2.4,
    )
    assert report.benchmark_status in {"pass", "warn", "fail", "n/a"}
    assert report.collision_status in {"pass", "warn", "fail", "n/a"}


def test_observed_order_and_richardson():
    cds = [1.40, 1.20, 1.15]
    order = observed_order_of_convergence(cds)
    ext = richardson_extrapolation(cds, observed_order=order)
    assert order is not None
    assert order > 0.0
    assert ext is not None
    assert ext < cds[-1]


def test_grid_study_runner_returns_summary():
    def runner(resolution: float):
        return {"Cd_mean": 1.0 + 0.2 / (resolution * resolution)}

    study = grid_study(runner, [0.5, 1.0, 2.0])
    assert study.status in {"pass", "warn", "fail"}
    assert len(study.cd_values) == 3
    assert study.observed_order is not None


def test_dean_correlation_and_channel_cf_positive():
    cf_ref = dean_correlation_cf(5000.0)
    cf_meas = channel_friction_coefficient(
        body_force_x=2.0e-5,
        hydraulic_diameter=20.0,
        bulk_velocity=0.08,
    )
    assert 0.0 < cf_ref < 0.1
    assert cf_meas > 0.0


def test_backward_facing_step_reattachment_length():
    ux = np.ones((8, 40), dtype=float)
    ux[1, 11:17] = -0.05
    length = backward_facing_step_reattachment_length(ux, step_index=10)
    assert length == pytest.approx(7.0)


def test_build_uncertainty_report_has_expected_components():
    report = build_uncertainty_report(
        mode="2d",
        shape="cylinder",
        params={
            "re": 100.0,
            "u0": 0.05,
            "radius": 20.0,
            "nx": 200,
            "ny": 100,
            "cx_frac": 1.0 / 3.0,
            "wall_bc": "slip",
            "inlet_bc": "velocity",
            "outlet_bc": "convective",
        },
        result={
            "Cd_mean": 1.4,
            "Cd_history": [1.4 + 1e-4 * math.sin(i) for i in range(200)],
            "Cl_history": [0.1 * math.sin(0.1 * i) for i in range(200)],
            "grid_cd_values": [1.6, 1.48, 1.43],
            "stop_reason": "auto_converged",
        },
    )
    assert report.overall_status in {"pass", "warn", "fail", "n/a"}
    assert set(report.components) == {
        "discretization",
        "blockage",
        "domain_length",
        "statistical",
        "convergence",
        "bc_sensitivity",
    }
    assert report.components["blockage"]["value"] > 0.0


def test_validate_scalar_diffusion_profile_linear_field_passes():
    ny, nx = 16, 5
    y = np.arange(ny, dtype=float)
    reference = 1.0 - y / (ny - 1)
    scalar = np.repeat(reference[:, None], nx, axis=1)
    report = validate_scalar_diffusion_profile(scalar, T_hot=1.0, T_cold=0.0, wall_axis=0)
    assert report.status == "pass"
    assert report.rmse < 1e-12


def test_run_scalar_diffusion_benchmark_2d_returns_validation():
    result = run_scalar_diffusion_benchmark_2d(Ny=24, Nx=8, steps=1500, alpha_T=0.1)
    assert result["scalar"] is not None
    assert result["scalar_validation"]["status"] in {"pass", "warn"}
    assert result["scalar_validation"]["rmse"] < 0.08
