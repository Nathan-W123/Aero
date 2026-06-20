"""Unit tests for literature / grid / BC validation helpers."""

import math

import pytest

from aero.benchmarks import (
    assess_collision,
    assess_grid_convergence,
    assess_literature,
    build_validation_report,
    schiller_naumann_cd,
    scale_params_for_grid,
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
