"""Tests for case/result reporting artifacts."""

import json

from aero.case import SimulationCase


def test_save_results_writes_uncertainty_and_validation_reports(tmp_path):
    case = SimulationCase(
        name="cylinder_re100_test",
        cases_root=str(tmp_path),
        config={
            "mode": "2d",
            "shape": "cylinder",
            "re": 100.0,
            "u0": 0.05,
            "radius": 20.0,
            "nx": 200,
            "ny": 100,
            "cx_frac": 1.0 / 3.0,
            "wall_bc": "slip",
            "inlet_bc": "velocity",
            "outlet_bc": "convective",
            "collision": "bgk",
        },
    )
    result = {
        "Cd_mean": 1.42,
        "Cd_std": 0.03,
        "Cl_mean": 0.0,
        "Cl_std": 0.01,
        "Cd_history": [1.4, 1.41, 1.43, 1.42],
        "Cl_history": [0.01, -0.01, 0.01, -0.01],
        "stop_reason": "max_steps",
    }
    case.save_results(result, elapsed_seconds=12.5)

    payload = json.loads(case.results_path.read_text())
    assert payload["shape"] == "cylinder"
    assert payload["validation_report"]["benchmark_status"] in {"pass", "warn", "fail", "n/a"}
    assert payload["uncertainty_report"]["overall_status"] in {"pass", "warn", "fail", "n/a"}
    assert "blockage" in payload["uncertainty_report"]["components"]

    history = json.loads((case.case_dir / "history.json").read_text())
    assert history["Cd_history"] == [1.4, 1.41, 1.43, 1.42]


def test_save_results_handles_null_3d_inlet_bc(tmp_path):
    """3D CLI cases may omit inlet_bc; validation must not crash on save."""
    case = SimulationCase(
        name="mesh_re10_test",
        cases_root=str(tmp_path),
        config={
            "mode": "3d",
            "shape": "mesh",
            "re": 10.0,
            "u0": 0.05,
            "nx": 64,
            "ny": 64,
            "nz": 64,
            "wall_bc": "slip",
            "inlet_bc": None,
            "outlet_bc": "convective",
            "collision": "bgk",
        },
    )
    result = {
        "Cd_mean": 0.8,
        "Cd_std": 0.001,
        "Cly_mean": 0.0,
        "Cly_std": 0.0,
        "Clz_mean": 0.0,
        "Clz_std": 0.0,
        "Cd_history": [0.81, 0.80, 0.80],
        "Cly_history": [0.0, 0.0, 0.0],
        "Clz_history": [0.0, 0.0, 0.0],
    }
    case.save_results(result, elapsed_seconds=37.0)

    payload = json.loads(case.results_path.read_text())
    assert payload["Cd_mean"] == 0.8
    assert "validation_report" in payload


def test_save_results_persists_scalar_stats(tmp_path):
    case = SimulationCase(
        name="scalar_case",
        cases_root=str(tmp_path),
        config={"mode": "2d", "shape": "rectangle", "re": 10.0, "nx": 40, "ny": 20},
    )
    result = {
        "Cd_mean": 0.1,
        "Cd_std": 0.0,
        "Cl_mean": 0.0,
        "Cl_std": 0.0,
        "Cd_history": [0.1],
        "Cl_history": [0.0],
        "scalar_stats": {"enabled": True, "mean": 0.5, "min": 0.0, "max": 1.0},
    }
    case.save_results(result, elapsed_seconds=1.0)

    payload = json.loads(case.results_path.read_text())
    assert payload["scalar_stats"]["enabled"] is True
    assert payload["scalar_stats"]["mean"] == 0.5


def test_save_results_persists_scalar_validation(tmp_path):
    case = SimulationCase(
        name="scalar_validation_case",
        cases_root=str(tmp_path),
        config={"mode": "2d", "shape": "rectangle", "re": 10.0, "nx": 40, "ny": 20},
    )
    result = {
        "Cd_mean": 0.1,
        "Cd_std": 0.0,
        "Cl_mean": 0.0,
        "Cl_std": 0.0,
        "Cd_history": [0.1],
        "Cl_history": [0.0],
        "scalar_validation": {"status": "pass", "rmse": 0.01, "max_error": 0.02},
    }
    case.save_results(result, elapsed_seconds=1.0)

    payload = json.loads(case.results_path.read_text())
    assert payload["scalar_validation"]["status"] == "pass"
    assert payload["scalar_validation"]["rmse"] == 0.01
