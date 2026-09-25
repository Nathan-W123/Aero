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



# ---------------------------------------------------------------------------
# Sphere confinement band
# ---------------------------------------------------------------------------

def test_sphere_band_is_unconfined_at_zero_blockage():
    from aero.benchmarks import sphere_expected_cd, schiller_naumann_cd
    sn, conf, _ = sphere_expected_cd(20.0, 0.0)
    assert sn == conf == schiller_naumann_cd(20.0)


def test_sphere_band_contains_the_measured_sweep_points():
    """
    The two confinement measurements the slope was calibrated on, with the
    frontal-area normalisation in place.  If someone retunes the slope or the
    allowance, these have to keep passing.
    """
    from aero.benchmarks import assess_literature
    for blockage, cd in ((14 / 48, 3.4456), (14 / 96, 3.0063), (14 / 144, 2.9457)):
        status, msg = assess_literature(mode="3d", shape="sphere", re=20.0, cd=cd, blockage=blockage)
        assert status == "pass", msg


def test_sphere_band_without_blockage_is_a_wide_envelope():
    from aero.benchmarks import literature_cd_range, schiller_naumann_cd
    lo, hi, note = literature_cd_range("3d", "sphere", 20.0)
    sn = schiller_naumann_cd(20.0)
    assert lo == pytest.approx(0.85 * sn) and hi == pytest.approx(1.6 * sn)
    assert "blockage" in note


def test_sphere_band_at_re100_accepts_a_ten_percent_blockage_run():
    """A 192^3 sphere, r=10, read 1.18 once normalised correctly."""
    from aero.benchmarks import assess_literature
    status, msg = assess_literature(mode="3d", shape="sphere", re=100.0, cd=1.182, blockage=20 / 192)
    assert status == "pass", msg


def test_the_old_d_squared_number_now_fails_the_sharp_band():
    """0.928 was the same run before the fix; it must no longer look right."""
    from aero.benchmarks import assess_literature
    status, _ = assess_literature(mode="3d", shape="sphere", re=100.0, cd=0.928, blockage=20 / 192)
    assert status != "pass"


def test_sphere_confinement_weakens_with_re():
    """Measured: blockage costs a sphere roughly half as much drag at Re=100 as at Re=20."""
    from aero.benchmarks import sphere_confinement_factor, sphere_confinement_slope
    assert sphere_confinement_slope(20.0) == pytest.approx(1.00)
    assert sphere_confinement_slope(100.0) == pytest.approx(0.58)
    assert sphere_confinement_slope(20.0) > sphere_confinement_slope(45.0) > sphere_confinement_slope(100.0)
    assert sphere_confinement_slope(5.0) == sphere_confinement_slope(20.0)       # held outside
    assert sphere_confinement_slope(300.0) == sphere_confinement_slope(100.0)
    assert sphere_confinement_factor(None, 100.0) == 1.0


def test_sphere_band_contains_the_re100_confinement_pair():
    """The Re=100 pair the slope was calibrated on (r=7, 96-long tunnel, Cd on pi r^2)."""
    from aero.benchmarks import assess_literature, sphere_expected_cd
    for blockage, cd in ((14 / 48, 1.3607), (14 / 96, 1.2629)):
        status, msg = assess_literature(mode="3d", shape="sphere", re=100.0, cd=cd, blockage=blockage)
        assert status == "pass", msg
    # 29% blockage at Re=100: the Re=20 slope said ~1.44; the measured one says ~1.28.
    _, conf, _ = sphere_expected_cd(100.0, 14 / 48)
    assert conf == pytest.approx(1.276, abs=0.01)


# ---------------------------------------------------------------------------
# Uncertainty of a correlated mean
# ---------------------------------------------------------------------------

def _ar1(phi, n, seed, burn=500):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + burn)
    x = np.empty_like(e)
    x[0] = 0.0
    for i in range(1, e.size):
        x[i] = phi * x[i - 1] + e[i]
    return x[burn:]


def test_autocorr_time_is_one_half_for_independent_samples():
    from aero.benchmarks import integrated_autocorr_time
    x = np.random.default_rng(1).standard_normal(20000)
    assert integrated_autocorr_time(x) == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("phi", [0.5, 0.9])
def test_autocorr_time_matches_ar1_exactly(phi):
    """AR(1) has tau_int = (1 + phi) / (2 (1 - phi)) in closed form."""
    from aero.benchmarks import integrated_autocorr_time
    tau = integrated_autocorr_time(_ar1(phi, 60000, seed=2))
    assert tau == pytest.approx((1 + phi) / (2 * (1 - phi)), rel=0.1)


def test_95_percent_interval_actually_covers_95_percent():
    """
    The old interval, 1.96 sigma / sqrt(N) over raw steps, claimed 95% and
    covered the true mean about one time in five on a correlated signal.
    A coefficient trace is exactly that kind of signal.
    """
    from aero.benchmarks import mean_uncertainty
    trials, hit, naive = 150, 0, 0
    for k in range(trials):
        x = _ar1(0.95, 1500, seed=100 + k)
        m = mean_uncertainty(x)
        hit += abs(m["mean"]) <= m["sem95"]
        naive += abs(x.mean()) <= 1.96 * x.std(ddof=1) / np.sqrt(x.size)
    assert hit / trials >= 0.85
    assert naive / trials <= 0.40          # documents what the old estimator did


def test_statistical_component_counts_effective_not_raw_samples():
    from aero.benchmarks import _statistical_component
    x = 1.3 + 0.05 * _ar1(0.98, 3000, seed=7) / 5.0
    comp = _statistical_component("3d", {"Cd_history": x.tolist(), "analysis_window": 1500})
    v = comp["value"]
    naive = 1.96 * np.std(x[-1500:], ddof=1) / np.sqrt(1500)
    assert v["n_eff"] < 1500 / 10
    assert v["cd_sem95"] > 3 * naive
    assert "effective samples" in comp["message"]


def test_drift_is_reported_when_the_run_has_not_settled():
    from aero.benchmarks import mean_uncertainty, _statistical_component
    rng = np.random.default_rng(3)
    x = 1.0 + np.linspace(0.0, 0.2, 2000) + 0.002 * rng.standard_normal(2000)
    assert mean_uncertainty(x)["stationary"] is False
    comp = _statistical_component("3d", {"Cd_history": x.tolist(), "analysis_window": 2000})
    assert comp["status"] in ("warn", "fail") and "drifting" in comp["message"]


def test_a_settled_signal_is_stationary():
    from aero.benchmarks import mean_uncertainty
    x = 1.2 + 0.01 * np.random.default_rng(4).standard_normal(3000)
    assert mean_uncertainty(x)["stationary"] is True


# ---------------------------------------------------------------------------
# Shedding advice and a-priori resolution
# ---------------------------------------------------------------------------

def test_sphere_below_its_onset_is_told_to_drop_the_perturbation():
    """
    47 is the 2D cylinder's shedding onset.  Applying it to a sphere told users
    to perturb a wake that is steady until Re ~ 210.
    """
    w = validate_bc_config(mode="3d", wall_bc="slip", outlet_bc="convective",
                           re=100.0, inlet_perturbation=0.02, shape="sphere")
    assert any("steady" in m and "set it to 0" in m for m in w)
    w0 = validate_bc_config(mode="3d", wall_bc="slip", outlet_bc="convective",
                            re=100.0, inlet_perturbation=0.0, shape="sphere")
    assert not any("shedding may not trigger" in m for m in w0)


def test_sphere_above_its_onset_still_gets_the_shedding_hint():
    w = validate_bc_config(mode="3d", wall_bc="slip", outlet_bc="convective",
                           re=300.0, inlet_perturbation=0.0, shape="sphere")
    assert any("270" in m for m in w)


def test_resolution_estimate_flags_a_thin_boundary_layer():
    from aero.benchmarks import build_uncertainty_report
    rep = build_uncertainty_report(
        mode="3d", shape="sphere",
        params={"radius": "7", "re": "100", "nx": "96", "ny": "48", "nz": "48"},
        result={"Cd_history": [1.0] * 100},
    )
    d = rep.components["discretization"]
    assert d["status"] == "warn"
    assert d["value"]["cells_across_boundary_layer"] == pytest.approx(1.4)
    assert "6-10% high" in d["message"]          # the measured grid bias, for scale


def test_blockage_message_quantifies_the_sphere_bias():
    from aero.benchmarks import build_uncertainty_report
    rep = build_uncertainty_report(
        mode="3d", shape="sphere",
        params={"radius": "7", "re": "100", "nx": "96", "ny": "48", "nz": "48"},
        result={"Cd_history": [1.0] * 100},
    )
    b = rep.components["blockage"]
    assert b["status"] == "fail" and "raise a sphere's Cd" in b["message"]
