"""Tests for time-averaged flow statistics."""

import numpy as np
import pytest

from aero.statistics import FlowStatistics
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D


def _cylinder(ny, nx, r=6.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((yy - ny / 2) ** 2 + (xx - nx / 4) ** 2) <= r * r


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

def test_mean_matches_numpy():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 4, 5)) * 0.01 + 0.05
    stats = FlowStatistics(shape=(4, 5), components=2)
    for s in samples:
        stats.update(s, 2.0 * s)
    np.testing.assert_allclose(stats.mean_velocity[0], samples.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.mean_velocity[1], 2.0 * samples.mean(axis=0), rtol=1e-12)


def test_variance_matches_numpy():
    rng = np.random.default_rng(1)
    samples = rng.standard_normal((300, 3, 3)) * 0.02 + 0.04
    stats = FlowStatistics(shape=(3, 3), components=2)
    for s in samples:
        stats.update(s, np.zeros_like(s))
    np.testing.assert_allclose(
        stats.reynolds_stress("uu"), samples.var(axis=0), rtol=1e-10
    )


def test_cross_stress_matches_numpy():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((250, 2, 2)) * 0.01 + 0.03
    b = rng.standard_normal((250, 2, 2)) * 0.02 - 0.01
    stats = FlowStatistics(shape=(2, 2), components=2)
    for ia, ib in zip(a, b):
        stats.update(ia, ib)
    expect = ((a - a.mean(axis=0)) * (b - b.mean(axis=0))).mean(axis=0)
    np.testing.assert_allclose(stats.reynolds_stress("uv"), expect, rtol=1e-9, atol=1e-18)


def test_welford_survives_a_tiny_fluctuation_on_a_large_mean():
    """
    Fluctuation/mean of 1e-5 is ordinary in LBM; the variance must survive it.

    A streaming sum-of-squares accumulator loses this in cancellation.
    """
    rng = np.random.default_rng(3)
    mean_level, fluct = 5e-2, 5e-7
    stats = FlowStatistics(shape=(1, 1), components=2)
    values = []
    for _ in range(5000):
        v = mean_level + fluct * rng.standard_normal()
        values.append(v)
        stats.update(np.array([[v]]), np.zeros((1, 1)))
    got = stats.reynolds_stress("uu")[0, 0]
    assert got > 0.0
    assert got == pytest.approx(np.var(values), rel=1e-6)


def test_constant_field_has_zero_fluctuation():
    stats = FlowStatistics(shape=(3, 3), components=2)
    for _ in range(50):
        stats.update(np.full((3, 3), 0.05), np.full((3, 3), -0.01))
    np.testing.assert_allclose(stats.reynolds_stress("uu"), 0.0, atol=1e-20)
    np.testing.assert_allclose(stats.turbulent_kinetic_energy, 0.0, atol=1e-20)


def test_tke_is_half_the_trace():
    rng = np.random.default_rng(4)
    stats = FlowStatistics(shape=(2, 2), components=3)
    for _ in range(100):
        stats.update(*(0.01 * rng.standard_normal((2, 2)) for _ in range(3)))
    expect = 0.5 * sum(stats.reynolds_stress(n) for n in ("uu", "vv", "ww"))
    np.testing.assert_allclose(stats.turbulent_kinetic_energy, expect)


def test_three_component_stress_names():
    stats = FlowStatistics(shape=(2, 2), components=3)
    assert tuple(stats.stress_names) == ("uu", "uv", "uw", "vv", "vw", "ww")


def test_start_step_gates_sampling():
    stats = FlowStatistics(shape=(2, 2), components=2, start_step=100)
    assert not stats.should_sample(99)
    assert stats.should_sample(100)


def test_summary_is_json_safe():
    import json
    rng = np.random.default_rng(5)
    stats = FlowStatistics(shape=(3, 3), components=2)
    for _ in range(20):
        stats.update(0.01 * rng.standard_normal((3, 3)), 0.01 * rng.standard_normal((3, 3)))
    json.dumps(stats.summary())              # must not raise


def test_summary_of_an_empty_accumulator():
    stats = FlowStatistics(shape=(2, 2), components=2)
    assert stats.summary()["samples"] == 0


def test_state_roundtrip():
    rng = np.random.default_rng(6)
    stats = FlowStatistics(shape=(3, 4), components=3, start_step=7)
    for _ in range(40):
        stats.update(*(0.01 * rng.standard_normal((3, 4)) for _ in range(3)),
                     rho=1.0 + 0.001 * rng.standard_normal((3, 4)))
    restored = FlowStatistics(shape=(3, 4), components=3)
    restored.load_state_dict(stats.state_dict())
    assert restored.count == stats.count
    assert restored.start_step == 7
    for name in stats.stress_names:
        np.testing.assert_allclose(
            restored.reynolds_stress(name), stats.reynolds_stress(name)
        )


def test_component_count_is_validated():
    with pytest.raises(ValueError):
        FlowStatistics(shape=(2, 2), components=4)
    stats = FlowStatistics(shape=(2, 2), components=2)
    with pytest.raises(ValueError):
        stats.update(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------

def test_solver_2d_collects_statistics():
    ny, nx = 40, 80
    solver = Solver(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx), omega=1.7, u0=0.06, D=12.0,
        backend="numpy", inlet_perturbation=0.05,
    )
    result = solver.run(steps=60, check_every=1000, verbose=False,
                        collect_statistics=True, stats_start=10)
    stats, fields = result["statistics"], result["mean_fields"]
    assert stats["samples"] == 51
    assert set(fields) >= {
        "mean_ux", "mean_uy", "rms_ux", "rms_uy",
        "reynolds_uu", "reynolds_uv", "reynolds_vv", "tke", "mean_pressure",
    }
    assert fields["mean_ux"].shape == (ny, nx)
    assert np.all(np.isfinite(fields["tke"]))
    assert np.all(fields["tke"] >= 0.0)


def test_statistics_are_off_by_default():
    ny, nx = 24, 48
    solver = Solver(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 4.0), omega=1.5, u0=0.05, D=8.0,
        backend="numpy",
    )
    result = solver.run(steps=10, check_every=100, verbose=False)
    assert result["statistics"] is None
    assert result["mean_fields"] is None


def test_solver_3d_collects_statistics():
    nz, ny, nx = 8, 10, 16
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((zz - 4) ** 2 + (yy - 5) ** 2 + (xx - 4) ** 2) <= 4
    solver = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=1.5, u0=0.05, D=4.0,
        backend="numpy",
    )
    result = solver.run(steps=20, check_every=1000, verbose=False,
                        collect_statistics=True, stats_start=5)
    fields = result["mean_fields"]
    assert result["statistics"]["samples"] == 16
    assert fields["mean_uz"].shape == (nz, ny, nx)
    assert "reynolds_uw" in fields and "reynolds_vw" in fields


def test_mean_of_a_steady_flow_equals_the_instantaneous_field():
    """With nothing unsteady happening, the average is the snapshot."""
    ny, nx = 20, 30
    solver = Solver(
        Ny=ny, Nx=nx, solid=np.zeros((ny, nx), dtype=bool), omega=1.5,
        u0=0.05, D=6.0, backend="numpy",
    )
    result = solver.run(steps=300, check_every=1000, verbose=False,
                        collect_statistics=True, stats_start=250)
    np.testing.assert_allclose(
        result["mean_fields"]["mean_ux"], result["ux"], atol=2e-6
    )
    assert float(np.max(result["mean_fields"]["tke"])) < 1e-10


def test_stats_every_thins_the_sampling():
    ny, nx = 20, 30
    solver = Solver(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 4.0), omega=1.5, u0=0.05, D=8.0,
        backend="numpy",
    )
    result = solver.run(steps=40, check_every=1000, verbose=False,
                        collect_statistics=True, stats_start=0, stats_every=4)
    assert result["statistics"]["samples"] == 10
